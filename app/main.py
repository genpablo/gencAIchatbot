"""FastAPI service.   uvicorn app.main:app --host 0.0.0.0 --port 8000

  /         staff chat app (sign in with a staff account; works on phone, iPad and desktop)
  /shop     customer assistant for the online shop (no login; product cards link to the shop)
  /widget.js  chat bubble that shows /shop on every BigCommerce page (Storefront > Script Manager)
  /store    customer assistant for screens in the store (no login; product links off by default)
  /admin   admin portal (ADMIN_PASSWORD)
  /docs     API documentation (only with API_DOCS=true; off by default so the API is not advertised)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import admin, chat, storefront
from .analytics import track
from .auth import require_staff
from .auth import router as staff_router
from .chat import check_feedback_owner
from .config import settings
from .db import connect, save_feedback
from .models import FeedbackRequest, RecommendRequest
from .embeddings import embed_model_label, embed_provider
from .providers import ProviderError, get_provider
from .recommender import recommend
from .security import HOW_TO_FIX, BodyLimit, admin_password_problem, is_https, secret_key_problem

log = logging.getLogger("main")


@asynccontextmanager
async def lifespan(_app):
    """Refuse to start with a missing or example SECRET_KEY: staff sessions could be forged."""
    problem = secret_key_problem()
    if problem:
        raise RuntimeError(f"{problem} {HOW_TO_FIX}")
    if admin_password_problem():
        log.error("Admin portal disabled: %s", admin_password_problem())
    yield


DOCS = {} if settings.api_docs else {"docs_url": None, "redoc_url": None, "openapi_url": None}
app = FastAPI(title="K-Beauty Product Recommender", version="0.5.0", lifespan=lifespan, **DOCS)
app.add_middleware(BodyLimit)
provider = get_provider()
STATIC = Path(__file__).parent / "static"
admin.set_provider(provider)
chat.set_provider(provider)
storefront.set_provider(provider)
app.include_router(admin.router)
app.include_router(staff_router)
app.include_router(chat.router)
app.include_router(storefront.router)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

NO_CACHE = {"Cache-Control": "no-cache"}


# What the pages may load: their own scripts and styles (inline included), Google Fonts, photos and the logo
# from any HTTPS address, and API calls to this app only. A script someone managed to inject could not
# send data anywhere else.
PAGE_POLICY = ("default-src 'self'; script-src 'self' 'unsafe-inline'; "
               "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; "
               "img-src 'self' data: https:; connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'")
DOC_PAGES = ("/docs", "/redoc")                       # Swagger loads its scripts from a CDN (only with API_DOCS)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Only the online shop may show /shop in a frame (the chat bubble); no other site may frame any page.
    Plus the usual browser protections on every response."""
    response = await call_next(request)
    path = request.url.path
    ancestors = "'self'"
    if path == "/shop":
        ancestors += " " + " ".join(settings.shop_embed_origins.split())
    policy = f"frame-ancestors {ancestors}" if path.startswith(DOC_PAGES) else f"{PAGE_POLICY}; frame-ancestors {ancestors}"
    h = response.headers
    h.setdefault("Content-Security-Policy", policy)
    h.setdefault("X-Content-Type-Options", "nosniff")
    h.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    h.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()")
    if is_https(request):
        h.setdefault("Strict-Transport-Security", "max-age=31536000")
    return response


@app.get("/")
def chat_ui():
    return FileResponse(STATIC / "chat.html", headers=NO_CACHE)


@app.api_route("/shop", methods=["GET", "HEAD"])      # HEAD: link checkers, shop embeds, health probes
@app.api_route("/store", methods=["GET", "HEAD"])
def customer_ui():
    """Customer assistant; the page reads its channel (online or store) from the path."""
    return FileResponse(STATIC / "assistant.html", headers=NO_CACHE)


@app.get("/widget.js")
def widget():
    """Chat bubble for the online shop: BigCommerce Script Manager loads it on every storefront page."""
    return FileResponse(STATIC / "widget.js", media_type="text/javascript", headers={"Cache-Control": "public, max-age=300"})


@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    return FileResponse(STATIC / "sw.js", media_type="text/javascript", headers=NO_CACHE)


@app.get("/admin")
def admin_ui():
    return FileResponse(STATIC / "admin.html", headers=NO_CACHE)


@app.get("/health")
def health(request: Request):
    """Up and the database reachable: {"ok": true} for anyone (load balancers, uptime checks). Which AI models
    the app uses and how many products are searchable is shown only to a manager signed in to the admin portal."""
    with connect() as conn:
        n = conn.execute("SELECT count(embedded_with = %(m)s OR NULL) AS current, "
                         "count(embedded_with IS DISTINCT FROM %(m)s OR NULL) AS other "
                         "FROM products WHERE embedding IS NOT NULL", {"m": embed_model_label()}).fetchone()
    try:
        admin.require_admin(request)
    except HTTPException:
        return {"ok": True}
    return {"ok": True, "provider": settings.provider, "chat_model": provider.model_label("rank"),
            "embed_provider": embed_provider(), "embed_model": embed_model_label(),
            "embedded_products": n["current"], "other_model_products": n["other"]}


@app.post("/recommend")
def post_recommend(req: RecommendRequest, user: dict = Depends(require_staff)):
    """Structured recommendation API (the chat uses the same engine). Requires a staff login."""
    if not req.profile and not (req.text and req.text.strip()):
        raise HTTPException(400, "Send a profile, free text, or both.")
    req = req.model_copy(update={"employee_id": user["username"]})
    try:
        with connect() as conn:
            res = recommend(conn, provider, req)
            track(conn, "recommendation", user=user, intent="api", message=req.text, profile=res.get("profile"),
                  skus=[o["sku"] for st in res.get("steps", []) for o in st["options"]],
                  status=res["status"], used_llm=res.get("used_llm"), model=res.get("model"), log_id=res.get("log_id"),
                  latency_ms=res.get("elapsed_ms"))
            return res
    except ProviderError as e:
        raise HTTPException(503, str(e))


@app.post("/feedback")
def post_feedback(fb: FeedbackRequest, user: dict = Depends(require_staff)):
    with connect() as conn:
        check_feedback_owner(conn, fb.log_id, user)
        save_feedback(conn, fb.log_id, fb.feedback, fb.note, fb.sku)
        track(conn, "feedback", user=user, intent=fb.feedback, skus=[fb.sku] if fb.sku else None, log_id=fb.log_id)
    return {"ok": True}
