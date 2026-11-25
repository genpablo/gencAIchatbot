"""Protection for an app that faces the internet.

  client_ip       who the visitor is, behind the hosting platform's proxy (never trusting what the visitor sends)
  take / failures rate limits kept in PostgreSQL, so they hold across restarts, workers and replicas
  BodyLimit       refuses oversized requests before they are read into memory
  signing_key     the key that signs staff sessions; the app will not run with an empty or example SECRET_KEY
"""
from __future__ import annotations

import hashlib
import hmac
import json
import random
import time
from datetime import datetime, timezone

from .config import settings

# Values from .env.example and the docs: anyone can read them in the public repository
EXAMPLE_SECRETS = {"replace-with-a-long-random-string", "change-me", "changeme", "secret", "password"}
MIN_SECRET_KEY = 32
MIN_ADMIN_PASSWORD = 12
PUBLIC_BODY_BYTES = 16 * 1024          # customer assistant requests: a message is at most 1000 characters


# ------------------------------------------------------------------ who the visitor is
def client_ip(request) -> str:
    """The visitor's address. Behind TRUSTED_PROXY_HOPS proxies, each proxy appends the address it received
    the request from to X-Forwarded-For, so the visitor is that many entries from the END of the list.
    Entries further left were written by the visitor and can be anything, so they are never used."""
    hops = settings.trusted_proxy_hops
    if hops > 0:
        chain = [h.strip() for v in request.headers.getlist("x-forwarded-for") for h in v.split(",") if h.strip()]
        if chain:
            return chain[-hops] if len(chain) >= hops else chain[0]
    return request.client.host if request.client else "?"


def is_https(request) -> bool:
    """Whether the visitor's connection is HTTPS (the platform proxy says so in X-Forwarded-Proto), so session
    cookies can be marked Secure. A visitor faking the header only affects their own cookie."""
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"


# ------------------------------------------------------------------ rate limits (PostgreSQL)
def _window(window_s: int) -> datetime:
    return datetime.fromtimestamp(int(time.time()) // window_s * window_s, tz=timezone.utc)


def take(conn, key: str, window_s: int, limit: int) -> bool:
    """Count one use of `key` in the current window of `window_s` seconds. False when that is over `limit`."""
    row = conn.execute(
        """INSERT INTO rate_limits (key, window_start, hits) VALUES (%s, %s, 1)
           ON CONFLICT (key, window_start) DO UPDATE SET hits = rate_limits.hits + 1 RETURNING hits""",
        (key[:200], _window(window_s))).fetchone()
    if random.random() < 0.01:                     # now and then, forget windows that have long ended
        conn.execute("DELETE FROM rate_limits WHERE window_start < now() - interval '2 days'")
    return row["hits"] <= limit


def failures(conn, key: str, window_s: int) -> int:
    """How often `key` was counted in the current window, without counting this call."""
    row = conn.execute("SELECT hits FROM rate_limits WHERE key = %s AND window_start = %s",
                       (key[:200], _window(window_s))).fetchone()
    return row["hits"] if row else 0


# ------------------------------------------------------------------ request size
class BodyLimit:
    """Refuses a request whose body is too large before anything reads it (413), and a body of unknown
    length (411): an attacker could otherwise send hundreds of MB that the app would load into memory
    before checking any field. Customer assistant requests may be 16 KB; the rest MAX_BODY_MB (Excel import)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] in ("POST", "PUT", "PATCH"):
            headers = dict(scope["headers"])
            limit = PUBLIC_BODY_BYTES if scope["path"].startswith("/api/public/") else settings.max_body_mb * 1024 * 1024
            length = headers.get(b"content-length")
            if length is None and b"chunked" in headers.get(b"transfer-encoding", b""):
                return await _refuse(send, 411, "Please send the request with a Content-Length.")
            try:
                too_big = length is not None and int(length) > limit
            except ValueError:
                return await _refuse(send, 400, "Invalid Content-Length.")
            if too_big:
                return await _refuse(send, 413, "This request is too large.")
        await self.app(scope, receive, send)


async def _refuse(send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


# ------------------------------------------------------------------ secrets
def secret_key_problem() -> str | None:
    k = settings.secret_key.strip()
    if not k:
        return "SECRET_KEY is not set."
    if k.lower() in EXAMPLE_SECRETS or len(k) < MIN_SECRET_KEY:
        return f"SECRET_KEY is the example value or shorter than {MIN_SECRET_KEY} characters."
    return None


def admin_password_problem() -> str | None:
    p = settings.admin_password
    if p and (p.strip().lower() in EXAMPLE_SECRETS or len(p) < MIN_ADMIN_PASSWORD):
        return f"ADMIN_PASSWORD is the example value or shorter than {MIN_ADMIN_PASSWORD} characters."
    return None


HOW_TO_FIX = ('Generate one with:  python -c "import secrets; print(secrets.token_hex(32))"  '
              "and put it in .env (or the hosting platform's secret settings), then restart.")


def signing_key() -> bytes:
    """Signs staff session cookies. There is deliberately no fallback: a missing key used to fall back to a
    value anyone could compute, which let anyone forge a staff login."""
    problem = secret_key_problem()
    if problem:
        raise RuntimeError(f"{problem} {HOW_TO_FIX}")
    return hashlib.sha256(settings.secret_key.encode()).digest()


def fingerprint(value: str) -> str:
    """A keyed fingerprint (not the value itself) to notice later that a setting such as ADMIN_PASSWORD changed."""
    return hmac.new(signing_key(), value.encode(), hashlib.sha256).hexdigest()[:32]
