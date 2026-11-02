"""Customer assistant (/shop online, /store in store): settings, product links, and the public chat end-to-end
(the end-to-end tests need TEST_DATABASE_URL and wipe that DB)."""
import os

import pytest

from app import storefront
from app.importer import read_table
from app.storefront import AssistantSettings, customer_card, product_link, voice
from tests.test_importer import xlsx

DB = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
CARD = {"sku": "SR-004", "name": "Cica Repair Ampoule", "category": "serum", "price": 24.0,
        "notes": ["Ingredient list not on file: check the label", "Expiry 2027-01-31"], "product_url": None}


def test_links_follow_the_channel_setting():
    s = AssistantSettings()
    assert s.links("online") and not s.links("store")
    on_file = {**CARD, "product_url": "https://gencbeauty.com/cica-repair-ampoule/"}
    assert product_link(on_file, s, "online") == "https://gencbeauty.com/cica-repair-ampoule/"
    assert product_link(on_file, s, "store") is None
    assert product_link(CARD, s, "online") == "https://gencbeauty.com/search.php?search_query=Cica+Repair+Ampoule"
    assert product_link(CARD, s.model_copy(update={"search_fallback": False}), "online") is None
    assert product_link(on_file, s.model_copy(update={"store_links": True}), "store") == on_file["product_url"]


def test_customer_card_hides_staff_details():
    c = customer_card({**CARD, "product_url": "https://gencbeauty.com/x/"}, AssistantSettings(), "online")
    assert "product_url" not in c and c["url"] == "https://gencbeauty.com/x/"
    assert c["notes"] == ["Ingredient list not on file: check the label"]


def test_shop_url_is_checked_and_trimmed():
    assert AssistantSettings(shop_url="https://gencbeauty.com/").shop_url == "https://gencbeauty.com"
    with pytest.raises(ValueError):
        AssistantSettings(shop_url="javascript:alert(1)")


def test_chips_are_editable_and_checked():
    from app.models import CustomerProfile
    s = AssistantSettings()
    assert s.starters == storefront.STARTERS and s.follow_ups == storefront.FOLLOW_UPS   # defaults = the old chips
    s = AssistantSettings(starters=["  Dry   skin ", "dry skin", "", "Start over", "Acne help"],
                          follow_ups=["Show me something cheaper", "Only for sensitive skin", "Any toners?"])
    assert s.starters == ["Dry skin", "Acne help"]                 # trimmed, de-duplicated, Start over is built in
    new, talked = CustomerProfile(), CustomerProfile(skin_type="dry", sensitive=True)
    assert storefront.customer_suggestions(new, False, s) == ["Dry skin", "Acne help"]
    assert storefront.customer_suggestions(talked, True, s) == ["Show me something cheaper", "Any toners?", "Start over"]
    assert storefront.customer_suggestions(talked, False, s) == ["Any toners?", "Start over"]   # nothing to be cheaper than
    with pytest.raises(ValueError):
        AssistantSettings(starters=["x" * 61])
    with pytest.raises(ValueError):
        AssistantSettings(follow_ups=[f"chip {i}" for i in range(9)])


def test_customer_prompts_format():
    v = voice("store")
    assert "Gen C Beauty" in v["open"] and '{"answer": "...", "recommend"' in v["open"]
    assert '{"answer": "..."}' in v["qa"] and "in store" in v["unknown"]


def test_import_reads_product_urls():
    p = read_table("p.xlsx", xlsx([("A-1", "Serum", 10, "https://gencbeauty.com/serum/"), ("A-2", "Toner", 9, "gencbeauty.com/toner")],
                                  headers=("SKU", "Name", "Price", "Product URL")))
    assert [r["url"] for r in p["rows"]] == ["https://gencbeauty.com/serum/", None]
    assert any("Product URL" in i["message"] for i in p["issues"])


def test_import_reads_image_urls():
    img = "https://cdn11.bigcommerce.com/s-x/products/1/images/2/serum.jpg"
    p = read_table("p.xlsx", xlsx([("A-1", "Serum", 10, img), ("A-2", "Toner", 9, "serum.jpg"), ("A-3", "Cream", 8, "")],
                                  headers=("SKU", "Name", "Price", "Image URL")))
    assert [r["image"] for r in p["rows"]] == [img, None, None]
    assert [i["message"] for i in p["issues"] if "Image URL" in i["message"]] and len(p["issues"]) == 1


def test_rate_limit(monkeypatch):
    import dataclasses
    monkeypatch.setattr(storefront, "settings", dataclasses.replace(storefront.settings, public_chat_per_minute=2))
    storefront._hits.pop("rate-test", None)
    storefront.check_rate("rate-test")
    storefront.check_rate("rate-test")
    with pytest.raises(Exception) as e:
        storefront.check_rate("rate-test")
    assert e.value.status_code == 429


# ------------------------------------------------------------------ end-to-end
@pytest.fixture(scope="module")
def client():
    if not os.environ.get("TEST_DATABASE_URL"):
        pytest.skip("TEST_DATABASE_URL not set")
    from fastapi.testclient import TestClient
    from app import db, ingest
    from app.main import app
    db.run_sql_file("schema.sql")
    db.run_sql_file("seed_sample.sql")
    with db.connect() as c:
        c.execute("DELETE FROM app_settings WHERE key = 'customer_assistant'")
        c.execute("UPDATE products SET product_url = 'https://gencbeauty.com/cica-repair-ampoule/' WHERE sku = 'SR-004'")
        c.execute("UPDATE products SET image_url = 'https://cdn11.bigcommerce.com/s-x/products/4/images/9/cica.jpg' WHERE sku = 'SR-004'")
    ingest.embed_products(reembed=True)
    storefront._hits.clear()
    return TestClient(app)


def ask(client, text, channel="online", conv=None):
    storefront._hits.clear()                     # the tests send faster than a customer may
    r = client.post("/api/public/chat", json={"channel": channel, "message": text, "conversation": conv})
    assert r.status_code == 200, r.text
    return r.json()["conversation"], r.json()["message"]


@DB
def test_online_customer_gets_answers_and_product_links(client):
    assert client.get("/shop").status_code == 200 and client.get("/store").status_code == 200
    assert client.head("/shop").status_code == 200 and client.head("/store").status_code == 200
    cfg = client.get("/api/public/config?channel=online").json()
    assert cfg["enabled"] and cfg["links"] and cfg["suggestions"]
    conv, m = ask(client, "hello")
    assert "assistant" in m["content"] and m["payload"]["suggestions"]
    conv, m = ask(client, "I have dry skin with fine lines, what would you suggest?", conv=conv)
    p = m["payload"]
    assert p["type"] == "products" and 1 <= len(p["products"]) <= 3 and "steps" not in p   # never a routine
    assert all(c["url"].startswith("https://gencbeauty.com/") for c in p["products"])
    conv, m = ask(client, "Is the Cica Repair Ampoule good for redness?", conv=conv)
    cica = next(c for c in m["payload"]["products"] if c["sku"] == "SR-004")
    assert cica["url"] == "https://gencbeauty.com/cica-repair-ampoule/"
    assert cica["image_url"] == "https://cdn11.bigcommerce.com/s-x/products/4/images/9/cica.jpg"
    assert all(c["image_url"] is None for c in m["payload"]["products"] if c["sku"] != "SR-004")   # no photo on file
    # the conversation remembers the customer: pregnancy-unsafe products are never suggested afterwards
    conv, m = ask(client, "I'm pregnant. Which serum would help with fine lines?", conv=conv)
    assert "SR-003" not in [c["sku"] for c in m["payload"]["products"]]


@DB
def test_store_channel_has_no_links_and_settings_apply(client):
    conv, m = ask(client, "Which products help with breakouts?", channel="store")
    assert m["payload"]["products"] and all(c["url"] is None for c in m["payload"]["products"])
    assert m["payload"]["links"] is False
    # a token from one channel does not open a conversation in the other
    other, _ = ask(client, "Which products help with breakouts?", channel="online", conv=conv)
    assert other != conv

    client.post("/admin/api/login", json={"name": "Tester", "password": "test-pass"})
    d = client.get("/admin/api/assistant").json()
    assert d["settings"]["online_links"] is True and d["products"]["with_url"] >= 1
    new = {**d["settings"], "online_links": False, "store_enabled": False}
    assert client.put("/admin/api/assistant", json=new).status_code == 200
    _, m = ask(client, "Which products help with breakouts?")
    assert all(c["url"] is None for c in m["payload"]["products"])
    assert client.get("/api/public/config?channel=store").json()["enabled"] is False
    r = client.post("/api/public/chat", json={"channel": "store", "message": "hi"})
    assert r.status_code == 404
    client.put("/admin/api/assistant", json=d["settings"])


@DB
def test_chips_set_in_the_admin_portal_reach_customers(client):
    client.post("/admin/api/login", json={"name": "Tester", "password": "test-pass"})
    d = client.get("/admin/api/assistant").json()
    new = {**d["settings"], "starters": ["Help with acne", "Gift ideas"], "follow_ups": ["Any cheaper option?", "Show toners"]}
    assert client.put("/admin/api/assistant", json=new).status_code == 200
    assert client.get("/api/public/config?channel=store").json()["suggestions"] == ["Help with acne", "Gift ideas"]
    _, m = ask(client, "I have oily skin with breakouts, what do you suggest?")
    assert m["payload"]["suggestions"] == ["Any cheaper option?", "Show toners", "Start over"]
    bad = client.put("/admin/api/assistant", json={**new, "starters": ["x" * 80]})
    assert bad.status_code == 422
    client.put("/admin/api/assistant", json=d["settings"])


@DB
def test_only_the_shop_may_frame_the_online_assistant(client):
    shop = client.get("/shop").headers["content-security-policy"]
    assert shop.startswith("frame-ancestors 'self' ") and "https://gencbeauty.com" in shop and "mybigcommerce.com" in shop
    for path in ("/", "/admin", "/store"):
        assert client.get(path).headers["content-security-policy"] == "frame-ancestors 'self'"
    w = client.get("/widget.js")
    assert w.status_code == 200 and "javascript" in w.headers["content-type"] and "/shop?embed=1" in w.text


@DB
def test_history_follows_the_customer_across_shop_pages(client):
    conv, m = ask(client, "Which products help with breakouts?")
    h = client.post("/api/public/history", json={"channel": "online", "conversation": conv})
    assert h.status_code == 200
    msgs = h.json()["messages"]
    assert [x["role"] for x in msgs] == ["user", "assistant"] and msgs[1]["payload"]["products"][0]["url"]
    assert client.post("/api/public/history", json={"channel": "store", "conversation": conv}).status_code == 404
    assert client.post("/api/public/history", json={"channel": "online", "conversation": "nope"}).status_code == 404


@DB
def test_customer_chats_stay_out_of_staff_history(client):
    from tests.conftest import staff_login
    ask(client, "What does niacinamide do?")
    staff_login(client, "store-staff", "store-pass-123")
    titles = [c["title"] for c in client.get("/api/conversations").json()["conversations"]]
    assert "What does niacinamide do?" not in titles
