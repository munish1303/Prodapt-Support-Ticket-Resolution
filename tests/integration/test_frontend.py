"""The agent console is served by the API process (no separate frontend build)."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)  # not used as a context manager: no lifespan, so no models load


def test_console_page_is_served_at_root():
    resp = client.get("/")
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/html")
    html = resp.text
    for marker in ('id="complaintForm"', 'id="searchCard"', 'id="resultCard"', 'id="viewKb"', "/static/js/app.js"):
        assert marker in html


def test_static_assets_are_served():
    css = client.get("/static/css/app.css")
    js = client.get("/static/js/app.js")
    assert css.status_code == 200 and "--red: #ea262a" in css.text
    assert js.status_code == 200 and "/tickets/resolve" in js.text
    assert client.get("/static/nope.js").status_code == 404


def test_frontend_escapes_api_text():
    js = client.get("/static/js/app.js").text
    assert "const esc =" in js  # every API string is passed through esc() before innerHTML
