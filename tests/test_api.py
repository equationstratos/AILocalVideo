"""Tests de l'API via TestClient (backend de génération non sollicité)."""

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_styles():
    r = client.get("/api/styles")
    assert r.status_code == 200
    names = {s["name"] for s in r.json()}
    assert {"animation", "realistic"} <= names


def test_list_backends():
    r = client.get("/api/backends")
    assert r.status_code == 200
    names = {b["name"] for b in r.json()}
    assert {"animatediff", "cogvideox"} <= names


def test_generate_unknown_style():
    r = client.post("/api/generate", json={"prompt": "x", "style": "nope"})
    assert r.status_code == 404


def test_generate_returns_job_id(monkeypatch):
    # On empêche tout traitement réel : le submit ne doit pas lancer de modèle.
    captured = {}

    def fake_submit(style, params, num_segments=1):
        captured["style"] = style
        captured["params"] = params
        captured["num_segments"] = num_segments
        return "fake-job-id"

    monkeypatch.setattr("backend.main.get_job_store", lambda: type(
        "S", (), {"submit": staticmethod(fake_submit)}
    )())

    r = client.post(
        "/api/generate",
        json={
            "prompt": "a robot dancing",
            "style": "animation",
            "num_frames": 8,
            "num_segments": 4,
        },
    )
    assert r.status_code == 200
    assert r.json()["job_id"] == "fake-job-id"
    # Les défauts du style sont fusionnés et les limites appliquées.
    assert captured["style"] == "animation"
    assert captured["params"].num_frames == 8
    assert captured["num_segments"] == 4
    assert "animated style" in captured["params"].prompt


def test_job_not_found():
    r = client.get("/api/jobs/does-not-exist")
    assert r.status_code == 404
