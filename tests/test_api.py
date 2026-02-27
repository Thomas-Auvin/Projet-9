from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main_mod


class FakeEngine:
    def ask(self, question: str, k: int | None = None):
        kk = 6 if k is None else k
        return {
            "turn_id": "testturn123",
            "question": question,
            "answer": "ok",
            "sources": [
                {
                    "id": "S1",
                    "title": "Fake event",
                    "url": "https://example.com",
                    "city": "Bordeaux",
                    "start": "2026-01-01T00:00:00Z",
                    "end": "2026-01-01T23:59:59Z",
                }
            ],
            "model": "fake-model",
            "k": kk,
            "rating": None,
        }

    def get_history(self):
        return []

    def rate(self, turn_id: str, vote: int) -> bool:
        return True


def test_health():
    client = TestClient(main_mod.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_with_fake_engine(monkeypatch):
    # inject de manière propre (auto-restore après le test)
    monkeypatch.setattr(main_mod, "ENGINE", FakeEngine())

    client = TestClient(main_mod.app)
    r = client.post("/ask", json={"question": "test", "k": 3})
    assert r.status_code == 200

    data = r.json()
    assert "turn_id" in data
    assert data["answer"] == "ok"
    assert data["k"] == 3
    assert len(data["sources"]) == 1
