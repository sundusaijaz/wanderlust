from dataclasses import replace

from fastapi.testclient import TestClient

from wanderlust.config import Settings
from wanderlust.llm.factory import build_provider
from wanderlust.web.app import create_app


def test_status_reports_the_model(web):
    status = web.get("/api/status").json()
    assert status["ready"] is True
    assert status["provider"] == "fake" and status["model"] == "scripted"
    assert status["problem"] is None and status["demo"] is False
    assert status["version"]


def test_index_and_static_files(web):
    page = web.get("/")
    assert page.status_code == 200 and "Wanderlust" in page.text


def test_status_problem_without_key(services, tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    app = create_app(settings, replace(services, provider=build_provider), demo=True)
    status = TestClient(app).get("/api/status").json()
    assert status["ready"] is False
    assert "OPENAI_API_KEY" in status["problem"]
    assert status["provider"] == "openai" and status["model"] is None
    assert status["demo"] is True
