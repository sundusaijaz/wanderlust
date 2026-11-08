import pytest
from typer.testing import CliRunner

from wanderlust import __version__
from wanderlust.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Run each test in an empty folder so a developer's .env is not picked up."""
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "WANDERLUST_LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_config_reports_missing_key():
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "openai" in result.output
    assert "missing" in result.output


def test_config_never_prints_the_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-value")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "configured" in result.output
    assert "test-secret-value" not in result.output


def test_config_reports_invalid_settings(monkeypatch):
    monkeypatch.setenv("WANDERLUST_LLM_PROVIDER", "not-a-provider")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 1
    assert "Invalid configuration" in result.output


def test_plan_is_a_placeholder_for_now():
    result = runner.invoke(app, ["plan", "4 days in Lisbon"])
    assert result.exit_code == 2
    assert "4 days in Lisbon" in result.output
