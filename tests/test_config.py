import pytest

from src.cli import env_setting
from src.prompts import PromptLoadError, active_prompt_version, load_prompt


def test_env_setting_default_when_unset(monkeypatch):
    monkeypatch.delenv("MRD_WARN_DELTA", raising=False)
    assert env_setting("MRD_WARN_DELTA", 0.03, float) == 0.03


def test_env_setting_blank_uses_default(monkeypatch):
    monkeypatch.setenv("MRD_WARN_DELTA", "  ")
    assert env_setting("MRD_WARN_DELTA", 0.03, float) == 0.03


def test_env_setting_casts(monkeypatch):
    monkeypatch.setenv("MRD_WARN_DELTA", "0.05")
    monkeypatch.setenv("MRD_CONCURRENCY", "4")
    assert env_setting("MRD_WARN_DELTA", 0.03, float) == 0.05
    assert env_setting("MRD_CONCURRENCY", 8, int) == 4


def test_env_setting_bad_value_names_the_variable(monkeypatch):
    monkeypatch.setenv("MRD_CONCURRENCY", "eight")
    with pytest.raises(SystemExit, match="MRD_CONCURRENCY='eight' is not a valid int"):
        env_setting("MRD_CONCURRENCY", 8, int)


def test_repo_active_prompt_loads():
    # Shipping a new prompt changes ACTIVE, so check that it points at a valid prompt, not at a fixed version.
    assert load_prompt(active_prompt_version()).version == active_prompt_version()


def test_active_prompt_strips_whitespace(tmp_path):
    (tmp_path / "ACTIVE").write_text("  v2\n", encoding="utf-8")
    assert active_prompt_version(tmp_path) == "v2"


def test_active_prompt_missing_or_empty(tmp_path):
    with pytest.raises(PromptLoadError, match="not found"):
        active_prompt_version(tmp_path)
    (tmp_path / "ACTIVE").write_text("\n", encoding="utf-8")
    with pytest.raises(PromptLoadError, match="is empty"):
        active_prompt_version(tmp_path)
