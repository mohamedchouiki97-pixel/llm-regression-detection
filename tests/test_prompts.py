from pathlib import Path

import pytest

from src.models import Category
from src.prompts import PromptLoadError, load_prompt

VALID = """\
version: {version}
created_at: 2026-09-23T00:00:00Z
model: gpt-4o-mini
system_prompt: Classify the email.
few_shot_examples:
  - email: I was charged twice.
    category: billing
    summary: The customer was double charged.
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_repo_v1_loads():
    config = load_prompt("v1")
    assert config.version == "v1"
    assert config.model == "gpt-4o-mini"
    assert config.few_shot_examples


def test_valid_file_loads(tmp_path):
    path = write(tmp_path, "v9.yaml", VALID.format(version="v9"))
    config = load_prompt(str(path))
    assert config.few_shot_examples[0].category is Category.BILLING


def test_missing_file(tmp_path):
    with pytest.raises(PromptLoadError, match="not found"):
        load_prompt(str(tmp_path / "nope.yaml"))


def test_malformed_yaml(tmp_path):
    path = write(tmp_path, "v9.yaml", "version: v9\nsystem_prompt: [unclosed\n")
    with pytest.raises(PromptLoadError, match="invalid YAML"):
        load_prompt(str(path))


def test_non_mapping_yaml(tmp_path):
    path = write(tmp_path, "v9.yaml", "- just\n- a list\n")
    with pytest.raises(PromptLoadError, match="expected a mapping"):
        load_prompt(str(path))


def test_missing_field_names_the_field(tmp_path):
    text = VALID.format(version="v9").replace("model: gpt-4o-mini\n", "")
    path = write(tmp_path, "v9.yaml", text)
    with pytest.raises(PromptLoadError, match=r"model: Field required"):
        load_prompt(str(path))


def test_bad_category_names_the_location(tmp_path):
    text = VALID.format(version="v9").replace("category: billing", "category: refunds")
    path = write(tmp_path, "v9.yaml", text)
    with pytest.raises(PromptLoadError, match=r"few_shot_examples\.0\.category"):
        load_prompt(str(path))


def test_unknown_key_rejected(tmp_path):
    text = VALID.format(version="v9") + "temprature: 0\n"
    path = write(tmp_path, "v9.yaml", text)
    with pytest.raises(PromptLoadError, match="temprature"):
        load_prompt(str(path))


def test_version_must_match_filename(tmp_path):
    path = write(tmp_path, "v9.yaml", VALID.format(version="v8"))
    with pytest.raises(PromptLoadError, match="does not match filename"):
        load_prompt(str(path))
