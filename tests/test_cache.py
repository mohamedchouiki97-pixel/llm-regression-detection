from datetime import datetime, timezone

from src.cache import Cache, make_key
from src.models import FewShotExample, PromptConfig
from src.prompts import prompt_hash


def config(**overrides) -> PromptConfig:
    data = dict(
        version="v1",
        created_at=datetime(2026, 9, 23, tzinfo=timezone.utc),
        model="gpt-4o-mini",
        system_prompt="Classify the email.",
        few_shot_examples=[FewShotExample(email="Charged twice.", category="billing", summary="Double charge.")],
    )
    data.update(overrides)
    return PromptConfig(**data)


def test_set_then_get(tmp_path):
    cache = Cache(tmp_path)
    cache.set("abc", {"x": 1})
    assert cache.get("abc") == {"x": 1}


def test_miss_returns_none(tmp_path):
    assert Cache(tmp_path).get("missing") is None


def test_disabled_cache_never_stores(tmp_path):
    cache = Cache(None)
    cache.set("abc", {"x": 1})
    assert cache.get("abc") is None


def test_corrupt_entry_is_a_miss(tmp_path):
    (tmp_path / "abc.json").write_text("{not json", encoding="utf-8")
    assert Cache(tmp_path).get("abc") is None


def test_key_is_stable():
    assert make_key("classify", "h", "m", "email") == make_key("classify", "h", "m", "email")


def test_key_changes_with_each_part():
    base = make_key("classify", "hash", "gpt-4o-mini", "email")
    assert make_key("classify", "hash", "gpt-4o-mini", "email!") != base
    assert make_key("classify", "hash", "gpt-4o", "email") != base
    assert make_key("classify", "hash2", "gpt-4o-mini", "email") != base
    assert make_key("judge", "hash", "gpt-4o-mini", "email") != base


def test_key_parts_do_not_run_together():
    assert make_key("ab", "c") != make_key("a", "bc")


def test_prompt_hash_changes_when_prompt_text_changes_without_version_bump():
    assert prompt_hash(config()) != prompt_hash(config(system_prompt="Classify the email carefully."))


def test_prompt_hash_changes_with_few_shots_and_model():
    base = prompt_hash(config())
    assert prompt_hash(config(few_shot_examples=[])) != base
    assert prompt_hash(config(model="gpt-4o")) != base


def test_prompt_hash_is_stable():
    assert prompt_hash(config()) == prompt_hash(config())
