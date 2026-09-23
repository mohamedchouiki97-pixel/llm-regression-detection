import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.classifier import ClassificationError, classify_email
from src.models import Category, ClassificationResult, FewShotExample, PromptConfig


class FakeCompletions:
    def __init__(self, parsed=None, refusal=None, usage=None):
        self.parsed = parsed
        self.refusal = refusal
        self.usage = usage
        self.calls: list[dict] = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(parsed=self.parsed, refusal=self.refusal)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=self.usage)


def fake_client(**kwargs) -> tuple[SimpleNamespace, FakeCompletions]:
    completions = FakeCompletions(**kwargs)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


@pytest.fixture
def config() -> PromptConfig:
    return PromptConfig(
        version="v1",
        created_at=datetime(2026, 9, 23, tzinfo=timezone.utc),
        model="gpt-4o-mini",
        system_prompt="Classify the email.",
        few_shot_examples=[
            FewShotExample(email="I was charged twice.", category=Category.BILLING, summary="Double charge."),
        ],
    )


def test_classify_returns_result_and_metadata(config):
    parsed = ClassificationResult(category=Category.TECHNICAL, summary="The app crashes on launch.")
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=18)
    client, _ = fake_client(parsed=parsed, usage=usage)

    output = asyncio.run(classify_email("The app crashes.", config, client=client))

    assert output.result == parsed
    assert output.prompt_version == "v1"
    assert output.model == "gpt-4o-mini"
    assert output.input_tokens == 120
    assert output.output_tokens == 18
    assert output.latency_ms >= 0


def test_request_shape(config):
    parsed = ClassificationResult(category=Category.TECHNICAL, summary="x")
    client, completions = fake_client(parsed=parsed, usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    asyncio.run(classify_email("The app crashes.", config, client=client))

    call = completions.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0
    assert call["response_format"] is ClassificationResult

    messages = call["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == "Classify the email."
    assert messages[1]["content"] == "I was charged twice."
    assert json.loads(messages[2]["content"]) == {"category": "billing", "summary": "Double charge."}
    assert messages[3]["content"] == "The app crashes."


def test_refusal_raises(config):
    client, _ = fake_client(refusal="I can't help with that.")
    with pytest.raises(ClassificationError, match="refused"):
        asyncio.run(classify_email("...", config, client=client))


def test_missing_parsed_output_raises(config):
    client, _ = fake_client()
    with pytest.raises(ClassificationError, match="no parsed output"):
        asyncio.run(classify_email("...", config, client=client))


def test_missing_usage_defaults_to_zero(config):
    parsed = ClassificationResult(category=Category.GENERAL, summary="x")
    client, _ = fake_client(parsed=parsed, usage=None)
    output = asyncio.run(classify_email("...", config, client=client))
    assert (output.input_tokens, output.output_tokens) == (0, 0)
