import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.models import (
    Category,
    ClassificationResult,
    ClassifyOutput,
    Difficulty,
    GoldenCase,
    JudgeOutput,
    JudgeVerdict,
    Split,
)
from src.scoring import JudgeError, build_judge_messages, estimate_cost, judge_summary, score_case

CASE = GoldenCase(
    id="gc-007",
    email="I was charged twice.",
    expected_category=Category.BILLING,
    ideal_summary="The customer was double charged and wants a refund.",
    difficulty=Difficulty.MEDIUM,
    tags=["typo"],
    split=Split.TEST,
)


def classify_output(category: Category = Category.BILLING, model: str = "gpt-4o-mini") -> ClassifyOutput:
    return ClassifyOutput(
        result=ClassificationResult(category=category, summary="The customer wants a refund."),
        prompt_version="v1",
        model=model,
        latency_ms=512.0,
        input_tokens=1000,
        output_tokens=100,
    )


def judge_output(score: int, model: str = "gpt-4o") -> JudgeOutput:
    return JudgeOutput(
        verdict=JudgeVerdict(reasoning="Close enough.", score=score),
        model=model,
        input_tokens=2000,
        output_tokens=50,
    )


# category_match and the pass rule


def test_match_and_high_score_passes():
    result = score_case(CASE, classify_output(), judge_output(5), threshold=4)
    assert result.category_match
    assert result.passed
    assert result.predicted_category is Category.BILLING
    assert result.summary_score == 5
    assert result.judge_reasoning == "Close enough."


def test_wrong_category_fails_even_with_perfect_summary():
    result = score_case(CASE, classify_output(Category.ACCOUNT), judge_output(5), threshold=4)
    assert not result.category_match
    assert not result.passed
    assert result.summary_score == 5  # still judged, so the two dimensions stay independent


@pytest.mark.parametrize("score, passed", [(3, False), (4, True), (5, True)])
def test_threshold_boundary(score, passed):
    assert score_case(CASE, classify_output(), judge_output(score), threshold=4).passed is passed


def test_threshold_is_configurable():
    assert score_case(CASE, classify_output(), judge_output(3), threshold=3).passed


def test_case_metadata_copied():
    result = score_case(CASE, classify_output(), judge_output(4), cached=True)
    assert result.case_id == "gc-007"
    assert result.expected_category is Category.BILLING
    assert result.difficulty is Difficulty.MEDIUM
    assert result.tags == ["typo"]
    assert result.split is Split.TEST
    assert result.latency_ms == 512.0
    assert (result.input_tokens, result.output_tokens) == (1000, 100)
    assert result.cached


def test_classify_error_fails_with_nothing_scored():
    result = score_case(CASE, None, None, error="classify: boom")
    assert not result.passed
    assert not result.category_match
    assert result.predicted_category is None
    assert result.summary_score is None
    assert result.error == "classify: boom"


def test_judge_error_keeps_classifier_output_but_fails():
    result = score_case(CASE, classify_output(), None, error="judge: boom")
    assert result.category_match
    assert result.summary_score is None
    assert not result.passed


# Cost


def test_cost_arithmetic():
    # 1000 * 0.15 / 1M + 100 * 0.60 / 1M
    assert estimate_cost("gpt-4o-mini", 1000, 100) == pytest.approx(0.000210)
    # 2000 * 2.50 / 1M + 50 * 10.00 / 1M
    assert estimate_cost("gpt-4o", 2000, 50) == pytest.approx(0.0055)


def test_dated_snapshot_uses_base_price():
    assert estimate_cost("gpt-4o-mini-2024-07-18", 1000, 100) == estimate_cost("gpt-4o-mini", 1000, 100)
    assert estimate_cost("gpt-4o-2024-08-06", 2000, 50) == estimate_cost("gpt-4o", 2000, 50)


def test_unknown_model_has_no_cost():
    assert estimate_cost("some-new-model", 1000, 100) is None
    assert estimate_cost("gpt-4", 1000, 100) is None


def test_score_case_fills_both_costs():
    result = score_case(CASE, classify_output(), judge_output(4))
    assert result.cost_usd == pytest.approx(0.000210)
    assert result.judge_cost_usd == pytest.approx(0.0055)


# Judge


@pytest.mark.parametrize("score", [0, 6])
def test_judge_score_out_of_range_rejected(score):
    with pytest.raises(ValidationError):
        JudgeVerdict(reasoning="x", score=score)


class FakeCompletions:
    def __init__(self, parsed=None, refusal=None, usage=None):
        self.parsed, self.refusal, self.usage = parsed, refusal, usage
        self.calls: list[dict] = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(parsed=self.parsed, refusal=self.refusal)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=self.usage)


def fake_client(**kwargs):
    completions = FakeCompletions(**kwargs)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_judge_messages_include_all_three_texts():
    messages = build_judge_messages("EMAIL", "IDEAL", "CANDIDATE")
    assert messages[0]["role"] == "system"
    assert "Rubric" in messages[0]["content"]
    user = messages[1]["content"]
    assert "<email>\nEMAIL\n</email>" in user
    assert "<reference_summary>\nIDEAL\n</reference_summary>" in user
    assert "<candidate_summary>\nCANDIDATE\n</candidate_summary>" in user


def test_judge_summary_returns_verdict_and_usage():
    verdict = JudgeVerdict(reasoning="Matches.", score=5)
    usage = SimpleNamespace(prompt_tokens=300, completion_tokens=20)
    client, completions = fake_client(parsed=verdict, usage=usage)
    output = asyncio.run(judge_summary("e", "i", "c", client, model="gpt-4o"))
    assert output.verdict == verdict
    assert (output.input_tokens, output.output_tokens) == (300, 20)
    call = completions.calls[0]
    assert call["model"] == "gpt-4o"
    assert call["temperature"] == 0
    assert call["response_format"] is JudgeVerdict


def test_judge_refusal_raises():
    client, _ = fake_client(refusal="no")
    with pytest.raises(JudgeError, match="refused"):
        asyncio.run(judge_summary("e", "i", "c", client))


def test_judge_missing_output_raises():
    client, _ = fake_client()
    with pytest.raises(JudgeError, match="no parsed output"):
        asyncio.run(judge_summary("e", "i", "c", client))
