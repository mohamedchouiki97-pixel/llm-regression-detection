import asyncio
from datetime import datetime, timezone

import httpx2
import openai
import pytest

from src.cache import Cache
from src.classifier import ClassificationError
from src.models import (
    Category,
    ClassificationResult,
    ClassifyOutput,
    GoldenDataset,
    JudgeOutput,
    JudgeVerdict,
    PromptConfig,
)
from src.runner import build_run_record, run_eval, with_retries

REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")


def rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError("slow down", response=httpx2.Response(429, request=REQUEST), body=None)


def bad_request_error() -> openai.BadRequestError:
    return openai.BadRequestError("bad", response=httpx2.Response(400, request=REQUEST), body=None)


class FakeSleep:
    def __init__(self):
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def flaky(failures: list[Exception], value="ok"):
    """An async call that raises each error in turn, then returns value."""
    state = {"calls": 0}

    async def call():
        state["calls"] += 1
        if failures:
            raise failures.pop(0)
        return value

    return call, state


# Retries


def test_retry_then_success():
    call, state = flaky([rate_limit_error(), openai.APIConnectionError(request=REQUEST)])
    sleep = FakeSleep()
    assert asyncio.run(with_retries(call, sleep=sleep)) == "ok"
    assert state["calls"] == 3
    assert len(sleep.delays) == 2


def test_default_retries_outlast_a_rate_limit_window():
    call, state = flaky([rate_limit_error() for _ in range(20)])
    sleep = FakeSleep()
    with pytest.raises(openai.RateLimitError):
        asyncio.run(with_retries(call, sleep=sleep))
    assert state["calls"] == 7  # first try plus 6 retries
    # 1 + 2 + 4 + 8 + 16 + 32 = 63s nominal; jitter can shorten each wait to 75%.
    assert sum(sleep.delays) >= 0.75 * 63


def test_retry_after_header_is_honored():
    response = httpx2.Response(429, request=REQUEST, headers={"retry-after-ms": "7000"})
    call, _ = flaky([openai.RateLimitError("slow down", response=response, body=None)])
    sleep = FakeSleep()
    asyncio.run(with_retries(call, base_delay=1.0, sleep=sleep))
    assert sleep.delays == [7.0]


def test_retry_after_seconds_header_and_cap():
    response = httpx2.Response(429, request=REQUEST, headers={"retry-after": "300"})
    call, _ = flaky([openai.RateLimitError("slow down", response=response, body=None)])
    sleep = FakeSleep()
    asyncio.run(with_retries(call, sleep=sleep))
    assert sleep.delays == [60.0]


def test_backoff_is_capped():
    call, _ = flaky([rate_limit_error() for _ in range(8)])
    sleep = FakeSleep()
    asyncio.run(with_retries(call, max_retries=8, base_delay=1.0, sleep=sleep))
    assert max(sleep.delays) == 60.0


def test_gives_up_after_max_retries():
    call, state = flaky([rate_limit_error() for _ in range(10)])
    sleep = FakeSleep()
    with pytest.raises(openai.RateLimitError):
        asyncio.run(with_retries(call, max_retries=4, sleep=sleep))
    assert state["calls"] == 5  # first try plus 4 retries
    assert len(sleep.delays) == 4


def test_backoff_doubles_with_jitter():
    call, _ = flaky([rate_limit_error() for _ in range(4)])
    sleep = FakeSleep()
    asyncio.run(with_retries(call, base_delay=1.0, sleep=sleep))
    for delay, expected in zip(sleep.delays, [1, 2, 4, 8]):
        assert expected * 0.75 <= delay <= expected * 1.25


@pytest.mark.parametrize(
    "error",
    [bad_request_error(), ClassificationError("model refused")],
)
def test_permanent_errors_not_retried(error):
    call, state = flaky([error])
    sleep = FakeSleep()
    with pytest.raises(type(error)):
        asyncio.run(with_retries(call, sleep=sleep))
    assert state["calls"] == 1
    assert sleep.delays == []


# run_eval


CONFIG = PromptConfig(
    version="v1",
    created_at=datetime(2026, 9, 23, tzinfo=timezone.utc),
    model="gpt-4o-mini",
    system_prompt="Classify.",
)


def dataset(n: int = 6) -> GoldenDataset:
    categories = list(Category)
    return GoldenDataset.model_validate(
        {
            "version": 3,
            "changelog": [{"version": 3, "date": "2026-09-23", "changes": "x"}],
            "cases": [
                {
                    "id": f"gc-{i:03d}",
                    "email": f"email {i}",
                    "expected_category": categories[i % 4].value,
                    "ideal_summary": f"ideal {i}",
                    "difficulty": "easy",
                    "split": "dev",
                }
                for i in range(1, n + 1)
            ],
        }
    )


class FakeClassifier:
    """Answers each email with its expected category unless told otherwise. Tracks concurrency."""

    def __init__(self, data: GoldenDataset, wrong: set[str] = frozenset(), errors: dict | None = None):
        self.expected = {c.email: c.expected_category for c in data.cases}
        self.wrong = wrong
        self.errors = errors or {}
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def __call__(self, email: str, config: PromptConfig) -> ClassifyOutput:
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        if email in self.errors:
            raise self.errors[email]
        category = self.expected[email]
        if email in self.wrong:
            category = Category.GENERAL if category is not Category.GENERAL else Category.BILLING
        return ClassifyOutput(
            result=ClassificationResult(category=category, summary=f"summary of {email}"),
            prompt_version=config.version,
            model=config.model,
            latency_ms=100.0,
            input_tokens=500,
            output_tokens=40,
        )


class FakeJudge:
    def __init__(self, score: int = 5, low: set[str] = frozenset()):
        self.score = score
        self.low = low
        self.calls = 0

    async def __call__(self, email: str, ideal: str, summary: str) -> JudgeOutput:
        self.calls += 1
        score = 2 if email in self.low else self.score
        return JudgeOutput(
            verdict=JudgeVerdict(reasoning="fake", score=score), model="gpt-4o", input_tokens=800, output_tokens=30
        )


def run(data, classifier, judge, cache=None, concurrency=8):
    return asyncio.run(
        run_eval(
            data,
            CONFIG,
            classifier,
            judge,
            cache or Cache(None),
            judge_model="gpt-4o",
            threshold=4,
            concurrency=concurrency,
            sleep=FakeSleep(),
        )
    )


def test_all_pass_in_dataset_order():
    data = dataset()
    results = run(data, FakeClassifier(data), FakeJudge())
    assert [r.case_id for r in results] == [c.id for c in data.cases]
    assert all(r.passed for r in results)


def test_concurrency_limit_respected():
    data = dataset(20)
    classifier = FakeClassifier(data)
    run(data, classifier, FakeJudge(), concurrency=3)
    assert classifier.max_in_flight == 3


def test_wrong_category_and_low_summary_fail():
    data = dataset()
    results = run(data, FakeClassifier(data, wrong={"email 2"}), FakeJudge(low={"email 3"}))
    by_id = {r.case_id: r for r in results}
    assert not by_id["gc-002"].passed and not by_id["gc-002"].category_match
    assert not by_id["gc-003"].passed and by_id["gc-003"].summary_score == 2
    assert sum(r.passed for r in results) == 4


def test_one_case_error_does_not_stop_the_run():
    data = dataset()
    classifier = FakeClassifier(data, errors={"email 4": bad_request_error()})
    judge = FakeJudge()
    results = run(data, classifier, judge)
    failed = [r for r in results if r.error]
    assert [r.case_id for r in failed] == ["gc-004"]
    assert failed[0].error.startswith("classify: BadRequestError")
    assert sum(r.passed for r in results) == 5
    assert judge.calls == 5  # no judge call for the failed case


def test_judge_error_recorded():
    data = dataset(2)

    async def broken_judge(email, ideal, summary):
        raise ClassificationError("judge exploded")

    results = run(data, FakeClassifier(data), broken_judge)
    assert all(r.error.startswith("judge: ") for r in results)
    assert all(r.category_match and not r.passed for r in results)


def test_bug_in_code_propagates():
    data = dataset(2)

    async def buggy(email, config):
        raise ZeroDivisionError

    with pytest.raises(ZeroDivisionError):
        run(data, buggy, FakeJudge())


def test_second_run_uses_cache(tmp_path):
    data = dataset()
    cache = Cache(tmp_path)
    first_classifier, first_judge = FakeClassifier(data), FakeJudge()
    first = run(data, first_classifier, first_judge, cache=cache)
    assert not any(r.cached for r in first)

    second_classifier, second_judge = FakeClassifier(data), FakeJudge()
    second = run(data, second_classifier, second_judge, cache=cache)
    assert second_classifier.calls == 0
    assert second_judge.calls == 0
    assert all(r.cached for r in second)
    assert [r.passed for r in second] == [r.passed for r in first]


def test_errors_are_not_cached(tmp_path):
    data = dataset(1)
    cache = Cache(tmp_path)
    run(data, FakeClassifier(data, errors={"email 1": bad_request_error()}), FakeJudge(), cache=cache)
    classifier = FakeClassifier(data)
    results = run(data, classifier, FakeJudge(), cache=cache)
    assert classifier.calls == 1
    assert results[0].passed


# Run record


def test_run_record_totals():
    data = dataset(4)
    classifier = FakeClassifier(data, wrong={"email 1"}, errors={"email 2": bad_request_error()})
    results = run(data, classifier, FakeJudge())
    now = datetime(2026, 9, 23, 14, 2, 11, tzinfo=timezone.utc)
    record = build_run_record(results, CONFIG, data, "gpt-4o", 4, ("abc123", "main", False), now=now)

    assert record.run_id.startswith("20260923T140211-")
    assert record.dataset_version == 3
    assert (record.n_cases, record.n_passed, record.n_errors) == (4, 2, 1)
    assert record.pass_rate == 0.5
    assert (record.git_sha, record.git_branch, record.git_dirty) == ("abc123", "main", False)
    # 3 classifier answers: 500 * 0.15 / 1M + 40 * 0.60 / 1M each
    assert record.classifier_cost_usd == pytest.approx(3 * 0.000099)
    # 3 judge calls: 800 * 2.50 / 1M + 30 * 10 / 1M each
    assert record.judge_cost_usd == pytest.approx(3 * 0.0023)


def test_unknown_model_price_gives_unknown_total():
    data = dataset(2)
    results = run(data, FakeClassifier(data), FakeJudge())
    results[0].cost_usd = None
    record = build_run_record(results, CONFIG, data, "gpt-4o", 4, ("a", "b", True))
    assert record.classifier_cost_usd is None
