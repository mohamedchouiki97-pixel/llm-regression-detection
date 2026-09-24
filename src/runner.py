"""Async eval runner: classify and judge every golden case with a concurrency limit and retries."""

import asyncio
import os
import random
import subprocess
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

import openai

from src.cache import Cache, make_key
from src.classifier import ClassificationError
from src.models import (
    CaseResult,
    ClassifyOutput,
    GoldenCase,
    GoldenDataset,
    JudgeOutput,
    PromptConfig,
    RunRecord,
)
from src.prompts import prompt_hash
from src.scoring import JUDGE_VERSION, JudgeError, score_case

# Temporary failures worth retrying. APITimeoutError is a subclass of APIConnectionError.
RETRYABLE_ERRORS = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)
# Failures that end one case but should not end the run. Anything else is a bug and propagates.
CASE_ERRORS = (openai.OpenAIError, ClassificationError, JudgeError)

DEFAULT_CONCURRENCY = 8
# Six retries with doubling waits span about two minutes, enough to outlast a per minute rate limit window.
MAX_RETRIES = 6
BASE_DELAY_S = 1.0
MAX_DELAY_S = 60.0


class QuotaExhaustedError(Exception):
    """The OpenAI account has no credits left. Every remaining call would fail, so the run stops."""


ClassifyFn = Callable[[str, PromptConfig], Awaitable[ClassifyOutput]]
JudgeFn = Callable[[str, str, str], Awaitable[JudgeOutput]]


def retry_after_seconds(exc: Exception) -> float | None:
    """The wait the API asked for, from the retry-after-ms or retry-after header, if present."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            return float(value) * scale
        except ValueError:
            # retry-after may also be an HTTP date; fall back to our own backoff then.
            return None
    return None


async def with_retries(
    call: Callable[[], Awaitable],
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY_S,
    sleep: Callable[[float], Awaitable] = asyncio.sleep,
):
    """Retry temporary API errors with exponential backoff and jitter (about 1s, 2s, 4s, ... capped at 60s).

    When the API says how long to wait, wait at least that long.
    """
    for attempt in range(max_retries + 1):
        try:
            return await call()
        except RETRYABLE_ERRORS as exc:
            # An empty balance also arrives as HTTP 429, but waiting cannot fix it.
            if getattr(exc, "code", None) == "insufficient_quota":
                raise QuotaExhaustedError(
                    "OpenAI account has no credits (insufficient_quota); add credits and rerun"
                ) from exc
            if attempt == max_retries:
                raise
            delay = base_delay * 2**attempt * random.uniform(0.75, 1.25)
            delay = max(delay, retry_after_seconds(exc) or 0.0)
            await sleep(min(delay, MAX_DELAY_S))


async def _cached(cache: Cache, key: str, call: Callable[[], Awaitable], model_type, sleep):
    """Return (value, from_cache). Only successful responses are cached."""
    hit = cache.get(key)
    if hit is not None:
        return model_type.model_validate(hit), True
    value = await with_retries(call, sleep=sleep)
    cache.set(key, value.model_dump(mode="json"))
    return value, False


async def run_case(
    case: GoldenCase,
    config: PromptConfig,
    classify: ClassifyFn,
    judge: JudgeFn,
    cache: Cache,
    judge_model: str,
    threshold: int,
    sleep: Callable[[float], Awaitable] = asyncio.sleep,
) -> CaseResult:
    try:
        output, cached = await _cached(
            cache,
            make_key("classify", prompt_hash(config), config.model, case.email),
            lambda: classify(case.email, config),
            ClassifyOutput,
            sleep,
        )
    except CASE_ERRORS as exc:
        return score_case(case, None, None, threshold, error=f"classify: {type(exc).__name__}: {exc}")

    try:
        verdict, _ = await _cached(
            cache,
            make_key("judge", judge_model, JUDGE_VERSION, case.email, case.ideal_summary, output.result.summary),
            lambda: judge(case.email, case.ideal_summary, output.result.summary),
            JudgeOutput,
            sleep,
        )
    except CASE_ERRORS as exc:
        return score_case(case, output, None, threshold, cached, error=f"judge: {type(exc).__name__}: {exc}")

    return score_case(case, output, verdict, threshold, cached)


async def run_eval(
    dataset: GoldenDataset,
    config: PromptConfig,
    classify: ClassifyFn,
    judge: JudgeFn,
    cache: Cache,
    judge_model: str,
    threshold: int,
    concurrency: int = DEFAULT_CONCURRENCY,
    on_progress: Callable[[int, int], None] | None = None,
    sleep: Callable[[float], Awaitable] = asyncio.sleep,
) -> list[CaseResult]:
    """Results come back in dataset order."""
    semaphore = asyncio.Semaphore(concurrency)
    total = len(dataset.cases)
    done = 0

    async def limited(case: GoldenCase) -> CaseResult:
        nonlocal done
        async with semaphore:
            result = await run_case(case, config, classify, judge, cache, judge_model, threshold, sleep)
        done += 1
        if on_progress:
            on_progress(done, total)
        return result

    return await asyncio.gather(*(limited(case) for case in dataset.cases))


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_info() -> tuple[str, str, bool]:
    """(sha, branch, dirty). CI checks out a detached commit, so GIT_SHA and GIT_BRANCH override git."""
    sha = os.environ.get("GIT_SHA") or _git("rev-parse", "HEAD") or "unknown"
    branch = os.environ.get("GIT_BRANCH") or _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    status = _git("status", "--porcelain")
    return sha, branch, bool(status)


def _sum_or_none(values: list[float | None]) -> float | None:
    """Total cost, or None if any case has an unknown price, so a partial total is never mistaken for the real one."""
    if any(v is None for v in values):
        return None
    return sum(values)


def build_run_record(
    results: list[CaseResult],
    config: PromptConfig,
    dataset: GoldenDataset,
    judge_model: str,
    threshold: int,
    git: tuple[str, str, bool],
    now: datetime | None = None,
) -> RunRecord:
    now = now or datetime.now(timezone.utc)
    sha, branch, dirty = git
    n_passed = sum(r.passed for r in results)
    return RunRecord(
        run_id=f"{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}",
        created_at=now,
        prompt_version=config.version,
        prompt_hash=prompt_hash(config),
        model=config.model,
        judge_model=judge_model,
        judge_version=JUDGE_VERSION,
        dataset_version=dataset.version,
        summary_threshold=threshold,
        git_sha=sha,
        git_branch=branch,
        git_dirty=dirty,
        n_cases=len(results),
        n_passed=n_passed,
        pass_rate=n_passed / len(results) if results else 0.0,
        n_errors=sum(r.error is not None for r in results),
        # Only calls that returned an answer have a cost.
        classifier_cost_usd=_sum_or_none([r.cost_usd for r in results if r.predicted_category is not None]),
        judge_cost_usd=_sum_or_none([r.judge_cost_usd for r in results if r.summary_score is not None]),
        n_cached=sum(r.cached for r in results),
        results=results,
    )
