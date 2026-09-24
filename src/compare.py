"""Diff a run against a baseline, test whether the change is real, watch for slow drift, and decide a status."""

from collections.abc import Callable
from math import comb

from src.models import (
    CaseChange,
    CaseResult,
    Category,
    Comparison,
    Difficulty,
    DriftResult,
    GroupDelta,
    MetricDelta,
    RunRecord,
    Split,
    Status,
)

# Drops in overall pass rate, in fractions of 1 (0.03 is 3 percentage points).
DEFAULT_WARN_DELTA = 0.03
DEFAULT_FAIL_DELTA = 0.08
DEFAULT_DRIFT_FLOOR = 0.88
DRIFT_WINDOW = 7
SIGNIFICANCE_LEVEL = 0.05
# Above this share of errored cases in either run, the comparison is too incomplete to trust.
DEFAULT_MAX_ERROR_RATE = 0.05

# Deltas are differences of fractions, so 0.92 - 0.84 can come out as 0.07999999999999996.
_EPSILON = 1e-9


def mcnemar_exact(regressions: int, improvements: int) -> float:
    """Two-sided exact McNemar p value from the discordant pairs.

    If the change made no real difference, each flipped case is equally likely to go either way,
    so the smaller flip count follows Binomial(n, 0.5). The p value is the chance of a split at
    least this lopsided in either direction.
    """
    n = regressions + improvements
    if n == 0:
        return 1.0
    k = min(regressions, improvements)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def pass_rate(results: list[CaseResult]) -> float:
    return sum(r.passed for r in results) / len(results) if results else 0.0


def category_accuracy(results: list[CaseResult]) -> float:
    return sum(r.category_match for r in results) / len(results) if results else 0.0


def mean_summary_score(results: list[CaseResult]) -> float:
    scores = [r.summary_score for r in results if r.summary_score is not None]
    return sum(scores) / len(scores) if scores else 0.0


def group_deltas(
    baseline: list[CaseResult],
    run: list[CaseResult],
    key: Callable[[CaseResult], str],
    groups: list[str],
) -> list[GroupDelta]:
    deltas = []
    for group in groups:
        before = [r for r in baseline if key(r) == group]
        after = [r for r in run if key(r) == group]
        deltas.append(
            GroupDelta(
                group=group,
                baseline_n=len(before),
                baseline_rate=pass_rate(before),
                run_n=len(after),
                run_rate=pass_rate(after),
                delta=pass_rate(after) - pass_rate(before),
            )
        )
    return deltas


def compute_drift(
    run: RunRecord, history: list[RunRecord], floor: float, window: int = DRIFT_WINDOW
) -> DriftResult:
    """Moving average of pass rate over the run plus the most recent earlier main runs.

    history must already be filtered to comparable main runs and sorted newest first.
    Returns no average when there is not yet a full window, rather than judging from a few points.
    """
    runs = [run] + [r for r in history if r.run_id != run.run_id][: window - 1]
    if len(runs) < window:
        return DriftResult(
            window=window, floor=floor, run_ids=[r.run_id for r in runs], moving_average=None, below_floor=False
        )
    average = sum(r.pass_rate for r in runs) / window
    return DriftResult(
        window=window,
        floor=floor,
        run_ids=[r.run_id for r in runs],
        moving_average=average,
        below_floor=average < floor - _EPSILON,
    )


def version_mismatches(run: RunRecord, baseline: RunRecord) -> list[str]:
    fields = ["dataset_version", "judge_version", "summary_threshold"]
    return [
        f"baseline {name} is {getattr(baseline, name)} but run has {getattr(run, name)}; "
        "results may not be comparable"
        for name in fields
        if getattr(run, name) != getattr(baseline, name)
    ]


def compare_runs(
    run: RunRecord,
    baseline: RunRecord | None,
    history: list[RunRecord],
    warn_delta: float = DEFAULT_WARN_DELTA,
    fail_delta: float = DEFAULT_FAIL_DELTA,
    drift_floor: float = DEFAULT_DRIFT_FLOOR,
    max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
) -> Comparison:
    """Pure function: all run lookups happen in the caller.

    Errored cases are infrastructure failures, not answers, so they are left out of every delta and of
    McNemar. Otherwise a rate limited run would look like a quality regression. Too many errors fail
    the comparison as incomplete, with a reason that says so.
    """
    reasons: list[str] = []
    notes: list[str] = []
    status = Status.PASS

    def raise_to(level: Status) -> None:
        nonlocal status
        order = [Status.PASS, Status.WARN, Status.FAIL]
        if order.index(level) > order.index(status):
            status = level

    drift = compute_drift(run, history, drift_floor)
    if drift.below_floor:
        raise_to(Status.WARN)
        reasons.append(
            f"drift: {drift.window} run average pass rate {drift.moving_average:.1%} "
            f"is below the floor {drift.floor:.0%}"
        )

    for label, checked in [("run", run), ("baseline", baseline)]:
        if checked is None or not checked.n_errors:
            continue
        if checked.n_errors / checked.n_cases > max_error_rate + _EPSILON:
            raise_to(Status.FAIL)
            reasons.append(
                f"eval incomplete: {checked.n_errors}/{checked.n_cases} cases errored in the {label} "
                f"(limit {max_error_rate:.0%}); rerun before trusting this comparison"
            )
        else:
            raise_to(Status.WARN)
            reasons.append(f"{checked.n_errors} cases errored in the {label} and are left out of the comparison")

    if baseline is None:
        notes.append("no baseline to compare against")
        return Comparison(
            run_id=run.run_id,
            baseline_id=None,
            status=status,
            reasons=reasons,
            notes=notes,
            warn_delta=warn_delta,
            fail_delta=fail_delta,
            overall=[],
            by_category=[],
            by_difficulty=[],
            by_split=[],
            regressions=[],
            improvements=[],
            unmatched_case_ids=[],
            prediction_changes=0,
            score_changes=0,
            mcnemar_p=None,
            drift=drift,
        )

    notes.extend(version_mismatches(run, baseline))

    before_by_id = {r.case_id: r for r in baseline.results}
    after_by_id = {r.case_id: r for r in run.results}
    shared = [case_id for case_id in after_by_id if case_id in before_by_id]
    unmatched = sorted(set(before_by_id) ^ set(after_by_id))
    if unmatched:
        notes.append(f"{len(unmatched)} cases appear in only one run and are not compared case by case")
    scored = [c for c in shared if before_by_id[c].error is None and after_by_id[c].error is None]
    if len(scored) < len(shared):
        notes.append(f"compared {len(scored)} cases scored in both runs")
    before_results = [before_by_id[c] for c in scored]
    after_results = [after_by_id[c] for c in scored]

    regressions, improvements = [], []
    prediction_changes = score_changes = 0
    for case_id in scored:
        before, after = before_by_id[case_id], after_by_id[case_id]
        if before.passed and not after.passed:
            regressions.append(CaseChange(case_id=case_id, before=before, after=after))
        elif after.passed and not before.passed:
            improvements.append(CaseChange(case_id=case_id, before=before, after=after))
        prediction_changes += before.predicted_category != after.predicted_category
        score_changes += before.summary_score != after.summary_score

    overall = []
    for name, metric in [
        ("pass_rate", pass_rate),
        ("category_accuracy", category_accuracy),
        ("mean_summary_score", mean_summary_score),
    ]:
        before_value, after_value = metric(before_results), metric(after_results)
        overall.append(MetricDelta(name=name, baseline=before_value, run=after_value, delta=after_value - before_value))

    p = mcnemar_exact(len(regressions), len(improvements))
    significance = "significant" if p < SIGNIFICANCE_LEVEL else "not significant"
    flips = f"{len(regressions)} regressions, {len(improvements)} improvements, McNemar p = {p:.3g}, {significance}"
    drop = -overall[0].delta
    if drop >= fail_delta - _EPSILON:
        raise_to(Status.FAIL)
        reasons.append(f"pass rate dropped {drop:.1%} (fail at {fail_delta:.0%}); {flips}")
    elif drop >= warn_delta - _EPSILON:
        raise_to(Status.WARN)
        reasons.append(f"pass rate dropped {drop:.1%} (warn at {warn_delta:.0%}); {flips}")
    else:
        notes.append(f"pass rate change {overall[0].delta:+.1%}; {flips}")

    return Comparison(
        run_id=run.run_id,
        baseline_id=baseline.run_id,
        status=status,
        reasons=reasons,
        notes=notes,
        warn_delta=warn_delta,
        fail_delta=fail_delta,
        overall=overall,
        by_category=group_deltas(
            before_results, after_results, lambda r: r.expected_category.value, [c.value for c in Category]
        ),
        by_difficulty=group_deltas(
            before_results, after_results, lambda r: r.difficulty.value, [d.value for d in Difficulty]
        ),
        by_split=group_deltas(before_results, after_results, lambda r: r.split.value, [s.value for s in Split]),
        regressions=regressions,
        improvements=improvements,
        unmatched_case_ids=unmatched,
        prediction_changes=prediction_changes,
        score_changes=score_changes,
        mcnemar_p=p,
        drift=drift,
    )
