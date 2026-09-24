from datetime import datetime, timedelta, timezone

import pytest

from src.compare import compare_runs, compute_drift, mcnemar_exact
from src.models import CaseResult, Category, Difficulty, RunRecord, Split, Status

T0 = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
CATEGORIES = list(Category)


def result(
    n: int,
    passed: bool,
    category: Category | None = None,
    difficulty: Difficulty = Difficulty.EASY,
    split: Split = Split.DEV,
    predicted: Category | None = None,
    score: int | None = 5,
    error: str | None = None,
) -> CaseResult:
    category = category or CATEGORIES[n % 4]
    return CaseResult(
        case_id=f"gc-{n:03d}",
        expected_category=category,
        predicted_category=predicted or category,
        category_match=(predicted or category) == category,
        summary_score=score,
        passed=passed,
        error=error,
        difficulty=difficulty,
        split=split,
    )


def make_run(run_id: str, results: list[CaseResult], minutes: int = 0, **overrides) -> RunRecord:
    n_passed = sum(r.passed for r in results)
    data = dict(
        run_id=run_id,
        created_at=T0 + timedelta(minutes=minutes),
        prompt_version="v1",
        prompt_hash="h",
        model="gpt-4o-mini",
        judge_model="gpt-4o",
        judge_version="j1",
        dataset_version=3,
        summary_threshold=4,
        git_sha="abc",
        git_branch="main",
        git_dirty=False,
        n_cases=len(results),
        n_passed=n_passed,
        pass_rate=n_passed / len(results),
        n_errors=sum(r.error is not None for r in results),
        classifier_cost_usd=0.0,
        judge_cost_usd=0.0,
        n_cached=0,
        results=results,
    )
    data.update(overrides)
    return RunRecord(**data)


def run_with_failures(run_id: str, failing: set[int], minutes: int = 0, n: int = 100, **overrides) -> RunRecord:
    return make_run(run_id, [result(i, i not in failing) for i in range(1, n + 1)], minutes, **overrides)


def run_with_rate(run_id: str, rate: float, minutes: int = 0) -> RunRecord:
    """A 100 case run where the first (1 - rate) * 100 cases fail."""
    return run_with_failures(run_id, set(range(1, round((1 - rate) * 100) + 1)), minutes)


BASE_FAILS = set(range(1, 9))  # baseline passes 92/100, like the real v1 run


# McNemar


@pytest.mark.parametrize(
    "b, c, expected",
    [
        (8, 0, 0.0078125),  # 2 * 0.5**8
        (10, 2, 2 * 79 / 4096),  # 2 * (C(12,0) + C(12,1) + C(12,2)) / 2**12 = 0.03857...
        (5, 5, 1.0),
        (0, 0, 1.0),
        (1, 0, 1.0),  # a single flip can never be significant
        (3, 0, 0.25),
    ],
)
def test_mcnemar_known_values(b, c, expected):
    assert mcnemar_exact(b, c) == pytest.approx(expected)


def test_mcnemar_is_symmetric():
    assert mcnemar_exact(3, 9) == mcnemar_exact(9, 3)


def test_mcnemar_more_lopsided_is_smaller():
    assert mcnemar_exact(9, 1) < mcnemar_exact(7, 3) < mcnemar_exact(5, 5)


# Thresholds


@pytest.mark.parametrize(
    "extra_fails, status",
    [
        (0, Status.PASS),
        (2, Status.PASS),  # -2 pp
        (3, Status.WARN),  # -3 pp, exactly at warn
        (7, Status.WARN),  # -7 pp
        (8, Status.FAIL),  # -8 pp, exactly at fail (0.92 - 0.84 is not exactly 0.08 in floating point)
        (20, Status.FAIL),
    ],
)
def test_threshold_boundaries(extra_fails, status):
    baseline = run_with_failures("base", BASE_FAILS)
    run = run_with_failures("new", BASE_FAILS | set(range(50, 50 + extra_fails)), minutes=1)
    assert compare_runs(run, baseline, []).status is status


def test_improvement_never_warns():
    baseline = run_with_failures("base", BASE_FAILS)
    run = run_with_failures("new", {1, 2}, minutes=1)
    comparison = compare_runs(run, baseline, [])
    assert comparison.status is Status.PASS
    assert len(comparison.improvements) == 6


def test_thresholds_are_configurable():
    baseline = run_with_failures("base", BASE_FAILS)
    run = run_with_failures("new", BASE_FAILS | {50, 51}, minutes=1)
    assert compare_runs(run, baseline, [], warn_delta=0.02).status is Status.WARN
    assert compare_runs(run, baseline, [], warn_delta=0.01, fail_delta=0.02).status is Status.FAIL


def test_fail_reason_reports_flips_and_p_value():
    baseline = run_with_failures("base", BASE_FAILS)
    run = run_with_failures("new", BASE_FAILS | set(range(50, 58)), minutes=1)
    comparison = compare_runs(run, baseline, [])
    assert comparison.mcnemar_p == pytest.approx(0.0078125)
    assert "8 regressions, 0 improvements, McNemar p = 0.00781, significant" in comparison.reasons[0]


def test_big_drop_from_mixed_flips_is_not_significant():
    baseline = run_with_failures("base", BASE_FAILS)
    # 8 new regressions and 3 improvements: -5 pp, but 8 vs 3 is plausible as noise.
    run = run_with_failures("new", (BASE_FAILS - {1, 2, 3}) | set(range(50, 58)), minutes=1)
    comparison = compare_runs(run, baseline, [])
    assert comparison.status is Status.WARN
    assert comparison.mcnemar_p > 0.05
    assert "not significant" in comparison.reasons[0]


# Diffing


def test_regressions_and_improvements_listed():
    baseline = run_with_failures("base", {1, 2, 3})
    run = run_with_failures("new", {3, 4, 5}, minutes=1)
    comparison = compare_runs(run, baseline, [])
    assert [c.case_id for c in comparison.regressions] == ["gc-004", "gc-005"]
    assert [c.case_id for c in comparison.improvements] == ["gc-001", "gc-002"]
    assert comparison.regressions[0].before.passed and not comparison.regressions[0].after.passed


def test_overall_metrics():
    t = Category.TECHNICAL
    baseline = make_run("base", [result(1, True, t, score=5), result(2, True, t, score=4), result(3, True, t, score=3)])
    run = make_run(
        "new",
        [
            result(1, True, t, score=5),
            result(2, False, t, score=2),  # right category, summary too weak
            result(3, False, t, score=3, predicted=Category.GENERAL),  # wrong category
        ],
        minutes=1,
    )
    overall = {m.name: m for m in compare_runs(run, baseline, []).overall}
    assert overall["pass_rate"].delta == pytest.approx(-2 / 3)
    assert overall["category_accuracy"].baseline == pytest.approx(1.0)
    assert overall["category_accuracy"].run == pytest.approx(2 / 3)
    assert overall["mean_summary_score"].baseline == pytest.approx(4.0)
    assert overall["mean_summary_score"].run == pytest.approx(10 / 3)


def test_per_group_deltas():
    b, t, a = Category.BILLING, Category.TECHNICAL, Category.ACCOUNT
    baseline = make_run(
        "base",
        [
            result(1, True, b, Difficulty.EASY, Split.DEV),
            result(2, True, b, Difficulty.HARD, Split.TEST),
            result(3, False, t, Difficulty.HARD, Split.DEV),
            result(4, True, a, Difficulty.MEDIUM, Split.DEV),
        ],
    )
    run = make_run(
        "new",
        [
            result(1, True, b, Difficulty.EASY, Split.DEV),
            result(2, False, b, Difficulty.HARD, Split.TEST),
            result(3, True, t, Difficulty.HARD, Split.DEV),
            result(4, True, a, Difficulty.MEDIUM, Split.DEV),
        ],
        minutes=1,
    )
    c = compare_runs(run, baseline, [])
    by_cat = {d.group: d for d in c.by_category}
    assert by_cat["billing"].delta == pytest.approx(-0.5)
    assert by_cat["technical"].delta == pytest.approx(1.0)
    assert by_cat["account"].delta == 0
    assert by_cat["general"].baseline_n == 0 and by_cat["general"].delta == 0
    by_diff = {d.group: d for d in c.by_difficulty}
    assert by_diff["hard"].baseline_rate == 0.5 and by_diff["hard"].run_rate == 0.5
    by_split = {d.group: d for d in c.by_split}
    assert by_split["test"].delta == pytest.approx(-1.0)
    assert by_split["dev"].delta == pytest.approx(1 / 3)


def test_category_drop_alone_does_not_set_status():
    # Billing loses 3 cases while other categories gain 3: overall unchanged, so the status stays pass.
    b = Category.BILLING
    base_results = [result(i, not (4 <= i <= 6), b if i <= 6 else None) for i in range(1, 101)]
    run_results = [result(i, not (1 <= i <= 3), b if i <= 6 else None) for i in range(1, 101)]
    baseline = make_run("base", base_results)
    run = make_run("new", run_results, minutes=1)
    comparison = compare_runs(run, baseline, [])
    assert comparison.status is Status.PASS
    assert {d.group: d.delta for d in comparison.by_category}["billing"] == 0
    assert len(comparison.regressions) == 3 and len(comparison.improvements) == 3


def test_changed_predictions_and_scores_counted():
    baseline = make_run("base", [result(1, True, score=5), result(2, True, score=5), result(3, True, score=4)])
    run = make_run(
        "new",
        [
            result(1, True, score=4),
            result(2, False, Category.TECHNICAL, predicted=Category.GENERAL, score=5),
            result(3, True, score=4),
        ],
        minutes=1,
    )
    comparison = compare_runs(run, baseline, [])
    assert comparison.prediction_changes == 1
    assert comparison.score_changes == 1


def test_unmatched_cases_noted():
    baseline = make_run("base", [result(1, True), result(2, True)])
    run = make_run("new", [result(1, True), result(3, True)], minutes=1)
    comparison = compare_runs(run, baseline, [])
    assert comparison.unmatched_case_ids == ["gc-002", "gc-003"]
    assert any("2 cases appear in only one run" in n for n in comparison.notes)


def test_version_mismatch_noted():
    baseline = run_with_failures("base", BASE_FAILS, dataset_version=2)
    run = run_with_failures("new", BASE_FAILS, minutes=1)
    comparison = compare_runs(run, baseline, [])
    assert any("baseline dataset_version is 2 but run has 3" in n for n in comparison.notes)


def test_no_baseline_passes_with_note():
    comparison = compare_runs(run_with_failures("new", BASE_FAILS), None, [])
    assert comparison.status is Status.PASS
    assert comparison.baseline_id is None
    assert "no baseline to compare against" in comparison.notes


def test_errors_warn():
    baseline = run_with_failures("base", BASE_FAILS)
    results = [result(i, i not in BASE_FAILS) for i in range(1, 100)] + [result(100, False, error="classify: boom")]
    run = make_run("new", results, minutes=1)
    comparison = compare_runs(run, baseline, [])
    assert comparison.status is Status.WARN
    assert "1 cases errored and were counted as fails" in comparison.reasons


# Drift


def test_drift_needs_full_window():
    history = [run_with_rate(f"h{i}", 0.5, minutes=-i) for i in range(1, 4)]
    drift = compute_drift(run_with_rate("new", 0.5), history, floor=0.88)
    assert drift.moving_average is None
    assert not drift.below_floor
    assert len(drift.run_ids) == 4


def test_steady_history_does_not_drift():
    history = [run_with_rate(f"h{i}", 0.92, minutes=-i) for i in range(1, 7)]
    drift = compute_drift(run_with_rate("new", 0.92), history, floor=0.88)
    assert drift.moving_average == pytest.approx(0.92)
    assert not drift.below_floor


def test_slow_decline_warns_even_when_each_step_is_small():
    rates = [0.90, 0.89, 0.88, 0.87, 0.86, 0.85]  # newest first after the run
    history = [run_with_rate(f"h{i}", rate, minutes=-(i + 1)) for i, rate in enumerate(reversed(rates))]
    history.sort(key=lambda r: r.created_at, reverse=True)
    run = run_with_rate("new", 0.84)
    comparison = compare_runs(run, history[0], history)
    # -1 pp against the previous run: no threshold alert, but the 7 run average is 87%.
    assert -comparison.overall[0].delta == pytest.approx(0.01)
    assert comparison.drift.moving_average == pytest.approx(0.87)
    assert comparison.status is Status.WARN
    assert comparison.reasons[0].startswith("drift:")


def test_one_bad_run_does_not_trigger_drift():
    history = [run_with_rate(f"h{i}", 0.94, minutes=-i) for i in range(1, 7)]
    drift = compute_drift(run_with_rate("new", 0.80), history, floor=0.88)
    assert drift.moving_average == pytest.approx((6 * 0.94 + 0.80) / 7)
    assert not drift.below_floor


def test_drift_window_uses_newest_runs_only():
    history = [run_with_rate(f"h{i}", 0.92, minutes=-i) for i in range(1, 7)]
    history += [run_with_rate(f"old{i}", 0.10, minutes=-100 - i) for i in range(5)]
    drift = compute_drift(run_with_rate("new", 0.92), history, floor=0.88)
    assert drift.run_ids == ["new"] + [f"h{i}" for i in range(1, 7)]


def test_drift_does_not_count_the_run_twice():
    run = run_with_rate("new", 0.92)
    history = [run] + [run_with_rate(f"h{i}", 0.92, minutes=-i) for i in range(1, 6)]
    assert compute_drift(run, history, floor=0.88).moving_average is None


def test_fail_outranks_drift_warning():
    rates = [0.84] * 6
    history = [run_with_rate(f"h{i}", rate, minutes=-(i + 1)) for i, rate in enumerate(rates)]
    baseline = run_with_rate("base", 0.92, minutes=-1)
    run = run_with_rate("new", 0.80)
    comparison = compare_runs(run, baseline, history)
    assert comparison.status is Status.FAIL
    assert any(r.startswith("drift:") for r in comparison.reasons)
