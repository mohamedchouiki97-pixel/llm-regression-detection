import csv
from collections import Counter
from datetime import date, datetime, timezone

import pytest

from src.agreement import (
    CSV_COLUMNS,
    AgreementInputError,
    confusion_matrix,
    exact_agreement,
    judged_results,
    pass_fail_agreement,
    quadratic_weighted_kappa,
    read_scored_csv,
    stratified_sample,
    write_sample_csv,
)
from src.models import (
    CaseResult,
    Category,
    ChangelogEntry,
    Difficulty,
    GoldenCase,
    GoldenDataset,
    RunRecord,
    Split,
)


def result(case_id: str, score: int | None, error: str | None = None) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        expected_category=Category.BILLING,
        predicted_category=None if error else Category.BILLING,
        category_match=error is None,
        summary=None if error else f"summary of {case_id}",
        summary_score=score,
        judge_reasoning=None if score is None else "reasoning",
        error=error,
        difficulty=Difficulty.EASY,
        split=Split.DEV,
    )


def make_run(results: list[CaseResult]) -> RunRecord:
    return RunRecord(
        run_id="run-a",
        created_at=datetime(2026, 10, 8, tzinfo=timezone.utc),
        prompt_version="v2",
        prompt_hash="hash",
        model="gpt-4o-mini",
        judge_model="gpt-4o-2024-08-06",
        judge_version="j2",
        dataset_version=3,
        summary_threshold=4,
        git_sha="abc",
        git_branch="main",
        git_dirty=False,
        n_cases=len(results),
        n_passed=0,
        pass_rate=0.0,
        n_errors=sum(r.error is not None for r in results),
        classifier_cost_usd=None,
        judge_cost_usd=None,
        n_cached=0,
        results=results,
    )


def write_csv(path, rows: list[tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for case_id, score in rows:
            writer.writerow([case_id, "email", "ideal", "summary", score])


# Agreement statistics, with expected values worked out by hand.


def test_perfect_agreement():
    pairs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    assert exact_agreement(pairs) == 1.0
    assert pass_fail_agreement(pairs, 4) == 1.0
    assert quadratic_weighted_kappa(pairs) == pytest.approx(1.0)


def test_kappa_with_one_near_miss():
    # Observed weighted disagreement 1/16; expected 80/64. Kappa = 1 - (1/16) / (80/64) = 0.95.
    pairs = [(1, 1), (3, 3), (5, 5), (5, 4)]
    assert quadratic_weighted_kappa(pairs) == pytest.approx(0.95)
    assert exact_agreement(pairs) == 0.75


def test_kappa_for_swapped_scores_is_minus_one():
    # Observed 2/16, expected 1/16, so kappa = 1 - 2 = -1.
    assert quadratic_weighted_kappa([(1, 2), (2, 1)]) == pytest.approx(-1.0)


def test_kappa_can_be_negative_despite_high_raw_agreement():
    # Both raters mostly say 5. Half the cases agree exactly, yet agreement is worse than chance.
    # Observed 2/16, expected 1.5/16, so kappa = 1 - 4/3.
    pairs = [(4, 5), (5, 5), (5, 5), (5, 4)]
    assert exact_agreement(pairs) == 0.5
    assert quadratic_weighted_kappa(pairs) == pytest.approx(-1 / 3)


def test_kappa_is_undefined_when_both_raters_use_one_score():
    assert quadratic_weighted_kappa([(5, 5), (5, 5), (5, 5)]) is None


def test_kappa_penalizes_distant_disagreement_more():
    near = [(1, 1), (3, 3), (5, 5), (5, 4), (2, 2)]
    far = [(1, 1), (3, 3), (5, 5), (5, 1), (2, 2)]
    assert exact_agreement(near) == exact_agreement(far)
    assert quadratic_weighted_kappa(near) > quadratic_weighted_kappa(far)


def test_pass_fail_agreement_at_threshold_edge():
    # 4 vs 5 both pass at threshold 4; 3 vs 4 straddles it; 2 vs 3 both fail.
    pairs = [(4, 5), (3, 4), (2, 3)]
    assert exact_agreement(pairs) == 0.0
    assert pass_fail_agreement(pairs, 4) == pytest.approx(2 / 3)
    assert pass_fail_agreement(pairs, 5) == pytest.approx(2 / 3)


def test_confusion_matrix_rows_are_human_columns_are_judge():
    matrix = confusion_matrix([(1, 5), (1, 5), (4, 4)])
    assert matrix[0][4] == 2
    assert matrix[3][3] == 1
    assert sum(map(sum, matrix)) == 3


# Sampling


def test_sample_spreads_evenly_across_judge_scores():
    results = [result(f"gc-{i:03d}", 5) for i in range(70)]
    results += [result(f"gc-{i:03d}", 4) for i in range(70, 95)]
    results += [result(f"gc-{i:03d}", 3) for i in range(95, 100)]
    sample = stratified_sample(results, 30)
    # 10 each, but only 5 threes exist, so their unused share goes to the other levels.
    counts = Counter(r.summary_score for r in sample)
    assert counts == {3: 5, 4: 13, 5: 12}
    assert len({r.case_id for r in sample}) == 30


def test_sample_is_reproducible_and_seed_dependent():
    results = [result(f"gc-{i:03d}", 4 + i % 2) for i in range(50)]
    first = [r.case_id for r in stratified_sample(results, 10, seed=1)]
    assert first == [r.case_id for r in stratified_sample(results, 10, seed=1)]
    assert first != [r.case_id for r in stratified_sample(results, 10, seed=2)]


def test_sample_larger_than_pool_returns_everything():
    results = [result("gc-001", 5), result("gc-002", 2)]
    assert {r.case_id for r in stratified_sample(results, 30)} == {"gc-001", "gc-002"}


def test_judged_results_skip_errors_and_unscored_cases():
    run = make_run([result("gc-001", 5), result("gc-002", None, error="judge: timeout"), result("gc-003", None)])
    assert [r.case_id for r in judged_results(run)] == ["gc-001"]


# CSV files


def test_sample_csv_hides_judge_score_and_reasoning(tmp_path):
    dataset = GoldenDataset(
        version=3,
        changelog=[ChangelogEntry(version=3, date=date(2026, 9, 23), changes="x")],
        cases=[
            GoldenCase(
                id="gc-001",
                email="Hola, me cobraron dos veces.",
                expected_category=Category.BILLING,
                ideal_summary="Charged twice.",
                difficulty=Difficulty.EASY,
                split=Split.DEV,
            )
        ],
    )
    path = tmp_path / "sample.csv"
    write_sample_csv([result("gc-001", 2)], dataset, path)
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == CSV_COLUMNS
    assert rows[0]["email"] == "Hola, me cobraron dos veces."
    assert rows[0]["model_summary"] == "summary of gc-001"
    assert rows[0]["human_score"] == ""
    assert "reasoning" not in path.read_text(encoding="utf-8-sig")


def test_read_scored_csv_pairs_human_with_judge(tmp_path):
    run = make_run([result("gc-001", 5), result("gc-002", 3)])
    path = tmp_path / "scored.csv"
    write_csv(path, [("gc-002", "2"), ("gc-001", " 4 ")])
    assert read_scored_csv(path, run) == [(2, 3), (4, 5)]


def test_read_scored_csv_reports_every_problem(tmp_path):
    run = make_run([result("gc-001", 5), result("gc-002", 3), result("gc-003", None, error="judge: timeout")])
    path = tmp_path / "scored.csv"
    write_csv(path, [("gc-001", ""), ("gc-002", "6"), ("gc-003", "4"), ("gc-999", "4"), ("gc-001", "5")])
    with pytest.raises(AgreementInputError) as exc:
        read_scored_csv(path, run)
    problems = exc.value.problems
    assert len(problems) == 5
    assert "gc-001" in problems[0] and "got ''" in problems[0]
    assert "got '6'" in problems[1]
    assert "gc-003" in problems[2] and "not a judged case" in problems[2]
    assert "gc-999" in problems[3]
    assert "duplicate" in problems[4]


def test_read_scored_csv_needs_columns(tmp_path):
    path = tmp_path / "scored.csv"
    path.write_text("id,score\ngc-001,5\n", encoding="utf-8")
    with pytest.raises(AgreementInputError, match="missing column"):
        read_scored_csv(path, make_run([result("gc-001", 5)]))
