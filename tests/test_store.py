import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.models import CaseResult, Category, Difficulty, RunRecord, Split
from src.store import comparable_main_runs, latest_run_id, load_run, save_run


def make_run(run_id: str = "20260923T140211-abc123") -> RunRecord:
    results = [
        CaseResult(
            case_id="gc-001",
            expected_category=Category.BILLING,
            predicted_category=Category.BILLING,
            category_match=True,
            summary="The customer wants a refund.",
            summary_score=5,
            judge_reasoning="Matches.",
            passed=True,
            difficulty=Difficulty.EASY,
            split=Split.DEV,
            tags=["short", "typo"],
            latency_ms=412.5,
            input_tokens=500,
            output_tokens=40,
            cost_usd=0.000099,
            judge_cost_usd=0.0023,
        ),
        CaseResult(
            case_id="gc-002",
            expected_category=Category.ACCOUNT,
            error="classify: BadRequestError: bad",
            difficulty=Difficulty.HARD,
            split=Split.TEST,
            cached=True,
        ),
    ]
    return RunRecord(
        run_id=run_id,
        created_at=datetime(2026, 9, 23, 14, 2, 11, tzinfo=timezone.utc),
        prompt_version="v1",
        prompt_hash="7edc03262ace1919",
        model="gpt-4o-mini",
        judge_model="gpt-4o",
        judge_version="j1",
        dataset_version=2,
        summary_threshold=4,
        git_sha="04d8293",
        git_branch="main",
        git_dirty=True,
        n_cases=2,
        n_passed=1,
        pass_rate=0.5,
        n_errors=1,
        classifier_cost_usd=0.000099,
        judge_cost_usd=None,
        n_cached=1,
        results=results,
    )


def test_round_trip(tmp_path):
    db = tmp_path / "runs.db"
    run = make_run()
    save_run(run, db)
    assert load_run(run.run_id, db) == run


def test_multiple_runs(tmp_path):
    db = tmp_path / "runs.db"
    save_run(make_run("run-a"), db)
    save_run(make_run("run-b"), db)
    assert load_run("run-b", db).run_id == "run-b"
    assert len(load_run("run-a", db).results) == 2


def test_missing_run(tmp_path):
    with pytest.raises(KeyError, match="not found"):
        load_run("nope", tmp_path / "runs.db")


def test_failed_save_leaves_nothing(tmp_path):
    db = tmp_path / "runs.db"
    run = make_run()
    run.results.append(run.results[0])  # duplicate case id violates the primary key
    with pytest.raises(sqlite3.IntegrityError):
        save_run(run, db)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM case_results").fetchone()[0] == 0


# Baseline and drift history lookup


def meta_run(run_id: str, minutes: int, **overrides) -> RunRecord:
    run = make_run(run_id)
    run.created_at = run.created_at + timedelta(minutes=minutes)
    run.git_dirty = False
    for name, value in overrides.items():
        setattr(run, name, value)
    return run


def test_comparable_main_runs_filters_and_orders(tmp_path):
    db = tmp_path / "runs.db"
    candidate = meta_run("candidate", 100, git_branch="feature/x")
    for run in [
        meta_run("main-old", 10),
        meta_run("main-new", 20),
        meta_run("dirty", 30, git_dirty=True),
        meta_run("branch", 40, git_branch="feature/y"),
        meta_run("old-dataset", 50, dataset_version=1),
        meta_run("other-judge", 60, judge_version="j2"),
        meta_run("other-threshold", 70, summary_threshold=3),
        meta_run("after-candidate", 200),
        candidate,
    ]:
        save_run(run, db)

    runs = comparable_main_runs(candidate, db)
    assert [r.run_id for r in runs] == ["main-new", "main-old"]
    assert [r.run_id for r in comparable_main_runs(candidate, db, limit=1)] == ["main-new"]


def test_comparable_main_runs_excludes_the_run_itself(tmp_path):
    db = tmp_path / "runs.db"
    earlier, run = meta_run("earlier", 0), meta_run("current", 10)
    save_run(earlier, db)
    save_run(run, db)
    assert [r.run_id for r in comparable_main_runs(run, db)] == ["earlier"]


def test_latest_run_id(tmp_path):
    db = tmp_path / "runs.db"
    assert latest_run_id(db) is None
    save_run(meta_run("a", 10), db)
    save_run(meta_run("b", 20, git_branch="feature"), db)
    assert latest_run_id(db) == "b"
