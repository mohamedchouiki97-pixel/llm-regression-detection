"""SQLite run history: one row per run, one row per case per run."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from src.models import CaseResult, RunRecord

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "runs.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    judge_version TEXT NOT NULL,
    dataset_version INTEGER NOT NULL,
    summary_threshold INTEGER NOT NULL,
    git_sha TEXT NOT NULL,
    git_branch TEXT NOT NULL,
    git_dirty INTEGER NOT NULL,
    n_cases INTEGER NOT NULL,
    n_passed INTEGER NOT NULL,
    pass_rate REAL NOT NULL,
    n_errors INTEGER NOT NULL,
    classifier_cost_usd REAL,
    judge_cost_usd REAL,
    n_cached INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS case_results (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    case_id TEXT NOT NULL,
    expected_category TEXT NOT NULL,
    predicted_category TEXT,
    category_match INTEGER NOT NULL,
    summary TEXT,
    summary_score INTEGER,
    judge_reasoning TEXT,
    passed INTEGER NOT NULL,
    error TEXT,
    difficulty TEXT NOT NULL,
    tags TEXT NOT NULL,
    latency_ms REAL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL,
    judge_cost_usd REAL,
    cached INTEGER NOT NULL,
    PRIMARY KEY (run_id, case_id)
);
"""

RUN_FIELDS = [name for name in RunRecord.model_fields if name != "results"]
CASE_FIELDS = list(CaseResult.model_fields)


def connect(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _case_row(run_id: str, result: CaseResult) -> tuple:
    data = result.model_dump(mode="json")
    data["tags"] = json.dumps(data["tags"])
    return (run_id, *(data[name] for name in CASE_FIELDS))


def save_run(run: RunRecord, path: Path | str = DEFAULT_DB_PATH) -> None:
    """Write the run and all its case results in one transaction, so a failure leaves nothing behind."""
    data = run.model_dump(mode="json", exclude={"results"})
    with closing(connect(path)) as conn, conn:
        conn.execute(
            f"INSERT INTO runs ({', '.join(RUN_FIELDS)}) VALUES ({', '.join('?' * len(RUN_FIELDS))})",
            [data[name] for name in RUN_FIELDS],
        )
        conn.executemany(
            f"INSERT INTO case_results (run_id, {', '.join(CASE_FIELDS)}) "
            f"VALUES (?, {', '.join('?' * len(CASE_FIELDS))})",
            [_case_row(run.run_id, result) for result in run.results],
        )


def load_run(run_id: str, path: Path | str = DEFAULT_DB_PATH) -> RunRecord:
    with closing(connect(path)) as conn:
        run_row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run_row is None:
            raise KeyError(f"run {run_id} not found")
        case_rows = conn.execute(
            "SELECT * FROM case_results WHERE run_id = ? ORDER BY rowid", (run_id,)
        ).fetchall()

    results = []
    for row in case_rows:
        data = {name: row[name] for name in CASE_FIELDS}
        data["tags"] = json.loads(data["tags"])
        results.append(CaseResult.model_validate(data))
    return RunRecord.model_validate({**dict(run_row), "results": results})
