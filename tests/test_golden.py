import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.golden import DatasetLoadError, check_dataset, coverage_table, load_dataset
from src.models import GoldenDataset

ROOT = Path(__file__).resolve().parent.parent


def make_case(n: int, category: str = "billing", difficulty: str = "easy", **overrides) -> dict:
    case = {
        "id": f"gc-{n:03d}",
        "email": f"Test email number {n}.",
        "expected_category": category,
        "ideal_summary": f"Summary {n}.",
        "difficulty": difficulty,
        "tags": [],
        "notes": "",
        "split": "dev",
    }
    case.update(overrides)
    return case


def balanced_raw() -> dict:
    """Four cases, one per category, so no balance warnings."""
    return {
        "version": 1,
        "changelog": [{"version": 1, "date": "2026-09-23", "changes": "First version."}],
        "cases": [
            make_case(1, "billing"),
            make_case(2, "technical", "medium"),
            make_case(3, "account", "hard"),
            make_case(4, "general"),
        ],
    }


def write(tmp_path: Path, data) -> Path:
    path = tmp_path / "golden.json"
    text = data if isinstance(data, str) else json.dumps(data)
    path.write_text(text, encoding="utf-8")
    return path


def load_raw(tmp_path: Path, raw: dict) -> GoldenDataset:
    return load_dataset(write(tmp_path, raw))


# Valid input


def test_valid_dataset_has_no_errors_or_warnings(tmp_path):
    dataset = load_raw(tmp_path, balanced_raw())
    report = check_dataset(dataset)
    assert report.ok
    assert report.warnings == []


def test_repo_dataset_passes():
    dataset = load_dataset(ROOT / "data" / "golden.json")
    report = check_dataset(dataset)
    assert report.ok, report.errors


# Loading


def test_missing_file(tmp_path):
    with pytest.raises(DatasetLoadError, match="not found"):
        load_dataset(tmp_path / "nope.json")


def test_malformed_json_reports_line(tmp_path):
    path = write(tmp_path, '{\n  "version": 1,\n  "cases": [\n}')
    with pytest.raises(DatasetLoadError, match=r"invalid JSON at line 4"):
        load_dataset(path)


def test_top_level_must_be_object(tmp_path):
    with pytest.raises(DatasetLoadError, match="expected an object"):
        load_dataset(write(tmp_path, "[1, 2]"))


# Schema


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"id": "case-1"}, r"cases\[0\] \(case-1\)\.id"),
        ({"expected_category": "refunds"}, r"cases\[0\] \(gc-001\)\.expected_category"),
        ({"difficulty": "extreme"}, r"cases\[0\] \(gc-001\)\.difficulty"),
        ({"email": "   "}, r"cases\[0\] \(gc-001\)\.email"),
        ({"ideal_summary": ""}, r"cases\[0\] \(gc-001\)\.ideal_summary"),
        ({"tags": ["Mixed-Language"]}, r"cases\[0\] \(gc-001\)\.tags\.0"),
        ({"tags": ["typo", "typo"]}, r"duplicate tags: typo"),
        ({"expected_catgory": "billing"}, r"expected_catgory"),
        ({"split": "train"}, r"cases\[0\] \(gc-001\)\.split"),
    ],
)
def test_bad_case_field_is_named(tmp_path, override, expected):
    raw = balanced_raw()
    raw["cases"][0].update(override)
    with pytest.raises(DatasetLoadError, match=expected):
        load_raw(tmp_path, raw)


def test_missing_id_is_labelled(tmp_path):
    raw = balanced_raw()
    del raw["cases"][2]["id"]
    with pytest.raises(DatasetLoadError, match=r"cases\[2\] \(no id\)\.id: Field required"):
        load_raw(tmp_path, raw)


def test_empty_cases_rejected(tmp_path):
    raw = balanced_raw()
    raw["cases"] = []
    with pytest.raises(DatasetLoadError, match=r"cases: List should have at least 1 item"):
        load_raw(tmp_path, raw)


def test_empty_changelog_rejected(tmp_path):
    raw = balanced_raw()
    raw["changelog"] = []
    with pytest.raises(DatasetLoadError, match=r"changelog: List should have at least 1 item"):
        load_raw(tmp_path, raw)


def test_version_must_be_a_number(tmp_path):
    raw = balanced_raw()
    raw["version"] = "one"
    with pytest.raises(DatasetLoadError, match=r"version: Input should be a valid integer"):
        load_raw(tmp_path, raw)


def test_all_schema_errors_reported_together(tmp_path):
    raw = balanced_raw()
    raw["cases"][0]["expected_category"] = "refunds"
    raw["cases"][1]["difficulty"] = "extreme"
    raw["cases"][3]["id"] = "bad"
    with pytest.raises(DatasetLoadError) as info:
        load_raw(tmp_path, raw)
    message = str(info.value)
    assert "(3 errors)" in message
    assert "cases[0] (gc-001).expected_category" in message
    assert "cases[1] (gc-002).difficulty" in message
    assert "cases[3] (bad).id" in message


# Cross-case errors


def test_duplicate_id(tmp_path):
    raw = balanced_raw()
    raw["cases"][3]["id"] = "gc-002"
    report = check_dataset(load_raw(tmp_path, raw))
    assert "duplicate id gc-002 at cases[1] and cases[3]" in report.errors


def test_version_missing_from_changelog(tmp_path):
    raw = balanced_raw()
    raw["version"] = 2
    report = check_dataset(load_raw(tmp_path, raw))
    assert not report.ok
    assert any("version 2 has no changelog entry" in e for e in report.errors)


def test_duplicate_changelog_version(tmp_path):
    raw = balanced_raw()
    raw["changelog"].append({"version": 1, "date": "2026-09-24", "changes": "Again."})
    report = check_dataset(load_raw(tmp_path, raw))
    assert "changelog lists version 1 2 times" in report.errors


# Warnings


def test_duplicate_email_ignores_case_and_whitespace(tmp_path):
    raw = balanced_raw()
    raw["cases"][0]["email"] = "Hello   there,\nmy card failed."
    raw["cases"][1]["email"] = "hello there, MY card failed."
    report = check_dataset(load_raw(tmp_path, raw))
    assert report.ok
    assert "gc-001 and gc-002 have the same email text" in report.warnings


def test_missing_category_warns(tmp_path):
    raw = balanced_raw()
    raw["cases"][3]["expected_category"] = "billing"
    report = check_dataset(load_raw(tmp_path, raw))
    assert report.ok
    assert 'no cases for category "general"' in report.warnings


def test_thin_category_warns(tmp_path):
    raw = balanced_raw()
    raw["cases"] += [make_case(n, "billing") for n in range(5, 8)]
    report = check_dataset(load_raw(tmp_path, raw))
    # 7 cases: billing is 4/7, the other three are 1/7 (14.3%) each, below the 15% default.
    thin = [w for w in report.warnings if "of cases (minimum 15%)" in w]
    assert len(thin) == 3
    assert 'category "technical" is 14.3% of cases (minimum 15%)' in thin


def test_share_above_default_threshold_does_not_warn(tmp_path):
    raw = balanced_raw()
    raw["cases"].append(make_case(5, "billing"))
    # 5 cases: the three smaller categories are 20% each, above the 15% default.
    report = check_dataset(load_raw(tmp_path, raw))
    assert report.warnings == []


def test_min_share_is_configurable(tmp_path):
    raw = balanced_raw()
    raw["cases"].append(make_case(5, "billing"))
    report = check_dataset(load_raw(tmp_path, raw), min_category_share=0.25)
    assert len(report.warnings) == 3


def test_placeholder_warning_lists_ids(tmp_path):
    raw = balanced_raw()
    raw["cases"][1]["tags"] = ["placeholder"]
    report = check_dataset(load_raw(tmp_path, raw))
    assert "1 placeholder cases remain: gc-002" in report.warnings


# Coverage


def test_coverage_table_counts(tmp_path):
    raw = balanced_raw()
    raw["cases"] += [
        make_case(5, "billing", "hard", tags=["typo", "short"]),
        make_case(6, "billing", "hard", tags=["typo"]),
        make_case(7, "technical", "easy", tags=["ambigous"]),
        make_case(8, "general", "medium"),
    ]
    table = coverage_table(load_raw(tmp_path, raw))
    lines = table.splitlines()
    assert lines[0] == "Dataset version 1, 8 cases"
    rows = {line.split()[0]: line.split()[1:] for line in lines[3:8]}
    assert rows["billing"] == ["1", "0", "2", "3", "37.5%"]
    assert rows["technical"] == ["1", "1", "0", "2", "25.0%"]
    assert rows["account"] == ["0", "0", "1", "1", "12.5%"]
    assert rows["general"] == ["1", "1", "0", "2", "25.0%"]
    assert rows["total"] == ["3", "2", "3", "8"]
    # Most common first, ties alphabetical, so the rare typo sinks to the end.
    assert lines[-2] == "split: dev 8, test 0"
    assert lines[-1] == "tags: typo 2, ambigous 1, short 1"


def test_coverage_table_without_tags(tmp_path):
    table = coverage_table(load_raw(tmp_path, balanced_raw()))
    assert table.splitlines()[-1] == "tags: (none)"


# CLI


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.cli", "validate-dataset", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_cli_passes_on_repo_dataset():
    result = run_cli()
    assert result.returncode == 0, result.stderr
    assert "OK: 0 errors" in result.stdout


def test_cli_fails_on_duplicate_id(tmp_path):
    raw = balanced_raw()
    raw["cases"][1]["id"] = "gc-001"
    result = run_cli("--path", str(write(tmp_path, raw)))
    assert result.returncode == 1
    assert "error: duplicate id gc-001" in result.stdout
    assert "FAILED: 1 errors" in result.stdout


def test_cli_fails_on_schema_error(tmp_path):
    raw = balanced_raw()
    raw["cases"][0]["difficulty"] = "extreme"
    result = run_cli("--path", str(write(tmp_path, raw)))
    assert result.returncode == 1
    assert "cases[0] (gc-001).difficulty" in result.stderr
