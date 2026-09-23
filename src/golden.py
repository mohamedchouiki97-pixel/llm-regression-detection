"""Load, check, and summarize the hand-written golden dataset."""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from src.models import Category, Difficulty, GoldenDataset, Split

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "golden.json"
PLACEHOLDER_TAG = "placeholder"


class DatasetLoadError(Exception):
    pass


@dataclass
class DatasetReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _case_label(raw: dict, index: int) -> str:
    """Name a case by position and id, e.g. 'cases[12] (gc-013)', so it can be found in the file."""
    cases = raw.get("cases")
    case_id = None
    if isinstance(cases, list) and index < len(cases) and isinstance(cases[index], dict):
        case_id = cases[index].get("id")
    return f"cases[{index}] ({case_id if isinstance(case_id, str) else 'no id'})"


def _format_location(raw: dict, loc: tuple) -> str:
    if len(loc) >= 2 and loc[0] == "cases" and isinstance(loc[1], int):
        rest = ".".join(str(part) for part in loc[2:])
        label = _case_label(raw, loc[1])
        return f"{label}.{rest}" if rest else label
    return ".".join(str(part) for part in loc) or "(root)"


def load_dataset(path: Path | str = DEFAULT_DATASET_PATH) -> GoldenDataset:
    path = Path(path)
    if not path.is_file():
        raise DatasetLoadError(f"{path}: dataset file not found")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetLoadError(
            f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(raw, dict):
        raise DatasetLoadError(f"{path}: expected an object at the top level, got {type(raw).__name__}")

    try:
        return GoldenDataset.model_validate(raw)
    except ValidationError as exc:
        lines = [f"{path}: invalid dataset ({exc.error_count()} errors)"]
        for err in exc.errors():
            lines.append(f"  {_format_location(raw, err['loc'])}: {err['msg']}")
        raise DatasetLoadError("\n".join(lines)) from exc


def _normalize_email(text: str) -> str:
    return " ".join(text.lower().split())


def check_dataset(dataset: GoldenDataset, min_category_share: float = 0.15) -> DatasetReport:
    """Checks that need the whole dataset. Assumes the schema already validated."""
    report = DatasetReport()

    positions: dict[str, list[int]] = {}
    for index, case in enumerate(dataset.cases):
        positions.setdefault(case.id, []).append(index)
    for case_id, indexes in positions.items():
        if len(indexes) > 1:
            where = " and ".join(f"cases[{i}]" for i in indexes)
            report.errors.append(f"duplicate id {case_id} at {where}")

    changelog_versions = Counter(entry.version for entry in dataset.changelog)
    for version, count in sorted(changelog_versions.items()):
        if count > 1:
            report.errors.append(f"changelog lists version {version} {count} times")
    if dataset.version not in changelog_versions:
        report.errors.append(
            f"version {dataset.version} has no changelog entry; add one describing what changed"
        )

    by_email: dict[str, list[str]] = {}
    for case in dataset.cases:
        by_email.setdefault(_normalize_email(case.email), []).append(case.id)
    for ids in by_email.values():
        if len(ids) > 1:
            report.warnings.append(f"{' and '.join(ids)} have the same email text")

    counts = Counter(case.expected_category for case in dataset.cases)
    total = len(dataset.cases)
    for category in Category:
        share = counts[category] / total
        if counts[category] == 0:
            report.warnings.append(f'no cases for category "{category.value}"')
        elif share < min_category_share:
            report.warnings.append(
                f'category "{category.value}" is {share:.1%} of cases (minimum {min_category_share:.0%})'
            )

    placeholders = [case.id for case in dataset.cases if PLACEHOLDER_TAG in case.tags]
    if placeholders:
        report.warnings.append(
            f"{len(placeholders)} placeholder cases remain: {', '.join(placeholders)}"
        )

    return report


def coverage_table(dataset: GoldenDataset) -> str:
    """Category by difficulty counts, category shares, and tag counts."""
    grid = Counter((case.expected_category, case.difficulty) for case in dataset.cases)
    total = len(dataset.cases)
    difficulties = list(Difficulty)

    header = f"{'':<12}" + "".join(f"{d.value:>8}" for d in difficulties) + f"{'total':>8}{'share':>9}"
    lines = [f"Dataset version {dataset.version}, {total} cases", "", header]
    for category in Category:
        row = [grid[(category, d)] for d in difficulties]
        row_total = sum(row)
        lines.append(
            f"{category.value:<12}"
            + "".join(f"{n:>8}" for n in row)
            + f"{row_total:>8}{row_total / total:>9.1%}"
        )
    column_totals = [sum(grid[(c, d)] for c in Category) for d in difficulties]
    lines.append(f"{'total':<12}" + "".join(f"{n:>8}" for n in column_totals) + f"{total:>8}")

    splits = Counter(case.split for case in dataset.cases)
    lines.append("")
    lines.append("split: " + ", ".join(f"{s.value} {splits[s]}" for s in Split))

    tags = Counter(tag for case in dataset.cases for tag in case.tags)
    # Sorted by count so rare tags, often typos, sink to the end.
    tag_text = ", ".join(f"{tag} {n}" for tag, n in sorted(tags.items(), key=lambda kv: (-kv[1], kv[0])))
    lines.append(f"tags: {tag_text or '(none)'}")
    return "\n".join(lines)
