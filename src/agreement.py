"""Judge vs human agreement: a sample for blind human scoring, and the agreement statistics."""

import csv
import random
from collections import defaultdict
from pathlib import Path

from src.models import CaseResult, GoldenDataset, RunRecord

SCORES = range(1, 6)
CSV_COLUMNS = ["case_id", "email", "ideal_summary", "model_summary", "human_score"]


class AgreementInputError(Exception):
    """A sample or filled CSV that cannot be used. Carries every problem found, not just the first."""

    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


def judged_results(run: RunRecord) -> list[CaseResult]:
    """Cases the judge actually scored. Errored cases have nothing for a human to compare against."""
    return [r for r in run.results if r.error is None and r.summary is not None and r.summary_score is not None]


def stratified_sample(results: list[CaseResult], n: int, seed: int = 0) -> list[CaseResult]:
    """Up to n cases, spread as evenly as possible across the judge's score levels, in shuffled order.

    Stratifying on the judge's score keeps rare low scores in the sample. A plain random sample of a
    good prompt's run would be almost all 4s and 5s, which leaves kappa with nothing to measure.
    Levels with fewer cases than their share give the rest to the other levels.
    """
    rng = random.Random(seed)
    by_score: dict[int, list[CaseResult]] = defaultdict(list)
    for result in sorted(results, key=lambda r: r.case_id):
        by_score[result.summary_score].append(result)

    quotas = {score: 0 for score in by_score}
    remaining = min(n, len(results))
    while remaining > 0:
        for score in sorted(by_score):
            if remaining > 0 and quotas[score] < len(by_score[score]):
                quotas[score] += 1
                remaining -= 1

    sample = [case for score in sorted(by_score) for case in rng.sample(by_score[score], quotas[score])]
    # Shuffled so the row order does not reveal the judge's score.
    rng.shuffle(sample)
    return sample


def write_sample_csv(sample: list[CaseResult], dataset: GoldenDataset, path: Path | str) -> None:
    """The judge's score and reasoning are left out on purpose, so the human scores blind."""
    cases = {case.id: case for case in dataset.cases}
    # utf-8-sig so Excel shows non-English emails correctly.
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in sample:
            case = cases[result.case_id]
            writer.writerow(
                {
                    "case_id": result.case_id,
                    "email": case.email,
                    "ideal_summary": case.ideal_summary,
                    "model_summary": result.summary,
                    "human_score": "",
                }
            )


def read_scored_csv(path: Path | str, run: RunRecord) -> list[tuple[int, int]]:
    """(human score, judge score) pairs from a filled sample, checked against the run it came from."""
    judged = {r.case_id: r.summary_score for r in judged_results(run)}
    problems = []
    pairs = []
    seen = set()

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ("case_id", "human_score") if c not in (reader.fieldnames or [])]
        if missing:
            raise AgreementInputError([f"missing column(s): {', '.join(missing)}"])
        for line, row in enumerate(reader, start=2):
            case_id = (row["case_id"] or "").strip()
            raw = (row["human_score"] or "").strip()
            where = f"line {line} ({case_id or 'no case_id'})"
            if case_id in seen:
                problems.append(f"{where}: duplicate case_id")
                continue
            seen.add(case_id)
            if case_id not in judged:
                problems.append(f"{where}: not a judged case in run {run.run_id}")
                continue
            if raw not in {str(s) for s in SCORES}:
                problems.append(f"{where}: human_score must be an integer from 1 to 5, got {raw!r}")
                continue
            pairs.append((int(raw), judged[case_id]))

    if not problems and not pairs:
        problems.append("no scored rows")
    if problems:
        raise AgreementInputError(problems)
    return pairs


def exact_agreement(pairs: list[tuple[int, int]]) -> float:
    return sum(h == j for h, j in pairs) / len(pairs)


def pass_fail_agreement(pairs: list[tuple[int, int]], threshold: int) -> float:
    """Share of cases where human and judge agree on pass/fail at the summary threshold."""
    return sum((h >= threshold) == (j >= threshold) for h, j in pairs) / len(pairs)


def confusion_matrix(pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Counts with rows for the human score and columns for the judge score, both 1 to 5."""
    matrix = [[0] * len(SCORES) for _ in SCORES]
    for h, j in pairs:
        matrix[h - SCORES.start][j - SCORES.start] += 1
    return matrix


def quadratic_weighted_kappa(pairs: list[tuple[int, int]]) -> float | None:
    """Cohen's kappa on the 1 to 5 scale, with disagreements weighted by squared distance.

    Returns None when kappa is undefined: chance alone would produce no disagreement, which happens
    when both raters gave every case the same single score.
    """
    observed = confusion_matrix(pairs)
    n = len(pairs)
    k = len(SCORES)
    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for i in range(k):
        for j in range(k):
            weight = (i - j) ** 2 / (k - 1) ** 2
            observed_disagreement += weight * observed[i][j]
            expected_disagreement += weight * row_totals[i] * col_totals[j] / n
    if expected_disagreement == 0:
        return None
    return 1 - observed_disagreement / expected_disagreement
