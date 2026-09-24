"""Command line entry point."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.alerts import build_slack_message, notify
from src.cache import DEFAULT_CACHE_DIR, Cache
from src.classifier import ClassificationError, classify_email
from src.compare import (
    DEFAULT_DRIFT_FLOOR,
    DEFAULT_FAIL_DELTA,
    DEFAULT_MAX_ERROR_RATE,
    DEFAULT_WARN_DELTA,
    DRIFT_WINDOW,
    compare_runs,
)
from src.golden import DEFAULT_DATASET_PATH, DatasetLoadError, check_dataset, coverage_table, load_dataset
from src.models import (
    CaseChange,
    ClassifyOutput,
    Comparison,
    GoldenDataset,
    GroupDelta,
    PromptConfig,
    RunRecord,
    Status,
)
from src.prompts import PromptLoadError, active_prompt_version, load_prompt
from src.report import DEFAULT_REPORT_DIR, TREND_RUNS, build_report_context, render_markdown, write_report
from src.runner import DEFAULT_CONCURRENCY, build_run_record, git_info, run_eval
from src.scoring import DEFAULT_SUMMARY_THRESHOLD, JUDGE_MODEL, judge_summary
from src.store import DEFAULT_DB_PATH, comparable_main_runs, latest_run_id, load_run, save_run


def env_setting(name: str, default, cast=str):
    """A setting from the environment, for Docker and CI. Command line flags still override it."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise SystemExit(f"error: environment variable {name}={raw!r} is not a valid {cast.__name__}")


def resolve_prompt(prompt: str | None) -> PromptConfig:
    """An explicit version or path, or else the active production prompt named in prompts/ACTIVE."""
    return load_prompt(prompt or active_prompt_version())


async def run_classify(email_text: str, config: PromptConfig) -> ClassifyOutput:
    async with AsyncOpenAI() as client:
        return await classify_email(email_text, config, client)


def cmd_classify(args: argparse.Namespace) -> int:
    try:
        config = resolve_prompt(args.prompt)
    except PromptLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    email_text = Path(args.email_file).read_text(encoding="utf-8") if args.email_file else args.email

    try:
        output = asyncio.run(run_classify(email_text, config))
    except ClassificationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"category:       {output.result.category.value}")
    print(f"summary:        {output.result.summary}")
    print(f"prompt_version: {output.prompt_version}")
    print(f"model:          {output.model}")
    print(f"latency_ms:     {output.latency_ms:.0f}")
    print(f"input_tokens:   {output.input_tokens}")
    print(f"output_tokens:  {output.output_tokens}")
    return 0


def cmd_validate_dataset(args: argparse.Namespace) -> int:
    try:
        dataset = load_dataset(args.path)
    except DatasetLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("FAILED: dataset could not be loaded", file=sys.stderr)
        return 1

    report = check_dataset(dataset, min_category_share=args.min_category_share)

    print(coverage_table(dataset))
    if report.warnings:
        print()
        for warning in report.warnings:
            print(f"warning: {warning}")
    if report.errors:
        print()
        for error in report.errors:
            print(f"error: {error}")

    print()
    status = "OK" if report.ok else "FAILED"
    print(f"{status}: {len(report.errors)} errors, {len(report.warnings)} warnings")
    return 0 if report.ok else 1


async def run_full_eval(
    dataset: GoldenDataset, config: PromptConfig, args: argparse.Namespace
) -> RunRecord:
    cache = Cache(None if args.no_cache else DEFAULT_CACHE_DIR)

    def progress(done: int, total: int) -> None:
        print(f"\r{done}/{total}", end="" if done < total else "\n", file=sys.stderr, flush=True)

    # Retries are handled by the runner, so the SDK's own retries are off.
    async with AsyncOpenAI(max_retries=0) as client:
        results = await run_eval(
            dataset,
            config,
            classify=lambda email, cfg: classify_email(email, cfg, client),
            judge=lambda email, ideal, summary: judge_summary(email, ideal, summary, client, JUDGE_MODEL),
            cache=cache,
            judge_model=JUDGE_MODEL,
            threshold=args.summary_threshold,
            concurrency=args.concurrency,
            on_progress=progress,
        )
    return build_run_record(results, config, dataset, JUDGE_MODEL, args.summary_threshold, git_info())


def format_cost(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.3f}"


def print_run_summary(run: RunRecord) -> None:
    scored = [r for r in run.results if r.summary_score is not None]
    answered = [r for r in run.results if r.predicted_category is not None]
    category_acc = sum(r.category_match for r in answered) / len(answered) if answered else 0.0
    mean_summary = sum(r.summary_score for r in scored) / len(scored) if scored else 0.0
    dirty = " (uncommitted changes)" if run.git_dirty else ""

    print(
        f"Run {run.run_id}  prompt {run.prompt_version}  dataset v{run.dataset_version}  "
        f"git {run.git_sha[:7]}{dirty}"
    )
    print(f"pass rate        {run.pass_rate:.1%}  ({run.n_passed}/{run.n_cases})")
    print(f"category acc     {category_acc:.1%}")
    print(f"mean summary     {mean_summary:.2f}")
    print(f"errors           {run.n_errors}")
    print(
        f"est. cost        {format_cost(run.classifier_cost_usd)} classifier"
        f" + {format_cost(run.judge_cost_usd)} judge"
    )
    print(f"cached           {run.n_cached}/{run.n_cases}")
    for result in run.results:
        if result.error:
            print(f"  error {result.case_id}: {result.error}")


def cmd_run(args: argparse.Namespace) -> int:
    try:
        config = resolve_prompt(args.prompt)
    except PromptLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        dataset = load_dataset(args.dataset)
    except DatasetLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = check_dataset(dataset)
    if not report.ok:
        for error in report.errors:
            print(f"error: {error}", file=sys.stderr)
        print("dataset has errors; run `mrd validate-dataset` and fix them first", file=sys.stderr)
        return 2

    try:
        run = asyncio.run(run_full_eval(dataset, config, args))
    except openai.OpenAIError as exc:
        # Reached only for setup problems such as a missing API key; per-case errors are recorded instead.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    save_run(run, args.db)
    print_run_summary(run)
    if args.run_id_file:
        Path(args.run_id_file).write_text(run.run_id, encoding="utf-8")
    return 0


def print_group_table(title: str, deltas: list[GroupDelta]) -> None:
    print(f"{title:<12}{'baseline':>10}{'run':>10}{'delta':>10}")
    for d in deltas:
        print(f"{d.group:<12}{d.baseline_rate:>10.1%}{d.run_rate:>10.1%}{d.delta * 100:>+9.1f}pp")


def print_case_changes(title: str, changes: list[CaseChange]) -> None:
    print(f"{title} ({len(changes)})")
    for change in changes:
        before, after = change.before, change.after
        print(
            f"  {change.case_id} [{after.expected_category.value}, {after.difficulty.value}, {after.split.value}]  "
            f"predicted {before.predicted_category.value if before.predicted_category else '-'}"
            f" -> {after.predicted_category.value if after.predicted_category else '-'}, "
            f"summary score {before.summary_score or '-'} -> {after.summary_score or '-'}"
        )
        if after.error:
            print(f"    error: {after.error}")


def print_comparison(comparison: Comparison) -> None:
    print(f"{comparison.status.value.upper()}: run {comparison.run_id} vs baseline {comparison.baseline_id or '(none)'}")
    for reason in comparison.reasons:
        print(f"  - {reason}")
    for note in comparison.notes:
        print(f"  note: {note}")

    if comparison.baseline_id is not None:
        print()
        print(f"{'metric':<20}{'baseline':>10}{'run':>10}{'delta':>10}")
        for m in comparison.overall:
            if m.name == "mean_summary_score":
                print(f"{m.name:<20}{m.baseline:>10.2f}{m.run:>10.2f}{m.delta:>+10.2f}")
            else:
                print(f"{m.name:<20}{m.baseline:>10.1%}{m.run:>10.1%}{m.delta * 100:>+9.1f}pp")
        for title, deltas in [
            ("category", comparison.by_category),
            ("difficulty", comparison.by_difficulty),
            ("split", comparison.by_split),
        ]:
            print()
            print_group_table(title, deltas)
        print()
        print(
            f"changed predictions: {comparison.prediction_changes}, "
            f"changed summary scores: {comparison.score_changes}"
        )
        print_case_changes("regressions", comparison.regressions)
        print_case_changes("improvements", comparison.improvements)

    drift = comparison.drift
    print()
    if drift.moving_average is None:
        print(f"drift: not enough history ({len(drift.run_ids)}/{drift.window} runs)")
    else:
        print(f"drift: {drift.window} run average {drift.moving_average:.1%} (floor {drift.floor:.0%})")


def cmd_compare(args: argparse.Namespace) -> int:
    run_id = args.run or latest_run_id(args.db)
    if run_id is None:
        print("error: no runs stored yet; run `mrd run` first", file=sys.stderr)
        return 2
    try:
        run = load_run(run_id, args.db)
        # Enough history for the trend chart; drift and the baseline only use the newest runs.
        history = comparable_main_runs(run, args.db, limit=TREND_RUNS + DRIFT_WINDOW - 2)
        baseline = load_run(args.baseline, args.db) if args.baseline else (history[0] if history else None)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2

    comparison = compare_runs(
        run,
        baseline,
        history,
        warn_delta=args.warn_delta,
        fail_delta=args.fail_delta,
        drift_floor=args.drift_floor,
        max_error_rate=args.max_error_rate,
    )
    print_comparison(comparison)

    try:
        dataset = load_dataset(args.dataset)
    except DatasetLoadError:
        dataset = None
    report_path = write_report(build_report_context(run, baseline, comparison, history, dataset), args.report_dir)
    print()
    print(f"report: {report_path}")

    # In CI, REPORT_URL points at the uploaded report; locally the file path is the best we have.
    report_url = os.environ.get("REPORT_URL") or str(report_path)
    if args.markdown:
        Path(args.markdown).write_text(render_markdown(comparison, run, baseline, report_url), encoding="utf-8")
        print(f"markdown: {args.markdown}")

    if args.notify:
        payload = build_slack_message(comparison, run, baseline, report_url)
        print()
        print("Slack message:")
        for line in payload["text"].splitlines():
            print(f"  {line}")
        print(notify(payload))

    return 1 if comparison.status is Status.FAIL else 0


def main() -> None:
    load_dotenv()
    # Windows consoles default to cp1252, which cannot print emoji or arrows. Replace them instead of
    # crashing, since a crash would exit 1 and look exactly like a failed regression check.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")

    db_path = env_setting("MRD_DB", str(DEFAULT_DB_PATH))

    parser = argparse.ArgumentParser(prog="mrd", description="Model regression detection system")
    sub = parser.add_subparsers(dest="command", required=True)

    classify = sub.add_parser("classify", help="Classify one email with a prompt version")
    classify.add_argument("--prompt", help="Prompt version (e.g. v1) or YAML path (default: prompts/ACTIVE)")
    source = classify.add_mutually_exclusive_group(required=True)
    source.add_argument("--email", help="Email text")
    source.add_argument("--email-file", help="Path to a file containing the email text")
    classify.set_defaults(func=cmd_classify)

    validate = sub.add_parser("validate-dataset", help="Check the golden dataset and print coverage")
    validate.add_argument("--path", default=str(DEFAULT_DATASET_PATH), help="Path to the dataset JSON")
    validate.add_argument(
        "--min-category-share",
        type=float,
        default=0.15,
        help="Warn when a category is below this share of cases (default 0.15)",
    )
    validate.set_defaults(func=cmd_validate_dataset)

    run = sub.add_parser("run", help="Run the full golden dataset and store the run")
    run.add_argument("--prompt", help="Prompt version (e.g. v1) or YAML path (default: prompts/ACTIVE)")
    run.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to the dataset JSON")
    run.add_argument("--db", default=db_path, help="Path to the SQLite run history (env MRD_DB)")
    run.add_argument(
        "--concurrency",
        type=int,
        default=env_setting("MRD_CONCURRENCY", DEFAULT_CONCURRENCY, int),
        help="Cases in flight at once (env MRD_CONCURRENCY)",
    )
    run.add_argument(
        "--summary-threshold",
        type=int,
        choices=range(1, 6),
        default=env_setting("MRD_SUMMARY_THRESHOLD", DEFAULT_SUMMARY_THRESHOLD, int),
        help=f"Minimum judge score for a case to pass (default {DEFAULT_SUMMARY_THRESHOLD}, env MRD_SUMMARY_THRESHOLD)",
    )
    run.add_argument("--no-cache", action="store_true", help="Always call the API; ignore cached responses")
    run.add_argument("--run-id-file", help="Write the new run id to this file, for scripts and CI")
    run.set_defaults(func=cmd_run)

    compare = sub.add_parser("compare", help="Compare a run against a baseline and decide pass, warn, or fail")
    compare.add_argument("--run", help="Run id to evaluate (default: the latest run)")
    compare.add_argument(
        "--baseline", help="Run id to compare against (default: latest comparable clean run on main)"
    )
    compare.add_argument("--db", default=db_path, help="Path to the SQLite run history (env MRD_DB)")
    compare.add_argument(
        "--warn-delta",
        type=float,
        default=env_setting("MRD_WARN_DELTA", DEFAULT_WARN_DELTA, float),
        help=f"Warn when overall pass rate drops by this much (default {DEFAULT_WARN_DELTA}, env MRD_WARN_DELTA)",
    )
    compare.add_argument(
        "--fail-delta",
        type=float,
        default=env_setting("MRD_FAIL_DELTA", DEFAULT_FAIL_DELTA, float),
        help=f"Fail when overall pass rate drops by this much (default {DEFAULT_FAIL_DELTA}, env MRD_FAIL_DELTA)",
    )
    compare.add_argument(
        "--drift-floor",
        type=float,
        default=env_setting("MRD_DRIFT_FLOOR", DEFAULT_DRIFT_FLOOR, float),
        help=f"Warn when the {DRIFT_WINDOW} run average pass rate is below this "
        f"(default {DEFAULT_DRIFT_FLOOR}, env MRD_DRIFT_FLOOR)",
    )
    compare.add_argument(
        "--max-error-rate",
        type=float,
        default=env_setting("MRD_MAX_ERROR_RATE", DEFAULT_MAX_ERROR_RATE, float),
        help=f"Fail as incomplete when more cases than this share errored (default {DEFAULT_MAX_ERROR_RATE}, "
        "env MRD_MAX_ERROR_RATE)",
    )
    compare.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Dataset JSON, for email text in the report")
    compare.add_argument(
        "--report-dir",
        default=env_setting("MRD_REPORT_DIR", str(DEFAULT_REPORT_DIR)),
        help="Where to write the HTML report (env MRD_REPORT_DIR)",
    )
    compare.add_argument("--markdown", help="Also write a Markdown summary here, for a PR comment")
    compare.add_argument("--notify", action="store_true", help="Send the result to Slack if SLACK_WEBHOOK_URL is set")
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
