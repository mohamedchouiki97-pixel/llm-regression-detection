"""Command line entry point."""

import argparse
import asyncio
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.cache import DEFAULT_CACHE_DIR, Cache
from src.classifier import ClassificationError, classify_email
from src.golden import DEFAULT_DATASET_PATH, DatasetLoadError, check_dataset, coverage_table, load_dataset
from src.models import ClassifyOutput, GoldenDataset, PromptConfig, RunRecord
from src.prompts import PromptLoadError, load_prompt
from src.runner import DEFAULT_CONCURRENCY, build_run_record, git_info, run_eval
from src.scoring import DEFAULT_SUMMARY_THRESHOLD, JUDGE_MODEL, judge_summary
from src.store import DEFAULT_DB_PATH, save_run


async def run_classify(email_text: str, config: PromptConfig) -> ClassifyOutput:
    async with AsyncOpenAI() as client:
        return await classify_email(email_text, config, client)


def cmd_classify(args: argparse.Namespace) -> int:
    try:
        config = load_prompt(args.prompt)
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
        config = load_prompt(args.prompt)
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
    return 0


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(prog="mrd", description="Model regression detection system")
    sub = parser.add_subparsers(dest="command", required=True)

    classify = sub.add_parser("classify", help="Classify one email with a prompt version")
    classify.add_argument("--prompt", default="v1", help="Prompt version (e.g. v1) or path to a YAML file")
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
    run.add_argument("--prompt", default="v1", help="Prompt version (e.g. v1) or path to a YAML file")
    run.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to the dataset JSON")
    run.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to the SQLite run history")
    run.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Cases in flight at once")
    run.add_argument(
        "--summary-threshold",
        type=int,
        choices=range(1, 6),
        default=DEFAULT_SUMMARY_THRESHOLD,
        help=f"Minimum judge score for a case to pass (default {DEFAULT_SUMMARY_THRESHOLD})",
    )
    run.add_argument("--no-cache", action="store_true", help="Always call the API; ignore cached responses")
    run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
