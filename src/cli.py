"""Command line entry point."""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.classifier import ClassificationError, classify_email
from src.golden import DEFAULT_DATASET_PATH, DatasetLoadError, check_dataset, coverage_table, load_dataset
from src.models import ClassifyOutput, PromptConfig
from src.prompts import PromptLoadError, load_prompt


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

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
