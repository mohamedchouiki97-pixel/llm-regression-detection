"""Slack alert for a comparison. Never raises and never changes the exit code."""

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable

from src.models import Comparison, RunRecord, Status

STATUS_ICON = {Status.PASS: "🟢", Status.WARN: "🟡", Status.FAIL: "🔴"}
MAX_LISTED_CASES = 5


def build_slack_message(
    comparison: Comparison, run: RunRecord, baseline: RunRecord | None, report_url: str
) -> dict:
    """Slack Block Kit payload. `text` is the fallback shown in notifications."""
    icon = STATUS_ICON[comparison.status]
    title = f"{icon} {comparison.status.value.upper()}: prompt {run.prompt_version}"
    if baseline is not None:
        title += f" vs {baseline.prompt_version}"
    title += f" (dataset v{run.dataset_version})"

    lines = []
    if baseline is None:
        lines.append(f"Pass rate {run.pass_rate:.1%} · no baseline to compare against")
    else:
        pass_rate = comparison.overall[0]
        headline = (
            f"Pass rate {pass_rate.baseline:.1%} → {pass_rate.run:.1%} ({pass_rate.delta * 100:+.1f} pp)"
            f" · {len(comparison.regressions)} regressions · {len(comparison.improvements)} improvements"
        )
        if comparison.mcnemar_p is not None:
            headline += f" · p = {comparison.mcnemar_p:.3g}"
        lines.append(headline)

        worst = min(comparison.by_category, key=lambda d: d.delta)
        if worst.delta < 0:
            lines.append(f"Worst hit: {worst.group} {worst.delta * 100:+.1f} pp")

        if comparison.regressions:
            ids = [change.case_id for change in comparison.regressions]
            listed = ", ".join(ids[:MAX_LISTED_CASES])
            more = len(ids) - MAX_LISTED_CASES
            lines.append(f"Regressed: {listed}" + (f" (+{more} more)" if more > 0 else ""))

    for reason in comparison.reasons:
        if reason.startswith("drift:") or "errored" in reason:
            lines.append(reason[0].upper() + reason[1:])

    link = f"<{report_url}|View report>" if report_url.startswith("http") else f"Report: {report_url}"
    lines.append(link)

    body = "\n".join(lines)
    return {
        "text": f"{title}\n{body}",
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*{title}*"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": body}},
        ],
    }


def _post(url: str, payload: dict) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()


def notify(
    payload: dict,
    webhook_url: str | None = None,
    post: Callable[[str, dict], None] = _post,
) -> str:
    """Send the payload if a webhook is configured. Returns a one line status for the terminal."""
    webhook_url = webhook_url if webhook_url is not None else os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return "Slack: skipped (SLACK_WEBHOOK_URL not set)"
    try:
        post(webhook_url, payload)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return f"Slack: failed to send ({type(exc).__name__}: {exc}); continuing"
    return "Slack: sent"
