import urllib.error

from src.alerts import build_slack_message, notify
from src.compare import compare_runs
from src.models import Category
from test_compare import BASE_FAILS, make_run, result, run_with_failures


def message(run, baseline, url="https://example.com/report.html"):
    return build_slack_message(compare_runs(run, baseline, []), run, baseline, url)


def test_fail_message_headline():
    baseline = run_with_failures("base", BASE_FAILS)
    run = run_with_failures("new", BASE_FAILS | set(range(50, 62)), minutes=1, prompt_version="v2")
    text = message(run, baseline)["text"]
    assert text.startswith("🔴 FAIL: prompt v2 vs v1 (dataset v3)")
    assert "Pass rate 92.0% → 80.0% (-12.0 pp) · 12 regressions · 0 improvements · p = 0.000488" in text
    assert "<https://example.com/report.html|View report>" in text


def test_regression_list_is_truncated():
    baseline = run_with_failures("base", BASE_FAILS)
    run = run_with_failures("new", BASE_FAILS | set(range(50, 62)), minutes=1)
    text = message(run, baseline)["text"]
    assert "Regressed: gc-050, gc-051, gc-052, gc-053, gc-054 (+7 more)" in text


def test_worst_category_named():
    b = Category.BILLING
    baseline = make_run("base", [result(i, True, b if i <= 4 else None) for i in range(1, 21)])
    run = make_run("new", [result(i, i > 2, b if i <= 4 else None) for i in range(1, 21)], minutes=1)
    # Billing has 8 cases (1 to 4, plus every 4th by the helper), so losing 2 is -25 pp.
    assert "Worst hit: billing -25.0 pp" in message(run, baseline)["text"]


def test_pass_and_warn_icons():
    baseline = run_with_failures("base", BASE_FAILS)
    assert message(run_with_failures("new", BASE_FAILS, minutes=1), baseline)["text"].startswith("🟢 PASS")
    warn = run_with_failures("new", BASE_FAILS | {50, 51, 52}, minutes=1)
    assert message(warn, baseline)["text"].startswith("🟡 WARN")


def test_no_baseline_message():
    text = message(run_with_failures("new", BASE_FAILS), None)["text"]
    assert "Pass rate 92.0% · no baseline to compare against" in text
    assert "Regressed" not in text


def test_local_path_is_not_a_link():
    text = message(run_with_failures("new", BASE_FAILS), None, url="reports/x.html")["text"]
    assert "Report: reports/x.html" in text


def test_blocks_mirror_text():
    payload = message(run_with_failures("new", BASE_FAILS), None)
    assert payload["blocks"][0]["text"]["text"].startswith("*🟢 PASS")


def test_notify_skips_without_webhook(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    calls = []
    assert notify({"text": "x"}, post=lambda url, p: calls.append(url)) == "Slack: skipped (SLACK_WEBHOOK_URL not set)"
    assert calls == []


def test_notify_posts_to_env_webhook(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/T/B/X")
    calls = []
    assert notify({"text": "x"}, post=lambda url, p: calls.append((url, p))) == "Slack: sent"
    assert calls == [("https://hooks.slack.com/services/T/B/X", {"text": "x"})]


def test_notify_failure_is_caught():
    def broken(url, payload):
        raise urllib.error.URLError("network down")

    status = notify({"text": "x"}, webhook_url="https://hooks.slack.com/x", post=broken)
    assert status.startswith("Slack: failed to send (URLError")
