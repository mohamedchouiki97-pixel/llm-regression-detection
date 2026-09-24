import re

from src.compare import compare_runs
from src.models import Category, GoldenDataset
from src.report import (
    TREND_RUNS,
    build_report_context,
    render_markdown,
    render_report,
    trend_points,
    trend_svg,
    write_report,
)
from test_compare import BASE_FAILS, make_run, result, run_with_failures, run_with_rate


def dataset_for(n: int = 100, version: int = 3, email: str = "Email {i}") -> GoldenDataset:
    categories = list(Category)
    return GoldenDataset.model_validate(
        {
            "version": version,
            "changelog": [{"version": version, "date": "2026-09-24", "changes": "x"}],
            "cases": [
                {
                    "id": f"gc-{i:03d}",
                    "email": email.format(i=i),
                    "expected_category": categories[i % 4].value,
                    "ideal_summary": f"Ideal summary {i}.",
                    "difficulty": "easy",
                    "split": "dev",
                }
                for i in range(1, n + 1)
            ],
        }
    )


def render(run, baseline, history=(), dataset=None) -> str:
    comparison = compare_runs(run, baseline, list(history))
    return render_report(build_report_context(run, baseline, comparison, list(history), dataset))


# Trend chart


def test_trend_points_order_and_current_run_last():
    history = [run_with_rate(f"h{i}", 0.9, minutes=-i) for i in range(1, 4)]  # newest first
    points = trend_points(run_with_rate("new", 0.92), history)
    assert [p.run_id for p in points] == ["h3", "h2", "h1", "new"]
    assert [p.current for p in points] == [False, False, False, True]


def test_trend_moving_average_only_with_full_window():
    history = [run_with_rate(f"h{i}", 0.9, minutes=-i) for i in range(1, 8)]
    points = trend_points(run_with_rate("new", 0.9), history)
    assert [p.moving_average is not None for p in points] == [False] * 6 + [True] * 2


def test_trend_is_capped():
    history = [run_with_rate(f"h{i}", 0.9, minutes=-i) for i in range(1, 40)]
    points = trend_points(run_with_rate("new", 0.9), history)
    assert len(points) == TREND_RUNS
    assert points[-1].run_id == "new"


def _circle_centres(svg: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in re.findall(r'class="marker[^"]*" cx="([\d.]+)" cy="([\d.]+)"', svg)]


def test_svg_single_point():
    svg = str(trend_svg(trend_points(run_with_rate("new", 0.92), []), floor=0.88))
    assert len(_circle_centres(svg)) == 1
    assert "<polyline" not in svg


def test_svg_all_points_inside_plot():
    history = [run_with_rate(f"h{i}", rate, minutes=-i) for i, rate in enumerate([0.5, 0.99, 0.8, 1.0, 0.7], 1)]
    svg = str(trend_svg(trend_points(run_with_rate("new", 0.6), history), floor=0.88))
    width, height = 720, 240
    centres = _circle_centres(svg)
    assert len(centres) == 6
    assert all(0 <= x <= width and 0 <= y <= height for x, y in centres)


def test_svg_equal_values_do_not_divide_by_zero():
    history = [run_with_rate(f"h{i}", 1.0, minutes=-i) for i in range(1, 5)]
    svg = str(trend_svg(trend_points(run_with_rate("new", 1.0), history), floor=1.0))
    assert "nan" not in svg.lower()


def test_svg_tooltips_escape_run_ids():
    run = run_with_rate("<b>x</b>", 0.9)
    svg = str(trend_svg(trend_points(run, []), floor=0.88))
    assert "<b>x</b>" not in svg
    assert "&lt;b&gt;x&lt;/b&gt;" in svg


# Report


def test_report_shows_verdict_and_regressed_cases():
    baseline = run_with_failures("base", BASE_FAILS)
    run = run_with_failures("new", BASE_FAILS | set(range(50, 60)), minutes=1)
    page = render(run, baseline, dataset=dataset_for())
    assert "FAIL" in page
    assert "Regressed cases (10)" in page
    for i in range(50, 60):
        assert f'id="gc-{i:03d}"' in page
    assert "Email 50" in page and "Ideal summary 50." in page


def test_report_shows_judge_reasoning_side_by_side():
    baseline = make_run("base", [result(1, True)])
    run = make_run("new", [result(1, False, score=2)], minutes=1)
    run.results[0].judge_reasoning = "Misses the refund request entirely."
    baseline.results[0].judge_reasoning = "Matches the reference."
    page = render(run, baseline, dataset=dataset_for(1))
    assert "Misses the refund request entirely." in page
    assert "Matches the reference." in page


def test_report_escapes_email_html():
    baseline = run_with_failures("base", set())
    run = run_with_failures("new", {1}, minutes=1)
    page = render(run, baseline, dataset=dataset_for(email="<script>alert('x')</script> {i}"))
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page


def test_report_without_baseline():
    page = render(run_with_failures("new", BASE_FAILS), None, dataset=dataset_for())
    assert "PASS" in page
    assert "no baseline to compare against" in page
    assert "Scorecard" not in page
    assert "All failing cases in this run (8)" in page


def test_report_dataset_mismatch_hides_email():
    baseline = run_with_failures("base", set())
    run = run_with_failures("new", {1}, minutes=1)
    page = render(run, baseline, dataset=dataset_for(version=9))
    assert "Email 1" not in page
    assert "Email text unavailable" in page


def test_write_report_uses_run_id(tmp_path):
    run = run_with_failures("20260924T120000-abc123", BASE_FAILS)
    comparison = compare_runs(run, None, [])
    path = write_report(build_report_context(run, None, comparison, [], None), tmp_path)
    assert path == tmp_path / "20260924T120000-abc123.html"
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")


# Markdown PR comment


def markdown(run, baseline, url="https://github.com/o/r/actions/runs/1/artifacts/2"):
    return render_markdown(compare_runs(run, baseline, []), run, baseline, url)


def test_markdown_has_marker_and_verdict():
    baseline = run_with_failures("base", BASE_FAILS)
    run = run_with_failures("new", BASE_FAILS | set(range(50, 62)), minutes=1, prompt_version="v2")
    text = markdown(run, baseline)
    assert text.startswith("<!-- mrd-eval-report -->\n### 🔴 FAIL: prompt v2 vs v1 (dataset v3)")
    assert "| Pass rate | 92.0% | 80.0% | -12.0 pp |" in text
    assert "[Full report](https://github.com/o/r/actions/runs/1/artifacts/2)" in text


def test_markdown_caps_listed_regressions():
    baseline = run_with_failures("base", BASE_FAILS)
    run = run_with_failures("new", BASE_FAILS | set(range(50, 62)), minutes=1)
    text = markdown(run, baseline)
    assert "**Regressed cases (12)**" in text
    assert "- `gc-059` (" in text
    assert "`gc-060`" not in text
    assert "- and 2 more in the report" in text


def test_markdown_without_baseline():
    text = markdown(run_with_failures("new", BASE_FAILS), None, url="reports/x.html")
    assert "### 🟢 PASS: prompt v1 (dataset v3)" in text
    assert "no baseline to compare against" in text
    assert "| Metric |" not in text
    assert "Full report: `reports/x.html`" in text
