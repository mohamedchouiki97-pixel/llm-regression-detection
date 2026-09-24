"""Self-contained HTML report for one comparison: verdict, scorecard, trend chart, and case evidence."""

import html
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup

from src.compare import DRIFT_WINDOW
from src.models import CaseResult, Comparison, GoldenCase, GoldenDataset, RunRecord

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = ROOT / "templates"
DEFAULT_REPORT_DIR = ROOT / "reports"
TREND_RUNS = 20


@dataclass
class TrendPoint:
    run_id: str
    prompt_version: str
    pass_rate: float
    moving_average: float | None
    current: bool


@dataclass
class CaseView:
    case_id: str
    golden: GoldenCase | None
    before: CaseResult | None
    after: CaseResult


def trend_points(run: RunRecord, history: list[RunRecord], window: int = DRIFT_WINDOW) -> list[TrendPoint]:
    """history is newest first. The run is always the last point, even when it is not on main.

    The moving average follows the drift rule: it only exists once a full window of runs is available.
    """
    series = list(reversed([r for r in history if r.run_id != run.run_id])) + [run]
    points = []
    for i, r in enumerate(series):
        window_runs = series[max(0, i - window + 1) : i + 1]
        average = sum(w.pass_rate for w in window_runs) / window if len(window_runs) == window else None
        points.append(TrendPoint(r.run_id, r.prompt_version, r.pass_rate, average, r.run_id == run.run_id))
    return points[-TREND_RUNS:]


def _y_domain(values: list[float], floor: float) -> tuple[float, float]:
    """Zoom to the data, in 5 point steps, always including the drift floor and 100%."""
    low = min(values + [floor]) - 0.02
    low = max(0.0, int(low * 20) / 20)
    return low, 1.0


def trend_svg(points: list[TrendPoint], floor: float, width: int = 720, height: int = 240) -> Markup:
    """Inline SVG line chart of pass rate and its moving average, with the drift floor as a reference line."""
    left, right, top, bottom = 44, 96, 12, 24
    plot_w, plot_h = width - left - right, height - top - bottom
    low, high = _y_domain([p.pass_rate for p in points], floor)

    def x(i: int) -> float:
        if len(points) == 1:
            return left + plot_w / 2
        return left + plot_w * i / (len(points) - 1)

    def y(value: float) -> float:
        return top + plot_h * (high - value) / (high - low)

    parts = [
        f'<svg class="trend" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Pass rate over the last {len(points)} runs">'
    ]

    step = 0.05 if high - low <= 0.3 else 0.1
    tick = high
    while tick >= low - 1e-9:
        ty = y(tick)
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{ty:.1f}" y2="{ty:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 6}" y="{ty + 4:.1f}" text-anchor="end">{tick:.0%}</text>')
        tick = round(tick - step, 9)

    fy = y(floor)
    parts.append(f'<line class="floor" x1="{left}" x2="{left + plot_w}" y1="{fy:.1f}" y2="{fy:.1f}"/>')
    parts.append(f'<text class="floor-label" x="{left + plot_w + 6}" y="{fy + 4:.1f}">drift floor {floor:.0%}</text>')

    average = [(x(i), y(p.moving_average)) for i, p in enumerate(points) if p.moving_average is not None]
    if len(average) >= 2:
        coords = " ".join(f"{px:.1f},{py:.1f}" for px, py in average)
        parts.append(f'<polyline class="series-2" points="{coords}"/>')

    if len(points) >= 2:
        coords = " ".join(f"{x(i):.1f},{y(p.pass_rate):.1f}" for i, p in enumerate(points))
        parts.append(f'<polyline class="series-1" points="{coords}"/>')

    for i, p in enumerate(points):
        px, py = x(i), y(p.pass_rate)
        radius = 6 if p.current else 4
        parts.append(f'<circle class="{"marker current" if p.current else "marker"}" cx="{px:.1f}" cy="{py:.1f}" r="{radius}"/>')
        label = f"{p.run_id} · prompt {p.prompt_version} · pass rate {p.pass_rate:.1%}"
        if p.moving_average is not None:
            label += f" · {DRIFT_WINDOW} run average {p.moving_average:.1%}"
        if p.current:
            label += " · this run"
        # Transparent hit area, bigger than the mark, carrying a native tooltip.
        parts.append(
            f'<circle class="hit" cx="{px:.1f}" cy="{py:.1f}" r="12" tabindex="0">'
            f"<title>{html.escape(label)}</title></circle>"
        )

    last = points[-1]
    parts.append(
        f'<text class="end-label" x="{x(len(points) - 1) + 10:.1f}" y="{y(last.pass_rate) - 8:.1f}">'
        f"{last.pass_rate:.1%}</text>"
    )
    parts.append(f'<text class="tick" x="{left}" y="{height - 6}">oldest</text>')
    parts.append(f'<text class="tick" x="{left + plot_w}" y="{height - 6}" text-anchor="end">this run</text>')
    parts.append("</svg>")
    return Markup("".join(parts))


def case_views(changes_or_results, golden_by_id: dict[str, GoldenCase]) -> list[CaseView]:
    views = []
    for item in changes_or_results:
        if isinstance(item, CaseResult):
            views.append(CaseView(item.case_id, golden_by_id.get(item.case_id), None, item))
        else:
            views.append(CaseView(item.case_id, golden_by_id.get(item.case_id), item.before, item.after))
    return views


def build_report_context(
    run: RunRecord,
    baseline: RunRecord | None,
    comparison: Comparison,
    history: list[RunRecord],
    dataset: GoldenDataset | None,
) -> dict:
    dataset_matches = dataset is not None and dataset.version == run.dataset_version
    golden_by_id = {case.id: case for case in dataset.cases} if dataset_matches else {}
    trend = trend_points(run, history)
    return {
        "run": run,
        "baseline": baseline,
        "comparison": comparison,
        "dataset_matches": dataset_matches,
        "trend": trend,
        "trend_svg": trend_svg(trend, comparison.drift.floor),
        "regressions": case_views(comparison.regressions, golden_by_id),
        "improvements": case_views(comparison.improvements, golden_by_id),
        "failures": case_views([r for r in run.results if not r.passed], golden_by_id),
    }


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _pp(delta: float) -> str:
    return f"{delta * 100:+.1f} pp"


def render_report(context: dict) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=True, trim_blocks=True, lstrip_blocks=True)
    env.filters["pct"] = _pct
    env.filters["pp"] = _pp
    return env.get_template("report.html.j2").render(**context)


def write_report(context: dict, report_dir: Path | str = DEFAULT_REPORT_DIR) -> Path:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{context['run'].run_id}.html"
    path.write_text(render_report(context), encoding="utf-8")
    return path
