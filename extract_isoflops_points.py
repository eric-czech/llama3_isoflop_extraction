from __future__ import annotations

import csv
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SVG_PATH = Path("isoflops.svg")
CSV_PATH = Path("isoflops_points.csv")
OUTPUT_PNG_PATH = Path("isoflops_reproduction.png")

# Plot-space bounds after the common SVG transform is applied.
PLOT_X_MIN = 65.359354188801
PLOT_X_MAX = 421.910020969368
PLOT_Y_MIN = 12.500099759912018
PLOT_Y_MAX = 277.675795785089

# Major tick calibration points read from the figure geometry/text.
X_TICK_PAGES = [171.92963222223, 293.49990594522603, 415.07017866981903]
X_TICK_VALUES = [1e10, 1e11, 1e12]
Y_TICK_PAGES = [
    257.35939697200104,
    208.38675637907602,
    159.414115786151,
    110.44538094576203,
    61.47274035283698,
    12.500099759912018,
]
Y_TICK_VALUES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Legend entries appear from top to bottom in this order.
LEGEND_ENTRIES = [
    ("6e18", "rgb(90.979004%, 95.292664%, 100%)"),
    ("1e19", "rgb(77.645874%, 90.586853%, 100%)"),
    ("3e19", "rgb(64.312744%, 84.70459%, 100%)"),
    ("6e19", "rgb(51.763916%, 73.724365%, 96.076965%)"),
    ("1e20", "rgb(28.234863%, 64.704895%, 98.03772%)"),
    ("3e20", "rgb(0%, 50.979614%, 98.429871%)"),
    ("6e20", "rgb(16.078186%, 38.430786%, 85.096741%)"),
    ("1e21", "rgb(11.372375%, 29.019165%, 69.802856%)"),
    ("3e21", "rgb(3.137207%, 11.372375%, 41.567993%)"),
    ("1e22", "rgb(0.784302%, 3.921509%, 30.195618%)"),
]


@dataclass(frozen=True)
class Point:
    compute_budget: str
    training_tokens: float
    validation_loss: float
    x_page: float
    y_page: float


def parse_transform(transform: str) -> tuple[float, float]:
    match = re.search(
        r"matrix\([^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*([^,]+),\s*([^)]+)\)",
        transform,
    )
    if not match:
        raise ValueError(f"Unsupported transform: {transform}")
    return float(match.group(1)), float(match.group(2))


def page_x_to_tokens(x_page: float) -> float:
    x0, x1, x2 = X_TICK_PAGES
    # The x-axis is logarithmic, with equal spacing per decade.
    if not math.isclose((x1 - x0), (x2 - x1), rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Unexpected x tick spacing")
    decades = math.log10(X_TICK_VALUES[0]) + (x_page - x0) / (x1 - x0)
    return 10 ** decades


def page_y_to_loss(y_page: float) -> float:
    y0, y1 = Y_TICK_PAGES[0], Y_TICK_PAGES[1]
    v0, v1 = Y_TICK_VALUES[0], Y_TICK_VALUES[1]
    slope = (v1 - v0) / (y1 - y0)
    intercept = v0 - slope * y0
    return slope * y_page + intercept


def rgb_percent_to_mpl(rgb: str) -> tuple[float, float, float]:
    values = [float(v.strip().rstrip("%")) / 100.0 for v in rgb[4:-1].split(",")]
    return tuple(values)  # type: ignore[return-value]


def extract_points() -> list[Point]:
    tree = ET.parse(SVG_PATH)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    color_to_budget = {color: budget for budget, color in LEGEND_ENTRIES}
    points: list[Point] = []

    for path in root.findall(".//svg:path", ns):
        fill = path.attrib.get("fill")
        stroke = path.attrib.get("stroke")
        transform = path.attrib.get("transform")
        d = path.attrib.get("d", "")

        if not transform or fill in (None, "none"):
            continue
        if fill != stroke or stroke not in color_to_budget:
            continue
        # Dot markers are stored as circular paths using cubic curves.
        if " C " not in d:
            continue

        x_page, y_page = parse_transform(transform)
        if not (PLOT_X_MIN <= x_page <= PLOT_X_MAX and PLOT_Y_MIN <= y_page <= PLOT_Y_MAX):
            continue

        points.append(
            Point(
                compute_budget=color_to_budget[stroke],
                training_tokens=page_x_to_tokens(x_page),
                validation_loss=page_y_to_loss(y_page),
                x_page=x_page,
                y_page=y_page,
            )
        )

    points.sort(key=lambda p: (float(p.compute_budget.replace("e", "E")), p.training_tokens, p.validation_loss))
    return points


def write_csv(points: list[Point]) -> None:
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["compute_budget", "training_tokens", "validation_loss", "x_page", "y_page"])
        for point in points:
            writer.writerow(
                [
                    point.compute_budget,
                    f"{point.training_tokens:.12g}",
                    f"{point.validation_loss:.12g}",
                    f"{point.x_page:.6f}",
                    f"{point.y_page:.6f}",
                ]
            )


def render_png(points: list[Point], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0887414, 4.5344151))

    for compute_budget, color in LEGEND_ENTRIES:
        series = [p for p in points if p.compute_budget == compute_budget]
        if not series:
            continue
        x_vals = np.array([p.training_tokens for p in series], dtype=float)
        y_vals = np.array([p.validation_loss for p in series], dtype=float)

        ax.scatter(
            x_vals,
            y_vals,
            s=28,
            color=rgb_percent_to_mpl(color),
            label=compute_budget,
            linewidths=0,
            zorder=3,
        )

        if len(series) >= 3:
            log_x = np.log10(x_vals)
            coeffs = np.polyfit(log_x, y_vals, 2)
            fit_log_x = np.linspace(log_x.min(), log_x.max(), 300)
            fit_y = np.polyval(coeffs, fit_log_x)
            fit_x = 10 ** fit_log_x
            ax.plot(
                fit_x,
                fit_y,
                color=rgb_percent_to_mpl(color),
                linewidth=3,
                solid_capstyle="round",
                zorder=2,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Training Tokens")
    ax.set_ylabel("Validation Loss")
    ax.set_xlim(page_x_to_tokens(PLOT_X_MIN), page_x_to_tokens(PLOT_X_MAX))
    ax.set_ylim(0.68, 0.96)
    ax.grid(True, which="major", linestyle=(0, (3.7, 1.6)), color="#cccccc", linewidth=0.6)
    ax.legend(title="Compute", loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    points = extract_points()
    write_csv(points)
    render_png(points, output_path=OUTPUT_PNG_PATH)
    print(f"Extracted {len(points)} points")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {OUTPUT_PNG_PATH}")


if __name__ == "__main__":
    main()
