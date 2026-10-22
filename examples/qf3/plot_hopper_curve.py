# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write an SVG of the finite Hopper eval curve next to the paper's QF3 means.

The paper series is figure ``fig:mujoco-curves`` for Hopper-v4, file
``Hopper-v4__FLOP.csv`` in the arXiv source (FLOP is the name in that source).
Our series are means of the eval episodes in ``evals.csv``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _means(path: Path) -> list[tuple[float, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[int(row["env_step"])].append(float(row["return"]))
    return [(float(step), sum(values) / len(values)) for step, values in sorted(grouped.items())]


def _paper(path: Path) -> list[tuple[float, float, float, float]]:
    points: list[tuple[float, float, float, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            points.append((float(row["step"]), float(row["mean"]), float(row["lo"]), float(row["hi"])))
    return points


def _svg(
    paper: list[tuple[float, float, float, float]],
    series: list[tuple[str, list[tuple[float, float]]]],
    out: Path,
    x_max: float,
) -> None:
    width, height = 760, 420
    left, right, top, bottom = 64, 20, 24, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    ys = [point[1] for point in paper if point[0] <= x_max]
    for _, points in series:
        ys.extend(point[1] for point in points if point[0] <= x_max)
    y_max = max(ys + [1.0]) * 1.05

    def x_of(step: float) -> float:
        return left + plot_w * step / x_max

    def y_of(value: float) -> float:
        return top + plot_h * (1.0 - value / y_max)

    colors = {"paper QF3": "#2A6F97", "qf3": "#C98A3B", "direct_q": "#009E73"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
    ]
    for tick in range(0, int(x_max) + 1, 20_000):
        x = x_of(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 4}" stroke="#222"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle" font-size="11">{tick // 1000}k</text>'
        )
    for fraction in (0.0, 0.5, 1.0):
        value = y_max * fraction
        y = y_of(value)
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#222"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{value:.0f}</text>'
        )

    def polyline(points: list[tuple[float, float]], color: str) -> str:
        kept = [(x_of(step), y_of(value)) for step, value in points if step <= x_max]
        body = " ".join(f"{x:.1f},{y:.1f}" for x, y in kept)
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{body}"/>'

    paper_points = [(step, mean) for step, mean, _, _ in paper]
    parts.append(polyline(paper_points, colors["paper QF3"]))
    legend = [("paper QF3", colors["paper QF3"])]
    for name, points in series:
        color = colors.get(name, "#222222")
        parts.append(polyline(points, color))
        legend.append((name, color))
    for index, (name, color) in enumerate(legend):
        y = 36 + index * 16
        parts.append(f'<line x1="520" y1="{y}" x2="548" y2="{y}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="554" y="{y + 4}" font-size="12">{name}</text>')
    parts.append(
        f'<text x="{left + plot_w / 2}" y="{height - 8}" text-anchor="middle" font-size="12">'
        "Hopper-v4 env steps</text>"
    )
    parts.append(
        f'<text x="16" y="{top + plot_h / 2}" font-size="12" transform="rotate(-90 16,{top + plot_h / 2})">'
        "eval return</text>"
    )
    parts.append("</svg>")
    out.write_text("\n".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the finite Hopper curve.")
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--run", action="append", nargs=2, metavar=("NAME", "EVALS_CSV"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--x-max", type=float, default=100_000)
    args = parser.parse_args()
    series = [(name, _means(Path(path))) for name, path in args.run]
    _svg(_paper(args.paper), series, args.out, args.x_max)


if __name__ == "__main__":
    main()
