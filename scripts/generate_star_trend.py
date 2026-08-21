#!/usr/bin/env python3
"""Generate a GitHub README-friendly SVG for repository star history."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_ROOT = "https://api.github.com"
DEFAULT_REPOSITORY = "Tencent/YOLO-Master"
SVG_WIDTH = 960
SVG_HEIGHT = 470


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/star-history.svg"),
        help="Output SVG path",
    )
    return parser.parse_args()


def github_json(
    url: str, token: Optional[str], accept: str = "application/vnd.github+json"
) -> Tuple[object, Dict[str, str]]:
    headers = {
        "Accept": accept,
        "User-Agent": "YOLO-Master-star-trend",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            return json.load(response), response_headers
    except (HTTPError, URLError) as error:
        detail = getattr(error, "read", lambda: b"")()
        raise RuntimeError(f"GitHub API request failed for {url}: {error}; {detail[:300]!r}") from error


def next_page(link_header: Optional[str]) -> Optional[str]:
    if not link_header:
        return None
    for link in link_header.split(","):
        parts = [part.strip() for part in link.split(";")]
        if len(parts) < 2 or 'rel="next"' not in parts[1:]:
            continue
        return parts[0].strip("<>")
    return None


def fetch_starred_at(repository: str, token: Optional[str]) -> List[datetime]:
    url = f"{API_ROOT}/repos/{repository}/stargazers?per_page=100"
    timestamps: List[datetime] = []
    while url:
        payload, headers = github_json(url, token, accept="application/vnd.github.star+json")
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected stargazer response for {repository}: {type(payload).__name__}")
        for item in payload:
            starred_at = item.get("starred_at") if isinstance(item, dict) else None
            if not starred_at:
                raise RuntimeError("GitHub returned a stargazer without starred_at; cannot build a time series")
            timestamps.append(datetime.fromisoformat(starred_at.replace("Z", "+00:00")).astimezone(timezone.utc))
        url = next_page(headers.get("link"))
    if not timestamps:
        raise RuntimeError(f"No stargazer timestamps returned for {repository}")
    return sorted(timestamps)


def daily_series(timestamps: Iterable[datetime]) -> List[Tuple[datetime, int, int]]:
    dates = [timestamp.date() for timestamp in timestamps]
    counts = Counter(dates)
    current = min(dates)
    end = max(dates)
    cumulative = 0
    series: List[Tuple[datetime, int, int]] = []
    while current <= end:
        daily = counts.get(current, 0)
        cumulative += daily
        series.append((datetime.combine(current, datetime.min.time(), tzinfo=timezone.utc), daily, cumulative))
        current += timedelta(days=1)
    return series


def number(value: float) -> str:
    return f"{value:,.0f}"


def svg_text(value: str) -> str:
    return html.escape(value, quote=True)


def render_svg(repository: str, series: List[Tuple[datetime, int, int]], observed_at: datetime) -> str:
    left, right = 72, 930
    top, cumulative_bottom = 82, 285
    daily_top, daily_bottom = 340, 410
    plot_width = right - left
    day_count = max(1, len(series) - 1)
    max_cumulative = max(item[2] for item in series)
    max_daily = max(item[1] for item in series)
    max_daily = max(1, ((max_daily + 4) // 5) * 5)

    def x(index: int) -> float:
        return left + plot_width * index / day_count

    def y_cumulative(value: int) -> float:
        return cumulative_bottom - (cumulative_bottom - top) * value / max_cumulative

    def y_daily(value: int) -> float:
        return daily_bottom - (daily_bottom - daily_top) * value / max_daily

    line_points = " ".join(f"{x(index):.1f},{y_cumulative(item[2]):.1f}" for index, item in enumerate(series))
    bar_width = max(1.8, plot_width / max(1, len(series)) * 0.78)
    bars = []
    for index, item in enumerate(series):
        if not item[1]:
            continue
        bar_height = daily_bottom - y_daily(item[1])
        bars.append(
            f'<rect x="{x(index) - bar_width / 2:.1f}" y="{y_daily(item[1]):.1f}" '
            f'width="{bar_width:.1f}" height="{bar_height:.1f}" rx="1" />'
        )

    tick_indices = sorted({round(index * (len(series) - 1) / 7) for index in range(8)})
    ticks = []
    for index in tick_indices:
        date = series[index][0]
        label = date.strftime("%b %d")
        ticks.append(
            f'<line class="x-grid" x1="{x(index):.1f}" y1="{top}" x2="{x(index):.1f}" y2="{daily_bottom}" />'
            f'<text class="x-label" x="{x(index):.1f}" y="442" text-anchor="middle">{svg_text(label)}</text>'
        )

    cumulative_ticks = [0, max_cumulative // 4, max_cumulative // 2, (max_cumulative * 3) // 4, max_cumulative]
    cumulative_grid = []
    for value in cumulative_ticks:
        y = y_cumulative(value)
        cumulative_grid.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" />'
            f'<text class="y-label" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{number(value)}</text>'
        )
    daily_grid = []
    for value in (0, max_daily // 2, max_daily):
        y = y_daily(value)
        daily_grid.append(
            f'<line class="daily-grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" />'
            f'<text class="daily-label" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{number(value)}</text>'
        )

    first_date = series[0][0].strftime("%Y-%m-%d")
    last_date = series[-1][0].strftime("%Y-%m-%d")
    total = series[-1][2]
    final_x = x(len(series) - 1)
    final_y = y_cumulative(total)
    title = f"{repository} GitHub Stars"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img" aria-labelledby="title desc">
  <title id="title">{svg_text(title)} history</title>
  <desc id="desc">Cumulative GitHub stars from {first_date} to {last_date}; {total} stars at the latest observation.</desc>
  <style>
    .background {{ fill: #ffffff; }}
    .title {{ fill: #202124; font: 700 22px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .subtitle, .caption {{ fill: #5f6368; font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .axis-title {{ fill: #202124; font: 600 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .grid, .daily-grid, .x-grid {{ stroke: #e6e8eb; stroke-width: 1; shape-rendering: crispEdges; }}
    .x-grid {{ stroke-dasharray: 2 4; }}
    .y-label, .daily-label, .x-label {{ fill: #5f6368; font: 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .line {{ fill: none; stroke: #0f9d58; stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
    .area {{ fill: #0f9d58; opacity: .10; }}
    .bar {{ fill: #1f6feb; opacity: .82; }}
    .point {{ fill: #0f9d58; stroke: #ffffff; stroke-width: 2; }}
    .frame {{ fill: none; stroke: #dfe1e5; stroke-width: 1; }}
  </style>
  <rect class="background" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" />
  <text class="title" x="{left}" y="34">{svg_text(title)}</text>
  <text class="subtitle" x="{left}" y="56">{svg_text(first_date)} to {svg_text(last_date)} · Data through {observed_at.strftime("%Y-%m-%d %H:%M UTC")}</text>
  <text class="caption" x="{left}" y="74">Cumulative stars</text>
  <rect class="frame" x="{left}" y="{top}" width="{plot_width}" height="{cumulative_bottom - top}" />
  {"".join(cumulative_grid)}
  {"".join(ticks)}
  <polygon class="area" points="{left},{cumulative_bottom} {line_points} {right},{cumulative_bottom}" />
  <polyline class="line" points="{line_points}" />
  <circle class="point" cx="{final_x:.1f}" cy="{final_y:.1f}" r="5" />
  <text class="axis-title" x="{min(right - 4, final_x - 8):.1f}" y="{max(top + 16, final_y - 12):.1f}" text-anchor="end">{number(total)} stars</text>
  <text class="caption" x="{left}" y="326">Daily new stars</text>
  <rect class="frame" x="{left}" y="{daily_top}" width="{plot_width}" height="{daily_bottom - daily_top}" />
  {"".join(daily_grid)}
  <g class="bar">{"".join(bars)}</g>
  <text class="caption" x="{right}" y="466" text-anchor="end">Source: GitHub stargazer timestamps · {svg_text(repository)}</text>
</svg>
'''


def main() -> int:
    args = parse_args()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    timestamps = fetch_starred_at(args.repo, token)
    series = daily_series(timestamps)
    output = render_svg(args.repo, series, timestamps[-1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"Generated {args.output} for {len(timestamps)} stars across {len(series)} days")
    return 0


if __name__ == "__main__":
    sys.exit(main())
