#!/usr/bin/env python3
"""按每日累计 Star 数生成 README 使用的静态 SVG。"""

import argparse
import json
import math
import os
import urllib.request
from datetime import date, timedelta
from pathlib import Path


def github_star_count(repository: str) -> int:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "math-modeling-skill-star-history",
        "X-GitHub-Api-Version": "2026-03-10",
    }
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"https://api.github.com/repos/{repository}", headers=headers)
    with urllib.request.urlopen(request, timeout=20) as response:
        count = json.load(response).get("stargazers_count")
    if not isinstance(count, int) or count < 0:
        raise ValueError("GitHub 返回了无效的 Star 数量")
    return count


def update_history(points: list[dict], day: str, count: int) -> list[dict]:
    updated = [point for point in points if point.get("date") != day]
    updated.append({"date": day, "count": count})
    return sorted(updated, key=lambda point: point["date"])


def nice_interval(value: int, target: int = 7) -> int:
    """选取 1/2/5×10^n 刻度间隔，使 0..value 约 target 条主刻度，避免 Y 轴过密。"""
    if value <= 0:
        return 1
    rough = value / target
    mag = 10 ** math.floor(math.log10(rough))
    for multiple in (1, 2, 5, 10):
        if rough <= multiple * mag:
            return multiple * mag
    return 10 * mag


def render_svg(points: list[dict], repository: str, updated: str) -> str:
    samples = [(date.fromisoformat(point["date"]), int(point["count"])) for point in points]
    if not samples:
        raise ValueError("Star 历史不能为空")

    width, height = 960, 540
    left, right, top, bottom = 70, 910, 105, 455
    start = samples[0][0]
    # 右边界延伸到数据最后一天与更新日期的较大者，保证 X 轴覆盖当前月
    end = max(samples[-1][0], date.fromisoformat(updated), samples[0][0] + timedelta(days=1))
    total = samples[-1][1]
    interval = nice_interval(total)
    maximum = math.ceil(total / interval) * interval + interval
    span = (end - start).days
    x = lambda day: left + (day - start).days / span * (right - left)
    y = lambda value: bottom - value / maximum * (bottom - top)
    coordinates = " ".join(f"{x(day):.1f},{y(count):.1f}" for day, count in samples)

    grid = []
    for value in range(0, maximum + 1, interval):
        position = y(value)
        grid.append(
            f'<path d="M{left} {position:.1f}H{right}" fill="none" stroke="#e4ddd0" stroke-width="1"/>'
            f'<text data-y-tick="true" x="58" y="{position + 4:.1f}" text-anchor="end">{value}</text>'
        )

    month = date(start.year, start.month, 1)
    if month < start:
        month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
    months = []
    while month <= end:
        months.append(month)
        month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
    stride = max(1, math.ceil(len(months) / 7))
    month_ticks = months[::stride]
    # 保证最新月份（当前月）始终显示，避免被 stride 跳跃取样跳过
    if months[-1] not in month_ticks:
        month_ticks.append(months[-1])
    month_grid = "".join(
        f'<path d="M{x(month):.1f} {top}V{bottom}" fill="none" stroke="#e4ddd0" stroke-width="1"/>'
        f'<text x="{x(month):.1f}" y="486" text-anchor="middle">{month.month:02d}月</text>'
        for month in month_ticks
    )
    area = f"{x(start):.1f},{bottom} {coordinates} {x(end):.1f},{bottom}"

    return f'''<svg xmlns="http://www.w3.org/2000/svg" data-y-max="{maximum}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{repository} GitHub Star 历史</title>
  <desc id="desc">{start.isoformat()} 至 {updated}，累计获得 {total} 个 Star。</desc>
  <defs>
    <linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#c54f35" stop-opacity=".28"/><stop offset="1" stop-color="#c54f35" stop-opacity=".02"/></linearGradient>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%"><feDropShadow dx="0" dy="4" stdDeviation="7" flood-color="#171917" flood-opacity=".08"/></filter>
  </defs>
  <rect width="960" height="540" rx="20" fill="#f3efe5"/>
  <rect x="20" y="20" width="920" height="500" rx="14" fill="#fffdf8" stroke="#d8d0c0" filter="url(#shadow)"/>
  <g font-family="Segoe UI, Microsoft YaHei, sans-serif">
    <text x="54" y="65" fill="#171917" font-size="20" font-weight="700">GitHub Star 历史</text>
    <text x="54" y="87" fill="#686961" font-size="12">{repository} · 数据更新于 {updated}</text>
    <text x="906" y="67" fill="#c54f35" font-size="28" font-weight="750" text-anchor="end">{total}</text>
    <text x="906" y="86" fill="#686961" font-size="11" text-anchor="end">累计 Stars</text>
    <g fill="#77786f" font-size="10">{''.join(grid)}{month_grid}</g>
    <polygon fill="url(#area)" points="{area}"/>
    <polyline fill="none" stroke="#c54f35" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" points="{coordinates}"/>
    <circle cx="{x(samples[-1][0]):.1f}" cy="{y(total):.1f}" r="5.5" fill="#fffdf8" stroke="#c54f35" stroke-width="3"/>
  </g>
</svg>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="XiaoMaColtAI/math-modeling-skill")
    parser.add_argument("--count", type=int)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--history", type=Path, default=Path("imgs/star-history.json"))
    parser.add_argument("--output", type=Path, default=Path("imgs/star-history.svg"))
    args = parser.parse_args()

    count = args.count if args.count is not None else github_star_count(args.repository)
    if count < 0:
        raise ValueError("Star 数量不能为负数")
    points = json.loads(args.history.read_text(encoding="utf-8"))
    points = update_history(points, args.date, count)
    args.history.write_text(json.dumps(points, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.write_text(render_svg(points, args.repository, args.date), encoding="utf-8")


if __name__ == "__main__":
    main()
