import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "update_star_history.py"


class StarHistoryTests(unittest.TestCase):
    def _render(self, points, count, day):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        history = Path(directory.name) / "history.json"
        output = Path(directory.name) / "star-history.svg"
        history.write_text(json.dumps(points), encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--count",
                str(count),
                "--date",
                day,
                "--history",
                str(history),
                "--output",
                str(output),
            ],
            check=True,
        )
        return output.read_text(encoding="utf-8"), json.loads(history.read_text(encoding="utf-8"))

    def test_generates_16_9_svg_with_dynamic_y_ticks(self):
        svg, points = self._render(
            [{"date": "2026-07-31", "count": 548}],
            558,
            "2026-08-01",
        )
        ET.fromstring(svg)
        self.assertIn('viewBox="0 0 960 540"', svg)
        # 558 → 动态间隔 100 → 0..700，共 8 条刻度
        self.assertIn('data-y-max="700"', svg)
        self.assertEqual(svg.count("data-y-tick"), 8)
        self.assertEqual(points[-1], {"date": "2026-08-01", "count": 558})

    def test_y_axis_interval_grows_with_star_count(self):
        # 5000 → 动态间隔 1000 → 0..6000，共 7 条刻度（不随总量线性膨胀）
        svg, _ = self._render(
            [{"date": f"2026-0{m}-01", "count": 100 * m} for m in range(1, 9)],
            5000,
            "2026-09-01",
        )
        self.assertIn('data-y-max="6000"', svg)
        self.assertEqual(svg.count("data-y-tick"), 7)

    def test_x_axis_includes_updated_month(self):
        # 数据止于 8 月，但更新日期为 9 月 → X 轴必须出现 09月
        svg, _ = self._render(
            [{"date": f"2026-0{m}-01", "count": 100 * m} for m in range(1, 9)],
            900,
            "2026-09-06",
        )
        self.assertIn("09月", svg)


if __name__ == "__main__":
    unittest.main()
