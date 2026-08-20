import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class DeliveryProtocolTests(unittest.TestCase):
    def test_protocol_preserves_a_submit_ready_version(self):
        text = read("references/交付与截止时间协议.md")
        for token in (
            "可提交性优于局部完美",
            "Checkpoint V1",
            "Checkpoint V2",
            "内容冻结",
            "最终导出",
            "不得再次回到没有任何可提交版本的状态",
            "未经用户或平台证据确认，不得声称已经提交",
        ):
            self.assertIn(token, text)

    def test_protocol_requires_official_deadline_and_end_to_end_rehearsal(self):
        text = read("references/交付与截止时间协议.md")
        for token in (
            "当前系统时间",
            "当届官方网站",
            "时区",
            "最小真实样例",
            "人工检查",
            "压缩",
            "上传",
            "显著缓冲",
        ):
            self.assertIn(token, text)
        self.assertNotRegex(text, r"9\s*月\s*13\s*日")

    def test_protocol_is_routed_through_root_and_paper_role(self):
        reference = "references/交付与截止时间协议.md"
        self.assertIn(reference, read("SKILL.md"))
        self.assertIn("交付与截止时间协议.md", read("references/README.md"))
        self.assertIn(reference, read("references/roles/论文手/SKILL.md"))

    def test_markdown_residual_and_support_package_checks_are_wired(self):
        writing = read("references/roles/论文手/SKILL.md")
        workflow = read("references/roles/论文手/references/工作流程.md")
        checklist = read("references/roles/论文手/references/自审框架.md")
        review = read("references/Subagent调度.md")
        for text in (writing, workflow, checklist, review):
            self.assertIn("Markdown 格式残留", text)
        self.assertIn("实际解压", workflow)
        self.assertIn("支撑材料完整性", review)


if __name__ == "__main__":
    unittest.main()
