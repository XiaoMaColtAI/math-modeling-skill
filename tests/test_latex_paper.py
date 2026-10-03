import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "tools" / "latex" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import latex_paper
from latex_paper import (
    _audit_pdf,
    _font_descriptor_embedded,
    _replace_pair,
    build_paper,
    doctor,
    inspect_paper,
    prepare_project,
    source_bundle_sha256,
)


class LatexPaperTests(unittest.TestCase):
    def test_prepares_bundled_template_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "完整论文-LaTeX"
            result = prepare_project(output, contest="cumcm")

            self.assertEqual(Path(result["main_tex"]), output / "main.tex")
            self.assertTrue((output / "references.bib").is_file())
            self.assertTrue((output / "latex-project.json").is_file())
            with self.assertRaises(FileExistsError):
                prepare_project(output, contest="cumcm")

    def test_copies_complete_official_template_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "official"
            template.mkdir()
            (template / "paper.tex").write_text(
                "\\documentclass{official}\\begin{document}x\\end{document}",
                encoding="utf-8",
            )
            (template / "cover.tex").write_text("cover", encoding="utf-8")
            (template / "official.cls").write_text("official class", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "主入口"):
                prepare_project(root / "ambiguous", template_path=template)
            result = prepare_project(
                root / "paper",
                template_path=template,
                main_file="paper.tex",
            )

            self.assertEqual(Path(result["main_tex"]).name, "paper.tex")
            self.assertTrue((root / "paper" / "official.cls").is_file())
            self.assertTrue((root / "paper" / "cover.tex").is_file())

    def test_project_entries_skip_configured_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            (root / ".git" / "HEAD").write_text("metadata", encoding="utf-8")
            (root / "sections" / ".cache").mkdir(parents=True)
            (root / "sections" / ".cache" / "data").write_text("cache", encoding="utf-8")
            (root / "sections" / "main.tex").write_text("source", encoding="utf-8")
            (root / "worktree").mkdir()
            (root / "worktree" / ".git").write_text("gitdir: elsewhere", encoding="utf-8")

            original_paths = {
                path.relative_to(root).as_posix() for path in root.rglob("*")
            }
            self.assertEqual(original_paths, {
                ".git", ".git/HEAD", "sections", "sections/.cache",
                "sections/.cache/data", "sections/main.tex", "worktree",
                "worktree/.git",
            })

            with patch.object(
                latex_paper, "PROJECT_IGNORED_NAMES", frozenset({".git", ".cache"})
            ):
                paths = {
                    path.relative_to(root).as_posix()
                    for path in latex_paper._project_entries(root)
                }

            self.assertEqual(paths, {"sections", "sections/main.tex", "worktree"})

    def test_project_files_only_yields_nonignored_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "main.tex").write_text("source", encoding="utf-8")
            (root / ".keep").write_text("kept", encoding="utf-8")
            (root / "figures").mkdir()
            (root / "figures" / "plot.png").write_bytes(b"image")
            (root / ".git").mkdir()
            (root / ".git" / "HEAD").write_text("metadata", encoding="utf-8")

            original_paths = {
                item.relative_to(root).as_posix()
                for item in root.rglob("*") if item.is_file()
            }
            self.assertEqual(original_paths, {
                "main.tex", ".keep", "figures/plot.png", ".git/HEAD",
            })

            paths = {
                path.relative_to(root).as_posix()
                for path in latex_paper._project_files(root)
            }
            self.assertEqual(paths, {"main.tex", ".keep", "figures/plot.png"})

    def test_supports_generic_template_with_nested_main_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "official"
            (template / "src").mkdir(parents=True)
            (template / "src" / "paper.tex").write_text(
                r"\documentclass{article}\begin{document}x\end{document}",
                encoding="utf-8",
            )

            result = prepare_project(
                root / "paper",
                contest="generic",
                template_path=template,
                main_file="src/paper.tex",
                template_source="https://contest.example/template",
                template_version="2026",
            )
            manifest = json.loads(
                (root / "paper" / "latex-project.json").read_text(encoding="utf-8")
            )

            self.assertEqual(Path(result["main_tex"]), root / "paper/src/paper.tex")
            self.assertEqual(manifest["main_tex"], "src/paper.tex")
            self.assertEqual(manifest["template"]["version"], "2026")
            with self.assertRaisesRegex(ValueError, "必须"):
                prepare_project(root / "missing", contest="generic")

    def test_inspects_nested_sources_figures_tables_and_bibliography(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sections").mkdir()
            (root / "figures").mkdir()
            (root / "figures" / "result.png").write_bytes(b"png")
            (root / "sections" / "model.tex").write_text(
                r"""
\section{模型}
正文引用图表~\cref{fig:result,tab:result}和文献~\cite{smith2026}。
\begin{equation} y=ax+b \end{equation}
\begin{figure}\includegraphics{result}\caption{结果}\label{fig:result}\end{figure}
\begin{longtable}{cc}\caption{参数}\label{tab:result}a&b\end{longtable}
""",
                encoding="utf-8",
            )
            (root / "references.bib").write_text(
                "@article{smith2026, title={Verified model}}",
                encoding="utf-8",
            )
            main = root / "main.tex"
            main.write_text(
                r"""
\documentclass{article}
\newcommand{\keywords}[1]{#1}
\graphicspath{{figures/}}
\begin{document}
\begin{abstract}摘要\keywords{模型；验证}\end{abstract}
\input{sections/model}
\bibliography{references}
\end{document}
""",
                encoding="utf-8",
            )

            report = inspect_paper(main)

            self.assertTrue(report["passed"], report["issues"])
            self.assertEqual(report["metrics"]["source_files"], 2)
            self.assertEqual(report["metrics"]["equations"], 1)
            self.assertEqual(report["metrics"]["figures"], 1)
            self.assertEqual(report["metrics"]["tables"], 1)
            self.assertEqual(report["metrics"]["citations"], 1)

    def test_reports_placeholders_orphans_and_missing_citations(self):
        with tempfile.TemporaryDirectory() as temporary:
            main = Path(temporary) / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}
\begin{abstract}LATEX_TEMPLATE_ABSTRACT\keywords{test}\end{abstract}
\begin{figure}\caption{result}\label{fig:orphan}\end{figure}
See \cite{missing}.\end{document}
""",
                encoding="utf-8",
            )

            issues = inspect_paper(main)["issues"]

            self.assertTrue(any("模板占位符" in item for item in issues))
            self.assertTrue(any("孤儿图" in item for item in issues))
            self.assertTrue(any("缺少参考文献条目" in item for item in issues))

    def test_reports_duplicate_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            main = Path(temporary) / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}\begin{abstract}summary\keywords{test}\end{abstract}
\section{A}\label{sec:duplicate}\section{B}\label{sec:duplicate}
\end{document}
""",
                encoding="utf-8",
            )

            issues = inspect_paper(main)["issues"]

            self.assertTrue(any("重复 label" in item for item in issues))

    def test_all_contests_share_the_eight_figure_quality_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            main = Path(temporary) / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}\begin{abstract}summary\keywords{test}\end{abstract}
\end{document}
""",
                encoding="utf-8",
            )

            for contest in ("cumcm", "mcm-icm"):
                report = inspect_paper(
                    main,
                    contest=contest,
                    quality_checks=True,
                    min_content_units=0,
                    min_pages=0,
                    min_equations=0,
                    min_tables=0,
                    require_pdf=False,
                    questions=["q1"],
                    override_reason="单元测试仅检查统一图数量默认值",
                )
                self.assertTrue(
                    any("图 0，低于质量目标 8" in issue for issue in report["issues"]),
                    (contest, report["issues"]),
                )

    def test_rejects_sources_outside_project_and_missing_engine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main = root / "paper" / "main.tex"
            main.parent.mkdir()
            main.write_text(
                "\\documentclass{article}\\begin{document}\\input{../secret}\\end{document}",
                encoding="utf-8",
            )
            (root / "secret.tex").write_text("secret", encoding="utf-8")
            with self.assertRaises(ValueError):
                inspect_paper(main)

            main.write_text(
                "\\documentclass{article}\\begin{document}ok\\end{document}",
                encoding="utf-8",
            )
            with patch("latex_paper.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "未找到"):
                    build_paper(main)

    def test_latexmk_build_uses_relative_output_and_disables_shell_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main = root / "main.tex"
            main.write_text(
                "\\documentclass{article}\\begin{document}ok\\end{document}",
                encoding="utf-8",
            )
            observed = {}

            def fake_run(command, **kwargs):
                observed.update({"command": command, **kwargs})
                build = Path(kwargs["cwd"]) / "build"
                (build / "main.pdf").write_bytes(b"pdf")
                (build / "main.log").write_text("clean build", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch(
                "latex_paper.shutil.which",
                side_effect=lambda name: name if name in {"latexmk", "xelatex"} else None,
            ), patch(
                "latex_paper._tool_version", return_value="test version"
            ), patch("latex_paper.subprocess.run", side_effect=fake_run):
                report = build_paper(main)

            self.assertTrue(report["passed"])
            self.assertIn("-norc", observed["command"])
            self.assertIn("-outdir=build", observed["command"])
            self.assertTrue(any("-no-shell-escape" in item for item in observed["command"]))
            self.assertEqual(observed["env"]["openin_any"], "p")
            self.assertEqual(observed["env"]["openout_any"], "p")

    def test_build_gate_rejects_layout_and_font_warnings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main = root / "main.tex"
            main.write_text(
                "\\documentclass{article}\\begin{document}ok\\end{document}",
                encoding="utf-8",
            )

            def fake_run(_command, **kwargs):
                build = Path(kwargs["cwd"]) / "build"
                (build / "main.pdf").write_bytes(b"pdf")
                (build / "main.log").write_text(
                    "Overfull \\hbox (2.0pt too wide)\n"
                    "LaTeX Font Warning: Font shape unavailable",
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch(
                "latex_paper.shutil.which",
                side_effect=lambda name: name if name in {"latexmk", "xelatex"} else None,
            ), patch("latex_paper.subprocess.run", side_effect=fake_run):
                report = build_paper(main)

        self.assertFalse(report["passed"])
        self.assertTrue(any("Overfull" in issue for issue in report["issues"]))
        self.assertTrue(any("Font Warning" in issue for issue in report["issues"]))

    def test_external_bibliography_requires_latexmk(self):
        with tempfile.TemporaryDirectory() as temporary:
            main = Path(temporary) / "main.tex"
            main.write_text(
                r"\documentclass{article}\begin{document}\cite{x}"
                r"\bibliography{references}\end{document}",
                encoding="utf-8",
            )
            (Path(temporary) / "references.bib").write_text(
                "@article{x,title={x}}",
                encoding="utf-8",
            )

            with patch(
                "latex_paper.shutil.which",
                side_effect=lambda name: None if name == "latexmk" else name,
            ):
                with self.assertRaisesRegex(RuntimeError, "latexmk"):
                    build_paper(main)

    def test_rejects_empty_figures_and_tables_even_when_counts_are_met(self):
        with tempfile.TemporaryDirectory() as temporary:
            main = Path(temporary) / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}\begin{abstract}摘要\keywords{测试}\end{abstract}
参见图~\ref{fig:q1-empty}与表~\ref{tab:empty}。
\begin{figure}\caption{空图}\label{fig:q1-empty}\end{figure}
\begin{table}\caption{空表}\label{tab:empty}\end{table}
\end{document}
""",
                encoding="utf-8",
            )

            report = inspect_paper(main)

            self.assertFalse(report["passed"])
            self.assertTrue(any("没有图片或绘图内容" in issue for issue in report["issues"]))
            self.assertTrue(any("没有表格数据" in issue for issue in report["issues"]))

    def test_counts_each_double_dollar_formula_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            main = Path(temporary) / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}\begin{abstract}摘要\keywords{测试}\end{abstract}
$$a=1$$
$$b=2$$
$$c=3$$
\end{document}
""",
                encoding="utf-8",
            )

            report = inspect_paper(main)

            self.assertEqual(report["metrics"]["equations"], 3)

    def test_line_break_spacing_is_not_counted_as_display_math(self):
        with tempfile.TemporaryDirectory() as temporary:
            main = Path(temporary) / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}\begin{abstract}摘要\\[0.4em]\keywords{测试}\end{abstract}
\[a=1\]
\end{document}
""",
                encoding="utf-8",
            )

            report = inspect_paper(main)

            self.assertEqual(report["metrics"]["equations"], 1)
            self.assertFalse(any(r"\[ 与 \]" in issue for issue in report["issues"]))

    def test_odd_backslash_run_still_starts_display_math(self):
        with tempfile.TemporaryDirectory() as temporary:
            main = Path(temporary) / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}\begin{abstract}摘要\keywords{测试}\end{abstract}
\\\[a=1\]
\end{document}
""",
                encoding="utf-8",
            )

            report = inspect_paper(main)

            self.assertEqual(report["metrics"]["equations"], 1)
            self.assertFalse(any(r"\[ 与 \]" in issue for issue in report["issues"]))

    def test_quality_thresholds_require_valid_values_and_override_reason(self):
        with tempfile.TemporaryDirectory() as temporary:
            main = Path(temporary) / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}\begin{abstract}摘要\keywords{测试}\end{abstract}\end{document}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "不能为负数"):
                inspect_paper(main, min_figures=-1)
            with self.assertRaisesRegex(ValueError, "override_reason"):
                inspect_paper(
                    main,
                    quality_checks=True,
                    min_figures=1,
                    require_pdf=False,
                    questions=["q1"],
                )
            with self.assertRaisesRegex(ValueError, "跳过 PDF"):
                inspect_paper(
                    main,
                    contest="mcm-icm",
                    quality_checks=True,
                    require_pdf=False,
                    questions=["q1"],
                )
            with self.assertRaisesRegex(ValueError, "min_image_dpi"):
                inspect_paper(
                    main,
                    contest="mcm-icm",
                    quality_checks=True,
                    min_image_dpi=150,
                    questions=["q1"],
                )
            report = inspect_paper(
                main,
                quality_checks=True,
                min_figures=1,
                require_pdf=False,
                questions=["q1"],
                override_reason="官方赛题只要求一张图",
            )
            self.assertEqual(report["threshold_overrides"][0]["name"], "min_figures")

    def test_requires_each_declared_question_to_have_a_formal_figure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "q1.png").write_bytes(b"png")
            main = root / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}\begin{abstract}摘要\keywords{测试}\end{abstract}
参见图~\ref{fig:q1-result}。
\begin{figure}\includegraphics{q1.png}\caption{问题一}\label{fig:q1-result}\end{figure}
\end{document}
""",
                encoding="utf-8",
            )

            report = inspect_paper(
                main,
                contest="mcm-icm",
                quality_checks=True,
                require_pdf=False,
                questions=["q1", "q2"],
                min_figures=1,
                override_reason="单元测试只构造一张图",
            )

            self.assertTrue(report["metrics"]["question_figure_coverage"]["q1"])
            self.assertFalse(report["metrics"]["question_figure_coverage"]["q2"])
            self.assertTrue(any("q2" in issue for issue in report["issues"]))

    def test_rejects_svg_in_safe_compilation_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "result.svg").write_text("<svg/>", encoding="utf-8")
            main = root / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}\begin{abstract}摘要\keywords{测试}\end{abstract}
参见图~\ref{fig:q1-svg}。
\begin{figure}\includegraphics{result.svg}\caption{结果}\label{fig:q1-svg}\end{figure}
\end{document}
""",
                encoding="utf-8",
            )

            report = inspect_paper(main)

            self.assertTrue(any("PDF/PNG/JPG" in issue for issue in report["issues"]))

    def test_warning_build_does_not_publish_unless_explicitly_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "project"
            root.mkdir()
            main = root / "main.tex"
            main.write_text(
                r"\documentclass{article}\begin{document}ok\end{document}",
                encoding="utf-8",
            )
            published = parent / "完整论文.pdf"

            def fake_run(_command, **kwargs):
                build = Path(kwargs["cwd"]) / "build"
                (build / "main.pdf").write_bytes(b"pdf")
                (build / "main.log").write_text(
                    "Overfull \\hbox (0.1pt too wide)", encoding="utf-8"
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            patches = (
                patch(
                    "latex_paper.shutil.which",
                    side_effect=lambda name: name if name in {"latexmk", "xelatex"} else None,
                ),
                patch("latex_paper.subprocess.run", side_effect=fake_run),
            )
            with patches[0], patches[1]:
                blocked = build_paper(main, publish_path=published)
            self.assertFalse(blocked["passed"])
            self.assertFalse(published.exists())

            with patch(
                "latex_paper.shutil.which",
                side_effect=lambda name: name if name in {"latexmk", "xelatex"} else None,
            ), patch("latex_paper.subprocess.run", side_effect=fake_run):
                allowed = build_paper(
                    main,
                    publish_path=published,
                    allow_warnings=[r"Overfull"],
                    override_reason="确认 0.1pt 不影响版面",
                )

            self.assertTrue(allowed["passed"])
            self.assertTrue(published.is_file())
            self.assertTrue(published.with_suffix(".build.json").is_file())

    def test_pdf_must_match_current_source_and_build_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "project"
            root.mkdir()
            main = root / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}\begin{abstract}摘要\keywords{测试}\end{abstract}\end{document}
""",
                encoding="utf-8",
            )
            pdf = parent / "完整论文.pdf"
            pdf.write_bytes(b"bound pdf")
            manifest = {
                "passed": True,
                "main_tex": "main.tex",
                "source_sha256": source_bundle_sha256(main),
                "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            }
            pdf.with_suffix(".build.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            audit = {
                "pages": 20,
                "blank_pages": [],
                "page_sizes_pt": [[595.3, 841.9]],
                "unembedded_fonts": [],
                "raster_images": 0,
                "min_image_dpi": None,
                "issues": [],
            }

            with patch("latex_paper._audit_pdf", return_value=audit):
                bound = inspect_paper(main, pdf_path=pdf)
            self.assertTrue(bound["passed"], bound["issues"])

            main.write_text(main.read_text(encoding="utf-8") + "% changed", encoding="utf-8")
            with patch("latex_paper._audit_pdf", return_value=audit):
                stale = inspect_paper(main, pdf_path=pdf)
            self.assertTrue(any("源码哈希" in issue for issue in stale["issues"]))

    def test_cumcm_page_limit_counts_body_without_abstract_or_appendix(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "project"
            root.mkdir()
            main = root / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}
\begin{abstract}摘要\keywords{测试}\end{abstract}
\section{正文}正文
\appendix
\section{支撑材料与代码清单}代码
\end{document}
""",
                encoding="utf-8",
            )
            pdf = parent / "完整论文.pdf"
            pdf.write_bytes(b"bound pdf")
            pdf.with_suffix(".build.json").write_text(
                json.dumps({
                    "passed": True,
                    "main_tex": "main.tex",
                    "source_sha256": source_bundle_sha256(main),
                    "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                }),
                encoding="utf-8",
            )
            audit = {
                "pages": 35,
                "blank_pages": [],
                "page_sizes_pt": [[595.3, 841.9]],
                "unembedded_fonts": [],
                "raster_images": 0,
                "min_image_dpi": None,
                "_page_texts": [
                    "摘要",
                    *(f"正文第{number}页" for number in range(1, 31)),
                    "支撑材料与代码清单",
                    "程序代码",
                    "程序代码",
                    "程序代码",
                ],
                "issues": [],
            }

            with patch("latex_paper._audit_pdf", return_value=audit):
                report = inspect_paper(main, pdf_path=pdf, max_pages=30)

            self.assertEqual(report["body_pages"], 30)
            self.assertFalse(any("超过官方上限" in issue for issue in report["issues"]))

    def test_appendix_page_argument_requires_appendix_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            main = Path(temporary) / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}\begin{abstract}摘要\keywords{测试}\end{abstract}
\section{正文}正文
\end{document}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"\\appendix"):
                inspect_paper(main, appendix_start_page=2)

    def test_non_cumcm_page_limit_uses_total_pdf_pages(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "project"
            root.mkdir()
            main = root / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}
\begin{abstract}Summary\keywords{test}\end{abstract}
\section{Body}Body
\appendix
\section{Appendix code}Code
\end{document}
""",
                encoding="utf-8",
            )
            pdf = parent / "paper.pdf"
            pdf.write_bytes(b"bound pdf")
            pdf.with_suffix(".build.json").write_text(
                json.dumps({
                    "passed": True,
                    "main_tex": "main.tex",
                    "source_sha256": source_bundle_sha256(main),
                    "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                }),
                encoding="utf-8",
            )
            audit = {
                "pages": 35,
                "blank_pages": [],
                "page_sizes_pt": [[595.3, 841.9]],
                "unembedded_fonts": [],
                "raster_images": 0,
                "min_image_dpi": None,
                "_page_texts": [
                    "Summary",
                    *(f"Body page {number}" for number in range(2, 31)),
                    "Appendix code",
                    "Code",
                    "Code",
                    "Code",
                    "Code",
                ],
                "issues": [],
            }

            for contest in ("mcm-icm", "generic"):
                with self.subTest(contest=contest), patch(
                    "latex_paper._audit_pdf",
                    return_value={**audit, "_page_texts": list(audit["_page_texts"])},
                ):
                    report = inspect_paper(
                        main,
                        contest=contest,
                        pdf_path=pdf,
                        max_pages=30,
                    )

                self.assertTrue(
                    any("PDF 总页数 35" in issue for issue in report["issues"]),
                    report["issues"],
                )

    def test_pdf_audit_detects_blank_pages_and_page_size_changes(self):
        from pypdf import PdfWriter

        with tempfile.TemporaryDirectory() as temporary:
            pdf = Path(temporary) / "paper.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=595, height=842)
            writer.add_blank_page(width=612, height=792)
            with pdf.open("wb") as handle:
                writer.write(handle)

            with patch(
                "latex_paper._pdf_image_dpi", return_value=([150, 150], [])
            ):
                audit = _audit_pdf(
                    pdf, min_image_dpi=300, require_image_audit=True
                )

            self.assertEqual(audit["blank_pages"], [1, 2])
            self.assertTrue(any("页面尺寸不一致" in issue for issue in audit["issues"]))
            self.assertTrue(any("150 DPI" in issue for issue in audit["issues"]))

    def test_pdf_font_embedding_check_distinguishes_standard_and_external_fonts(self):
        from pypdf.generic import DictionaryObject, NameObject

        standard = DictionaryObject({
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        external = DictionaryObject({
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/ExternalFont"),
        })

        self.assertTrue(_font_descriptor_embedded(standard))
        self.assertFalse(_font_descriptor_embedded(external))

    def test_doctor_reports_selected_toolchain(self):
        with patch(
            "latex_paper.shutil.which",
            side_effect=lambda name: name if name in {"latexmk", "xelatex"} else None,
        ), patch("latex_paper._tool_version", return_value="test version"):
            report = doctor(need_pdf_audit=False)

        self.assertTrue(report["passed"])
        self.assertEqual(report["tools"]["latexmk"]["path"], "latexmk")

    def test_unusable_latexmk_falls_back_to_engine_without_bibliography(self):
        with tempfile.TemporaryDirectory() as temporary:
            main = Path(temporary) / "main.tex"
            main.write_text(
                r"\documentclass{article}\begin{document}ok\end{document}",
                encoding="utf-8",
            )
            commands = []

            def fake_run(command, **kwargs):
                commands.append(command)
                build = Path(kwargs["cwd"]) / "build"
                (build / "main.pdf").write_bytes(b"pdf")
                (build / "main.log").write_text("clean build", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch(
                "latex_paper.shutil.which",
                side_effect=lambda name: name if name in {"latexmk", "xelatex"} else None,
            ), patch(
                "latex_paper._tool_version",
                side_effect=lambda path: None if path == "latexmk" else "test version",
            ), patch("latex_paper.subprocess.run", side_effect=fake_run):
                report = build_paper(main)

            self.assertTrue(report["passed"], report["issues"])
            self.assertEqual([command[0] for command in commands], ["xelatex", "xelatex"])

    def test_bound_resource_drift_blocks_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            latex_root = project_root / "完整论文-LaTeX"
            result = prepare_project(latex_root, contest="cumcm")
            main = Path(result["main_tex"])
            source = project_root / "figures"
            source.mkdir()
            (source / "result_q1.png").write_bytes(b"version 1")
            destination = latex_root / "figs"
            destination.mkdir()
            (destination / "result_q1.png").write_bytes(b"version 1")

            try:
                latex_paper.bind_resource(main, source, "figs")
            except AttributeError as error:
                self.fail(f"缺少 LaTeX 资源绑定功能：{error}")

            (source / "result_q1.png").write_bytes(b"version 2")
            report = inspect_paper(main)

            self.assertTrue(any("资源副本已漂移" in issue for issue in report["issues"]))

    def test_single_file_binding_allows_destination_rename(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            latex_root = project_root / "完整论文-LaTeX"
            result = prepare_project(latex_root, contest="cumcm")
            source = project_root / "figures" / "result_q1.png"
            source.parent.mkdir()
            source.write_bytes(b"same image")
            destination = latex_root / "figs" / "figure1.png"
            destination.parent.mkdir()
            destination.write_bytes(b"same image")

            binding = latex_paper.bind_resource(
                Path(result["main_tex"]), source, "figs/figure1.png"
            )

            self.assertTrue(binding["passed"])
            self.assertEqual(binding["binding"]["hash_mode"], "content")

    def test_binding_rejects_invalid_existing_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            latex_root = project_root / "完整论文-LaTeX"
            result = prepare_project(latex_root, contest="cumcm")
            source = project_root / "figures" / "result_q1.png"
            source.parent.mkdir()
            source.write_bytes(b"same image")
            destination = latex_root / "figs" / "result_q1.png"
            destination.parent.mkdir()
            destination.write_bytes(b"same image")
            manifest_path = latex_root / "latex-project.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["resource_bindings"] = [1]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"resource_bindings\[0\].*对象"):
                latex_paper.bind_resource(
                    Path(result["main_tex"]), source, "figs/result_q1.png"
                )

    def test_legacy_manifest_requires_existing_resource_copies_to_be_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            latex_root = project_root / "完整论文-LaTeX"
            result = prepare_project(latex_root, contest="cumcm")
            manifest_path = latex_root / "latex-project.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("resource_bindings")
            manifest.pop("template_resources", None)
            manifest.pop("template_files", None)
            manifest.pop("template_resource_files", None)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (latex_root / "figs").mkdir()
            (latex_root / "figs" / "result_q1.png").write_bytes(b"legacy copy")

            report = inspect_paper(Path(result["main_tex"]))

            self.assertTrue(any("资源副本未绑定：figs" in issue for issue in report["issues"]))

    def test_unbound_resource_in_any_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            latex_root = Path(temporary) / "完整论文-LaTeX"
            result = prepare_project(latex_root, contest="cumcm")
            image = latex_root / "images" / "result_q1.png"
            image.parent.mkdir()
            image.write_bytes(b"unbound")

            report = inspect_paper(Path(result["main_tex"]))

            self.assertTrue(
                any("资源副本未绑定：images/result_q1.png" in issue for issue in report["issues"]),
                report["issues"],
            )

    def test_unbound_common_code_and_data_formats_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            latex_root = Path(temporary) / "完整论文-LaTeX"
            result = prepare_project(latex_root, contest="cumcm")
            resources = {
                "code/model.ipynb": b"{}",
                "code/solver.cpp": b"int main(){}",
                "data/input.txt": b"1 2 3",
            }
            for relative, content in resources.items():
                path = latex_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            report = inspect_paper(Path(result["main_tex"]))
            messages = "\n".join(report["issues"])

            for relative in resources:
                self.assertIn(f"资源副本未绑定：{relative}", messages)

    def test_resource_binding_rejects_project_root_as_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            latex_root = project_root / "完整论文-LaTeX"
            result = prepare_project(latex_root, contest="cumcm")
            (latex_root / "code").mkdir()

            with self.assertRaisesRegex(ValueError, "PROJECT_ROOT 根目录"):
                latex_paper.bind_resource(
                    Path(result["main_tex"]),
                    project_root,
                    "code",
                )

    def test_compilation_runs_in_isolated_copy_and_cannot_modify_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "project"
            root.mkdir()
            main = root / "main.tex"
            original = r"\documentclass{article}\begin{document}safe\end{document}"
            main.write_text(original, encoding="utf-8")
            published = parent / "paper.pdf"

            def fake_run(_command, **kwargs):
                isolated = Path(kwargs["cwd"])
                (isolated / "main.tex").write_text("modified", encoding="utf-8")
                build = isolated / "build"
                (build / "main.pdf").write_bytes(b"pdf")
                (build / "main.log").write_text("clean", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch(
                "latex_paper.shutil.which",
                side_effect=lambda name: name
                if name in {"latexmk", "xelatex"}
                else None,
            ), patch("latex_paper.subprocess.run", side_effect=fake_run):
                report = build_paper(main, publish_path=published)

            self.assertEqual(main.read_text(encoding="utf-8"), original)
            self.assertFalse(report["passed"])
            self.assertFalse(published.exists())
            self.assertTrue(any("隔离编译" in issue for issue in report["issues"]))

    def test_rejects_custom_build_directory_before_compilation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main = root / "main.tex"
            main.write_text(
                r"\documentclass{article}\begin{document}x\end{document}",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "build"):
                build_paper(main, output_dir=root / "out")

    def test_pair_publish_restores_old_pdf_and_manifest_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "new.pdf"
            target = root / "paper.pdf"
            manifest = root / "paper.build.json"
            source.write_bytes(b"new")
            target.write_bytes(b"old")
            manifest.write_text('{"old": true}', encoding="utf-8")
            real_replace = os.replace

            def fail_manifest(source_path, target_path):
                if (
                    Path(target_path) == manifest
                    and ".new-" in Path(source_path).name
                ):
                    raise OSError("simulated manifest failure")
                return real_replace(source_path, target_path)

            with patch("latex_paper.os.replace", side_effect=fail_manifest):
                with self.assertRaisesRegex(OSError, "simulated"):
                    _replace_pair(source, target, {"new": True}, overwrite=True)

            self.assertEqual(target.read_bytes(), b"old")
            self.assertEqual(manifest.read_text(encoding="utf-8"), '{"old": true}')

    def test_verbatim_inputs_and_nocite_star_do_not_create_false_issues(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "references.bib").write_text(
                "@article{x,title={x}}", encoding="utf-8"
            )
            main = root / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}\begin{abstract}摘要\keywords{测试}\end{abstract}
\begin{verbatim}\input{ghost}\end{verbatim}
\verb|\input{also-ghost}|
\nocite{*}\bibliography{references}
\end{document}
""",
                encoding="utf-8",
            )

            report = inspect_paper(main)

            self.assertTrue(report["passed"], report["issues"])

    def test_percent_inside_verb_does_not_hide_following_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main = root / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}\begin{abstract}摘要\keywords{测试}\end{abstract}
\verb|\input{ignored}%| \input{ghost}
\end{document}
""",
                encoding="utf-8",
            )

            report = inspect_paper(main)

            self.assertFalse(report["passed"])
            self.assertTrue(
                any("ghost.tex" in issue for issue in report["issues"]),
                report["issues"],
            )
            self.assertFalse(
                any("ignored.tex" in issue for issue in report["issues"]),
                report["issues"],
            )

    def test_explicit_missing_pdf_always_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main = root / "main.tex"
            main.write_text(
                r"""
\documentclass{article}\newcommand{\keywords}[1]{#1}
\begin{document}\begin{abstract}摘要\keywords{测试}\end{abstract}\end{document}
""",
                encoding="utf-8",
            )

            report = inspect_paper(main, pdf_path=root / "missing.pdf")

            self.assertFalse(report["passed"])
            self.assertTrue(any("指定的 PDF 不存在" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
