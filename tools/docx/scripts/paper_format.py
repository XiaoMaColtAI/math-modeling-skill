#!/usr/bin/env python3
"""Tiny python-docx helpers for the default math-modeling paper format."""

import argparse
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt
from lxml import etree


SKILL_ROOT = Path(__file__).resolve().parents[3]


def set_run_font(run, font="宋体", size=12, bold=False):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    r_fonts = run._element.get_or_add_rPr().get_or_add_rFonts()
    r_fonts.set(qn("w:ascii"), "Times New Roman")
    r_fonts.set(qn("w:hAnsi"), "Times New Roman")
    r_fonts.set(qn("w:eastAsia"), font)
    return run


@dataclass(frozen=True)
class ContestProfile:
    name: str
    paper: str
    margins: tuple[float, float, float, float]
    required_markers: tuple[str, ...]
    rules_source: str


CONTEST_PROFILES = {
    "cumcm": ContestProfile(
        name="全国大学生数学建模竞赛",
        paper="A4",
        margins=(2.54, 2.54, 3.18, 3.18),
        required_markers=("摘 要", "关键词："),
        rules_source="http://www.mcm.edu.cn/",
    ),
    "mcm-icm": ContestProfile(
        name="MCM/ICM",
        paper="LETTER",
        margins=(2.54, 2.54, 2.54, 2.54),
        required_markers=("Summary",),
        rules_source="https://www.comap.com/contests/mcm-icm",
    ),
}


def get_profile(contest="cumcm"):
    try:
        return CONTEST_PROFILES[contest.lower()]
    except KeyError as exc:
        raise ValueError(f"未知竞赛配置: {contest}") from exc


def setup_page(doc, contest="cumcm"):
    profile = get_profile(contest)
    section = doc.sections[0]
    if profile.paper == "LETTER":
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
    else:
        section.page_width = Cm(21)
        section.page_height = Cm(29.7)
    top, bottom, left, right = profile.margins
    section.top_margin = Cm(top)
    section.bottom_margin = Cm(bottom)
    section.left_margin = Cm(left)
    section.right_margin = Cm(right)


def paragraph(doc, text="", align=None, first_line=False, line_spacing=1.25):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = line_spacing
    if first_line:
        p.paragraph_format.first_line_indent = Pt(24)
    if align is not None:
        p.alignment = align
    if text:
        set_run_font(p.add_run(text))
    return p


def title(doc, text):
    p = paragraph(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    set_run_font(p.add_run(text), "黑体", 14, False)
    return p


def abstract_title(doc):
    return title(doc, "摘 要")


def body(doc, text):
    return paragraph(doc, text, first_line=True)


def _latex2omml(latex):
    try:
        from .equations import latex2omml
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from equations import latex2omml
    return latex2omml(latex)


def equation(doc, latex):
    p = paragraph(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    math_para = OxmlElement("m:oMathPara")
    math = OxmlElement("m:oMath")
    for child in etree.fromstring(_latex2omml(latex)):
        math.append(child)
    math_para.append(math)
    p._element.append(math_para)
    return p


def equation_placeholder(doc, latex, prefix="EQ"):
    placeholder = f"{prefix}_{uuid.uuid4().hex[:8].upper()}"
    body(doc, placeholder)
    return placeholder, latex


def keywords(doc, text):
    paragraph(doc)
    p = paragraph(doc)
    set_run_font(p.add_run("关键词："), bold=True)
    set_run_font(p.add_run(text))
    return p


def heading1(doc, text):
    p = paragraph(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    p.paragraph_format.page_break_before = True
    set_run_font(p.add_run(text), size=16, bold=True)
    return p


def heading2(doc, text):
    p = paragraph(doc)
    set_run_font(p.add_run(text), size=14, bold=False)
    return p


def heading3(doc, text):
    p = paragraph(doc, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    set_run_font(p.add_run(text), size=12, bold=True)
    return p


def page_break(doc):
    doc.add_page_break()


def section_break(doc):
    doc.add_section(WD_SECTION.NEW_PAGE)


def image(doc, path, width_cm=12):
    p = paragraph(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    with open(path, "rb") as image_file:
        p.add_run().add_picture(image_file, width=Cm(width_cm))
    return p


def figure_caption(doc, text):
    p = paragraph(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    set_run_font(p.add_run(text), size=10)
    return p


def count_chinese_chars(doc):
    text = "\n".join(p.text for p in doc.paragraphs)
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def _border(val="nil", size="0"):
    elem = OxmlElement("w:bottom")
    elem.set(qn("w:val"), val)
    elem.set(qn("w:sz"), size)
    elem.set(qn("w:space"), "0")
    elem.set(qn("w:color"), "000000" if val != "nil" else "auto")
    return elem


def _set_cell_bottom(cell, val="nil", size="0"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for old in list(borders):
        if old.tag == qn("w:bottom"):
            borders.remove(old)
    borders.append(_border(val, size))


def _set_table_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is not None:
        tbl_pr.remove(borders)
    borders = OxmlElement("w:tblBorders")
    for name, val, size in [
        ("top", "single", "12"),
        ("start", "nil", "0"),
        ("left", "nil", "0"),
        ("bottom", "single", "12"),
        ("end", "nil", "0"),
        ("right", "nil", "0"),
        ("insideH", "nil", "0"),
        ("insideV", "nil", "0"),
    ]:
        elem = OxmlElement(f"w:{name}")
        elem.set(qn("w:val"), val)
        elem.set(qn("w:sz"), size)
        elem.set(qn("w:space"), "0")
        elem.set(qn("w:color"), "000000" if val != "nil" else "auto")
        borders.append(elem)
    tbl_look = tbl_pr.find(qn("w:tblLook"))
    if tbl_look is None:
        tbl_pr.append(borders)
    else:
        tbl_pr.insert(tbl_pr.index(tbl_look), borders)


def three_line_table(doc, rows):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_borders(table)
    for row_i, row in enumerate(rows):
        for col_i, text in enumerate(row):
            cell = table.cell(row_i, col_i)
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_run_font(p.add_run(str(text)), size=12, bold=(row_i == 0))
            if row_i == 0:
                _set_cell_bottom(cell, "single", "4")
    return table


def _clear_template_body(doc):
    body_element = doc._element.body
    for child in list(body_element):
        if child.tag != qn("w:sectPr"):
            body_element.remove(child)


def new_document(contest="cumcm", template_path=None, preserve_template_content=False):
    """从空白文档或参考模板创建论文，可保留官方模板的固定正文。"""
    doc = Document(str(template_path)) if template_path else Document()
    if template_path and not preserve_template_content:
        _clear_template_body(doc)
    zoom = doc.settings.element.find(qn("w:zoom"))
    if zoom is not None and zoom.get(qn("w:percent")) is None:
        zoom.set(qn("w:percent"), "100")
    if not template_path:
        setup_page(doc, contest)
    return doc


def _document_texts(doc):
    texts = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            texts.extend(cell.text.strip() for cell in row.cells if cell.text.strip())
    return texts


def _content_units(text):
    """按中文字符和连续拉丁字母/数字词计数，用于中英文混排篇幅预警。"""
    return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text))


def _markdown_residual_issues(doc):
    """识别转换后不应继续出现在 DOCX 正文中的常见 Markdown 标记。"""
    patterns = (
        (re.compile(r"```"), "代码围栏 ```"),
        (re.compile(r"\*\*[^*\n]+\*\*"), "粗体标记 **...**"),
        (re.compile(r"`[^`\n]+`"), "行内代码标记 `...`"),
        (re.compile(r"^\s{0,3}#{1,6}\s+\S"), "标题标记 #"),
        (re.compile(r"^\s*[-+*]\s+\S"), "行首列表标记"),
        (re.compile(r"^\s*\$\$.+\$\$\s*$"), "行间公式定界符 $$"),
        (re.compile(r"(?<!\$)\$(?!\$)\S(?:.*?\S)?\$(?!\$)"), "行内公式定界符 $"),
        (
            re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"),
            "Markdown 表格分隔行",
        ),
        (re.compile(r"^\s*\|[^\n]*\|[^\n]*\|?\s*$"), "Markdown 表格行"),
    )
    issues = []
    for index, text in enumerate(_document_texts(doc), start=1):
        for pattern, label in patterns:
            if pattern.search(text):
                excerpt = text if len(text) <= 80 else text[:77] + "..."
                issues.append(f"疑似 Markdown 格式残留（文本 {index}，{label}）：{excerpt}")
    return issues


def _numbered_object_issues(doc, kind, object_count):
    caption_pattern = re.compile(rf"^\s*{kind}\s*(\d+)(?!\d)")
    reference_pattern = re.compile(rf"{kind}\s*(\d+)(?!\d)")
    captions = {}
    body_references = set()
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        caption = caption_pattern.match(text)
        if caption:
            captions[int(caption.group(1))] = text
        else:
            body_references.update(int(number) for number in reference_pattern.findall(text))

    issues = []
    expected = set(range(1, object_count + 1))
    missing_captions = sorted(expected - set(captions))
    if missing_captions:
        issues.append(f"{kind}编号不完整，缺少题注: {missing_captions}")
    extra_captions = sorted(set(captions) - expected)
    if extra_captions:
        issues.append(f"{kind}题注没有对应对象或编号跳跃: {extra_captions}")
    for number in sorted(set(captions) - body_references):
        issues.append(f"{kind}{number} 已插入但未在正文引用")
    return issues


def _reference_issues(paragraphs):
    split_at = next(
        (index for index, p in enumerate(paragraphs) if p.text.strip().lower() in {"参考文献", "references"}),
        None,
    )
    if split_at is None:
        return ["未找到参考文献章节"]
    body = "\n".join(p.text for p in paragraphs[:split_at])
    bibliography = [p.text.strip() for p in paragraphs[split_at + 1:] if p.text.strip()]
    cited = set()
    for group in re.findall(r"\[([0-9,，\-–—\s]+)\]", body):
        for item in re.split(r"[,，]", group):
            item = item.strip()
            if not item:
                continue
            bounds = re.split(r"[\-–—]", item)
            if len(bounds) == 2 and all(bound.strip().isdigit() for bound in bounds):
                start, end = (int(bound.strip()) for bound in bounds)
                if start <= end:
                    cited.update(range(start, end + 1))
            elif item.isdigit():
                cited.add(int(item))
    listed = {
        int(match.group(1))
        for text in bibliography
        if (match := re.match(r"^\[(\d+)\]", text))
    }
    issues = [f"正文引用 [{number}] 未出现在参考文献表" for number in sorted(cited - listed)]
    issues.extend(f"参考文献 [{number}] 未在正文引用" for number in sorted(listed - cited))
    return issues


def validate_paper_structure(
    doc,
    contest="cumcm",
    *,
    quality_checks=True,
    min_content_units=None,
    min_equations=None,
    min_figures=None,
    min_tables=None,
    rendered_pages=None,
    target_pages=None,
    official_max_pages=None,
    require_rendered_pages=True,
):
    """检查官方结构、Markdown 残留、篇幅目标、公式图表、编号引用和文献。"""
    profile = get_profile(contest)
    texts = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]
    errors = []
    if not texts:
        errors.append("缺少论文标题")
    full_text = "\n".join(texts)
    for marker in profile.required_markers:
        if marker == "Summary":
            present = any(text.lower() in {"summary", "summary sheet"} for text in texts)
        elif marker.endswith("："):
            present = any(text.startswith(marker) for text in texts)
        else:
            present = marker in texts
        if not present:
            label = "摘要" if marker == "摘 要" else "关键词" if marker == "关键词：" else marker
            errors.append(f"缺少官方结构项: {label}")
    if "[待补充" in full_text:
        errors.append("论文仍含 [待补充] 占位符")
    errors.extend(_markdown_residual_issues(doc))
    if not quality_checks:
        return errors

    if contest.lower() == "cumcm":
        min_content_units = 15000 if min_content_units is None else min_content_units
        min_equations = 5 if min_equations is None else min_equations
        min_figures = 8 if min_figures is None else min_figures
        min_tables = 3 if min_tables is None else min_tables
        target_pages = 20 if target_pages is None else target_pages
        official_max_pages = 30 if official_max_pages is None else official_max_pages
    else:
        min_content_units = 0 if min_content_units is None else min_content_units
        min_equations = 0 if min_equations is None else min_equations
        min_figures = 8 if min_figures is None else min_figures
        min_tables = 0 if min_tables is None else min_tables

    all_text = "\n".join(_document_texts(doc))
    units = _content_units(all_text)
    equations = len(doc._element.findall(f".//{qn('m:oMath')}"))
    figures = len(doc._element.findall(f".//{qn('a:blip')}"))
    tables = len(doc.tables)
    if units < min_content_units:
        errors.append(
            f"预警：正文约 {units} 字词单位，低于 {min_content_units} 的质量目标；"
            "该目标不是 CUMCM 官方最低字数，可按当届规则或用户要求覆盖"
        )
    if equations < min_equations:
        errors.append(f"预警：仅检测到 {equations} 个可编辑公式，低于质量目标 {min_equations}")
    if figures < min_figures:
        errors.append(f"预警：仅检测到 {figures} 幅图，低于质量目标 {min_figures}")
    if tables < min_tables:
        errors.append(f"预警：仅检测到 {tables} 个表，低于质量目标 {min_tables}")

    errors.extend(_numbered_object_issues(doc, "图", figures))
    errors.extend(_numbered_object_issues(doc, "表", tables))
    errors.extend(_reference_issues(doc.paragraphs))

    if rendered_pages is None and require_rendered_pages and target_pages is not None:
        errors.append(
            f"预警：未提供渲染页数，无法检查约 {target_pages} 页质量目标和 "
            f"{official_max_pages} 页官方上限"
        )
    elif rendered_pages is not None:
        if target_pages is not None and rendered_pages < target_pages:
            errors.append(
                f"预警：渲染后共 {rendered_pages} 页，低于约 {target_pages} 页质量目标；"
                "该目标不是官方最低页数"
            )
        if official_max_pages is not None and rendered_pages > official_max_pages:
            errors.append(
                f"渲染后共 {rendered_pages} 页，超过当前核验的官方上限 {official_max_pages} 页"
            )
    return errors


def _is_within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def save_document(doc, project_root, filename="完整论文.docx", contest="cumcm", overwrite=False):
    """校验后原子保存到 PROJECT_ROOT，并拒绝写入 Skill 目录。"""
    project = Path(project_root).resolve()
    if _is_within(project, SKILL_ROOT):
        raise ValueError("PROJECT_ROOT 不能位于 SKILL_ROOT 内部")
    output = (project / filename).resolve()
    if not _is_within(output, project):
        raise ValueError("论文输出必须位于 PROJECT_ROOT 内部")
    if _is_within(output, SKILL_ROOT):
        raise ValueError("论文输出不能位于 SKILL_ROOT 内部")
    issues = validate_paper_structure(doc, contest)
    errors = [issue for issue in issues if not issue.startswith("预警：")]
    if errors:
        raise ValueError("论文结构校验失败: " + "；".join(errors))
    if output.exists() and not overwrite:
        raise FileExistsError(f"输出已存在，未覆盖: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    doc.save(temporary)
    os.replace(temporary, output)
    return output


def validate_document(path, *, contest="cumcm", rendered_pages=None):
    """校验现有 DOCX，并返回可供完成门禁使用的结构化结果。"""
    source = Path(path).resolve()
    if not source.is_file() or source.suffix.casefold() != ".docx":
        raise FileNotFoundError(f"DOCX 论文不存在：{source}")
    doc = Document(source)
    issues = validate_paper_structure(
        doc,
        contest,
        quality_checks=True,
        rendered_pages=rendered_pages,
        require_rendered_pages=True,
    )
    text = "\n".join(_document_texts(doc))
    return {
        "path": str(source),
        "metrics": {
            "content_units": _content_units(text),
            "rendered_pages": rendered_pages,
            "equations": len(doc._element.findall(f".//{qn('m:oMath')}")),
            "figures": len(doc._element.findall(f".//{qn('a:blip')}")),
            "tables": len(doc.tables),
        },
        "issues": issues,
        "passed": not issues,
    }


def main():
    parser = argparse.ArgumentParser(description="生成或校验数学建模 DOCX")
    commands = parser.add_subparsers(dest="action", required=True)
    validate = commands.add_parser("validate", help="执行 DOCX 完成门禁")
    validate.add_argument("path", type=Path)
    validate.add_argument("--contest", choices=sorted(CONTEST_PROFILES), default="cumcm")
    validate.add_argument("--rendered-pages", type=int, required=True)
    arguments = parser.parse_args()
    result = validate_document(
        arguments.path,
        contest=arguments.contest,
        rendered_pages=arguments.rendered_pages,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
