"""初始化、编译、诊断并校验模板驱动的 LaTeX 数学建模论文。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

TEMPLATE_ROOT = Path(__file__).resolve().parents[1] / "assets" / "templates"
SKILL_ROOT = Path(__file__).resolve().parents[3]
PROJECT_MANIFEST = "latex-project.json"
PROJECT_IGNORED_NAMES = frozenset({".git"})
CONTESTS = {"cumcm", "mcm-icm", "generic"}
ENGINES = {"xelatex", "lualatex", "pdflatex"}
GRAPHIC_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg")
UNSUPPORTED_GRAPHIC_SUFFIXES = (".svg", ".eps")
LATEX_AUTHORING_SUFFIXES = {
    ".tex", ".bib", ".cls", ".sty", ".bst", ".def", ".cfg", ".clo", ".fd",
    ".bbx", ".cbx", ".lbx", ".dtx", ".ins", ".ltx",
}
GENERATED_SUFFIXES = {
    ".aux", ".log", ".out", ".toc", ".bbl", ".blg", ".bcf", ".fls",
    ".fdb_latexmk", ".synctex.gz", ".run.xml", ".lof", ".lot", ".nav",
    ".snm", ".vrb", ".xdv", ".dvi", ".ps",
}
DANGEROUS_TEX = re.compile(
    r"\\(?:immediate\s*)?write18\b|\\(?:openin|openout|read|write)\b|"
    r"\\usepackage\s*\{(?:minted|shellesc)\}",
    re.I,
)
INPUT_RE = re.compile(
    r"\\(input|include)(?![A-Za-z@])\s*(?:\{([^}]+)\}|([^\s%]+))"
)
LITERAL_ENV_RE = re.compile(
    r"\\begin\s*\{(?P<env>verbatim\*?|lstlisting|minted|comment)\}.*?"
    r"\\end\s*\{(?P=env)\}",
    re.S,
)
VERB_RE = re.compile(r"\\verb\*?(?P<delimiter>[^\w\s]).*?(?P=delimiter)")
APPENDIX_RE = re.compile(r"\\appendix\b|\\begin\s*\{appendices\}")
QUALITY_DEFAULTS = {
    "cumcm": {
        "min_content_units": 15_000,
        "min_pages": 20,
        "min_equations": 5,
        "min_figures": 8,
        "min_tables": 3,
    },
    "mcm-icm": {
        "min_content_units": 0,
        "min_pages": 0,
        "min_equations": 0,
        "min_figures": 8,
        "min_tables": 0,
    },
    "generic": {
        "min_content_units": 0,
        "min_pages": 0,
        "min_equations": 0,
        "min_figures": 8,
        "min_tables": 0,
    },
}
STANDARD_14_FONTS = {
    "Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique",
    "Helvetica", "Helvetica-Bold", "Helvetica-Oblique",
    "Helvetica-BoldOblique", "Times-Roman", "Times-Bold", "Times-Italic",
    "Times-BoldItalic", "Symbol", "ZapfDingbats",
}


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"路径超出 LaTeX 项目：{path}")
    return resolved


def _writable(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == SKILL_ROOT or SKILL_ROOT in resolved.parents:
        raise ValueError("拒绝向 SKILL_ROOT 写入 LaTeX 产物")
    return resolved


def _project_entries(root: Path) -> Iterator[Path]:
    """按统一的忽略规则遍历项目条目。"""
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in PROJECT_IGNORED_NAMES)
        for name in sorted(dirs + [
            name for name in files if name not in PROJECT_IGNORED_NAMES
        ]):
            yield Path(directory) / name


def _project_files(root: Path) -> Iterator[Path]:
    return (path for path in _project_entries(root) if path.is_file())


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink() or any(path.is_symlink() for path in _project_entries(root)):
        raise ValueError("LaTeX 项目包含符号链接，拒绝处理")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    items = [path] if path.is_file() else sorted(_project_files(path))
    for item in items:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_hashes(path: Path) -> dict[str, str]:
    if path.is_file():
        return {"main.tex": _sha256(path)}
    return {
        item.relative_to(path).as_posix(): _sha256(item)
        for item in sorted(_project_files(path))
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _replace_pair(
    source: Path,
    target: Path,
    manifest_payload: dict,
    *,
    overwrite: bool,
) -> Path:
    """Replace one artifact and its JSON manifest, restoring the old pair on error."""
    manifest = _build_manifest_path(target)
    if not overwrite and (target.exists() or manifest.exists()):
        raise FileExistsError(f"发布产物已存在，拒绝覆盖：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staged_target = target.parent / f".{target.name}.new-{token}"
    staged_manifest = target.parent / f".{manifest.name}.new-{token}"
    backups = {
        target: target.parent / f".{target.name}.bak-{token}",
        manifest: target.parent / f".{manifest.name}.bak-{token}",
    }
    moved = set()
    installed = set()
    try:
        shutil.copyfile(source, staged_target)
        staged_manifest.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for current, backup in backups.items():
            if current.exists():
                os.replace(current, backup)
                moved.add(current)
        os.replace(staged_target, target)
        installed.add(target)
        os.replace(staged_manifest, manifest)
        installed.add(manifest)
    except Exception:
        for current in installed:
            if current.exists():
                current.unlink()
        for current in moved:
            backup = backups[current]
            if backup.exists():
                os.replace(backup, current)
        raise
    finally:
        for path in (staged_target, staged_manifest, *backups.values()):
            if path.exists():
                path.unlink()
    return manifest


def _mask_literals(text: str) -> str:
    def blank(match: re.Match) -> str:
        return re.sub(r"[^\r\n]", " ", match.group(0))

    return VERB_RE.sub(blank, LITERAL_ENV_RE.sub(blank, text))


def _entry(directory: Path, main_file: str | None = None) -> Path:
    root = directory.resolve()
    if main_file:
        relative = Path(main_file)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("主入口必须位于模板根目录内")
        selected = _inside(root / relative, root)
        if not selected.is_file() or selected.suffix.casefold() != ".tex":
            raise FileNotFoundError(f"指定的 LaTeX 主入口不存在：{main_file}")
        return selected
    preferred = root / "main.tex"
    if preferred.is_file():
        return preferred
    candidates = sorted(root.glob("*.tex"))
    if len(candidates) != 1:
        raise ValueError("未确定 LaTeX 主入口：请提供 main.tex、唯一顶层 .tex 或显式 --main")
    return candidates[0]


def prepare_project(
    output_dir: Path,
    *,
    contest: str = "cumcm",
    template_path: Path | None = None,
    main_file: str | None = None,
    template_source: str | None = None,
    template_version: str | None = None,
) -> dict:
    """复制官方或内置模板，并写入可追溯的项目清单。"""
    if contest not in CONTESTS:
        raise ValueError(f"不支持的竞赛配置：{contest}")
    if contest == "generic" and template_path is None:
        raise ValueError("generic 配置必须通过 --template 提供官方模板")
    output = _writable(output_dir)
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output}")
    source = (template_path or (TEMPLATE_ROOT / contest)).resolve()
    if not source.exists():
        raise FileNotFoundError(f"LaTeX 模板不存在：{source}")
    items = [source] if source.is_file() else [source, *_project_entries(source)]
    if any(item.is_symlink() for item in items):
        raise ValueError("模板中包含符号链接，拒绝复制")
    if source.is_file() and source.suffix.casefold() != ".tex":
        raise ValueError("单文件模板必须是 .tex 文件")
    selected = _entry(source, main_file) if source.is_dir() else None
    if source.is_file() and main_file:
        raise ValueError("单文件模板不需要指定主入口")

    entry_relative = (
        selected.relative_to(source).as_posix() if selected is not None else "main.tex"
    )
    manifest = {
        "schema_version": 1,
        "contest": contest,
        "main_tex": entry_relative,
        "template": {
            "path": str(source),
            "source": template_source,
            "version": template_version,
            "sha256": _tree_sha256(source),
        },
        "template_files": _file_hashes(source),
        "resource_bindings": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    try:
        if source.is_dir():
            shutil.copytree(
                source, temporary,
                ignore=shutil.ignore_patterns(*PROJECT_IGNORED_NAMES),
            )
        else:
            temporary.mkdir()
            shutil.copy2(source, temporary / "main.tex")
        _atomic_json(temporary / PROJECT_MANIFEST, manifest)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "project_dir": str(output),
        "main_tex": str(output / Path(entry_relative)),
        "template": str(source),
        "contest": contest,
        "manifest": str(output / PROJECT_MANIFEST),
    }


def _project_root(main_tex: Path) -> Path:
    main = main_tex.resolve()
    for candidate in (main.parent, *main.parents):
        manifest_path = candidate / PROJECT_MANIFEST
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            configured = _inside(candidate / manifest["main_tex"], candidate)
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"LaTeX 项目清单无效：{error}") from error
        if configured != main:
            raise ValueError(
                f"当前入口与 {PROJECT_MANIFEST} 不一致：{manifest['main_tex']}"
            )
        return candidate
    return main.parent


def _validated_resource_bindings(manifest: dict) -> list[dict]:
    bindings = manifest.get("resource_bindings", [])
    if not isinstance(bindings, list):
        raise ValueError("resource_bindings 必须为列表")
    for index, binding in enumerate(bindings):
        if not isinstance(binding, dict):
            raise ValueError(f"resource_bindings[{index}] 必须为对象")
        for key in ("source", "destination", "sha256"):
            if not isinstance(binding.get(key), str) or not binding[key]:
                raise ValueError(f"resource_bindings[{index}].{key} 必须为非空字符串")
        if binding.get("hash_mode", "tree") not in {"content", "tree"}:
            raise ValueError(
                f"resource_bindings[{index}].hash_mode 必须是 content 或 tree"
            )
    return bindings


def _resource_hash(path: Path, mode: str) -> str:
    if mode == "content":
        if not path.is_file():
            raise ValueError("content 哈希只能用于单文件资源")
        return _sha256(path)
    return _tree_sha256(path)


def _resource_binding_issues(root: Path) -> list[str]:
    manifest_path = root / PROJECT_MANIFEST
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bindings = _validated_resource_bindings(manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [f"LaTeX 资源绑定清单无效：{error}"]

    issues = []
    bound_destinations = set()
    project_root = root.parent.resolve()
    for binding in bindings:
        try:
            source = _inside(project_root / binding["source"], project_root)
            destination = _inside(root / binding["destination"], root)
            if source == project_root:
                raise ValueError("权威资源不能是 PROJECT_ROOT 根目录")
            if source == root or root in source.parents:
                raise ValueError("权威资源不能位于 LaTeX 项目副本内")
            _reject_symlinks(source)
            expected = binding["sha256"]
            hash_mode = binding.get("hash_mode", "tree")
            bound_destinations.add(destination.relative_to(root))
            if not source.exists() or not destination.exists():
                issues.append(
                    f"资源绑定缺失：{binding['source']} -> {binding['destination']}"
                )
            elif (
                _resource_hash(source, hash_mode) != expected
                or _resource_hash(destination, hash_mode) != expected
            ):
                issues.append(
                    f"资源副本已漂移：{binding['source']} -> {binding['destination']}；"
                    "同步两端后重新执行 bind"
                )
        except (KeyError, OSError, TypeError, ValueError) as error:
            issues.append(f"LaTeX 资源绑定无效：{error}")

    template_files = manifest.get(
        "template_files", manifest.get("template_resource_files", {})
    )
    if not isinstance(template_files, dict):
        issues.append("LaTeX 资源绑定清单无效：template_files 必须为对象")
        template_files = {}
    for path in sorted(_project_files(root)):
        if path == manifest_path:
            continue
        relative = path.relative_to(root)
        if relative.parts[0] == "build":
            continue
        suffix = path.suffix.casefold()
        compound_suffix = "".join(path.suffixes[-2:]).casefold()
        if (
            suffix in LATEX_AUTHORING_SUFFIXES
            or suffix in GENERATED_SUFFIXES
            or compound_suffix in GENERATED_SUFFIXES
            or path.name.endswith((".build.json", ".conversion.json"))
        ):
            continue
        if any(relative == bound or bound in relative.parents for bound in bound_destinations):
            continue
        template_hash = template_files.get(relative.as_posix())
        if template_hash and _sha256(path) == template_hash:
            continue
        issues.append(
            f"LaTeX 资源副本未绑定：{relative.as_posix()}；"
            "必须绑定到 PROJECT_ROOT 权威来源"
        )
    return issues


def bind_resource(main_tex: Path, source_path: Path, destination: str) -> dict:
    """把 LaTeX 项目内的现有资源副本绑定到 PROJECT_ROOT 权威来源。"""
    main = main_tex.resolve()
    root = _project_root(main)
    manifest_path = root / PROJECT_MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"缺少 {PROJECT_MANIFEST}，无法记录资源绑定")
    project_root = root.parent.resolve()
    source = _inside(source_path, project_root)
    if source == project_root:
        raise ValueError("权威资源不能是 PROJECT_ROOT 根目录")
    if source == root or root in source.parents:
        raise ValueError("权威资源必须位于 LaTeX 项目之外的 PROJECT_ROOT 中")
    _reject_symlinks(source)
    relative_destination = Path(destination)
    if relative_destination.is_absolute() or ".." in relative_destination.parts:
        raise ValueError("资源副本路径必须位于 LaTeX 项目内")
    target = _inside(root / relative_destination, root)
    if not source.exists() or not target.exists():
        raise FileNotFoundError("权威资源或 LaTeX 资源副本不存在")
    if source.is_file() != target.is_file():
        raise ValueError("权威资源与 LaTeX 资源副本类型不一致")
    hash_mode = "content" if source.is_file() else "tree"
    source_hash = _resource_hash(source, hash_mode)
    if _resource_hash(target, hash_mode) != source_hash:
        raise ValueError("权威资源与 LaTeX 资源副本内容不一致，不能建立绑定")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_bindings = _validated_resource_bindings(manifest)
    bindings = [
        item
        for item in existing_bindings
        if item.get("destination") != target.relative_to(root).as_posix()
    ]
    binding = {
        "source": source.relative_to(project_root).as_posix(),
        "destination": target.relative_to(root).as_posix(),
        "sha256": source_hash,
        "hash_mode": hash_mode,
        "bound_at": datetime.now(timezone.utc).isoformat(),
    }
    bindings.append(binding)
    manifest["resource_bindings"] = bindings
    _atomic_json(manifest_path, manifest)
    return {
        "passed": True,
        "main_tex": str(main),
        "binding": binding,
        "manifest": str(manifest_path),
    }


def _strip_comments(text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        cut = len(line)
        for index, character in enumerate(line):
            if character != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                cut = index
                break
        cleaned.append(line[:cut])
    return "\n".join(cleaned)


def _collect_sources(main_tex: Path) -> tuple[str, list[Path], list[str]]:
    main = main_tex.resolve()
    root = _project_root(main)
    seen: set[Path] = set()
    stack: set[Path] = set()
    files: list[Path] = []
    issues: list[str] = []

    def load(path: Path) -> str:
        current = _inside(path, root)
        if current in stack:
            issues.append(f"LaTeX 子文件循环包含：{current.relative_to(root).as_posix()}")
            return ""
        if not current.is_file():
            issues.append(f"缺少 LaTeX 子文件：{current.relative_to(root).as_posix()}")
            return ""
        if current not in seen:
            seen.add(current)
            files.append(current)
        stack.add(current)
        text = _strip_comments(
            _mask_literals(current.read_text(encoding="utf-8"))
        )

        def expand(match: re.Match) -> str:
            raw = (match.group(2) or match.group(3)).strip()
            included = Path(raw)
            if not included.suffix:
                included = included.with_suffix(".tex")
            content = load(root / included)
            return f"\n\\clearpage\n{content}\n" if match.group(1) == "include" else content

        expanded = INPUT_RE.sub(expand, text)
        stack.remove(current)
        return expanded

    return load(main), files, issues


def _bibliography_keys(
    source: str, root: Path, issues: list[str]
) -> tuple[set[str], set[str]]:
    manual = set(re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", source))
    keys = set(manual)
    raw_files = re.findall(r"\\bibliography\s*\{([^}]+)\}", source)
    raw_files += re.findall(
        r"\\addbibresource(?:\[[^]]*\])?\s*\{([^}]+)\}", source
    )
    for group in raw_files:
        for raw in group.split(","):
            path = Path(raw.strip())
            if not path.suffix:
                path = path.with_suffix(".bib")
            try:
                bibliography = _inside(root / path, root)
            except ValueError as error:
                issues.append(str(error))
                continue
            if not bibliography.is_file():
                issues.append(f"缺少参考文献库：{path.as_posix()}")
                continue
            content = _strip_comments(bibliography.read_text(encoding="utf-8"))
            keys.update(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", content))
    return keys, manual


def _graphic_roots(source: str, root: Path) -> list[Path]:
    directories = [root]
    for group in re.findall(
        r"\\graphicspath\s*\{((?:\s*\{[^{}]*\}\s*)+)\}", source
    ):
        for raw in re.findall(r"\{([^{}]*)\}", group):
            directories.append(_inside(root / raw.strip(), root))
    return directories


def _graphic_status(root: Path, directories: list[Path], raw: str) -> str:
    requested = Path(raw.strip())
    if requested.suffix.casefold() in UNSUPPORTED_GRAPHIC_SUFFIXES:
        return "unsupported"
    unsupported_found = False
    for directory in directories:
        candidate = _inside(directory / requested, root)
        if candidate.is_file():
            return "ok" if candidate.suffix.casefold() in GRAPHIC_SUFFIXES else "unsupported"
        if not candidate.suffix:
            if any(candidate.with_suffix(suffix).is_file() for suffix in GRAPHIC_SUFFIXES):
                return "ok"
            unsupported_found = unsupported_found or any(
                candidate.with_suffix(suffix).is_file()
                for suffix in UNSUPPORTED_GRAPHIC_SUFFIXES
            )
    return "unsupported" if unsupported_found else "missing"


def _environment_matches(source: str, names: str) -> list[re.Match]:
    return list(re.finditer(
        rf"\\begin\s*\{{(?P<env>{names})\}}(?P<body>.*?)"
        rf"\\end\s*\{{(?P=env)\}}",
        source,
        re.S,
    ))


def _display_equations(source: str, issues: list[str]) -> int:
    environments = len(re.findall(
        r"\\begin\s*\{(?:equation\*?|align\*?|alignat\*?|gather\*?|"
        r"multline\*?|displaymath|eqnarray\*?)\}",
        source,
    ))
    opens = len(re.findall(r"(?<!\\)(?:\\\\)*\\\[", source))
    closes = len(re.findall(r"(?<!\\)(?:\\\\)*\\\]", source))
    if opens != closes:
        issues.append(r"行间公式分隔符 \[ 与 \] 数量不一致")
    dollars = len(re.findall(r"(?<!\\)\$\$", source))
    if dollars % 2:
        issues.append("行间公式分隔符 $$ 数量为奇数")
    return environments + min(opens, closes) + dollars // 2


def _normalise_questions(questions: list[str] | tuple[str, ...] | None) -> list[str]:
    result = []
    for raw in questions or []:
        question = raw.strip().casefold()
        if not re.fullmatch(r"q[1-9]\d*", question):
            raise ValueError(f"子问题编号必须使用 q1、q2…格式：{raw}")
        if question not in result:
            result.append(question)
    return result


def _thresholds(
    contest: str,
    quality_checks: bool,
    values: dict[str, int | None],
    max_pages: int | None,
    override_reason: str | None,
) -> tuple[dict[str, int], list[dict]]:
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages 必须为正整数")
    defaults = QUALITY_DEFAULTS[contest] if quality_checks else {}
    thresholds: dict[str, int] = {}
    overrides = []
    for key, explicit in values.items():
        if explicit is not None and explicit < 0:
            raise ValueError(f"{key} 不能为负数")
        default = int(defaults.get(key, 0))
        value = default if explicit is None else explicit
        if quality_checks and explicit is not None and explicit < default:
            if not override_reason or not override_reason.strip():
                raise ValueError(f"降低 {key} 质量门槛必须提供 override_reason")
            overrides.append({
                "name": key,
                "default": default,
                "value": explicit,
                "reason": override_reason.strip(),
            })
        thresholds[key] = int(value)
    return thresholds, overrides


def source_bundle_sha256(main_tex: Path) -> str:
    if main_tex.is_symlink():
        raise ValueError("LaTeX 主入口不能是符号链接")
    root = _project_root(main_tex)
    _reject_symlinks(root)
    digest = hashlib.sha256()
    for path in sorted(_project_files(root)):
        relative = path.relative_to(root)
        if relative.parts[0] == "build":
            continue
        suffix = path.suffix.casefold()
        compound_suffix = "".join(path.suffixes[-2:]).casefold()
        if suffix in GENERATED_SUFFIXES or compound_suffix in GENERATED_SUFFIXES:
            continue
        if path.name.endswith((".build.json", ".conversion.json")):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _font_descriptor_embedded(font) -> bool:
    font = font.get_object()
    if font.get("/Subtype") == "/Type3":
        return True
    if font.get("/Subtype") == "/Type0":
        descendants = font.get("/DescendantFonts", [])
        return bool(descendants) and all(
            _font_descriptor_embedded(item) for item in descendants
        )
    base = str(font.get("/BaseFont", "")).lstrip("/").split("+")[-1]
    if base in STANDARD_14_FONTS:
        return True
    descriptor = font.get("/FontDescriptor")
    if descriptor is None:
        return False
    descriptor = descriptor.get_object()
    return any(descriptor.get(key) is not None for key in (
        "/FontFile", "/FontFile2", "/FontFile3"
    ))


def _pdf_image_dpi(path: Path) -> tuple[list[int], list[str]]:
    executable = shutil.which("pdfimages")
    if executable is None:
        return [], ["未找到 pdfimages，无法检查 PDF 内嵌位图分辨率"]
    try:
        completed = subprocess.run(
            [executable, "-list", str(path)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return [], [f"PDF 位图分辨率检查失败：{error}"]
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-500:]
        return [], [f"pdfimages 检查失败：{detail or completed.returncode}"]
    values = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 14 and fields[0].isdigit() and fields[1].isdigit():
            try:
                values.extend((int(float(fields[12])), int(float(fields[13]))))
            except ValueError:
                continue
    return values, []


def _audit_pdf(path: Path, *, min_image_dpi: int, require_image_audit: bool) -> dict:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("缺少 pypdf，无法检查 PDF") from error
    reader = PdfReader(path)
    blank_pages = []
    page_sizes = []
    page_texts = []
    unembedded_fonts = set()
    for number, page in enumerate(reader.pages, 1):
        width = round(float(page.mediabox.width), 1)
        height = round(float(page.mediabox.height), 1)
        page_sizes.append((width, height))
        resources = page.get("/Resources")
        resources = resources.get_object() if resources else {}
        xobjects = resources.get("/XObject", {})
        xobjects = xobjects.get_object() if hasattr(xobjects, "get_object") else xobjects
        try:
            contents = page.get_contents()
            content_bytes = contents.get_data() if contents else b""
        except Exception:
            content_bytes = b""
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        page_texts.append(text)
        if not text.strip() and not xobjects and not content_bytes.strip():
            blank_pages.append(number)
        fonts = resources.get("/Font", {})
        fonts = fonts.get_object() if hasattr(fonts, "get_object") else fonts
        for name, font in fonts.items():
            try:
                if not _font_descriptor_embedded(font):
                    base = str(font.get_object().get("/BaseFont", name)).lstrip("/")
                    unembedded_fonts.add(base)
            except Exception:
                unembedded_fonts.add(str(name).lstrip("/"))

    dpi_values, dpi_issues = _pdf_image_dpi(path)
    issues = list(dpi_issues) if require_image_audit else []
    if blank_pages:
        issues.append("PDF 存在空白页：" + "、".join(map(str, blank_pages)))
    unique_sizes = sorted(set(page_sizes))
    if len(unique_sizes) > 1:
        issues.append("PDF 页面尺寸不一致：" + "、".join(
            f"{width}×{height}pt" for width, height in unique_sizes
        ))
    if unembedded_fonts:
        issues.append("PDF 字体未嵌入：" + "、".join(sorted(unembedded_fonts)))
    if dpi_values and min(dpi_values) < min_image_dpi:
        issues.append(
            f"PDF 内嵌位图最低分辨率 {min(dpi_values)} DPI，低于 {min_image_dpi} DPI"
        )
    return {
        "pages": len(reader.pages),
        "blank_pages": blank_pages,
        "page_sizes_pt": [list(size) for size in unique_sizes],
        "unembedded_fonts": sorted(unembedded_fonts),
        "raster_images": len(dpi_values) // 2,
        "min_image_dpi": min(dpi_values) if dpi_values else None,
        "_page_texts": page_texts,
        "issues": issues,
    }


def _appendix_start_page(source: str, page_texts: list[str]) -> int | None:
    match = APPENDIX_RE.search(source)
    if not match:
        return None
    tail = source[match.end():]
    headings = re.findall(
        r"\\(?:part|chapter|section)\*?(?:\[[^]]*\])?\s*\{([^{}]+)\}",
        tail,
    )
    candidates = []
    for heading in headings[:2]:
        plain = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", "", heading)
        plain = re.sub(r"[{}~\s]", "", plain).casefold()
        if len(plain) >= 2:
            candidates.append(plain)
    candidates.extend(("附录", "appendix"))
    matches = {
        number
        for number, text in enumerate(page_texts, 1)
        if any(
            candidate in re.sub(r"\s+", "", text).casefold()
            for candidate in candidates
        )
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _build_manifest_path(pdf: Path) -> Path:
    return pdf.with_suffix(".build.json")


def _verify_pdf_binding(
    main: Path, pdf: Path, source_sha256: str
) -> tuple[dict | None, list[str]]:
    path = _build_manifest_path(pdf)
    if not path.is_file():
        return None, [f"PDF 缺少构建清单：{path.name}"]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"PDF 构建清单无效：{error}"]
    issues = []
    root = _project_root(main)
    expected_main = main.relative_to(root).as_posix()
    if not manifest.get("passed"):
        issues.append("PDF 对应的构建未通过发布门禁")
    if manifest.get("source_sha256") != source_sha256:
        issues.append("PDF 对应源码哈希与当前 LaTeX 项目不一致")
    if manifest.get("pdf_sha256") != _sha256(pdf):
        issues.append("PDF 文件哈希与构建清单不一致")
    if manifest.get("main_tex") != expected_main:
        issues.append("PDF 构建清单记录的主入口与当前项目不一致")
    return manifest, issues


def inspect_paper(
    main_tex: Path,
    *,
    contest: str = "cumcm",
    pdf_path: Path | None = None,
    quality_checks: bool = False,
    min_content_units: int | None = None,
    min_pages: int | None = None,
    max_pages: int | None = None,
    body_start_page: int | None = None,
    appendix_start_page: int | None = None,
    min_equations: int | None = None,
    min_figures: int | None = None,
    min_tables: int | None = None,
    min_image_dpi: int = 300,
    require_pdf: bool | None = None,
    questions: list[str] | tuple[str, ...] | None = None,
    override_reason: str | None = None,
) -> dict:
    """检查源码、引用、图表、子问题覆盖和已绑定的渲染 PDF。"""
    if contest not in CONTESTS:
        raise ValueError(f"不支持的竞赛配置：{contest}")
    if min_image_dpi <= 0:
        raise ValueError("min_image_dpi 必须为正整数")
    for name, value in (
        ("body_start_page", body_start_page),
        ("appendix_start_page", appendix_start_page),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} 必须为正整数")
    main = main_tex.resolve()
    if not main.is_file() or main.suffix.casefold() != ".tex":
        raise FileNotFoundError(f"LaTeX 入口不存在：{main}")
    root = _project_root(main)
    source, source_files, issues = _collect_sources(main)
    _reject_symlinks(root)
    issues.extend(_resource_binding_issues(root))
    has_appendix = bool(APPENDIX_RE.search(source))
    if appendix_start_page is not None and not has_appendix:
        raise ValueError("--appendix-start-page 只能用于包含 \\appendix 或 appendices 环境的源码")
    if contest != "cumcm" and (
        body_start_page is not None or appendix_start_page is not None
    ):
        raise ValueError("正文与附录页码参数仅适用于已核验正文上限规则的 CUMCM")
    if DANGEROUS_TEX.search(source):
        issues.append("源码包含被禁用的 TeX 文件或命令执行指令")
    if "\\begin{document}" not in source or "\\end{document}" not in source:
        issues.append("LaTeX 入口缺少完整的 document 环境")
    if not re.search(r"\\begin\s*\{abstract\}", source):
        issues.append("缺少摘要环境")
    if not re.search(r"\\(?:keywords|keyword)\s*\{", source, re.I):
        issues.append("缺少关键词命令")

    placeholders = sorted(set(re.findall(
        r"LATEX_TEMPLATE_[A-Z_]+|\bTODO\b|待补充|请填写|PLACEHOLDER",
        source,
        re.I,
    )))
    if placeholders:
        issues.append("仍含模板占位符：" + "、".join(placeholders[:8]))

    equation_count = _display_equations(source, issues)
    figures = _environment_matches(source, r"figure\*?")
    tables = _environment_matches(source, r"table\*?|longtable")
    object_spans = sorted(
        [match.span() for match in figures + tables], reverse=True
    )
    outside = source
    for start, end in object_spans:
        outside = outside[:start] + (" " * (end - start)) + outside[end:]
    references: set[str] = set()
    for group in re.findall(
        r"\\(?:ref|pageref|autoref|cref|Cref|eqref)\*?\s*\{([^}]+)\}", outside
    ):
        references.update(label.strip() for label in group.split(",") if label.strip())
    all_labels = re.findall(r"\\label\s*\{([^}]+)\}", source)
    duplicate_labels = sorted(
        label for label, count in Counter(all_labels).items() if count > 1
    )
    if duplicate_labels:
        issues.append("重复 label：" + "、".join(duplicate_labels))

    figure_labels = []
    for index, match in enumerate(figures, 1):
        block = match.group("body")
        labels = re.findall(r"\\label\s*\{([^}]+)\}", block)
        figure_labels.extend(labels)
        if not re.search(r"\\caption(?:\[[^]]*\])?\s*\{", block):
            issues.append(f"第 {index} 个图环境缺少 caption")
        if not re.search(
            r"\\includegraphics(?:\[[^]]*\])?\s*\{|"
            r"\\begin\s*\{(?:tikzpicture|axis)\}",
            block,
        ):
            issues.append(f"第 {index} 个图环境没有图片或绘图内容")
        if not labels:
            issues.append(f"第 {index} 个图环境缺少 label")
        elif not any(label in references for label in labels):
            issues.append(f"孤儿图：{labels[0]} 未在图环境外的正文引用")

    for index, match in enumerate(tables, 1):
        block = match.group("body")
        labels = re.findall(r"\\label\s*\{([^}]+)\}", block)
        if not re.search(r"\\caption(?:\[[^]]*\])?\s*\{", block):
            issues.append(f"第 {index} 个表环境缺少 caption")
        has_data = (
            match.group("env").startswith("longtable") and "&" in block
        ) or bool(re.search(
            r"\\begin\s*\{(?:tabular\*?|tabularx|array|tblr)\}", block
        ))
        if not has_data:
            issues.append(f"第 {index} 个表环境没有表格数据")
        if not labels:
            issues.append(f"第 {index} 个表环境缺少 label")
        elif not any(label in references for label in labels):
            issues.append(f"孤儿表：{labels[0]} 未在表环境外的正文引用")

    graphic_roots = _graphic_roots(source, root)
    for raw in re.findall(
        r"\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}", source
    ):
        try:
            status = _graphic_status(root, graphic_roots, raw)
        except ValueError as error:
            issues.append(str(error))
            continue
        if status == "missing":
            issues.append(f"图片文件不存在：{raw}")
        elif status == "unsupported":
            issues.append(f"图片格式不受安全编译链支持，请先转为 PDF/PNG/JPG：{raw}")

    citations: set[str] = set()
    for group in re.findall(
        r"\\(?:(?i:cite|citep|citet|citealp|citealt|citeauthor|citeyear|"
        r"parencite|textcite|autocite|footcite|smartcite|supercite))\*?"
        r"(?:\[[^]]*\]){0,2}\s*\{([^}]+)\}",
        source,
    ):
        citations.update(key.strip() for key in group.split(",") if key.strip())
    bibliography, manual_bibliography = _bibliography_keys(source, root, issues)
    missing_citations = sorted(citations - bibliography)
    if missing_citations:
        issues.append("正文引用缺少参考文献条目：" + "、".join(missing_citations))
    unused_manual = sorted(manual_bibliography - citations)
    if unused_manual:
        issues.append("手工参考文献未被正文引用：" + "、".join(unused_manual))

    document = source.split("\\begin{document}", 1)[-1]
    document = re.sub(r"\\begin\s*\{[^}]+\}|\\end\s*\{[^}]+\}", " ", document)
    document = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", document)
    document = re.sub(r"[{}$^_&~#]", " ", document)
    content_units = len(re.findall(
        r"[\u3400-\u9fff]|[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?",
        document,
    ))

    question_ids = _normalise_questions(questions)
    if quality_checks and not question_ids:
        issues.append("质量校验必须通过 --questions 声明全部子问题（如 q1 q2）")
    question_coverage = {}
    for question in question_ids:
        matched = any(re.match(
            rf"fig:{re.escape(question)}(?:[-_:]|$)", label, re.I
        ) for label in figure_labels)
        question_coverage[question] = matched
        if not matched:
            issues.append(f"子问题 {question} 缺少正式结果图（label 应以 fig:{question}- 开头）")

    threshold_values, threshold_overrides = _thresholds(
        contest,
        quality_checks,
        {
            "min_content_units": min_content_units,
            "min_pages": min_pages,
            "min_equations": min_equations,
            "min_figures": min_figures,
            "min_tables": min_tables,
        },
        max_pages,
        override_reason,
    )
    if quality_checks and min_image_dpi < 300:
        if not override_reason or not override_reason.strip():
            raise ValueError("降低 min_image_dpi 质量门槛必须提供 override_reason")
        threshold_overrides.append({
            "name": "min_image_dpi",
            "default": 300,
            "value": min_image_dpi,
            "reason": override_reason.strip(),
        })

    source_hash = source_bundle_sha256(main)
    rendered_pages = None
    body_pages = None
    pdf_audit = None
    build_manifest = None
    pdf = pdf_path.resolve() if pdf_path else None
    must_have_pdf = quality_checks if require_pdf is None else require_pdf
    if quality_checks and not must_have_pdf:
        if not override_reason or not override_reason.strip():
            raise ValueError("跳过 PDF 质量校验必须提供 override_reason")
        threshold_overrides.append({
            "name": "require_pdf",
            "default": True,
            "value": False,
            "reason": override_reason.strip(),
        })
    if pdf is not None and pdf.is_file():
        build_manifest, binding_issues = _verify_pdf_binding(main, pdf, source_hash)
        issues.extend(binding_issues)
        try:
            pdf_audit = _audit_pdf(
                pdf,
                min_image_dpi=min_image_dpi,
                require_image_audit=quality_checks,
            )
            rendered_pages = pdf_audit["pages"]
            page_texts = pdf_audit.pop("_page_texts", [])
            issues.extend(pdf_audit["issues"])
            if contest == "cumcm":
                body_start = body_start_page or 2
                detected_appendix = (
                    _appendix_start_page(source, page_texts) if has_appendix else None
                )
                appendix_start = appendix_start_page or detected_appendix
                body_end = appendix_start or (rendered_pages + 1)
                if (
                    appendix_start_page is not None
                    and detected_appendix is not None
                    and appendix_start_page != detected_appendix
                ):
                    issues.append(
                        f"显式附录起始页 {appendix_start_page} 与 PDF 自动定位页 "
                        f"{detected_appendix} 不一致"
                    )
                if body_start > rendered_pages + 1:
                    issues.append(
                        f"正文起始页 {body_start} 超出 PDF 总页数 {rendered_pages}"
                    )
                elif body_end < body_start or body_end > rendered_pages + 1:
                    issues.append(
                        f"附录起始页 {body_end} 与正文起始页 {body_start}、"
                        f"PDF 总页数 {rendered_pages} 不一致"
                    )
                elif has_appendix and appendix_start is None and max_pages is not None:
                    issues.append(
                        "无法自动定位附录起始页；请通过 --appendix-start-page "
                        "提供附录在 PDF 中的第一页"
                    )
                else:
                    body_pages = body_end - body_start
        except Exception as error:
            issues.append(f"PDF 检查失败：{error}")
    elif pdf is not None:
        issues.append(f"指定的 PDF 不存在：{pdf}")
    elif must_have_pdf:
        issues.append("缺少编译后的 PDF，无法执行实际页数和版面检查")

    metrics = {
        "content_units": content_units,
        "equations": equation_count,
        "figures": len(figures),
        "tables": len(tables),
        "citations": len(citations),
        "source_files": len(source_files),
        "question_figure_coverage": question_coverage,
    }
    labels = {
        "content_units": "字词单位",
        "equations": "公式",
        "figures": "图",
        "tables": "表",
    }
    for key, metric_key in (
        ("min_content_units", "content_units"),
        ("min_equations", "equations"),
        ("min_figures", "figures"),
        ("min_tables", "tables"),
    ):
        minimum = threshold_values[key]
        if minimum and metrics[metric_key] < minimum:
            issues.append(
                f"预警：{labels[metric_key]} {metrics[metric_key]}，低于质量目标 {minimum}"
            )
    minimum_pages = threshold_values["min_pages"]
    if rendered_pages is not None and minimum_pages and rendered_pages < minimum_pages:
        issues.append(f"预警：实际页数 {rendered_pages}，低于质量目标 {minimum_pages}")
    if max_pages is not None:
        if contest == "cumcm" and body_pages is not None and body_pages > max_pages:
            issues.append(f"正文页数 {body_pages}，超过官方上限 {max_pages}")
        elif contest != "cumcm" and rendered_pages is not None and rendered_pages > max_pages:
            issues.append(f"PDF 总页数 {rendered_pages}，超过官方上限 {max_pages}")

    return {
        "main_tex": str(main),
        "project_root": str(root),
        "pdf_path": str(pdf) if pdf else None,
        "source_sha256": source_hash,
        "rendered_pages": rendered_pages,
        "body_pages": body_pages,
        "page_limit_scope": "body" if contest == "cumcm" else "total",
        "pdf_audit": pdf_audit,
        "build_manifest": build_manifest,
        "thresholds": threshold_values,
        "threshold_overrides": threshold_overrides,
        "metrics": metrics,
        "issues": list(dict.fromkeys(issues)),
        "passed": not issues,
    }


def _tool_version(executable: str | None) -> str | None:
    if executable is None:
        return None
    name = Path(executable).stem.casefold()
    flag = "-v" if name in {"pdfimages", "pdftoppm"} else "--version"
    try:
        process = subprocess.Popen(
            [executable, flag],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output, _ = process.communicate(timeout=10)
    except (OSError, subprocess.SubprocessError):
        if "process" in locals():
            process.kill()
        return None
    return output.splitlines()[0].strip() if process.returncode == 0 and output else None


def _warning_lines(log: str) -> tuple[list[str], list[str]]:
    fatal = []
    if re.search(
        r"undefined references|Citation .+ undefined|There were undefined citations",
        log,
        re.I,
    ):
        fatal.append("编译日志仍有未解析的引用或文献")
    if re.search(r"Undefined control sequence|! LaTeX Error", log, re.I):
        fatal.append("编译日志包含 LaTeX 错误")
    warnings = []
    for line in log.splitlines():
        stripped = line.strip()
        if re.search(
            r"(?:LaTeX|Package .+|Class .+) Warning:|"
            r"(?:Over|Under)full \\[hv]box|font warning",
            stripped,
            re.I,
        ):
            warnings.append(stripped)
    return fatal, list(dict.fromkeys(warnings))


def _warning_gate(
    warnings: list[str],
    patterns: list[str] | tuple[str, ...] | None,
    reason: str | None,
) -> tuple[list[str], list[str]]:
    raw_patterns = list(patterns or [])
    if raw_patterns and (not reason or not reason.strip()):
        raise ValueError("允许编译预警必须提供 override_reason")
    try:
        compiled = [re.compile(pattern, re.I) for pattern in raw_patterns]
    except re.error as error:
        raise ValueError(f"无效的预警允许正则：{error}") from error
    allowed, blocked = [], []
    for warning in warnings:
        (allowed if any(pattern.search(warning) for pattern in compiled) else blocked).append(
            warning
        )
    return allowed, blocked


def build_paper(
    main_tex: Path,
    *,
    engine: str = "xelatex",
    output_dir: Path | None = None,
    publish_path: Path | None = None,
    timeout: int = 180,
    allow_warnings: list[str] | tuple[str, ...] | None = None,
    override_reason: str | None = None,
    overwrite: bool = False,
) -> dict:
    """在禁用 shell escape 的条件下编译，并仅发布通过门禁的 PDF。"""
    if engine not in ENGINES:
        raise ValueError(f"不支持的 LaTeX 引擎：{engine}")
    if timeout <= 0:
        raise ValueError("timeout 必须为正整数")
    if main_tex.is_symlink():
        raise ValueError("LaTeX 主入口不能是符号链接")
    main = _writable(main_tex)
    if not main.is_file() or main.suffix.casefold() != ".tex":
        raise FileNotFoundError(f"LaTeX 入口不存在：{main}")
    root = _project_root(main)
    _reject_symlinks(root)
    binding_issues = _resource_binding_issues(root)
    if binding_issues:
        raise ValueError("；".join(binding_issues))
    source, _, source_issues = _collect_sources(main)
    if source_issues:
        raise ValueError("；".join(source_issues))
    if DANGEROUS_TEX.search(source):
        raise ValueError("源码包含被禁用的 TeX 文件或命令执行指令")
    graphic_roots = _graphic_roots(source, root)
    for raw in re.findall(
        r"\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}", source
    ):
        status = _graphic_status(root, graphic_roots, raw)
        if status == "unsupported":
            raise ValueError(f"不支持直接编译 SVG/EPS，请先转为 PDF/PNG/JPG：{raw}")
        if status == "missing":
            raise FileNotFoundError(f"图片文件不存在：{raw}")

    output = _inside((output_dir or (root / "build")), root)
    if output != (root / "build").resolve():
        raise ValueError("--output-dir 仅允许使用项目根目录下的 build/")
    output_argument = "build"
    publish_target = None
    if publish_path is not None:
        publish_target = _writable(publish_path)
        if publish_target.parent != root.parent:
            raise ValueError("发布 PDF 必须位于 LaTeX 项目目录的直接父目录")
        if publish_target.suffix.casefold() != ".pdf":
            raise ValueError("发布路径必须使用 .pdf 扩展名")
        publish_manifest = _build_manifest_path(publish_target)
        if not overwrite and (publish_target.exists() or publish_manifest.exists()):
            raise FileExistsError(f"发布产物已存在，拒绝覆盖：{publish_target}")

    latexmk_path = shutil.which("latexmk")
    latexmk_version = _tool_version(latexmk_path)
    latexmk = latexmk_path if latexmk_version else None
    executable = shutil.which(engine)
    if executable is None:
        raise RuntimeError(f"未找到 {engine}，无法编译 LaTeX 论文")
    main_argument = main.relative_to(root).as_posix()
    if latexmk:
        mode = {"xelatex": "-xelatex", "lualatex": "-lualatex", "pdflatex": "-pdf"}[
            engine
        ]
        commands = [[
            latexmk,
            "-norc",
            "-gg",
            mode,
            "-e",
            f"${engine} = '{engine} -no-shell-escape %O %S'",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-outdir={output_argument}",
            main_argument,
        ]]
    elif executable:
        if re.search(r"\\(?:bibliography|addbibresource)\b", source):
            raise RuntimeError("未找到可执行的 latexmk，含外部参考文献的论文无法完成可靠编译")
        command = [
            executable,
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-output-directory={output_argument}",
            main_argument,
        ]
        commands = [command, command]
    else:
        raise RuntimeError(f"未找到 {engine}，无法编译 LaTeX 论文")

    pdf = output / f"{main.stem}.pdf"
    log_path = output / f"{main.stem}.log"
    build_manifest_path = _build_manifest_path(pdf)
    source_hash = source_bundle_sha256(main)
    outputs = []
    runs = []
    with tempfile.TemporaryDirectory(prefix="latex-build-") as temporary_directory:
        isolated_root = Path(temporary_directory) / "project"
        shutil.copytree(
            root,
            isolated_root,
            ignore=shutil.ignore_patterns("build", *PROJECT_IGNORED_NAMES),
        )
        isolated_main = isolated_root / main_argument
        isolated_output = isolated_root / "build"
        isolated_output.mkdir()
        for command in commands:
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    command,
                    cwd=isolated_root,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env={**os.environ, "openin_any": "p", "openout_any": "p"},
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"LaTeX 编译超过 {timeout} 秒") from error
            combined = (completed.stdout or "") + (completed.stderr or "")
            outputs.append(combined)
            runs.append({
                "command": command,
                "cwd": str(root),
                "returncode": completed.returncode,
                "duration_seconds": round(time.monotonic() - started, 3),
            })
            if completed.returncode != 0:
                detail = combined.strip()[-2000:]
                raise RuntimeError(f"LaTeX 编译失败：{detail or completed.returncode}")

        isolated_pdf = isolated_output / f"{main.stem}.pdf"
        isolated_log = isolated_output / f"{main.stem}.log"
        if not isolated_pdf.is_file():
            raise RuntimeError("LaTeX 命令执行成功但没有生成 PDF")
        log = (
            isolated_log.read_text(encoding="utf-8", errors="replace")
            if isolated_log.is_file()
            else "\n".join(outputs)
        )
        if not isolated_log.is_file():
            isolated_log.write_text(log, encoding="utf-8")
        fatal_issues, warnings = _warning_lines(log)
        compiled_source_hash = source_bundle_sha256(isolated_main)
        post_build_hash = source_bundle_sha256(main)
        if compiled_source_hash != source_hash:
            fatal_issues.append("隔离编译过程修改了源码副本，拒绝发布")
        if post_build_hash != source_hash:
            fatal_issues.append("编译期间原始 LaTeX 项目发生变化，拒绝发布")
        allowed_warnings, blocked_warnings = _warning_gate(
            warnings, allow_warnings, override_reason
        )
        issues = fatal_issues + [
            f"编译预警：{warning}" for warning in blocked_warnings
        ]
        passed = not issues
        project_manifest = root / PROJECT_MANIFEST
        reproduce = [
            sys.executable,
            str(Path(__file__).resolve()),
            "build",
            str(main),
            "--engine",
            engine,
            "--timeout",
            str(timeout),
        ]
        if publish_target is not None:
            reproduce.extend(["--publish", str(publish_target)])
        for pattern in allow_warnings or []:
            reproduce.extend(["--allow-warning", pattern])
        if override_reason:
            reproduce.extend(["--override-reason", override_reason.strip()])
        if overwrite:
            reproduce.append("--overwrite")
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "passed": passed,
            "main_tex": main_argument,
            "project_root": str(root),
            "source_sha256": source_hash,
            "compiled_source_sha256": compiled_source_hash,
            "post_build_source_sha256": post_build_hash,
            "pdf_sha256": _sha256(isolated_pdf),
            "project_manifest_sha256": (
                _sha256(project_manifest) if project_manifest.is_file() else None
            ),
            "engine": engine,
            "tools": {
                "latexmk": {
                    "path": latexmk,
                    "version": latexmk_version,
                },
                engine: {
                    "path": executable,
                    "version": _tool_version(executable),
                },
            },
            "commands": runs,
            "warnings": warnings,
            "allowed_warnings": allowed_warnings,
            "warning_override": {
                "patterns": list(allow_warnings or []),
                "reason": override_reason.strip() if override_reason else None,
            },
            "issues": issues,
            "reproduce": subprocess.list2cmdline(reproduce),
        }

        output.mkdir(parents=True, exist_ok=True)
        log_temporary = output / f".{log_path.name}.new-{uuid.uuid4().hex}"
        try:
            shutil.copyfile(isolated_log, log_temporary)
            os.replace(log_temporary, log_path)
        finally:
            if log_temporary.exists():
                log_temporary.unlink()
        build_manifest_path = _replace_pair(
            isolated_pdf, pdf, manifest, overwrite=True
        )

        published_pdf = None
        published_manifest = None
        if publish_target is not None and passed:
            published_manifest = _replace_pair(
                isolated_pdf,
                publish_target,
                manifest,
                overwrite=overwrite,
            )
            published_pdf = str(publish_target)

    return {
        "main_tex": str(main),
        "project_root": str(root),
        "pdf_path": str(pdf),
        "build_manifest": str(build_manifest_path),
        "published_pdf": published_pdf,
        "published_manifest": str(published_manifest) if published_manifest else None,
        "log_path": str(log_path),
        "engine": engine,
        "warnings": warnings,
        "allowed_warnings": allowed_warnings,
        "issues": issues,
        "passed": passed,
    }


def doctor(
    *,
    engine: str = "xelatex",
    bibliography_backend: str = "none",
    need_pandoc: bool = False,
    need_pdf_audit: bool = True,
) -> dict:
    """检查当前环境是否具备所选 LaTeX 工作流需要的工具。"""
    if engine not in ENGINES:
        raise ValueError(f"不支持的 LaTeX 引擎：{engine}")
    if bibliography_backend not in {"none", "bibtex", "biber"}:
        raise ValueError("bibliography_backend 必须为 none、bibtex 或 biber")
    paths = {
        "latexmk": shutil.which("latexmk"),
        engine: shutil.which(engine),
        "bibtex": shutil.which("bibtex"),
        "biber": shutil.which("biber"),
        "pandoc": shutil.which("pandoc"),
        "pdfimages": shutil.which("pdfimages"),
        "pdftoppm": shutil.which("pdftoppm"),
    }
    tools = {
        name: {"path": path, "version": _tool_version(path)}
        for name, path in paths.items()
    }
    tools["pypdf"] = {
        "available": importlib.util.find_spec("pypdf") is not None,
        "version": None,
    }
    try:
        if tools["pypdf"]["available"]:
            from importlib.metadata import version
            tools["pypdf"]["version"] = version("pypdf")
    except Exception:
        pass
    issues = []
    unavailable = lambda name: not paths[name] or not tools[name]["version"]
    if unavailable(engine):
        issues.append(f"缺少或无法执行 LaTeX 引擎 {engine}")
    if bibliography_backend != "none" and unavailable("latexmk"):
        issues.append("外部参考文献工作流缺少 latexmk")
    if bibliography_backend != "none" and unavailable(bibliography_backend):
        issues.append(f"缺少或无法执行参考文献工具 {bibliography_backend}")
    if need_pandoc and unavailable("pandoc"):
        issues.append("缺少或无法执行 pandoc，无法生成 DOCX")
    if need_pdf_audit and not tools["pypdf"]["available"]:
        issues.append("缺少 pypdf，无法检查 PDF")
    if need_pdf_audit and unavailable("pdfimages"):
        issues.append("缺少或无法执行 pdfimages，无法检查位图分辨率")
    if need_pdf_audit and unavailable("pdftoppm"):
        issues.append("缺少或无法执行 pdftoppm，无法进行 PDF 渲染抽查")
    return {
        "engine": engine,
        "bibliography_backend": bibliography_backend,
        "tools": tools,
        "issues": issues,
        "passed": not issues,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    init = commands.add_parser("init", help="从官方或内置模板初始化 LaTeX 项目")
    init.add_argument("output_dir", type=Path)
    init.add_argument("--contest", choices=sorted(CONTESTS), default="cumcm")
    init.add_argument("--template", type=Path)
    init.add_argument("--main", dest="main_file")
    init.add_argument("--template-source")
    init.add_argument("--template-version")

    bind = commands.add_parser("bind", help="绑定 LaTeX 资源副本与 PROJECT_ROOT 权威来源")
    bind.add_argument("main_tex", type=Path)
    bind.add_argument("source", type=Path)
    bind.add_argument("destination")

    build = commands.add_parser("build", help="编译并按门禁发布 LaTeX 项目")
    build.add_argument("main_tex", type=Path)
    build.add_argument("--engine", choices=sorted(ENGINES), default="xelatex")
    build.add_argument("--output-dir", type=Path)
    build.add_argument("--publish", type=Path)
    build.add_argument("--timeout", type=int, default=180)
    build.add_argument("--allow-warning", action="append", default=[])
    build.add_argument("--override-reason")
    build.add_argument("--overwrite", action="store_true")

    validate = commands.add_parser("validate", help="校验 LaTeX 源码和编译结果")
    validate.add_argument("main_tex", type=Path)
    validate.add_argument("--contest", choices=sorted(CONTESTS), default="cumcm")
    validate.add_argument("--pdf", type=Path)
    validate.add_argument("--quality-checks", action="store_true")
    validate.add_argument("--min-content-units", type=int)
    validate.add_argument("--min-pages", type=int)
    validate.add_argument("--max-pages", type=int)
    validate.add_argument("--body-start-page", type=int)
    validate.add_argument("--appendix-start-page", type=int)
    validate.add_argument("--min-equations", type=int)
    validate.add_argument("--min-figures", type=int)
    validate.add_argument("--min-tables", type=int)
    validate.add_argument("--min-image-dpi", type=int, default=300)
    validate.add_argument("--questions", nargs="+")
    validate.add_argument("--override-reason")
    validate.add_argument(
        "--no-require-pdf",
        action="store_false",
        dest="require_pdf",
        default=None,
    )

    check = commands.add_parser("doctor", help="检查 LaTeX、PDF 与 DOCX 工具链")
    check.add_argument("--engine", choices=sorted(ENGINES), default="xelatex")
    check.add_argument(
        "--bibliography-backend",
        choices=("none", "bibtex", "biber"),
        default="none",
    )
    check.add_argument("--need-pandoc", action="store_true")
    check.add_argument(
        "--no-pdf-audit",
        action="store_false",
        dest="need_pdf_audit",
        default=True,
    )
    return parser


def main() -> int:
    _configure_utf8_stdio()
    arguments = _parser().parse_args()
    try:
        if arguments.action == "init":
            result = prepare_project(
                arguments.output_dir,
                contest=arguments.contest,
                template_path=arguments.template,
                main_file=arguments.main_file,
                template_source=arguments.template_source,
                template_version=arguments.template_version,
            )
        elif arguments.action == "bind":
            result = bind_resource(
                arguments.main_tex,
                arguments.source,
                arguments.destination,
            )
        elif arguments.action == "build":
            result = build_paper(
                arguments.main_tex,
                engine=arguments.engine,
                output_dir=arguments.output_dir,
                publish_path=arguments.publish,
                timeout=arguments.timeout,
                allow_warnings=arguments.allow_warning,
                override_reason=arguments.override_reason,
                overwrite=arguments.overwrite,
            )
        elif arguments.action == "validate":
            result = inspect_paper(
                arguments.main_tex,
                contest=arguments.contest,
                pdf_path=arguments.pdf,
                quality_checks=arguments.quality_checks,
                min_content_units=arguments.min_content_units,
                min_pages=arguments.min_pages,
                max_pages=arguments.max_pages,
                body_start_page=arguments.body_start_page,
                appendix_start_page=arguments.appendix_start_page,
                min_equations=arguments.min_equations,
                min_figures=arguments.min_figures,
                min_tables=arguments.min_tables,
                min_image_dpi=arguments.min_image_dpi,
                require_pdf=arguments.require_pdf,
                questions=arguments.questions,
                override_reason=arguments.override_reason,
            )
        else:
            result = doctor(
                engine=arguments.engine,
                bibliography_backend=arguments.bibliography_backend,
                need_pandoc=arguments.need_pandoc,
                need_pdf_audit=arguments.need_pdf_audit,
            )
    except (OSError, RuntimeError, ValueError) as error:
        result = {
            "passed": False,
            "issues": [str(error)],
            "error_type": type(error).__name__,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
