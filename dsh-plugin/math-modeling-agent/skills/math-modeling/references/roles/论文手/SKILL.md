---
name: 论文手
description: 根据题目、建模分析和真实代码结果生成完整 Word 数学建模论文，用户显式要求时同时生成 LaTeX 论文。
---

# 论文手

## 开始门禁

开始写作前先在进度更新中报告输入检查结果。题目、两个建模交付物、可运行代码、实际结果表格、三类每类至少 3 张且覆盖全部子问题的候选图或已核验文献任一缺失时，回退补齐，不要直接写论文。

当届官方规则或模板尚未核验时，论文手先完成核验；用户已明确启用"官方规则核验 Subagent"时，可让其与输入盘点并行取得官方 URL、适用届次、硬约束、模板路径与哈希。无论是否启用可选 Subagent，论文构建都必须等待规则核验完成。

真实竞赛或其他有明确截止时间的任务还必须读取 `../../../references/交付与截止时间协议.md`，核验当前系统时间、官方截止时间与时区，并在主要内容完成前用最小真实样例走通代码、图表、论文导出和支撑材料打包链路。

- 用户显式要求 LaTeX 时，必须先调用 `<SKILL_ROOT>/tools/latex/scripts/latex_paper.py init` 复制官方或内置模板；禁止另写一个临时 `main.tex` 替代模板链路。
- 写正文前先调用 `latex_paper.py doctor` 检查所选引擎、参考文献后端、PDF 审计工具和 Pandoc（需要 Word 时）；环境不完整立即报告，不把工具链检查拖到交付前。
- 引用前必须运行双引擎检索并打开 DOI 或出版机构页面核验元数据；仅复制上游参考文献列表不算核验。
- 写作过程中持续核对篇幅和主张—证据映射，不要等文档生成后才第一次统计。

## 路径

- `ROLE_ROOT`：本文件所在目录。
- `SKILL_ROOT`：`ROLE_ROOT/../../..`，只读。
- `PROJECT_ROOT`：用户项目目录，论文只写这里。

## 格式与交付物

默认只生成 Word 论文；用户显式要求时同时生成 LaTeX 论文，确保正文内容、数据、图表和结论一致。

- Word：`PROJECT_ROOT/完整论文.docx`。
- LaTeX（可选）：`PROJECT_ROOT/完整论文-LaTeX/` 源码项目和由它实际编译的 `PROJECT_ROOT/完整论文.pdf`。

用户明确只要 Word 时，只生成 Word。当届官方提交要求仍决定实际可提交的版本；论文仅供参考，格式和内容必须服从当届官方规则。

## 内容边界（不得写入论文）

论文正文、摘要、结论和官方要求的附录只包含题目答案、建模、求解、结果、结论和参考文献。本 Skill 内部的流程管控信息——门禁（M1/P1/P2/W1/W2）、Subagent 质检、质检简报/回执、证据大纲、复现清单、哈希/版本、编译日志、Checkpoint、内容冻结——只存在于对话交互与内部检查，严禁出现在任何论文交付物中。写作或验证时若发现这些词被带入正文，必须清除并重新导出。

## 官方规则优先级

1. 用户提供的当届官方模板和规则。
2. 从目标竞赛官方网站取得的当届模板和规则。
3. Word 分支以 `references/论文模板.docx` 和 `../../../tools/docx/scripts/paper_format.py` 为无官方模板时的构建基线；LaTeX 分支以 `../../../tools/latex/assets/templates/` 为构建基线。两者均不得声称替代当届官方文件。

在生成前明确竞赛名称、届次、语言和官方规则来源。官方结构、页型、页边距、摘要页、页数、编号和提交格式均以当届规则为准。

同时明确篇幅质量目标。CUMCM 未取得当届更具体要求时，可用“约 15000 字词单位、约 20 页”规划完整度，但必须标注为可调整的质量目标，不能写成官方最低要求。以 2026 年官方规范为例，正文要求是不超过 30 页，并未规定最低 15000 字或最低 20 页。

无论目标竞赛为何，默认用至少 8 幅正式图规划论文证据链，其他竞赛不低于 CUMCM；每幅图都要有连续编号、题注和正文引用。当届官方规则或用户明确要求与此冲突时，以其要求为准并记录依据。

## 执行顺序

1. 读取题目、`题目分析报告.md`、`术语表格.md`、全部真实运行表格、图和代码。
2. 建立内部 Claim-Evidence 映射和论文大纲，为每个子问题列出核心主张、公式、结果表位置、至少 1 幅拟入文正式图、代码输出或已核验文献；不得只集中使用问题一的图。缺证据时回退到建模手或编程手，禁止编造。
3. 在开始长篇正文和双格式排版前，派发独立质检 Subagent 执行 `W1` 证据大纲门禁；未返回 `PASS` 不得先写后补。
4. 按官方结构写完整正文，引用由双引擎搜索结果和原始出版页面核验。
5. 默认先确定同一份正文、数据、图表和参考文献，再生成 Word；用户显式要求 LaTeX 时同时生成 LaTeX，禁止两份论文出现不同结论。
6. Word 使用 `../../../tools/docx/SKILL.md` 构建 DOCX，公式使用原生 OMML；已有完整 LaTeX 主稿时可通过 `convert_latex` 生成内容一致的 Word 初稿，再按官方 DOCX 模板修正。LaTeX（可选）使用 `../../../tools/latex/SKILL.md` 复制完整官方模板项目、填充源码并真实编译 PDF。
7. 首次生成完整且满足已知硬约束的论文和支撑材料后，保存为独立的 Checkpoint V1；后续优化在新版本上进行，不得原地覆盖唯一可提交版本。条件允许时记录文件清单和 SHA-256。
8. 检查 Word 的结构、篇幅、公式、图表、全部子问题覆盖、编号引用、参考文献、Markdown 格式残留和实际渲染页数。用户显式要求 LaTeX 时，还必须消除编译错误及未解析的引用，核对权威资源—源码—PDF 哈希、PDF 总页数、正文页数、附录边界、字体嵌入、空白页、页面尺寸和图片 DPI；所有预警必须修正，或根据当届官方规则记录明确覆盖理由后才能继续。
9. 在安全时间内完成受控优化并复验为 Checkpoint V2；到达内部冻结时间后停止非必要修改，完成最终导出、完整性与合规检查，并提醒用户为上传和提交确认预留时间。
10. 完成下列确定性门禁后，派发独立质检 Subagent 执行 `W2` 论文终检；未返回 `PASS` 不得宣称完成或交付论文。

## 阶段内独立门禁

- `W1`：质检 Subagent 核对每个必须回答的结论类型都有精确证据路径，摘要拟用关键数值与结果表一致，图表、公式和引用都有章节落点。用户显式要求 LaTeX 时，Word 与 LaTeX 共用同一证据源。证据大纲只保留在内部检查中，不新增固定交付物。
- `W2`：用户要求的全部格式冻结且各自确定性命令全部返回 0 后，质检 Subagent 核对当届规则、主张—证据、数值与单位、图表引用、文献和实际渲染效果；同时生成两种格式时再检查 Word/LaTeX 一致性。失败时精确到页码、章节、命令或来源并返回对应角色。

两次门禁均按 `../../../references/Subagent调度.md` 返回证据；正文、数据、图表或规则发生实质变化时重跑受影响门禁。

## 完成门禁

交付 Word 时依次运行：

```powershell
python "<SKILL_ROOT>/tools/docx/scripts/paper_format.py" validate "<PROJECT_ROOT>/完整论文.docx" --contest cumcm --rendered-pages <DOCX实际渲染页数>
python "<SKILL_ROOT>/tools/docx/scripts/office/validate.py" "<PROJECT_ROOT>/完整论文.docx"
python "<SKILL_ROOT>/tools/docx/scripts/equations.py" verify-conversion "<PROJECT_ROOT>/完整论文.docx"
```

最后一条仅适用于由 `equations.py generate/convert-latex` 生成的 DOCX；若 Word 由 `paper_format.py` 直接构建，则改为核对其自身质量报告和复现清单。交付 LaTeX 时依次运行：

```powershell
python "<SKILL_ROOT>/tools/latex/scripts/latex_paper.py" doctor --engine xelatex --bibliography-backend <none|bibtex|biber> --need-pandoc
python "<SKILL_ROOT>/tools/latex/scripts/latex_paper.py" build "<PROJECT_ROOT>/完整论文-LaTeX/main.tex" --engine xelatex --publish "<PROJECT_ROOT>/完整论文.pdf"
python "<SKILL_ROOT>/tools/latex/scripts/latex_paper.py" validate "<PROJECT_ROOT>/完整论文-LaTeX/main.tex" --pdf "<PROJECT_ROOT>/完整论文.pdf" --contest cumcm --quality-checks --questions q1 q2 q3 --min-image-dpi 300 --max-pages <当届官方正文上限> --body-start-page <正文起始页> --appendix-start-page <附录起始页>
```

根据目标竞赛、实际子问题和官方模板替换 `contest`、`--questions`、引擎、参考文献后端、正文起始页、附录起始页及正文页数上限；没有附录时省略 `--appendix-start-page`。复制权威代码或图表到 LaTeX 项目后，先按 LaTeX 工具说明执行 `bind`，再编译和校验。构建警告默认阻断 `完整论文.pdf` 发布；只有已核对警告才能同时提供 `--allow-warning` 与 `--override-reason`。降低默认质量目标也必须通过 `--override-reason` 记录官方条款或用户要求。任一命令退出码非零即回到论文构建步骤修正；缺少运行环境即明确报告阻塞，不得交付未通过版本。最终回复必须报告篇幅、PDF 总页数、正文页数、公式数、图数、表数、子问题图覆盖、引用核验、资源/源码/PDF 哈希绑定和全部命令退出码。

Word 门禁通过后还必须提取实际 DOCX 文本并检查 Markdown 格式残留与占位符，包括 `**`、反引号、标题 `#`、表格竖线、公式定界符 `$`/`$$`、行首列表标记和 `[待补充]`；命中项逐条判断，非正文所需的残留必须在源文件中清除、重新导出并复验。

上述命令只完成作者侧技术校验，不替代 `W2` 独立验收。

## 何时加载

| 情形 | 读取 |
|---|---|
| 开始写作 | `references/工作流程.md` |
| 组织章节 | `references/章节模板.md` |
| 生成 Word | `references/论文格式规范.md`、`../../../tools/docx/SKILL.md` |
| 生成 LaTeX | `references/LaTeX格式规范.md`、`../../../tools/latex/SKILL.md` |
| 中文写作检查 | `references/写作规范.md` |
| 抓分检查（摘要/结论/图表） | `references/写作规范.md`（第二节 评阅人抓分方法论） |
| 英文 MCM/ICM | `references/英文化工作流.md` |
| 交付前 | `references/自审框架.md` |
| 真实竞赛、截止时间或最终提交 | `../../../references/交付与截止时间协议.md` |
| 阶段内独立验收 | `../../../references/Subagent调度.md` |

内部分析表、核对清单和临时 Markdown 不作为交付物。
