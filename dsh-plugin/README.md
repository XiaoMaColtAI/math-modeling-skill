# 数学建模 Workbench — Agent 预设分享包

一个**完全自包含**的数学建模 Agent 预设：内置完整知识库（角色规范、算法资料、工具说明）+ 工作流引擎（三阶段 / 五门禁 / 任务看板 / 进度看板）。**不依赖任何外部 skill 仓库路径**，可整体复制到任何机器使用。

## 目录结构

```
math-modeling-agent/                  # 预设根（整体复制即可用）
├── agent.cordis.yml                  # 预设 composition（插件行 + 系统提示）
├── preset.yml                        # 预设元数据（显示名/描述）
├── plugins/
│   ├── math-modeling.js              # 工作流引擎（零依赖 ESM 插件）
│   └── dsh-math-modeling-ui/         # 进度看板 UI 组合包（可选，重启自动加载）
│       ├── lib/index.js              # host 侧：Connection RPC 读 state.json
│       └── client/client.js          # client 侧：右侧悬浮看板 + 设置页开关
├── skills/
│   └── math-modeling/                # 内置知识库（随预设持久化）
│       ├── SKILL.md                  # 主协议（name: math-modeling）
│       ├── references/               # 三角色规范 / 算法索引 / Subagent 调度 / 质检清单
│       ├── assets/                   # 7 类算法资料（优化/预测/评价/图论/统计/综合/ML）
│       └── tools/                    # docx/figure/latex/paper_search/pdf/xlsx 说明
└── ui-dock/                          # UI 设计说明与安装文档（可选）
```

---

## 复制安装提示词（给 DeepSeek Harness）

在 DeepSeek Harness 中**新建一个会话**，把下面整段内容复制粘贴给 agent，它会自动完成安装：

````text
请帮我安装「数学建模 Workbench」Agent 预设（math-modeling 数学建模工作台）。

源目录：<在这里填写分享包的实际路径，例如 F:\xxx\math-modeling-agent>

请按以下步骤执行：

1. 定位预设根目录：用 agentPresets 服务调用 list()，查看现有预设的 path，推断出本机
   <dsh-home>\.agent-presets\ 根目录（dsh-home 通常是 $env:DSH_HOME，未设置时默认
   C:\Users\<用户名>\AppData\Roaming\dsh-desktop\dsh-home）。

2. 复制源目录：把整个源目录复制到预设根下，目标目录名即预设 id（建议 math-modeling），
   最终结构为 <预设根>\math-modeling\ 且包含 agent.cordis.yml、preset.yml、plugins\、skills\。
   如果源目录名不是想要的目标 id，复制其全部内容到目标 id 目录即可。

3. 校验：
   - 用 agentPresets.resolve('math-modeling') 确认该预设可见（trust 为 user）；
   - 用 agentPresets.standingKeyFor('math-modeling') 做 mount 校验，确认 composition
     可加载、所有插件行正常激活、无报错。

4. 若复制或写文件被沙箱拦截：因为预设根位于会话工作区之外，这是预期的沙箱策略，
   请按权限提示升级（danger-full-access）后重试；不得绕过其他安全机制。

5. 完成后向我报告：预设 id、安装路径、mount 校验结果（成功/失败及错误信息）。

安装后的使用方式：新建会话，在预设选择器中选择「数学建模 Workbench」，即可使用
mm_project_init / mm_phase_enter / mm_todo / mm_gate / mm_check_deliverables /
mm_complete / mm_state 等工具，并可通过 skill 工具加载内置 math-modeling 知识库。
````

> 提示：对方机器上若预设 id 已被占用，把目标 id 换成其他名字（仅小写字母/数字/连字符）即可，
> 插件会通过 `import.meta.url` 自动定位同目录知识库，**无需改任何配置**。

---

## 手动安装（两步）

### 1. 找到你的预设根目录

预设根 = `<dsh-home>\.agent-presets\`，其中 `dsh-home` 通常是：

- Windows：`C:\Users\<你的用户名>\AppData\Roaming\dsh-desktop\dsh-home`
- 环境变量：`$env:DSH_HOME`（若设置了则以此为准）

### 2. 复制预设

把整个 `math-modeling-agent` 目录复制到预设根下（目录名即预设 id，可改名）：

```powershell
# 示例（按你的实际路径调整）
$dst = "$env:APPDATA\dsh-desktop\dsh-home\.agent-presets"
Copy-Item -Path ".\math-modeling-agent" -Destination "$dst\math-modeling" -Recurse -Force
```

> 预设 ID（目标目录名）可改成你喜欢的名字（仅小写字母/数字/连字符），
> 插件会通过 `import.meta.url` 自动定位同目录下的 `skills/math-modeling` 知识库，**无需改任何配置**。

### 3.（可选）验证

重启 DSH（或新建会话）后，预设选择器应出现「数学建模 Workbench」。

## 使用

新建会话 → 选择预设 **「数学建模 Workbench」**，然后：

1. **知识加载**：让 agent "加载 math-modeling skill"（`skill` 工具，读内置库）；
2. **工作流执行**：
   - `mm_project_init` —— 初始化项目（竞赛/届次/子问题/论文格式）
   - `mm_phase_enter phase=modeling` —— 进入建模手（返回角色规范全文 + 任务清单）
   - `mm_todo list / check` —— 任务看板
   - `mm_gate` —— 五门禁质检（M1/P1/P2/W1/W2，派发独立只读 Subagent）
   - `mm_state` —— 三阶段进度看板
   - `mm_complete` —— 完成判定（门禁全 PASS + 交付物齐全 + 无漂移）
3. 状态持久化在 `<PROJECT_ROOT>/.math-modeling/state.json`，跨会话可续接。

## UI 进度看板（可选）

额外把 `dsh-math-modeling-ui` 组合包安装进 DSH profile 后，数学建模会话右侧会出现可折叠的进度看板（三阶段时间线 / 五门禁状态 / 任务完成度），并可在「设置 → 插件 → 数学建模」里开关（`mm_ui_toggle` 亦可控制，持久化）。看板只读 `<PROJECT_ROOT>/.math-modeling/state.json`，不修改业务数据。

## 内置工具

`mm_project_init` · `mm_state` · `mm_phase_enter` · `mm_skill_read` · `mm_gate` · `mm_check_deliverables` · `mm_complete` · `mm_log` · `mm_todo` · `mm_ui_toggle`

## 可移植性说明

- 插件通过 `import.meta.url` 推导自身位置 → 知识库路径**运行时自动解析**，无任何硬编码绝对路径；整个预设目录可整体复制/改名到任意位置；
- 内置 skill 通过 `skills.register` 注册（apply 时读 `skills/math-modeling/SKILL.md`），不依赖文件系统发现；
- 若内置知识库缺失（例如只复制了 `plugins/` 未复制 `skills/`），`mm_project_init` 会要求显式传入 `skillRoot`。

## 已知限制

- 编程手阶段需要本机 Python 环境（`mm_check_deliverables` 会如实报告 `blocked`）；
- 论文手 Word/LaTeX 生成依赖 skill 工具链脚本（内置了说明文档，可执行脚本请按需从知识库 `tools/*/SKILL.md` 指引准备）。
