# 数学建模 Workbench — UI 设计与现状

> 本文档描述与 `math-modeling-agent` 预设配套的 UI 插件的设计思路与当前功能。

## 一、总体思路

### 1. 两个插件的分工

DSH 架构分两个平面：
- **预设（宿主侧）**：`math-modeling.js` 注册 mm_* 工具、系统提示、状态机；
- **Client UI**：`dsh-math-modeling-ui` 组合包提供浏览器看板。

所以"数学建模 Agent"与"数学建模 UI 看板"是两个包：
- `math-modeling-agent`（预设）：干活的工作流引擎（三阶段/五门禁/任务/交付物）；
- `dsh-math-modeling-ui`（组合包）：给人看的进度看板。

### 2. 持久化机制（关键）

`dsh-math-modeling-ui` 是 **DSH 组合包**（`package.json` 同时声明 `dsh.bundle.patch` + `dsh.client`），
通过 `dsh plugin --profile web add` 安装到 profile，**重启后自动加载，无需创造模式手动开启**。

### 3. 数据流（状态文件是唯一真相）

```
工作流引擎（预设插件）                  UI 看板（组合包）
mm_project_init / mm_todo / mm_gate
        │                                   │
        │ 写 state.json                      │ host connection.rpc.handle
        ▼                                   ▼
<PROJECT_ROOT>/.math-modeling/state.json ──▶ ctx.connection.rpc.call('/math-modeling-ui', 'mm.state')
   （持久化，跨会话/重启有效）                    （右侧悬浮面板渲染）
```

- **状态持久化**：所有进度真相在 `<PROJECT_ROOT>/.math-modeling/state.json`；
- **UI 只读**：看板通过 `connection.rpc` 读 state.json，不修改业务数据；
- **host RPC**：`ctx.connection.rpc.handle('/math-modeling-ui', (endpoint, payload) => {...})`（组合包持久 RPC，非动态插件 harness.handle）。

### 4. 位置：右侧悬浮面板（shell.overlay）

- 用 `shell.overlay`（list 类型、无冲突）做**右侧 fixed 悬浮面板**（360px，可折叠）；
- 默认**折叠成小条**（「📐 数学建模项目」+「展开」按钮），点击展开完整看板；
- 不占用官方 `details` 右侧详情列（它是 single，被系统占用，第三方注册会崩溃）。

## 二、当前功能清单

### 看板内容（展开后）
- **Header**：项目标题、进行中/已交付、竞赛/届次/当前阶段标签、收起按钮；
- **NEXT**：下一动作提示；
- **阶段进度**：01 建模 · 题目分析与模型设计 / 02 编程 · 代码求解与结果验证 / 03 论文 · 撰写与排版交付（时间线 + 任务完成度）；
- **五门禁**：M1 建模终检 / P1 最小可运行 / P2 编程终检 / W1 证据大纲 / W2 论文终检（PASS/FAIL/BLOCK/待）；
- **刷新数据** 按钮。

### 设置开关
- 设置 → 插件 → **数学建模** tab：进度看板开关（on/off，持久化）；
- 通过 `mm.setEnabled` RPC 读/写 DSH settings。

### 会话门控（仅数学建模预设会话显示）
- client 用 `ctx.sessions.list.getSnapshot()` 读当前活跃会话的 `agentPreset`；
- 仅 `math-modeling-agent` 显示，其余返回 null（不渲染，不误占空间）。

## 三、技术要点（已踩坑）

| 坑 | 解决 |
|---|---|
| 官方 `details` 右侧列被 system 占用 | 改用 `shell.overlay`（list、无冲突；而非 single 的 details，后者多 occupant 注册直接崩溃） |
| 组合包 RPC（重启持久） | 用 `ctx.connection.rpc.handle`/`.call`（区别于动态插件 `harness.handle`/`host.call`） |
| `shell.overlay` 是 root scope 无 sessionId | 用 `ctx.sessions.list.getSnapshot().current` 拿当前会话做门控 |
| overlay 容器 click-through | 条目自带 `pointer-events:auto`，fixed 定位右侧 |
| 会话门控误放行 | 严格 `preset !== TARGET → return null`（不区分数据是否已加载） |
| 展开占满屏幕 | `.mm-overlay` fixed 360px 宽 + max-height:calc(100vh-24px)，可折叠 |

## 四、安装方式（一次安装持久生效）

```bash
# 完全关闭 DSH 后执行
$env:DSH_HOME = "C:\Users\86198\AppData\Roaming\dsh-desktop\dsh-home"
dsh plugin --profile web add "C:\Users\86198\dsh-math-modeling-ui" -w
```

## 五、局限与改进方向（待做）

| 方向 | 现状 | 可改进 |
|---|---|---|
| 交付物检查显示 | 无 | 在 UI 显示 `mm_check_deliverables` 结果 |
| 门禁回执详情 | 只显示状态 | 展开显示回执证据数/发现数 |
| 返工历史 | 无 | 从 ledger 显示 FAIL→返工次数 |
| 漂移状态 | 无 | 显示"需复验"警告 |
| 任务列表 | 折叠仅显示计数 | 展开显示任务条（✅/⬜） |
