// 数学建模 Workbench — 标准宿主插件（agent preset 持久化版）
//
// 零依赖 ESM 模块：通过 ctx 服务注册 mm_* 工具与系统提示段。
// 与动态插件版的差异：
//   1. 无 harness builtin —— 用 ctx.get('tools').register(手写 definition) 注册工具；
//   2. 无 harness.handle / Client —— 预设是宿主侧，GUI 面板不随预设持久化；
//   3. 无进程级共享状态 —— 每次调用从 <PROJECT_ROOT>/.math-modeling/state.json 加载，
//      操作后写回，多会话加入同一预设时互不串扰；
//   4. PROJECT_ROOT 默认解析自调用者的会话 cwd（agents.currentInitiator()）。

const META_DIR = '.math-modeling'
const STATE_NAME = 'state.json'
// 内置 SKILL_ROOT：从插件自身位置推导（插件位于 <预设>/plugins/ 下，知识库位于 <预设>/skills/math-modeling）
// 推导失败时为 ''（不携带任何本机路径），此时 mm_project_init 会要求显式传入 skillRoot。
const __here = typeof import.meta !== 'undefined' && import.meta.url ? import.meta.url : ''
let __bundleRoot = ''
try {
  if (__here) {
    const sk = new URL('../skills/math-modeling', __here)
    let p = sk.pathname
    if (p.startsWith('/') && /^\/[A-Za-z]:/.test(p)) p = p.slice(1)
    __bundleRoot = decodeURIComponent(p)
  }
} catch (e) { __bundleRoot = '' }
const BUNDLED_SKILL_ROOT = __bundleRoot || ''

const PHASES = {
  modeling: { key: 'modeling', label: '建模手', skillMd: 'references/roles/建模手/SKILL.md', gates: ['M1'], deliverables: ['题目分析报告.md', '术语表格.md'] },
  programming: { key: 'programming', label: '编程手', skillMd: 'references/roles/编程手/SKILL.md', gates: ['P1', 'P2'], deliverables: ['可运行代码(.py/.m)', '结果表格(.csv/.xlsx)', '候选图(三类各≥3张,覆盖全部子问题)', 'results/复现清单.json'] },
  paper: { key: 'paper', label: '论文手', skillMd: 'references/roles/论文手/SKILL.md', gates: ['W1', 'W2'], deliverables: ['完整论文.docx', '正式图≥8幅且覆盖全部子问题', 'LaTeX(可选): 完整论文-LaTeX/ + PDF + build.json'] },
}
const GATES = {
  M1: { phase: 'modeling', title: '建模终检', after: '进入编程或交付建模产物', checks: ['子问题覆盖', '假设依据', '公式与符号', '单位与约束', '模型数量', '可实现性', '验证方案', '文献可追溯性'] },
  P1: { phase: 'programming', title: '最小可运行结果', after: '开始全量计算、正式出图和参数扫描', checks: ['从 PROJECT_ROOT 执行最小命令', '真实输入或等价小实例', '退出码', '关键结果', '单位/范围/约束', '模型合同'] },
  P2: { phase: 'programming', title: '编程终检', after: '进入论文阶段或交付编程产物', checks: ['独立复现', '输入哈希', '随机种子', '关键数值', '边界与量纲', '各子问题图表数量与语义', '复现清单', '文件完整性'] },
  W1: { phase: 'paper', title: '证据大纲', after: '开始长篇正文和排版', checks: ['每个子问题结论有公式/结果表位置/图文件/代码输出/已核验文献支撑'] },
  W2: { phase: 'paper', title: '论文终检', after: '按用户要求交付论文并宣称完整完成', checks: ['当届规则', '主张—证据', '数值与单位', '图表引用', '文献', '实际渲染', 'Word/LaTeX 一致性(双格式时)'] },
}
const GATE_ORDER = ['M1', 'P1', 'P2', 'W1', 'W2']
const GATE_PREREQ = { M1: null, P1: 'M1', P2: 'P1', W1: 'P2', W2: 'W1' }
const RECEIPT_FIELDS = ['scope', 'inputSnapshot', 'status', 'evidence', 'findings', 'rework']

// 每阶段的标准任务清单（源自角色 SKILL.md 的执行顺序），用于 mm_todo 看板
const PHASE_TASKS = {
  modeling: [
    '读取题目与全部附件，建立问题/目标/约束/数据字段/输出要求清单',
    '数据驱动检查缺失、异常、量纲、时间与空间范围',
    '写出每个子问题应回答的结论类型，再评估候选模型',
    '每子问题选定 ≤2 个独立模型体系（含公式/参数/输入输出/验证方式/风险）',
    '用 paper_search 双引擎检索理论依据并核验文献可追溯性',
    '写入 题目分析报告.md 与 术语表格.md，按 质检清单.md 作者自检',
    '派发独立只读 Subagent 执行 M1 建模终检并记录回执',
  ],
  programming: [
    '选择 Python/MATLAB（按模型依赖与环境），动态检查依赖（check_env.py）',
    '实现数据读取/预处理/核心求解链，用真实输入跑通最小命令',
    '派发独立只读 Subagent 执行 P1 最小可运行结果门禁',
    '将子问题规范为 q1..qN，读取 tools/figure/SKILL.md 做数据剖析与图表契约',
    '用 plot_style 生成三类候选图（raw/process/result 各≥3、覆盖全部子问题）',
    'export_figure 布局门禁 + figure_audit.py --strict 机械审图 + 实际读图',
    '生成 results/复现清单.json（种子/输入哈希/版本/参数/唯一命令）',
    '按 质检清单.md 作者自检，派发独立只读 Subagent 执行 P2 编程终检',
  ],
  paper: [
    '核对输入：题目/建模交付物/代码/结果表/三类图/文献，缺失则回退补齐',
    '核验当届官方规则与模板；显式要求 LaTeX 时先 latex_paper.py init/doctor',
    '建立 Claim-Evidence 映射与论文大纲（每子问题有公式/表/图/文献证据）',
    '派发独立只读 Subagent 执行 W1 证据大纲门禁',
    '按官方结构写完整正文（默认至少 8 幅正式图，全部子问题覆盖）',
    '用 docx 工具生成 完整论文.docx（原生 OMML 公式）；LaTeX 可选用官方模板真实编译',
    '运行确定性门禁（paper_format validate / office validate / equations 校验等）',
    '按 自审框架.md 自审，派发独立只读 Subagent 执行 W2 论文终检',
  ],
}

export const name = 'math-modeling-workbench'

export async function apply(ctx) {
  const fs = ctx.get('fs')
  const shell = ctx.get('shell')
  const agents = ctx.get('agents')
  const sandboxPolicy = ctx.get('sandboxPolicy')
  const systemPrompt = ctx.get('systemPrompt')
  const settings = ctx.get('settings')
  if (!fs || !shell) return

  const join = (...parts) => parts.filter(Boolean).join('/')
  const nowIso = () => new Date().toISOString()
  const msg = (e) => String((e && e.message) || e)
  let wsRoot = null

  async function workspaceRoot() {
    if (wsRoot) return wsRoot
    try {
      if (agents) {
        const initiator = agents.currentInitiator()
        const cwd = initiator && initiator.session && initiator.session.header && initiator.session.header.cwd
        if (cwd && String(cwd).trim()) { wsRoot = String(cwd); return wsRoot }
      }
    } catch (e) { /* fallthrough */ }
    try {
      if (sandboxPolicy && sandboxPolicy.workspaceRoot) { wsRoot = sandboxPolicy.workspaceRoot; return wsRoot }
    } catch (e) { /* fallthrough */ }
    try {
      const t = await fs.resolve('.')
      wsRoot = fs.processPath(t)
    } catch (e) { wsRoot = null }
    return wsRoot
  }
  async function targetOf(p) { return fs.resolve(p) }
  async function pathOf(p) { return fs.processPath(await targetOf(p)) }
  async function exists(p) { try { const t = await targetOf(p); const info = await fs.stat(t); return info || null } catch (e) { return null } }
  async function readFile(p) { return fs.readText(await targetOf(p)) }
  async function writeFile(p, content) {
    const t = await targetOf(p)
    const root = state.project && state.project.projectRoot ? state.project.projectRoot : (await workspaceRoot())
    if (!root) return fs.writeText(t, content)
    return fs.writeText(t, content, undefined, undefined, { mode: 'workspace-write', workspaceRoot: root })
  }
  async function listDir(p) { return fs.listDir(await targetOf(p)) }
  async function runShell(command, cwd, timeoutMs) {
    const req = { command: String(command), description: 'math-modeling 插件内部命令' }
    if (cwd) req.cwd = String(cwd)
    if (timeoutMs) req.timeoutMs = timeoutMs
    return shell.run(shell.resolve(req))
  }

  // 会话隔离状态：每次调用从磁盘加载
  let state = freshState()
  function freshState() {
    return {
      version: 6,
      initializedAt: null,
      project: null,
      currentPhase: 'modeling',
      phases: {
        modeling: { enteredAt: null, skillMdRead: false, skillMdPath: null, gateM1: null, tasks: [] },
        programming: { enteredAt: null, skillMdRead: false, skillMdPath: null, gateP1: null, gateP2: null, tasks: [] },
        paper: { enteredAt: null, skillMdRead: false, skillMdPath: null, gateW1: null, gateW2: null, tasks: [] },
      },
      deliverables: null,
      completed: false,
      completedAt: null,
      ledger: [],
    }
  }
  async function loadState(projectRoot) {
    const info = await exists(join(projectRoot, META_DIR, STATE_NAME))
    if (!info) { state = freshState(); return state }
    try { state = JSON.parse(await readFile(join(projectRoot, META_DIR, STATE_NAME))) } catch (e) { state = freshState() }
    if (!state || typeof state !== 'object') state = freshState()
    // 迁移：旧版本状态补齐 tasks 字段
    if (state.version < 6) {
      state.version = 6
      for (const k of Object.keys(PHASES)) {
        if (!state.phases[k]) state.phases[k] = {}
        if (!Array.isArray(state.phases[k].tasks)) state.phases[k].tasks = []
      }
    }
    return state
  }
  async function ensureMetaDir() {
    const root = state.project && state.project.projectRoot ? state.project.projectRoot : await workspaceRoot()
    if (!root) return null
    const dir = join(root, META_DIR)
    const info = await exists(dir)
    if (info) return dir
    try { await runShell("New-Item -ItemType Directory -Force -Path '" + String(dir).replace(/'/g, "''") + "' | Out-Null") } catch (e) { /* 交给 writeText 尝试 */ }
    return dir
  }
  async function persist() {
    const dir = await ensureMetaDir()
    if (!dir) return
    await writeFile(join(dir, STATE_NAME), JSON.stringify(state, null, 2))
  }
  function gateNode(g) { return state.phases[GATES[g].phase]['gate' + g] }
  function gateStatus(g) { const n = gateNode(g); return n ? n.status : 'pending' }
  function log(event, detail) {
    state.ledger.push({ at: nowIso(), event, detail: detail || null })
    if (state.ledger.length > 500) state.ledger = state.ledger.slice(-500)
  }

  function summaryOf() {
    const project = state.project
    const gates = {}
    for (const g of GATE_ORDER) {
      const n = gateNode(g)
      gates[g] = {
        title: GATES[g].title,
        phase: GATES[g].phase,
        status: n ? n.status : 'pending',
        at: n ? n.at : null,
        receiptSummary: n && n.receipt ? { status: n.receipt.status, findingsCount: (n.receipt.findings || []).length, evidenceCount: (n.receipt.evidence || []).length } : null,
      }
    }
    const phases = {}
    for (const k of Object.keys(PHASES)) {
      const p = state.phases[k]
      phases[k] = { label: PHASES[k].label, entered: !!p.enteredAt, skillMdRead: !!p.skillMdRead, gates: PHASES[k].gates.map((g) => gates[g].status) }
    }
    // 进度看板：阶段步进 + 任务完成度
    const steps = ['modeling', 'programming', 'paper'].map((k) => {
      const done = state.phases[k].enteredAt && PHASES[k].gates.every((g) => gateStatus(g) === 'pass')
      return { key: k, label: PHASES[k].label, status: done ? 'done' : (state.currentPhase === k ? 'current' : (state.phases[k].enteredAt ? 'inprogress' : 'pending')) }
    })
    const progress = {
      steps,
      currentPhase: state.currentPhase,
      tasks: {},
    }
    for (const k of Object.keys(PHASES)) {
      const list = state.phases[k].tasks && state.phases[k].tasks.length ? state.phases[k].tasks : PHASE_TASKS[k].map((text) => ({ text, done: false }))
      const doneCount = list.filter((t) => t.done).length
      progress.tasks[k] = { total: list.length, done: doneCount, pct: list.length ? Math.round(doneCount / list.length * 100) : 0 }
    }
    // 下一动作提示
    const cur = state.currentPhase
    const curGate = PHASES[cur].gates.find((g) => gateStatus(g) !== 'pass')
    progress.nextAction = curGate ? ('进入 ' + curGate + ' 门禁：mm_gate gate=' + curGate + ' mode=prepare（' + GATES[curGate].title + '）') : ('完成 ' + PHASES[cur].label + ' 任务后，用 mm_check_deliverables 检查交付物')
    if (cur === 'programming') progress.nextAction = PHASES[cur].gates.every((g) => gateStatus(g) === 'pass') ? '进入论文手：mm_phase_enter phase=paper' : progress.nextAction
    if (cur === 'paper' && PHASES[cur].gates.every((g) => gateStatus(g) === 'pass')) progress.nextAction = '运行 mm_complete 完成判定'
    return {
      initialized: !!project,
      currentPhase: state.currentPhase,
      project: project ? {
        title: project.title, competition: project.competition, edition: project.edition,
        skillRoot: project.skillRoot, projectRoot: project.projectRoot,
        subproblems: project.subproblems, paperFormat: project.paperFormat, optionalCollab: project.optionalCollab,
      } : null,
      phases,
      gates,
      progress,
      deliverables: state.deliverables,
      completed: state.completed,
      completedAt: state.completedAt,
      ledgerTail: state.ledger.slice(-8).map((e) => ({ at: e.at, event: e.event, detail: e.detail && typeof e.detail === 'object' ? JSON.stringify(e.detail) : String(e.detail || '') })),
    }
  }

  async function resolveProjectRoot(args) {
    const root = await workspaceRoot()
    return String(args && args.projectRoot ? args.projectRoot : (root || '')).trim()
  }

  async function initProject(args) {
    // SKILL_ROOT 解析：显式参数 > 内置知识库（随预设持久化）> 报错（无本机路径回退）
    let skillRootArg = ''
    if (args.skillRoot && String(args.skillRoot).trim()) {
      skillRootArg = String(args.skillRoot).trim()
    } else if (BUNDLED_SKILL_ROOT) {
      const bundledOk = await exists(join(BUNDLED_SKILL_ROOT, 'SKILL.md'))
      skillRootArg = bundledOk ? BUNDLED_SKILL_ROOT : ''
    }
    if (!skillRootArg) return { ok: false, error: 'SKILL_ROOT 无法解析：请在 mm_project_init 显式传入 skillRoot（内置知识库缺失）' }
    const projectRootArg = await resolveProjectRoot(args)
    if (!projectRootArg) return { ok: false, error: 'projectRoot 为空，且无法探测工作区' }
    let skillRoot, projectRoot
    try { skillRoot = await pathOf(skillRootArg) } catch (e) { return { ok: false, error: 'skillRoot 无法解析: ' + msg(e) } }
    try { projectRoot = await pathOf(projectRootArg) } catch (e) { return { ok: false, error: 'projectRoot 无法解析: ' + msg(e) } }
    const skillMd = await exists(join(skillRoot, 'SKILL.md'))
    if (!skillMd) return { ok: false, error: 'SKILL_ROOT 缺少 SKILL.md: ' + skillRoot }
    const prInfo = await exists(projectRoot)
    if (!prInfo) return { ok: false, error: 'PROJECT_ROOT 不存在: ' + projectRoot + '（请先创建目录）' }
    if (prInfo.type !== 'directory') return { ok: false, error: 'PROJECT_ROOT 不是目录: ' + projectRoot }
    const skillT = await targetOf(skillRoot)
    const projT = await targetOf(projectRoot)
    if (fs.contains(skillT, projT) || fs.contains(projT, skillT)) {
      return { ok: false, error: 'SKILL_ROOT 与 PROJECT_ROOT 必须不同且互不包含；禁止把项目建在 skill 仓库内' }
    }
    await loadState(projectRoot)
    if (state.project && state.project.projectRoot) {
      if (args.title) state.project.title = String(args.title).trim()
      if (args.competition) state.project.competition = String(args.competition).trim()
      if (args.edition) state.project.edition = String(args.edition).trim()
      if (args.paperFormat === 'word' || args.paperFormat === 'word+latex') state.project.paperFormat = args.paperFormat
      log('resume', { skillRoot: state.project.skillRoot, projectRoot: state.project.projectRoot })
      await persist()
      return Object.assign({ ok: true, resumed: true, notice: '已续接既有项目（不覆盖既有门禁/交付物状态）。当前阶段=' + state.currentPhase + '。' }, summaryOf())
    }
    let subproblems = []
    if (Array.isArray(args.subproblems)) subproblems = args.subproblems.map(String).filter(Boolean)
    else if (typeof args.subproblems === 'string') subproblems = args.subproblems.split(/[,，\s]+/).filter(Boolean)
    if (subproblems.length === 0) subproblems = ['q1']
    const collab = { rulesCheck: false, attachmentInventory: false, literature: false, prototype: false, experiments: false, bilingual: false, terminology: false }
    if (Array.isArray(args.optionalCollab)) for (const k of args.optionalCollab) if (k in collab) collab[k] = true
    else if (typeof args.optionalCollab === 'string') for (const k of args.optionalCollab.split(/[,，\s]+/).filter(Boolean)) if (k in collab) collab[k] = true
    else if (args.optionalCollab && typeof args.optionalCollab === 'object') Object.assign(collab, args.optionalCollab)
    const paperFormat = args.paperFormat === 'word+latex' ? 'word+latex' : 'word'
    state = freshState()
    state.initializedAt = nowIso()
    state.project = {
      title: String(args.title || '数学建模项目').trim(),
      competition: String(args.competition || 'CUMCM').trim(),
      edition: String(args.edition || '').trim(),
      skillRoot, projectRoot, subproblems, paperFormat, optionalCollab: collab,
    }
    state.currentPhase = 'modeling'
    log('init', { skillRoot, projectRoot, competition: state.project.competition, edition: state.project.edition, subproblems, paperFormat })
    await persist()
    return Object.assign({ ok: true, notice: '项目已初始化：已激活 math-modeling Skill 协议。当前阶段=建模手；目标竞赛=' + state.project.competition + (state.project.edition ? ' ' + state.project.edition : '') + '；计划读取的入口=references/roles/建模手/SKILL.md 与 tools/*/SKILL.md；官方规则未核验项请在首轮进度更新中标为待核验。\n\n💡 如需进度看板，请粘贴：读取 ui-dock/加载UI看板提示词.md 中的 prompt 并执行' }, summaryOf())
  }

  async function enterPhase(phase) {
    const def = PHASES[phase]
    if (!def) return { ok: false, error: '未知阶段: ' + phase + '（可选 modeling/programming/paper）' }
    if (!state.project) return { ok: false, error: '项目未初始化，请先运行 mm_project_init' }
    if (phase === 'programming' && gateStatus('M1') !== 'pass') return { ok: false, error: '进入编程手前必须通过 M1 建模终检（当前 ' + gateStatus('M1') + '）' }
    if (phase === 'paper' && gateStatus('P2') !== 'pass') return { ok: false, error: '进入论文手前必须通过 P2 编程终检（当前 ' + gateStatus('P2') + '）' }
    state.currentPhase = phase
    if (!state.phases[phase].enteredAt) { state.phases[phase].enteredAt = nowIso(); log('phase_enter', { phase, label: def.label }) }
    // 注入标准任务清单（保留已勾选状态）
    if (!state.phases[phase].tasks || state.phases[phase].tasks.length === 0) {
      state.phases[phase].tasks = PHASE_TASKS[phase].map((text) => ({ text, done: false }))
    }
    const p = join(state.project.skillRoot, def.skillMd)
    let content = null, readError = null
    try {
      content = await readFile(p)
      state.phases[phase].skillMdRead = true
      state.phases[phase].skillMdPath = p
      log('skill_md_read', { phase, path: p })
    } catch (e) { readError = msg(e) }
    await persist()
    return {
      ok: true, phase, label: def.label, skillMdPath: p, skillMdRead: !readError, skillMd: content, readError,
      tasks: state.phases[phase].tasks,
      gates: def.gates.map((g) => ({ gate: g, status: gateStatus(g), title: GATES[g].title })),
      deliverables: def.deliverables,
    }
  }

  async function skillRead(pathArg) {
    if (!state.project) return { ok: false, error: '项目未初始化' }
    const rel = String(pathArg || '').trim()
    if (!rel) return { ok: false, error: 'path 不能为空' }
    const skillRoot = state.project.skillRoot
    const full = /^[A-Za-z]:[\\/]|^[\\/]/.test(rel) ? rel : join(skillRoot, rel)
    const rootT = await targetOf(skillRoot)
    const fileT = await targetOf(full)
    if (!fs.contains(rootT, fileT)) return { ok: false, error: 'path 必须在 SKILL_ROOT 内: ' + rel }
    let content
    try { content = await readFile(full) } catch (e) { return { ok: false, error: '读取失败: ' + msg(e) } }
    log('skill_read', { path: rel })
    await persist()
    return { ok: true, path: rel, resolved: fs.processPath(fileT), content }
  }

  async function snapshotArtifacts(phase) {
    const root = state.project.projectRoot
    const out = {}
    const snap = async (p) => { try { const info = await exists(p); if (info) out[p] = info.version || String(info.size || 0) } catch (e) {} }
    if (phase === 'modeling') {
      await snap(join(root, '题目分析报告.md')); await snap(join(root, '术语表格.md'))
    } else if (phase === 'programming') {
      try { const es = await listDir(root); for (const e of es) if (e.type === 'file' && !e.name.startsWith('.')) out[join(root, e.name)] = e.version || String(e.size || 0) } catch (e) {}
      for (const sub of ['results', 'figures']) {
        try { const es = await listDir(join(root, sub)); for (const e of es) if (e.type === 'file') out[join(root, sub, e.name)] = e.version || String(e.size || 0) } catch (e) {}
      }
    } else {
      for (const f of ['完整论文.docx', '完整论文.conversion.json', '完整论文-LaTeX/main.tex', '完整论文.pdf', '完整论文.build.json']) await snap(join(root, f))
    }
    return out
  }

  async function gatePrepare(gate, evidence) {
    const def = GATES[gate]
    if (!def) return { ok: false, error: '未知门禁: ' + gate + '（可选 ' + GATE_ORDER.join('/') + '）' }
    if (!state.project) return { ok: false, error: '项目未初始化' }
    const prereq = GATE_PREREQ[gate]
    if (prereq && gateStatus(prereq) !== 'pass') return { ok: false, error: '门禁 ' + gate + ' 的前置门禁 ' + prereq + ' 未通过（当前 ' + gateStatus(prereq) + '）' }
    const ev = evidence && typeof evidence === 'object' ? evidence : {}
    const arr = (v) => Array.isArray(v) ? v : (v ? [String(v)] : [])
    const snapshot = {
      at: nowIso(), projectRoot: state.project.projectRoot, subproblems: state.project.subproblems,
      inputs: arr(ev.inputs), deliverablePaths: arr(ev.deliverablePaths), commands: arr(ev.commands),
      hashes: ev.hashes || null, notes: String(ev.notes || ''),
    }
    const brief = [
      '# ' + def.title + '（' + gate + '）质检简报 — 数学建模 Workbench',
      '',
      '目标竞赛: ' + state.project.competition + (state.project.edition ? ' ' + state.project.edition : '') + ' | 项目: ' + state.project.title,
      'PROJECT_ROOT: ' + state.project.projectRoot,
      '',
      '## 你的角色',
      '你是本门禁的独立只读质检员。你未参与被审产物的编写或修正；默认只读并返回证据；禁止修改任何交付物。',
      '',
      '## 验收条款',
    ].concat(def.checks.map((c, i) => (i + 1) + '. ' + c), [
      '',
      '## 输入快照',
      '- 题目与只读附件路径: ' + (snapshot.inputs.length ? snapshot.inputs.join('；') : '（未提供）'),
      '- 权威产物路径: ' + (snapshot.deliverablePaths.length ? snapshot.deliverablePaths.join('；') : '（未提供）'),
      '- 关键命令: ' + (snapshot.commands.length ? snapshot.commands.join('；') : '（未提供）'),
      snapshot.hashes ? '- SHA-256/版本: ' + JSON.stringify(snapshot.hashes) : '',
      snapshot.notes ? '- 备注: ' + snapshot.notes : '',
      '',
      '## 固定回执（必须按此 JSON 结构返回）',
      '{ "scope": "本次检查的门禁与未覆盖内容", "inputSnapshot": "路径/版本/SHA-256", "status": "PASS|FAIL|BLOCKED", "evidence": ["来源URL/命令/退出码/文件位置/页码/关键数值"], "findings": [{ "level": "P0|P1|P2", "text": "..." }], "rework": "应回到的角色、必须修正项、复验入口" }',
      '',
      '## 判定规则',
      '存在未解决 P0 或 P1 只能返回 FAIL；缺少依赖/权限/输入或可验证证据返回 BLOCKED。',
    ]).filter(Boolean).join('\n')
    return {
      ok: true, gate, title: def.title, phase: def.phase, after: def.after, brief, snapshot,
      recordInstruction: '用主 Agent 的 subagent 工具派发独立只读质检，随后调用 mm_gate mode=record 记录回执（gate=' + gate + '，receipt 为 Subagent 返回的固定回执 JSON）',
    }
  }

  async function gateRecord(gate, receipt) {
    const def = GATES[gate]
    if (!def) return { ok: false, error: '未知门禁: ' + gate }
    if (!state.project) return { ok: false, error: '项目未初始化' }
    if (!receipt || typeof receipt !== 'object' || Array.isArray(receipt)) return { ok: false, error: 'receipt 必须是 JSON 对象' }
    const status = String(receipt.status || '').toUpperCase()
    if (!['PASS', 'FAIL', 'BLOCKED'].includes(status)) return { ok: false, error: 'receipt.status 必须是 PASS|FAIL|BLOCKED' }
    const missing = RECEIPT_FIELDS.filter((f) => !(f in receipt))
    if (missing.length) return { ok: false, error: '回执缺少字段: ' + missing.join(', ') }
    if (!Array.isArray(receipt.evidence)) return { ok: false, error: 'receipt.evidence 必须是数组' }
    if (!Array.isArray(receipt.findings)) return { ok: false, error: 'receipt.findings 必须是数组' }
    if (status === 'PASS' && receipt.evidence.length === 0) return { ok: false, error: 'PASS 回执必须附带证据（evidence 不能为空）' }
    const node = gateNode(gate)
    const prevStatus = node ? node.status : 'pending'
    const record = {
      status: status.toLowerCase(),
      at: nowIso(),
      artifactVersions: await snapshotArtifacts(def.phase),
      receipt: {
        scope: String(receipt.scope || ''),
        inputSnapshot: String(receipt.inputSnapshot || ''),
        status,
        evidence: receipt.evidence.map(String),
        findings: receipt.findings.map((f) => typeof f === 'string' ? { level: 'P2', text: f } : { level: String((f && f.level) || 'P2').toUpperCase(), text: String((f && f.text) || '') }),
        rework: String(receipt.rework || ''),
      },
    }
    state.phases[def.phase]['gate' + gate] = record
    log('gate_record', { gate, status: record.status, prevStatus })
    await persist()
    return { ok: true, gate, status: record.status, at: record.at, prevStatus, phase: def.phase, after: def.after }
  }

  async function checkFigures() {
    const root = state.project.projectRoot
    const items = []
    const dirs = [root]
    const fi = await exists(join(root, 'figures'))
    if (fi) dirs.push(join(root, 'figures'))
    const files = []
    for (const d of dirs) {
      try { const es = await listDir(d); for (const e of es) if (e.type === 'file') files.push({ dir: d, name: e.name }) } catch (e) {}
    }
    const figFiles = files.filter((f) => /\.(svg|png)$/i.test(f.name) && !/^\./.test(f.name))
    const byCat = { raw: new Set(), process: new Set(), result: new Set() }
    const bySub = {}
    for (const f of figFiles) {
      const m = /^(raw|process|result)_(q\d+)([._-].*)?\.(svg|png)$/i.exec(f.name)
      if (!m) continue
      const cat = m[1].toLowerCase(), sub = m[2].toLowerCase()
      const stem = f.name.replace(/\.(svg|png)$/i, '')
      byCat[cat].add(sub + ':' + stem)
      bySub[sub] = bySub[sub] || { raw: new Set(), process: new Set(), result: new Set() }
      bySub[sub][cat].add(stem)
    }
    const pass = (name, note) => items.push({ name, ok: true, note: note || '' })
    const fail = (name, note) => items.push({ name, ok: false, note: note || '' })
    for (const cat of ['raw', 'process', 'result']) {
      const n = byCat[cat].size
      n >= 3 ? pass(cat + '_q* 候选图', n + ' 张') : fail(cat + '_q* 候选图', n + ' 张（需≥3）')
    }
    const total = byCat.raw.size + byCat.process.size + byCat.result.size
    total >= 9 ? pass('候选图合计', total + ' 张') : fail('候选图合计', total + ' 张（需≥9）')
    if (state.project.subproblems.length) {
      for (const sp of state.project.subproblems) {
        const s = bySub[String(sp).toLowerCase()]
        const cov = s ? ['raw', 'process', 'result'].filter((c) => s[c].size > 0) : []
        cov.length === 3 ? pass('子问题 ' + sp + ' 图覆盖', '三类齐全') : fail('子问题 ' + sp + ' 图覆盖', '缺少: ' + (['raw', 'process', 'result'].filter((c) => !cov.includes(c)).join('/') || '该子问题无图'))
      }
    }
    if (state.currentPhase === 'paper') {
      const formal = byCat.result.size
      formal >= 8 ? pass('正式图基线', formal + ' 张 result 图') : fail('正式图基线', formal + ' 张（默认需≥8）')
    }
    return { items, total }
  }

  async function checkDeliverables(phaseArg) {
    if (!state.project) return { ok: false, error: '项目未初始化' }
    const phase = phaseArg || state.currentPhase
    const def = PHASES[phase]
    if (!def) return { ok: false, error: '未知阶段: ' + phase }
    const root = state.project.projectRoot
    const items = []
    const pass = (name, note) => items.push({ name, ok: true, note: note || '' })
    const fail = (name, note, blocked) => items.push({ name, ok: false, blocked: !!blocked, note: note || '' })
    if (phase === 'modeling') {
      for (const f of ['题目分析报告.md', '术语表格.md']) {
        const info = await exists(join(root, f))
        info ? pass(f, '已存在') : fail(f, '缺失')
      }
    } else if (phase === 'programming') {
      let codeFiles = []
      try { codeFiles = (await listDir(root)).filter((e) => e.type === 'file' && /\.(py|m)$/i.test(e.name) && !e.name.startsWith('.')) } catch (e) { fail('代码文件', '无法列出目录: ' + msg(e), true) }
      if (codeFiles.length) pass('可运行代码', codeFiles.map((e) => e.name).join(', '))
      else if (!items.some((i) => i.name === '代码文件')) fail('可运行代码', 'PROJECT_ROOT 下未发现 .py/.m 文件')
      let tables = []
      try {
        tables = (await listDir(root)).filter((e) => e.type === 'file' && /\.(csv|xlsx)$/i.test(e.name))
        const rinfo = await exists(join(root, 'results'))
        if (rinfo) tables = tables.concat((await listDir(join(root, 'results'))).filter((e) => e.type === 'file' && /\.(csv|xlsx)$/i.test(e.name)))
      } catch (e) {}
      tables.length ? pass('结果表格', tables.map((e) => e.name).slice(0, 12).join(', ')) : fail('结果表格', '未发现 .csv/.xlsx')
      const fig = await checkFigures()
      items.push.apply(items, fig.items)
      const rj = await exists(join(root, 'results', '复现清单.json'))
      if (rj) {
        try {
          const data = JSON.parse(await readFile(join(root, 'results', '复现清单.json')))
          const keys = Object.keys(data)
          const has = (re) => keys.some((k) => re.test(k))
          const checks = [['随机种子', /seed|随机种子|random/i], ['输入 SHA-256', /sha|hash|哈希/i], ['运行时/依赖版本', /version|版本|runtime|依赖/i], ['关键参数', /param|参数/i], ['复现命令', /command|命令|repro/i]]
          const missing = checks.filter((c) => !has(c[1])).map((c) => c[0])
          missing.length ? fail('复现清单.json', '缺少字段: ' + missing.join('、')) : pass('复现清单.json', '必备字段齐全')
        } catch (e) { fail('复现清单.json', '解析失败: ' + msg(e)) }
      } else fail('复现清单.json', 'results/复现清单.json 缺失')
      const py = await probePython()
      py.ok ? pass('Python 环境', py.version) : fail('Python 环境', py.error || '未检测到 Python（技能脚本依赖）', true)
    } else if (phase === 'paper') {
      const doc = await exists(join(root, '完整论文.docx'))
      doc ? pass('完整论文.docx', '已存在') : fail('完整论文.docx', '缺失')
      const fig = await checkFigures()
      items.push.apply(items, fig.items)
      if (state.project.paperFormat === 'word+latex') {
        const main = await exists(join(root, '完整论文-LaTeX', 'main.tex'))
        main ? pass('LaTeX 源码项目', 'main.tex 存在') : fail('LaTeX 源码项目', '完整论文-LaTeX/main.tex 缺失')
        const pdf = await exists(join(root, '完整论文.pdf'))
        pdf ? pass('完整论文.pdf', '已存在') : fail('完整论文.pdf', '缺失（需实际编译）')
        const bj = await exists(join(root, '完整论文.build.json'))
        bj ? pass('完整论文.build.json', '已存在') : fail('完整论文.build.json', '缺失（哈希绑定清单）')
      }
    }
    const overall = items.some((i) => i.blocked) ? 'blocked' : (items.every((i) => i.ok) ? 'ok' : 'missing')
    state.deliverables = { at: nowIso(), phase, overall, items }
    await persist()
    return { ok: true, phase, overall, items }
  }

  async function probePython() {
    try {
      const res = await runShell('python --version', undefined, 15000)
      if (res.exitCode === 0) return { ok: true, version: String(res.stdout || res.stderr || '').trim() }
      return { ok: false, error: 'python --version 退出码 ' + res.exitCode }
    } catch (e) { return { ok: false, error: msg(e) } }
  }

  async function completeProject() {
    if (!state.project) return { ok: false, error: '项目未初始化' }
    const gateResults = {}
    for (const g of GATE_ORDER) {
      const n = gateNode(g)
      gateResults[g] = { status: n ? n.status : 'pending', at: n ? n.at : null }
    }
    const failedGates = GATE_ORDER.filter((g) => gateResults[g].status !== 'pass')
    const checks = {}
    for (const ph of ['modeling', 'programming', 'paper']) {
      const r = await checkDeliverables(ph)
      checks[ph] = r.overall
    }
    const drift = []
    for (const g of GATE_ORDER) {
      const n = gateNode(g)
      if (n && n.status === 'pass' && n.artifactVersions) {
        const cur = await snapshotArtifacts(GATES[g].phase)
        for (const p of Object.keys(n.artifactVersions)) {
          if (p in cur && cur[p] !== n.artifactVersions[p]) drift.push({ gate: g, path: p })
        }
      }
    }
    const blockers = []
    if (failedGates.length) blockers.push('未通过门禁: ' + failedGates.join(', '))
    for (const ph of Object.keys(checks)) if (checks[ph] !== 'ok') blockers.push('交付物未齐全(' + PHASES[ph].label + '): ' + checks[ph])
    if (drift.length) blockers.push('门禁通过后产物发生变化（需复验）: ' + drift.map((d) => d.gate + '@' + d.path).join('; '))
    const done = blockers.length === 0
    if (done) { state.completed = true; state.completedAt = nowIso(); log('complete', { gateResults }) }
    await persist()
    const reads = state.ledger.filter((e) => e.event === 'skill_md_read' || e.event === 'skill_read').map((e) => e.detail)
    const report = [
      '# 数学建模项目完成报告',
      '',
      '- 项目: ' + state.project.title + ' (' + state.project.competition + (state.project.edition ? ' ' + state.project.edition : '') + ')',
      '- SKILL_ROOT: ' + state.project.skillRoot,
      '- PROJECT_ROOT: ' + state.project.projectRoot,
      '- 完成状态: ' + (done ? 'COMPLETE' : 'INCOMPLETE'),
      '',
      '## 门禁状态',
    ].concat(GATE_ORDER.map((g) => '- ' + g + ' ' + GATES[g].title + ': ' + gateResults[g].status + (gateResults[g].at ? ' @ ' + gateResults[g].at : '')), [
      '',
      '## 交付物检查',
    ].concat(Object.keys(checks).map((ph) => '- ' + PHASES[ph].label + ': ' + checks[ph]), [
      '',
      '## 实际读取的入口',
      reads.length ? reads.map((r) => '- ' + (typeof r === 'object' ? JSON.stringify(r) : r)).join('\n') : '- （无记录）',
      '',
      '## 阻塞与待办',
      blockers.length ? blockers.map((b) => '- ' + b).join('\n') : '- 无',
    ])).join('\n')
    return { ok: true, done, blockers, gates: gateResults, deliverableChecks: checks, drift, report }
  }

  // 任务清单管理：list（看板）/ check（勾选完成）/ reset（重置）
  async function todoManage(action, phaseArg, index, note) {
    if (!state.project) return { ok: false, error: '项目未初始化' }
    const phase = phaseArg || state.currentPhase
    if (!PHASES[phase]) return { ok: false, error: '未知阶段: ' + phase }
    if (!state.phases[phase].tasks || state.phases[phase].tasks.length === 0) {
      state.phases[phase].tasks = PHASE_TASKS[phase].map((text) => ({ text, done: false }))
    }
    const list = state.phases[phase].tasks
    if (action === 'check') {
      const i = Number(index)
      if (!Number.isInteger(i) || i < 0 || i >= list.length) return { ok: false, error: 'index 越界: ' + index }
      list[i].done = true
      if (note) list[i].note = String(note)
      log('todo_check', { phase, index: i, text: list[i].text })
    } else if (action === 'reset') {
      state.phases[phase].tasks = PHASE_TASKS[phase].map((text) => ({ text, done: false }))
      log('todo_reset', { phase })
    }
    await persist()
    const doneCount = list.filter((t) => t.done).length
    const pct = list.length ? Math.round(doneCount / list.length * 100) : 0
    return { ok: true, phase, action, total: list.length, done: doneCount, pct, tasks: list }
  }

  function renderTool(value) {
    if (!value || typeof value !== 'object') return [{ type: 'text', text: JSON.stringify(value) }]
    if (value.ok === false) return [{ type: 'text', text: '❌ ' + (value.error || '操作失败') }]
    if (value.brief) return [{ type: 'text', text: '门禁 ' + value.gate + '（' + value.title + '）质检简报已生成。\n\n' + value.brief }]
    if (value.items && Array.isArray(value.items)) {
      const lines = value.items.map((i) => (i.ok ? '✅' : (i.blocked ? '⛔' : '❌')) + ' ' + i.name + ' — ' + (i.note || ''))
      return [{ type: 'text', text: '交付物检查（' + value.phase + '）: ' + value.overall + '\n' + lines.join('\n') }]
    }
    if (value.report) return [{ type: 'text', text: value.report }]
    if (value.skillMd !== undefined && value.skillMd !== null) {
      const head = String(value.skillMd).split('\n').slice(0, 3).join(' ')
      return [{ type: 'text', text: '已读取 ' + (value.label || value.phase || '') + ' 角色 SKILL.md（' + (value.skillMdRead ? '成功' : '失败') + '）: ' + value.skillMdPath + '\n开头: ' + head }]
    }
    if (value.content !== undefined) return [{ type: 'text', text: '已读取 ' + value.path + '（' + String(value.content).length + ' 字符）' }]
    if (value.tasks && Array.isArray(value.tasks)) {
      const lines = ['任务看板（' + value.phase + '，' + value.done + '/' + value.total + '，' + value.pct + '%）']
      value.tasks.forEach((t, i) => lines.push((t.done ? '✅' : '⬜') + ' ' + i + '. ' + t.text + (t.note ? ' — ' + t.note : '')))
      return [{ type: 'text', text: lines.join('\n') }]
    }
    if (value.progress) {
      const lines = ['【进度看板】']
      value.progress.steps.forEach((s) => {
        const mark = s.status === 'done' ? '✅' : s.status === 'current' ? '▶' : s.status === 'inprogress' ? '🔄' : '⬜'
        lines.push('  ' + mark + ' ' + s.label + ' (' + (value.progress.tasks[s.key] ? value.progress.tasks[s.key].pct + '%' : '-') + ')')
      })
      lines.push('  当前阶段: ' + value.currentPhase)
      lines.push('  下一动作: ' + (value.progress.nextAction || '-'))
      return [{ type: 'text', text: lines.join('\n') }]
    }
    if (value.initialized !== undefined) {
      const lines = []
      lines.push('项目: ' + (value.initialized ? (value.project.title + ' (' + value.project.competition + (value.project.edition ? ' ' + value.project.edition : '') + ')') : '未初始化'))
      if (value.initialized) {
        lines.push('当前阶段: ' + value.currentPhase)
        for (const g of GATE_ORDER) lines.push('- ' + g + ' ' + (value.gates[g] ? value.gates[g].status : 'pending'))
      }
      return [{ type: 'text', text: lines.join('\n') }]
    }
    return [{ type: 'text', text: JSON.stringify(value) }]
  }

  function toolDef(t) {
    // 规范化参数为标准 JSON Schema：属性级 required 标记收敛到顶层 required 数组
    const properties = {}
    const required = []
    const raw = t.parameters || {}
    for (const key of Object.keys(raw)) {
      const p = Object.assign({}, raw[key])
      if (p.required === true) { required.push(key); delete p.required }
      properties[key] = p
    }
    const parameters = { type: 'object', properties }
    if (required.length > 0) parameters.required = required
    return {
      name: t.name,
      description: t.description,
      parameters,
      output: { schema: { type: 'object', additionalProperties: true }, render: (args, value) => renderTool(value) },
      execute: t.execute,
    }
  }

  const tools = [
    {
      name: 'mm_project_init',
      description: '初始化或续接数学建模项目：配置 SKILL_ROOT（默认预设内置 math-modeling 知识库，只读）、PROJECT_ROOT（项目/题目目录，默认当前会话工作区）、竞赛与届次、子问题列表、论文格式与可选协作开关；校验两根目录不同；回显激活清单。若 PROJECT_ROOT 下已有状态文件则续接（不覆盖既有门禁/交付物状态）。',
      parameters: {
        skillRoot: { type: 'string', description: 'SKILL_ROOT：math-modeling 知识库路径（默认预设内置，随预设持久化；可指定外部仓库）' },
        projectRoot: { type: 'string', description: 'PROJECT_ROOT：项目/题目目录（默认当前会话工作区；必须已存在，且与 SKILL_ROOT 不同）' },
        title: { type: 'string', description: '项目标题（可选）' },
        competition: { type: 'string', description: '目标竞赛：CUMCM / MCM-ICM / APMCM / MathorCup 等（可选，默认 CUMCM）' },
        edition: { type: 'string', description: '届次，如 2026（可选）' },
        subproblems: { type: 'string', description: '子问题列表，逗号分隔，如 q1,q2,q3（可选，默认 q1）' },
        paperFormat: { type: 'string', enum: ['word', 'word+latex'], description: '论文格式（默认 word；word+latex 需用户显式要求）' },
        optionalCollab: { type: 'string', description: '可选协作 Subagent 开关（逗号分隔，默认全部关闭）: rulesCheck,attachmentInventory,literature,prototype,experiments,bilingual,terminology' },
      },
      execute: async (args) => { await loadState(await resolveProjectRoot(args)); return initProject(args || {}) },
    },
    {
      name: 'mm_state',
      description: '读取当前数学建模项目状态快照：阶段、五门禁状态、交付物检查结果、项目配置与最近账本。',
      parameters: {},
      execute: async () => { await loadState(await resolveProjectRoot({})); return summaryOf() },
    },
    {
      name: 'mm_phase_enter',
      description: '进入建模手(modeling)/编程手(programming)/论文手(paper)阶段：校验前置门禁，实际读取并返回该角色 SKILL.md 全文（强制执行“先读角色规范”协议），记录阶段进入。',
      parameters: {
        phase: { type: 'string', enum: ['modeling', 'programming', 'paper'], required: true, description: '要进入的阶段' },
      },
      required: ['phase'],
      execute: async (args) => { await loadState(await resolveProjectRoot({})); return enterPhase(args.phase) },
    },
    {
      name: 'mm_skill_read',
      description: '按需读取 SKILL_ROOT 内的参考资料/脚本说明（如 references/算法索引.md、tools/figure/SKILL.md），路径必须在 SKILL_ROOT 内，只读。',
      parameters: {
        path: { type: 'string', required: true, description: 'SKILL_ROOT 内的相对或绝对路径' },
      },
      required: ['path'],
      execute: async (args) => { await loadState(await resolveProjectRoot({})); return skillRead(args.path) },
    },
    {
      name: 'mm_gate',
      description: '门禁质检：mode=prepare 生成 M1/P1/P2/W1/W2 的独立只读质检简报（含验收条款、输入快照、固定回执格式），由主 Agent 用 subagent 工具派发；mode=record 校验并记录 Subagent 回执（PASS/FAIL/BLOCKED），FAIL 需返工复验。',
      parameters: {
        gate: { type: 'string', enum: ['M1', 'P1', 'P2', 'W1', 'W2'], required: true, description: '门禁名' },
        mode: { type: 'string', enum: ['prepare', 'record'], description: 'prepare=生成简报（默认）；record=记录回执' },
        evidence: { type: 'object', additionalProperties: true, description: 'mode=prepare 时的证据快照对象: { inputs:[], deliverablePaths:[], commands:[], hashes:{}, notes:"" }' },
        receipt: { type: 'object', additionalProperties: true, description: 'mode=record 时的固定回执对象: { scope, inputSnapshot, status:"PASS|FAIL|BLOCKED", evidence:[], findings:[{level,text}], rework }' },
      },
      required: ['gate'],
      execute: async (args) => { await loadState(await resolveProjectRoot({})); return (args.mode === 'record') ? gateRecord(args.gate, args.receipt) : gatePrepare(args.gate, args.evidence) },
    },
    {
      name: 'mm_check_deliverables',
      description: '按阶段运行确定性交付物检查：建模手=两个 Markdown；编程手=代码/结果表/三类候选图(各≥3 且覆盖全部子问题)/复现清单/环境；论文手=完整论文.docx/正式图≥8/LaTeX(可选)。任何缺失或阻塞如实报告，不静默降级。',
      parameters: {
        phase: { type: 'string', enum: ['modeling', 'programming', 'paper'], description: '要检查的阶段（默认当前阶段）' },
      },
      execute: async (args) => { await loadState(await resolveProjectRoot({})); return checkDeliverables(args.phase) },
    },
    {
      name: 'mm_complete',
      description: '完成判定：校验五门禁全部 PASS、三阶段交付物齐全、门禁通过后产物未发生实质变化（漂移检测）；生成最终完成报告。未满足时返回阻塞清单。',
      parameters: {},
      execute: async () => { await loadState(await resolveProjectRoot({})); return completeProject() },
    },
    {
      name: 'mm_log',
      description: '向项目账本追加一条事件（如返工/决策记录），用于“修正后复验”追踪。',
      parameters: {
        event: { type: 'string', required: true, description: '事件名，如 rework / decision' },
        details: { type: 'string', description: '事件详情' },
      },
      required: ['event'],
      execute: async (args) => { await loadState(await resolveProjectRoot({})); if (!state.project) return { ok: false, error: '项目未初始化' }; log(String(args.event || 'log'), String(args.details || '')); await persist(); return { ok: true, event: args.event } },
    },
    {
      name: 'mm_todo',
      description: '阶段任务清单看板：list 查看当前阶段标准任务与完成度；check 勾选完成某项（index 从 0 开始，可选 note 记录备注）；reset 重置该阶段任务。进入阶段时 mm_phase_enter 已自动注入标准任务。',
      parameters: {
        action: { type: 'string', enum: ['list', 'check', 'reset'], required: true, description: 'list=查看看板（默认）；check=勾选完成；reset=重置' },
        phase: { type: 'string', enum: ['modeling', 'programming', 'paper'], description: '阶段（默认当前阶段）' },
        index: { type: 'number', description: 'check 时勾选的任务下标（从 0 开始）' },
        note: { type: 'string', description: 'check 时的备注（可选）' },
      },
      required: ['action'],
      execute: async (args) => { await loadState(await resolveProjectRoot({})); return todoManage(args.action || 'list', args.phase, args.index, args.note) },
    },
    {
      name: 'mm_ui_toggle',
      description: '数学建模 UI 看板开关：get 查当前状态；on 开启；off 关闭。设置持久化（重起 DSH 仍生效）。',
      parameters: {
        action: { type: 'string', enum: ['get', 'on', 'off'], required: true, description: 'get=查状态；on=开启；off=关闭' },
      },
      required: ['action'],
      execute: async (args) => {
        if (!settings) return { ok: false, error: 'settings 服务不可用' }
        const action = args.action || 'get'
        const current = (() => { try { return settings.get('math-modeling-ui') || {} } catch (e) { return {} } })()
        if (action === 'get') return { ok: true, enabled: current.enabled !== false }
        const next = action === 'on'
        try {
          await settings.update('math-modeling-ui', { ...current, enabled: next })
        } catch (e) { return { ok: false, error: msg(e) } }
        return { ok: true, enabled: next, notice: next ? 'UI 看板已开启' : 'UI 看板已关闭' }
      },
    },
  ]

  const toolsReg = ctx.get('tools')
  if (toolsReg) {
    for (const t of tools) toolsReg.register(toolDef(t))
  }

  // 注册 UI 开关设置
  if (settings) {
    try {
      const uiScope = settings.register('math-modeling-ui', {
        type: 'object',
        properties: {
          enabled: { type: 'boolean', default: true, description: '在数学建模会话中显示进度看板' },
        },
        additionalProperties: false,
      }, {
        label: '数学建模 Workbench',
        description: '控制数学建模会话中的进度看板显示',
      })
      console.error('[mathm] UI 设置已注册')
    } catch (e) {
      console.error('[mathm] UI 设置注册失败: ' + String((e && e.message) || e))
    }
  }

  if (systemPrompt) {
    systemPrompt.section({
      name: 'math-modeling-workbench',
      order: 5000,
      text: [
        '## 数学建模工作流（Workbench 插件）',
        '当任务涉及数学建模/建模竞赛时，必须按以下协议执行，不要降级为建议：',
        '1. 用 mm_project_init 初始化或续接项目（SKILL_ROOT 默认预设内置 math-modeling 知识库，随预设持久化，不依赖外部路径；PROJECT_ROOT 默认当前会话工作区；两者必须不同且互不包含；状态文件位于 PROJECT_ROOT/.math-modeling/state.json）。',
        '2. 阶段顺序：建模手 → 编程手 → 论文手。进入每个阶段前必须调用 mm_phase_enter（它会实际读取并返回该角色 SKILL.md 全文，并注入该阶段标准任务清单）。',
        '3. 任务进度：每阶段用 mm_todo list 查看看板，完成一项用 mm_todo check index=N 勾选；用 mm_state 查看三阶段进度看板（阶段步进、门禁状态、任务完成度、下一动作）。',
        '4. 门禁顺序：M1 → P1 → P2 → W1 → W2。到门禁节点必须用 mm_gate mode=prepare 生成质检简报并派发独立只读 Subagent（作者自检不能替代独立验收），再用 mm_gate mode=record 记录回执；FAIL 必须按证据返工并复验。',
        '5. 建模约束：先完整理解题目/附件/目标/约束/评价口径再定方案；每道子问题最多使用两个独立模型体系（物理题同一机理不同精度按一个模型族计数）；避免直接套用常见简单模型冒充创新；不因两个模型结论相近就强制删除。',
        '6. 渐进式加载：用 mm_skill_read 按需读取 SKILL_ROOT 内资料——选模型/查算法读 references/算法索引.md 再读 assets/*.md；搜索论文读 tools/paper_search/SKILL.md；读题 PDF 读 tools/pdf/SKILL.md；处理 Excel 读 tools/xlsx/SKILL.md；画图读 tools/figure/SKILL.md；生成 Word 读 tools/docx/SKILL.md；生成 LaTeX 读 tools/latex/SKILL.md；禁止一次性加载全部资料。',
        '7. 交付物用 mm_check_deliverables 检查；任何校验预警视为未完成，不得静默降低篇幅/图表/公式/引用/编译质量目标；环境缺依赖时如实报告阻塞并继续可验证部分。',
        '8. 宣称完成前必须运行 mm_complete：所有门禁 PASS、交付物齐全、通过后产物未发生实质变化（漂移检测）；SKILL_ROOT 禁止改写。',
      ].join('\n'),
    })
  }

  // 注册内置 math-modeling skill（content 来自随预设持久化的 SKILL.md）
  // 使 skill 工具可直接加载「math-modeling」，不依赖文件系统发现或外部路径。
  const skills = ctx.get('skills')
  if (skills && BUNDLED_SKILL_ROOT) {
    try {
      const mainMd = join(BUNDLED_SKILL_ROOT, 'SKILL.md')
      const info = await exists(mainMd)
      if (info) {
        const content = await readFile(mainMd)
        skills.register({
          name: 'math-modeling',
          description: '数学建模（Workbench 内置）：三阶段工作流（建模手/编程手/论文手）、五门禁质检（M1/P1/P2/W1/W2）、固定交付物与完成判定。',
          whenToUse: '当用户要求数学建模、建模竞赛、建模分析、代码求解、结果可视化或生成数学建模论文时使用。',
          source: 'math-modeling (builtin skill bundle)',
          content,
        })
        console.error('[mathm] 内置 skill math-modeling 已注册')
      }
    } catch (e) {
      console.error('[mathm] 内置 skill 注册失败: ' + String((e && e.message) || e))
    }
  }
}
