// dsh-math-modeling-ui 插件入口（单包单插件：数学建模进度看板）
//
// 用 DSH Host Connection RPC（ctx.connection.rpc.handle）注册看板数据通道，
// client 侧用 ctx.connection.rpc.call 调用。这是组合包 host 插件可用的
// 持久化 RPC（区别于动态插件的 harness.handle），重启后仍然生效。
//
// 读取 <PROJECT_ROOT>/.math-modeling/state.json，独立于 math-modeling.js 预设
// 插件内部状态机。

// channel：client 与 host 共享的 RPC 通道标识
const MM_RPC_CHANNEL = '/math-modeling-ui';
const MM_ENDPOINTS = Object.freeze({
  state: 'mm.state',
  setEnabled: 'mm.setEnabled',
  getEnabled: 'mm.getEnabled',
});

const META_DIR = '.math-modeling';
const STATE_NAME = 'state.json';

const name = 'dsh-math-modeling-ui';
const inject = ['connection', 'webServer'];

function ok(value) {
  return { ok: true, value };
}
function fail(message) {
  return { ok: false, error: { code: 'bad-request', message, details: { issues: [{ message }] } } };
}

export function apply(ctx) {
  const logger = ctx.logger?.(name) ?? console;
  if (!ctx?.connection?.rpc?.handle) {
    logger.error('dsh-math-modeling-ui: Host Connection RPC unavailable — board disabled | 无 Connection RPC，看板不可用');
    return () => {};
  }

  const fs = ctx.get('fs');
  const sessions = ctx.get('sessions');
  const agents = ctx.get('agents');
  const settings = ctx.get('settings');

  const join = (...parts) => parts.filter(Boolean).join('/');
  const msg = (e) => String((e && e.message) || e);

  async function ex(p) {
    try { const t = await fs.resolve(p); return (await fs.stat(t)) || null; } catch (e) { return null; }
  }
  async function readText(p) { return fs.readText(await fs.resolve(p)); }

  function readEnabled() {
    if (!settings) return true;
    try { const s = settings.get('math-modeling-ui'); return !(s && s.enabled === false); } catch (e) { return true; }
  }

  async function resolveCwd(sessionId) {
    let cwd = null;
    if (sessionId && sessions) {
      try {
        const s = sessions.get(sessionId);
        if (s && s.header && s.header.cwd) cwd = s.header.cwd;
      } catch (e) {}
    }
    if (!cwd) {
      try {
        const initiator = agents && agents.currentInitiator();
        if (initiator && initiator.session && initiator.session.header && initiator.session.header.cwd) cwd = initiator.session.header.cwd;
      } catch (e) {}
    }
    if (!cwd) { try { cwd = fs.processPath(await fs.resolve('.')); } catch (e) {} }
    return cwd;
  }

  async function readState(cwd) {
    const direct = join(cwd, META_DIR, STATE_NAME);
    if (await ex(direct)) return { path: direct, json: JSON.parse(await readText(direct)) };
    for (const c of ['B题', 'A题', 'C题', 'D题', 'E题']) {
      const p = join(cwd, c, META_DIR, STATE_NAME);
      if (await ex(p)) return { path: p, json: JSON.parse(await readText(p)) };
    }
    return null;
  }

  async function getState(sessionId) {
    try {
      if (!readEnabled()) return ok({ ok: true, hidden: true, reason: 'disabled' });
      const cwd = await resolveCwd(sessionId);
      if (!cwd) return fail('cannot detect workspace');
      const st = await readState(cwd);
      if (!st) return ok({ ok: true, initialized: false, cwd, projectRoot: cwd });
      const data = st.json;
      const project = data.project;
      const GATE_ORDER = ['M1', 'P1', 'P2', 'W1', 'W2'];
      const PHASES = { modeling: '建模手', programming: '编程手', paper: '论文手' };
      const gates = {};
      for (const g of GATE_ORDER) {
        const pk = ({ M1: 'modeling', P1: 'programming', P2: 'programming', W1: 'paper', W2: 'paper' })[g];
        const gn = data.phases && data.phases[pk] && data.phases[pk]['gate' + g];
        gates[g] = { title: ({ M1: '建模终检', P1: '最小可运行', P2: '编程终检', W1: '证据大纲', W2: '论文终检' })[g], status: gn ? gn.status : 'pending' };
      }
      const steps = ['modeling', 'programming', 'paper'].map((k) => {
        const p = data.phases && data.phases[k];
        const gk = { modeling: ['M1'], programming: ['P1', 'P2'], paper: ['W1', 'W2'] }[k];
        const done = p && p.enteredAt && gk.every((g) => gates[g].status === 'pass');
        return {
          key: k, label: PHASES[k],
          status: done ? 'done' : (data.currentPhase === k ? 'current' : (p && p.enteredAt ? 'inprogress' : 'pending')),
          tasks: p && Array.isArray(p.tasks) ? p.tasks.map((t) => ({ text: t.text, done: !!t.done })) : [],
        };
      });
      return ok({
        ok: true, initialized: !!project, statePath: st.path,
        project: project ? { title: project.title, competition: project.competition, edition: project.edition, projectRoot: project.projectRoot } : null,
        currentPhase: data.currentPhase, steps, gates, completed: !!data.completed, completedAt: data.completedAt,
      });
    } catch (e) { return fail(msg(e)); }
  }

  async function setEnabled(enabled) {
    if (!settings) return fail('settings unavailable');
    try {
      const cur = settings.get('math-modeling-ui') || {};
      await settings.update('math-modeling-ui', Object.assign({}, cur, { enabled: !!enabled }));
      return ok({ enabled: !!enabled });
    } catch (e) { return fail(msg(e)); }
  }

  function getEnabled() {
    return ok({ enabled: readEnabled() });
  }

  return ctx.connection.rpc.handle(MM_RPC_CHANNEL, async (endpoint, payload = {}) => {
    try {
      if (endpoint === MM_ENDPOINTS.state) {
        return await getState(payload?.sessionId);
      }
      if (endpoint === MM_ENDPOINTS.getEnabled) {
        return getEnabled();
      }
      if (endpoint === MM_ENDPOINTS.setEnabled) {
        return await setEnabled(payload?.enabled === true);
      }
      return fail(`unknown endpoint: ${endpoint}`);
    } catch (err) {
      logger.error('dsh-math-modeling-ui: rpc %s failed | RPC 失败: %s', endpoint, err?.message ?? err);
      return fail(err?.message ?? String(err));
    }
  }, { authority: 'loopback' });
}

export { name, inject, MM_RPC_CHANNEL, MM_ENDPOINTS };
