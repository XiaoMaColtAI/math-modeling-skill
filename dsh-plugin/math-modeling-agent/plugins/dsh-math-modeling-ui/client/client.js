window.__ModuleLoader__.load({
  id: "dsh-math-modeling-ui",
  factory: (require) => {
    var module = { exports: {} };
    var exports = module.exports;
    var React = require("react");
    var __defProp = Object.defineProperty;
    var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
    var __getOwnPropNames = Object.getOwnPropertyNames;
    var __hasOwnProp = Object.prototype.hasOwnProperty;
    var __export = (target, all) => {
      for (var name2 in all)
        __defProp(target, name2, { get: all[name2], enumerable: true });
    };
    var __copyProps = (to, from, except, desc) => {
      if (from && typeof from === "object" || typeof from === "function") {
        for (let key of __getOwnPropNames(from))
          if (!__hasOwnProp.call(to, key) && key !== except)
            __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
      }
      return to;
    };
    var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

    var index_exports = {};
    __export(index_exports, {
      name: () => name,
      inject: () => inject,
      apply: () => apply,
    });
    module.exports = __toCommonJS(index_exports);

    var name = "dsh-math-modeling-ui";
    var inject = ["slots", "connection", "locale", "sessions"];

    var MM_CHANNEL = "/math-modeling-ui";
    var MM_ENDPOINTS = {
      state: "mm.state",
      setEnabled: "mm.setEnabled",
      getEnabled: "mm.getEnabled",
    };
    var TARGET = "math-modeling-agent";

    var CSS = ".mm-overlay{pointer-events:auto;position:fixed;top:12px;right:12px;width:360px;max-height:calc(100vh - 24px);z-index:30;box-shadow:0 8px 30px rgba(0,0,0,.18);border-radius:12px;overflow:hidden;background:#F8FAFC}.mm-overlay.mm-collapsed{width:auto;box-shadow:0 4px 16px rgba(0,0,0,.15)}.mm-overlay .mmui{height:auto;max-height:calc(100vh - 24px);overflow-y:auto}.mmui{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:14px;line-height:1.5;color:#1E3A8A;padding:20px;overflow-y:auto;height:100%;background:#F8FAFC;box-sizing:border-box}.mmui *{box-sizing:border-box}.mmui .hdr{background:linear-gradient(135deg,#1E40AF,#3B82F6);border-radius:12px;padding:20px 22px;color:#fff;margin-bottom:20px;box-shadow:0 1px 3px rgba(30,64,175,.12),0 4px 12px rgba(30,64,175,.08)}.mmui .hdr-eyebrow{font-size:11px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;opacity:.8;margin-bottom:6px;display:flex;align-items:center;gap:6px}.mmui .hdr-dot{width:6px;height:6px;border-radius:50%;background:#10B981;box-shadow:0 0 8px #10B981}.mmui .hdr-top{display:flex;align-items:center;justify-content:space-between;gap:8px}.mmui .hdr-toggle{font-size:11px;font-weight:600;color:#fff;background:rgba(255,255,255,.2);border:none;border-radius:6px;padding:3px 8px;cursor:pointer;flex-shrink:0}.mmui .hdr-toggle:hover{background:rgba(255,255,255,.32)}.mmui .mmui-collapsed{padding:10px 14px}.mmui .mmui-collapsed .hdr-top{margin-bottom:0}.mmui .hdr-title{font-size:20px;font-weight:700;line-height:1.3;margin-bottom:8px;word-break:break-word}.mmui .hdr-meta{display:flex;flex-wrap:wrap;gap:8px;font-size:12px;font-weight:500;opacity:.95}.mmui .hdr-tag{background:rgba(255,255,255,.18);padding:3px 10px;border-radius:6px;display:inline-flex;align-items:center;gap:4px}.mmui .hdr-tag-accent{background:#D97706;color:#fff}.mmui .card{background:#fff;border:1px solid #E9EEF6;border-radius:10px;padding:18px;margin-bottom:14px;box-shadow:0 1px 2px rgba(15,23,42,.04)}.mmui .card-h{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid #F1F5F9}.mmui .card-title{font-size:13px;font-weight:700;color:#1E3A8A;display:flex;align-items:center;gap:8px}.mmui .card-action{font-size:12px;font-weight:600;color:#1E40AF;cursor:pointer;padding:4px 10px;border-radius:6px;background:#EEF2FF;border:none}.mmui .card-action:hover{background:#1E40AF;color:#fff}.mmui .step{display:flex;align-items:stretch;gap:14px;padding:12px 0;border-bottom:1px solid #F8FAFC}.mmui .step:last-child{border-bottom:none;padding-bottom:0}.mmui .step:first-child{padding-top:0}.mmui .step-rail{display:flex;flex-direction:column;align-items:center;width:32px;flex-shrink:0}.mmui .step-circle{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;border:2px solid #E9EEF6;background:#fff;color:#94A3B8}.mmui .step.done .step-circle{background:#10B981;border-color:#10B981;color:#fff}.mmui .step.cur .step-circle{background:#1E40AF;border-color:#1E40AF;color:#fff;box-shadow:0 0 0 4px rgba(30,64,175,.15)}.mmui .step-line{width:2px;flex:1;background:#E9EEF6;margin:4px 0}.mmui .step.done .step-line{background:#10B981}.mmui .step-body{flex:1;min-width:0;padding:6px 0}.mmui .step-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:4px}.mmui .step-name{font-size:14px;font-weight:600;color:#1E3A8A}.mmui .step.cur .step-name{color:#1E40AF}.mmui .step.done .step-name{color:#64748B}.mmui .step-meta{font-size:11px;font-weight:600;color:#94A3B8;font-family:monospace;background:#F1F5F9;padding:2px 8px;border-radius:4px}.mmui .step.cur .step-meta{background:rgba(30,64,175,.1);color:#1E40AF}.mmui .step-bar{height:6px;background:#F1F5F9;border-radius:3px;overflow:hidden;margin-top:6px}.mmui .step-bar-fill{height:100%;background:linear-gradient(90deg,#1E40AF,#3B82F6);border-radius:3px}.mmui .step.done .step-bar-fill{background:#10B981}.mmui .gate-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.mmui .gate{background:#fff;border:1px solid #E9EEF6;border-radius:8px;padding:12px 14px;display:flex;align-items:center;gap:10px}.mmui .gate-ico{width:28px;height:28px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;font-family:monospace;flex-shrink:0}.mmui .gate.pass .gate-ico{background:#D1FAE5;color:#047857}.mmui .gate.fail .gate-ico{background:#FEE2E2;color:#B91C1C}.mmui .gate.blocked .gate-ico{background:#FEF3C7;color:#B45309}.mmui .gate.pending .gate-ico{background:#F1F5F9;color:#94A3B8}.mmui .gate-body{flex:1;min-width:0}.mmui .gate-id{font-size:12px;font-weight:700;color:#1E3A8A;font-family:monospace}.mmui .gate-label{font-size:10px;font-weight:600;color:#64748B;letter-spacing:.5px;margin-top:1px}.mmui .gate-status{font-size:10px;font-weight:700;padding:3px 8px;border-radius:4px;flex-shrink:0;font-family:monospace}.mmui .gate.pass .gate-status{background:#D1FAE5;color:#047857}.mmui .gate.fail .gate-status{background:#FEE2E2;color:#B91C1C}.mmui .gate.blocked .gate-status{background:#FEF3C7;color:#B45309}.mmui .gate.pending .gate-status{background:#F1F5F9;color:#94A3B8}.mmui .tasks-empty{padding:24px 12px;text-align:center;color:#94A3B8;font-size:13px;background:#F8FAFC;border-radius:8px;border:1px dashed #E9EEF6}.mmui .tasks{display:flex;flex-direction:column;gap:6px;max-height:240px;overflow-y:auto}.mmui .task{display:flex;align-items:flex-start;gap:12px;padding:10px 12px;border-radius:8px;background:#F8FAFC;border:1px solid #F1F5F9}.mmui .task-check{width:18px;height:18px;border-radius:5px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0;margin-top:1px;background:#fff;border:2px solid #CBD5E1;color:#fff}.mmui .task.done{opacity:.6}.mmui .task.done .task-check{background:#10B981;border-color:#10B981}.mmui .task-text{flex:1;font-size:13px;color:#1E3A8A}.mmui .task.done .task-text{text-decoration:line-through;color:#94A3B8}.mmui .completed{background:linear-gradient(135deg,#D1FAE5,#A7F3D0);border:1px solid #10B981;border-radius:10px;padding:16px 20px;margin-bottom:14px;display:flex;align-items:center;gap:12px}.mmui .completed-ico{width:36px;height:36px;border-radius:50%;background:#10B981;color:#fff;display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;flex-shrink:0}.mmui .completed-title{font-size:14px;font-weight:700;color:#065F46}.mmui .completed-sub{font-size:12px;color:#047857;margin-top:2px}.mmui .next{background:linear-gradient(135deg,#EEF2FF,#DBEAFE);border:1px solid #BFDBFE;border-radius:10px;padding:14px 18px;margin-bottom:14px;display:flex;align-items:flex-start;gap:12px}.mmui .next-ico{width:32px;height:32px;border-radius:8px;background:#1E40AF;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0}.mmui .next-label{font-size:11px;font-weight:700;color:#1E40AF;letter-spacing:.8px;margin-bottom:4px}.mmui .next-text{font-size:13px;color:#1E3A8A;line-height:1.5}.mmui .actions{display:flex;gap:8px;margin-top:8px}.mmui .btn{flex:1;padding:10px 14px;border:1px solid #E9EEF6;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;background:#fff;color:#475569}.mmui .btn-primary{background:#1E40AF;color:#fff;border-color:#1E40AF}.mmui .err{background:#FEE2E2;border:1px solid #FCA5A5;border-radius:8px;padding:14px 16px;font-size:13px;color:#B91C1C}.mmui .init{background:#EEF2FF;border:1px dashed #BFDBFE;border-radius:10px;padding:20px;text-align:center;color:#1E40AF;font-size:13px;line-height:1.6}.mmui .init-title{font-weight:700;margin-bottom:4px;font-size:14px}.mmu-set{font-family:Inter,sans-serif;padding:4px}.mmu-set .card{background:#fff;border:1px solid #E9EEF6;border-radius:10px;padding:18px;max-width:560px}.mmu-set .row{display:flex;align-items:center;justify-content:space-between;gap:16px}.mmu-set .title{font-size:14px;font-weight:700;color:#1E3A8A}.mmu-set .desc{font-size:12px;color:#64748B;margin-top:4px;line-height:1.5}.mmu-set .toggle{position:relative;width:44px;height:24px;border-radius:12px;background:#CBD5E1;border:none;cursor:pointer;flex-shrink:0;transition:background .2s}.mmu-set .toggle.on{background:#1E40AF}.mmu-set .toggle::after{content:\"\";position:absolute;top:2px;left:2px;width:20px;height:20px;border-radius:50%;background:#fff;transition:transform .2s;box-shadow:0 1px 3px rgba(0,0,0,.2)}.mmu-set .toggle.on::after{transform:translateX(20px)}.mmu-set .toggle:disabled{opacity:.5;cursor:default}.mmu-set .note{margin-top:12px;font-size:12px;color:#64748B;background:#F8FAFC;border:1px solid #E9EEF6;border-radius:8px;padding:10px 12px;line-height:1.5}@media (prefers-reduced-motion:reduce){.mmui *{transition:none!important;animation:none!important}}";

    function useRpc(ctx) {
      return function (endpoint, payload, signal) {
        return ctx.connection.rpc.call(MM_CHANNEL, endpoint, payload, signal);
      };
    }

    function MathModelingDock(props) {
      var h = React.createElement;
      var ctx = props.ctx;
      var rpcCall = props.rpcCall;
      // 当前会话 preset：shell.overlay 是 root scope 无 sessionId prop，
      // 用 ctx.sessions.list.getSnapshot() 读当前活跃会话的 agentPreset 做门控。
      var currentSession = function () {
        try {
          var st = ctx.sessions.list.getSnapshot();
          var id = st.current;
          return (id && st.byId[id]) ? st.byId[id] : null;
        } catch (e) { return null; }
      };
      var cs = currentSession();
      var sid = cs ? cs.id : null;
      var preset = cs ? cs.agentPreset : undefined;
      var state = React.useState(null);
      var data = state[0], setData = state[1];
      var expPair = React.useState(false);
      var exp = expPair[0], setExp = expPair[1];
      var colPair = React.useState(true);
      var collapsed = colPair[0], setCollapsed = colPair[1];
      var errPair = React.useState(null);
      var err = errPair[0], setErr = errPair[1];

      var ref = React.useCallback(function () {
        rpcCall(MM_ENDPOINTS.state, { sessionId: sid }).then(function (res) {
          if (res && res.ok) { setData(res.value); setErr(null); }
          else setErr((res && res.error && res.error.message) || 'RPC failed');
        }).catch(function (e) { setErr(String((e && e.message) || e)); });
      }, [sid]);
      React.useEffect(function () {
        ref();
        var timer = setInterval(ref, 10000);
        return function () { clearInterval(timer); };
      }, [ref]);

      if (data && data.hidden) return null;
      if (err) return h("div", { className: "mm-overlay" }, h("div", { className: "mmui" }, h("div", { className: "err" }, "ERR " + err)));
      // 门控：仅数学建模 agent 预设会话显示（不区分数据是否已加载）
      if (preset !== undefined && preset !== TARGET) return null;
      // 折叠状态：即使数据未加载也显示折叠条（标题 + 展开按钮）
      if (collapsed) {
        var colTitle = (data && data.project && data.project.title) || "数学建模 Workbench";
        return h("div", { className: "mm-overlay mm-collapsed" },
          h("div", { className: "mmui" },
            h("div", { className: "hdr mmui-collapsed" },
              h("div", { className: "hdr-top" },
                h("div", { className: "hdr-eyebrow" }, h("div", { className: "hdr-dot" }), "📐 " + colTitle),
                h("button", { className: "hdr-toggle", onClick: function () { setCollapsed(false); setExp(true); }, "aria-expanded": false }, "展开")
              )
            )
          )
        );
      }
      if (!data) return h("div", { className: "mm-overlay" }, h("div", { className: "mmui" }, h("div", { className: "init" }, "加载中…")));
      if (data.ok === false) return h("div", { className: "mm-overlay" }, h("div", { className: "mmui" }, h("div", { className: "err" }, data.error || "")));
      if (!data.initialized) return h("div", { className: "mm-overlay" }, h("div", { className: "mmui" }, h("div", { className: "init" }, h("div", { className: "init-title" }, "项目未初始化"), "运行 mm_project_init 启动数学建模工作流")));

      var SC = { done: "done", current: "cur", inprogress: "inprogress", pending: "pending" };
      var GC = { pass: "pass", fail: "fail", blocked: "blocked" };
      var PI = { modeling: "建模 · 题目分析与模型设计", programming: "编程 · 代码求解与结果验证", paper: "论文 · 撰写与排版交付" };
      var PHASE_NUM = { modeling: "01", programming: "02", paper: "03" };
      var PHASE_SHORT = { modeling: "建模手", programming: "编程手", paper: "论文手" };
      var GL = { pass: "PASS", fail: "FAIL", blocked: "BLOCK", pending: "待" };
      var NEXT = { modeling: "进入 M1 门禁质检 → 准备编程实现", programming: "通过 P1/P2 门禁 → 进入论文阶段", paper: "通过 W1/W2 门禁 → 准备完成判定" };

      var cur = data.steps && data.steps.find(function (s) { return s.status === "current"; });
      var ct = cur ? (cur.tasks || []) : [];
      var proj = data.project;

      var metaTags = [];
      if (proj && proj.competition) metaTags.push(h("span", { className: "hdr-tag", key: "c" }, proj.competition));
      if (proj && proj.edition) metaTags.push(h("span", { className: "hdr-tag", key: "e" }, proj.edition));
      if (cur) metaTags.push(h("span", { className: "hdr-tag hdr-tag-accent", key: "p" }, "▶ " + (PHASE_SHORT[cur.key] || cur.label)));

      var headerEl = h("div", { className: "hdr" + (collapsed ? " mmui-collapsed" : ""), key: "hdr" },
        h("div", { className: "hdr-top" },
          h("div", { className: "hdr-eyebrow" }, h("div", { className: "hdr-dot" }), data.completed ? "已交付" : "进行中"),
          h("button", { className: "hdr-toggle", onClick: function () { setCollapsed(!collapsed); }, "aria-expanded": !collapsed }, collapsed ? "▶ 展开" : "◀ 收起")
        ),
        !collapsed ? [
          h("div", { className: "hdr-title" }, proj ? proj.title : "数学建模 Workbench"),
          h("div", { className: "hdr-meta" }, metaTags)
        ] : null
      );

      var nextEl = cur && !data.completed ? h("div", { className: "next", key: "next" },
        h("div", { className: "next-ico" }, "→"),
        h("div", null, h("div", { className: "next-label" }, "NEXT"), h("div", { className: "next-text" }, NEXT[cur.key] || ""))
      ) : null;

      var phaseItems = (data.steps || []).map(function (s, i, arr) {
        var tasks = s.tasks || [];
        var doneCnt = tasks.filter(function (t) { return t.done; }).length;
        var pct = tasks.length ? Math.round(doneCnt / tasks.length * 100) : 0;
        var isLast = i === arr.length - 1;
        var statusLabel = s.status === "done" ? "已完成" : s.status === "current" ? "进行中" : s.status === "inprogress" ? "已启动" : "未开始";
        var icon = s.status === "done" ? "✓" : s.status === "current" ? "▶" : s.status === "inprogress" ? "◐" : "○";
        var barEl = tasks.length > 0 ? h("div", { className: "step-bar" }, h("div", { className: "step-bar-fill", style: { width: pct + "%" } })) : null;
        return h("div", { className: "step " + SC[s.status], key: s.key },
          h("div", { className: "step-rail" }, h("div", { className: "step-circle" }, icon), !isLast ? h("div", { className: "step-line" }) : null),
          h("div", { className: "step-body" },
            h("div", { className: "step-head" },
              h("div", { className: "step-name" }, (PHASE_NUM[s.key] || "") + " " + (PI[s.key] || s.label)),
              h("div", { className: "step-meta" }, statusLabel + (tasks.length ? " · " + doneCnt + "/" + tasks.length : ""))
            ),
            barEl
          )
        );
      });

      var phaseCard = h("div", { className: "card", key: "ph" },
        h("div", { className: "card-h" }, h("div", { className: "card-title" }, "阶段进度")),
        h("div", { className: "stepper" }, phaseItems)
      );

      var gateItems = ["M1", "P1", "P2", "W1", "W2"].map(function (g) {
        var st = data.gates && data.gates[g] ? data.gates[g].status : "pending";
        var title = data.gates && data.gates[g] ? data.gates[g].title : "";
        var ico = GL[st] === "PASS" ? "✓" : GL[st] === "FAIL" ? "✗" : GL[st] === "BLOCK" ? "!" : "·";
        return h("div", { className: "gate " + (GC[st] || "pending"), key: g, title: title },
          h("div", { className: "gate-ico" }, ico),
          h("div", { className: "gate-body" }, h("div", { className: "gate-id" }, g), h("div", { className: "gate-label" }, title)),
          h("div", { className: "gate-status" }, GL[st])
        );
      });
      var gateCard = h("div", { className: "card", key: "g" },
        h("div", { className: "card-h" }, h("div", { className: "card-title" }, "五门禁状态")),
        h("div", { className: "gate-grid" }, gateItems)
      );

      var completedEl = data.completed ? h("div", { className: "completed", key: "cb" },
        h("div", { className: "completed-ico" }, "✓"),
        h("div", null, h("div", { className: "completed-title" }, "项目已完成"), h("div", { className: "completed-sub" }, "所有门禁通过，交付物齐全"))
      ) : null;

      var taskCard = null;
      if (cur && ct.length > 0) {
        var taskListEl = exp
          ? h("div", { className: "tasks", role: "list" }, ct.map(function (t, i) { return h("div", { className: "task" + (t.done ? " done" : ""), key: i, role: "listitem" },
              h("div", { className: "task-check" }, t.done ? "✓" : ""),
              h("div", { className: "task-text" }, t.text)
            ); }))
          : h("div", { className: "tasks-empty" }, "共 " + ct.length + " 项任务 · 点击展开查看详情");
        taskCard = h("div", { className: "card", key: "t" },
          h("div", { className: "card-h" },
            h("div", { className: "card-title" }, "当前任务 · " + (PHASE_SHORT[cur.key] || cur.label)),
            h("button", { className: "card-action", onClick: function () { setExp(!exp); }, "aria-expanded": exp }, exp ? "收起" : "展开")
          ),
          taskListEl
        );
      }

      var actions = h("div", { className: "actions", key: "a" },
        h("button", { className: "btn btn-primary", onClick: ref }, "↻ 刷新数据")
      );

      // 展开时显示完整内容（右侧悬浮面板）
      return h("div", { className: "mm-overlay" },
        h("div", { className: "mmui" }, headerEl, nextEl, phaseCard, gateCard, completedEl, taskCard, actions)
      );
    }

    function SettingsToggle(props) {
      var h = React.createElement;
      var rpcCall = props.rpcCall;
      var enabledPair = React.useState(null);
      var enabled = enabledPair[0], setEnabled = enabledPair[1];
      var busyPair = React.useState(false);
      var busy = busyPair[0], setBusy = busyPair[1];
      React.useEffect(function () {
        rpcCall(MM_ENDPOINTS.getEnabled, {}).then(function (res) {
          if (res && res.ok) setEnabled(!!res.value.enabled);
          else setEnabled(true);
        }).catch(function () { setEnabled(true); });
      }, []);
      function toggle() {
        if (busy || enabled === null) return;
        setBusy(true);
        rpcCall(MM_ENDPOINTS.setEnabled, { enabled: !enabled }).then(function (res) {
          if (res && res.ok) setEnabled(!!res.value.enabled);
          setBusy(false);
        }).catch(function () { setBusy(false); });
      }
      var noteText = enabled ? "看板已开启：切换到数学建模 Workbench 预设的会话即可自动显示。" : "看板已关闭：数学建模会话不再显示看板。";
      return h("div", { className: "mmu-set" },
        h("div", { className: "card" },
          h("div", { className: "row" },
            h("div", null,
              h("div", { className: "title" }, "数学建模进度看板"),
              h("div", { className: "desc" }, "在「数学建模 Workbench」会话的右侧详情面板显示三阶段进度、五门禁状态与当前任务清单。")
            ),
            h("button", { className: "toggle" + (enabled ? " on" : ""), onClick: toggle, disabled: busy || enabled === null, "aria-pressed": enabled === true, "aria-label": "开关数学建模看板" })
          ),
          h("div", { className: "note" }, noteText)
        )
      );
    }

    function apply(ctx) {
      if (typeof document !== "undefined") {
        var tag = document.createElement("style");
        tag.dataset.mmui = "dsh-math-modeling-ui";
        tag.textContent = CSS;
        document.head.append(tag);
      }
      var rpcCall = useRpc(ctx);
      // 右侧悬浮面板（shell.overlay 无冲突，能常驻；用 sessions.list.getSnapshot().current 门控当前会话）
      ctx.slots.inject("shell.overlay", function () {
        return ctx.slots.register(
          { name: "shell.overlay", id: "mathmodeling-board", order: 50 },
          function () {
            return React.createElement(MathModelingDock, { ctx: ctx, rpcCall: rpcCall });
          }
        );
      });
      // 设置 → 插件页内的「数学建模」tab（不污染一级导航）
      ctx.slots.inject("settings.plugins.tab", function () {
        return ctx.slots.register(
          {
            name: "settings.plugins.tab",
            id: "math-modeling",
            order: 20,
            label: function () { return "数学建模"; },
            inject: function () { return { rpcCall: rpcCall }; },
          },
          SettingsToggle
        );
      });
    }

    return module.exports;
  }
});
