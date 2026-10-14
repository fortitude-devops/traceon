// Traceon — логика интерфейса. Без сборщиков и зависимостей, кроме vis-network.
"use strict";

const $ = (id) => document.getElementById(id);
const LEVEL_COLOR = { low: "var(--low)", medium: "var(--medium)", high: "var(--high)" };
const LEVEL_NAME = { low: "Низкий риск", medium: "Средний риск", high: "Высокий риск" };

// Пример подписи: тестовый ключ, созданный для демонстрации (на адресе нет средств)
const SAMPLE_SIG = {
  address: "bc1qu70n2hgxx66w2vuvae9g7w0e0mjevfh34y7s42",
  message: "Я, клиент №12345, подтверждаю владение этим адресом для проверки источника средств. 13.10.2026",
  signature: "J+IyFUSGeavwsYXCzvoC/RgKZo60DOolrtWyv/LJjgA28dMmh5XlU6M0Z7Yah/ZKSnyQtqW/PIEqteZbyWT2QbE=",
};

let network = null;
let graphData = null;          // DataSet графа, чтобы подсвечивать участников схемы
let baseNodes = [];            // исходные цвета узлов
let current = null;           // последний результат проверки
let verifiedOwnership = null; // подпись, прошедшая проверку

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = `Ошибка сервера (${res.status}).`;
    try { const body = await res.json(); if (body.detail) detail = typeof body.detail === "string" ? body.detail : detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

const fmt = (n, digits = 3) => (n === null || n === undefined) ? "—" : Number(n).toFixed(digits).replace(".", ",");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ---------- Состояние сервиса и модель ---------- */
async function loadStatus() {
  const el = $("status"), txt = $("status-text");
  try {
    const h = await api("/api/health");
    if (h.status !== "ok") {
      el.classList.add("bad");
      txt.textContent = h.model_error || (h.database ? "Сервис работает не полностью" : "База данных недоступна");
      return;
    }
    const m = await api("/api/model");
    el.classList.add("ok");
    txt.textContent = `Модель в работе · ${m.dataset}`;
    renderTicker(m, h);
    renderModel(m);
    loadDemo();
  } catch (e) {
    el.classList.add("bad");
    txt.textContent = "Сервер недоступен";
  }
}

function renderTicker(m, h) {
  const x = m.test.xgboost, glass = m.paper_baselines.GLASS;
  const top1 = m.top_k && m.top_k["1%"];
  const hi = m.levels_stats.high, med = m.levels_stats.medium;
  const caught = Math.round((hi.share_of_all_suspicious + med.share_of_all_suspicious) * 100);
  const items = [
    ["Схем в базе", h.subgraphs.toLocaleString("ru-RU"), ""],
    ["PR-AUC", fmt(x.pr_auc), "acc", glass ? `GLASS ${fmt(glass.pr_auc)}` : ""],
    ["ROC-AUC", fmt(x.roc_auc), ""],
    top1 ? ["Точность в топ-1%", `${fmt(top1.precision * 100, 1)}%`, "up", `×${fmt(top1.lift, 0)} к среднему`] : null,
    ["Находит подозрительных", `${caught}%`, "up", "в среднем и высоком риске"],
    ["Подозрительных в тесте", `${fmt(m.suspicious_share_test * 100, 1)}%`, ""],
    ["Признаков", String(m.n_features), ""],
  ].filter(Boolean);
  $("ticker").innerHTML = items.map(([k, v, cls, sub]) =>
    `<div class="t"><small>${esc(k)}</small><b class="${cls}">${esc(v)}</b>${sub ? `<i>${esc(sub)}</i>` : ""}</div>`).join("");
  $("ticker").hidden = false;
}

function renderModel(m) {
  const x = m.test.xgboost, lr = m.test.logistic_regression, glass = m.paper_baselines.GLASS;
  const hi = m.levels_stats.high, med = m.levels_stats.medium;
  const flagged = hi.count + med.count;
  const caught = Math.round((hi.share_of_all_suspicious + med.share_of_all_suspicious) * 100);
  const top1 = m.top_k && m.top_k["1%"];
  $("model-info").innerHTML = `
    <table>
      <thead><tr><th>Модель (тест)</th><th>PR-AUC</th><th>ROC-AUC</th></tr></thead>
      <tbody>
        <tr><td><b>XGBoost — Traceon</b></td><td>${fmt(x.pr_auc)}</td><td>${fmt(x.roc_auc)}</td></tr>
        <tr><td>Логистическая регрессия</td><td>${fmt(lr.pr_auc)}</td><td>${fmt(lr.roc_auc)}</td></tr>
        <tr><td>GLASS (авторы датасета)</td><td>${fmt(glass.pr_auc)}</td><td>${fmt(glass.roc_auc)}</td></tr>
      </tbody>
    </table>
    <p class="note">Тест: ${m.n_test.toLocaleString("ru-RU")} подграфов, подозрительных ${fmt(m.suspicious_share_test * 100, 1)}%.
    Проверяя вручную ${flagged.toLocaleString("ru-RU")} подграфов среднего и высокого риска, комплаенс находит ${caught}% подозрительных.
    ${top1 ? `Среди 1% самых рискованных подозрительных ${fmt(top1.precision * 100, 0)}% — в ${fmt(top1.lift, 1)} раза больше среднего.` : ""}
    Авторы датасета обучали GLASS без признаков кластеров, поэтому сравнение ориентировочное.</p>`;
}

async function loadDemo() {
  try {
    const demo = await api("/api/demo");
    if (!demo.length) return;
    $("chips").innerHTML = demo.map((d) =>
      `<button type="button" class="chip" style="--c:${LEVEL_COLOR[d.level]}" data-q="${esc(d.cl_id)}">
        <span>${esc(d.scheme || LEVEL_NAME[d.level])}</span><span class="cid">${esc(d.cl_id)}</span></button>`).join("");
    $("empty-examples").innerHTML = demo.map((d) =>
      `<button type="button" class="ex-card" style="--c:${LEVEL_COLOR[d.level]}" data-q="${esc(d.cl_id)}">
        <b>${esc(d.scheme || LEVEL_NAME[d.level])}</b><span>${esc(d.cl_id)}</span></button>`).join("");
    $("examples").hidden = false;
    document.querySelectorAll("#chips .chip, #empty-examples .ex-card").forEach((b) => b.addEventListener("click", () => {
      $("query").value = b.dataset.q;
      runCheck();
    }));
  } catch (_) { /* примеры необязательны */ }
}

/* ---------- Шаг 1: подпись ---------- */
function defaultMessage() {
  const d = new Date().toLocaleDateString("ru-RU");
  return `Я, клиент №[номер], подтверждаю владение этим адресом для проверки источника средств. ${d}`;
}

$("sig-sample").addEventListener("click", () => {
  $("sig-address").value = SAMPLE_SIG.address;
  $("sig-message").value = SAMPLE_SIG.message;
  $("sig-signature").value = SAMPLE_SIG.signature;
  $("sig-result").textContent = "";
});

$("sig-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const out = $("sig-result");
  const body = { address: $("sig-address").value.trim(), message: $("sig-message").value, signature: $("sig-signature").value.trim() };
  out.className = "sig-result";
  out.textContent = "Проверяю…";
  try {
    const r = await api("/api/verify-signature", { method: "POST", body: JSON.stringify(body) });
    if (r.valid) {
      verifiedOwnership = body;
      setSigState(true);
      out.classList.add("ok");
      out.textContent = `Подпись верна: клиент владеет адресом, тип адреса — ${r.address_type}. Подтверждение попадёт в отчёт.`;
    } else {
      verifiedOwnership = null;
      setSigState(false);
      out.classList.add("bad");
      out.textContent = r.error;
    }
  } catch (e) {
    out.classList.add("bad");
    out.textContent = e.message;
  }
});

/* ---------- Шаг 2: проверка ---------- */
$("check-form").addEventListener("submit", (ev) => { ev.preventDefault(); runCheck(); });

async function runCheck() {
  const q = $("query").value.trim();
  const err = $("check-error");
  err.textContent = "";
  if (!q) { err.textContent = "Введите ID кластера или подграфа."; $("query").focus(); return; }
  const btn = $("check-btn");
  btn.disabled = true;
  btn.textContent = "Проверяю…";
  $("result").classList.add("loading");
  try {
    const body = { query: q };
    if (verifiedOwnership) body.ownership = verifiedOwnership;
    const r = await api("/api/check", { method: "POST", body: JSON.stringify(body) });
    current = r;
    renderResult(r);
    loadGraph(r.cc_id, r.matched_by === "cluster" ? r.query : null);
    loadHistory();
  } catch (e) {
    err.textContent = e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Проверить";
    $("result").classList.remove("loading");
  }
}

const GROUP_CLASS = { hidden: "g-hidden", network: "g-network", structure: "g-structure" };

function factorItems(list, emptyText) {
  if (!list.length) return `<li class="empty-item">${emptyText}</li>`;
  return list.map((f) => {
    const meta = f.group
      ? `<span class="f-meta"><span class="f-group ${GROUP_CLASS[f.group]}">${esc(f.group_name)}</span>`
        + ` · влияние ${esc(f.strength)}, ${fmt(f.share * 100, 0)}% оценки</span>`
      : "";
    const proxy = f.proxy_note ? `<span class="proxy">${esc(f.proxy_note[0].toUpperCase() + f.proxy_note.slice(1))}</span>` : "";
    const hint = f.hint ? `<span class="hint">${esc(f.hint)}</span>` : "";
    return `<li>${meta}${esc(f.text)}${proxy}${hint}</li>`;
  }).join("");
}

function renderDrivers(groups) {
  const box = $("r-drivers");
  const list = (groups || []).filter((g) => g.share >= 0.005);
  box.hidden = !list.length;
  $("r-drivers-bar").innerHTML = list.map((g) =>
    `<span class="${GROUP_CLASS[g.key]}" style="flex-basis:${(g.share * 100).toFixed(1)}%"></span>`).join("");
  $("r-drivers-legend").innerHTML = list.map((g) =>
    `<li><span class="sw ${GROUP_CLASS[g.key]}"></span>${esc(g.name)} — <b>${fmt(g.share * 100, 0)}%</b></li>`).join("");
}

function renderResult(r) {
  $("empty").hidden = true;
  $("report").hidden = false;
  const color = LEVEL_COLOR[r.level];
  $("report").style.setProperty("--lvl", color);

  $("r-title").textContent = r.matched_by === "cluster" ? `Кластер ${r.query}` : `Подграф ${r.cc_id}`;
  $("r-meta").textContent = `Проверка №${r.check_id} · подграф ${r.cc_id}: ${r.subgraph.n_nodes} кластеров, ${r.subgraph.n_edges} переводов`;
  $("r-num").textContent = r.score;
  $("r-level").textContent = r.level_ru;
  requestAnimationFrame(() => { $("r-marker").style.left = `${r.score}%`; });

  const badges = [];
  if (r.signature && r.signature.valid) badges.push(`<span class="badge ok">Владение адресом подтверждено подписью</span>`);
  else if (r.signature) badges.push(`<span class="badge warn">Подпись не прошла проверку</span>`);
  else badges.push(`<span class="badge">Владение адресом не подтверждено</span>`);
  if (r.blacklisted) badges.push(`<span class="badge warn">Есть совпадение с чёрным списком</span>`);
  badges.push(`<span class="badge">Метка в датасете: ${esc(r.dataset_label)}</span>`);
  $("r-badges").innerHTML = badges.join("");

  $("r-conclusion").textContent = r.conclusion;
  $("r-conclusion-src").textContent = r.conclusion_source === "llm"
    ? "Текст сформулирован языковой моделью по результатам ML-модели."
    : "Текст собран по шаблону из результатов ML-модели.";
  const sl = $("r-scheme");
  sl.hidden = !r.scheme;
  sl.innerHTML = r.scheme ? `<span>Тип схемы:</span> ${esc(r.scheme)}` : "";
  $("r-patterns").innerHTML = (r.patterns || []).map((p, i) => {
    let stat = "";
    if (p.suspicious_share !== undefined && p.base_rate) {
      const ratio = p.suspicious_share / p.base_rate;
      const cmp = ratio >= 1.25 ? `в ${fmt(ratio, 1)} раза больше среднего`
        : ratio <= 0.8 ? `в ${fmt(1 / ratio, 1)} раза меньше среднего` : "на уровне среднего";
      stat = `<span class="pattern-stat">Среди таких схем в данных подозрительных ${fmt(p.suspicious_share * 100, 1)}% — ${cmp}</span>`;
    }
    const clickable = (p.nodes || []).length ? ` role="button" tabindex="0" data-i="${i}"` : "";
    return `<div class="pattern${clickable ? " clickable" : ""}${i === 0 && clickable ? " active" : ""}"${clickable}><b>${esc(p.name)}</b><span>${esc(p.description)}</span>${stat}</div>`;
  }).join("") + ((r.patterns || []).length
    ? `<span class="pattern-stat">Форма схемы сама по себе риск почти не определяет: скор рассчитан моделью по признакам кластеров и переводов.</span>`
    : "");
  document.querySelectorAll("#r-patterns .pattern.clickable").forEach((el) => {
    const pick = () => {
      document.querySelectorAll("#r-patterns .pattern").forEach((x) => x.classList.toggle("active", x === el));
      highlightPattern(r.patterns[+el.dataset.i], true);
    };
    el.addEventListener("click", pick);
    el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
  });
  renderDrivers(r.reasons.groups);
  $("c-risk").textContent = r.reasons.risk.length;
  $("c-mitigating").textContent = r.reasons.mitigating.length;
  selectTab("risk");
  $("r-risk").innerHTML = factorItems(r.reasons.risk, "Существенных факторов риска нет.");
  $("r-mitigating").innerHTML = factorItems(r.reasons.mitigating, "Смягчающих факторов нет.");
  const note = $("r-hidden-note");
  note.hidden = !r.hidden_note;
  note.textContent = r.hidden_note || "";

  $("r-pdf").href = `/api/report/${r.check_id}`;
  const bl = $("r-blacklist");
  bl.hidden = r.matched_by !== "cluster";
  bl.disabled = false;
  bl.textContent = "В чёрный список";
  $("report").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
}

$("r-blacklist").addEventListener("click", async () => {
  if (!current || current.matched_by !== "cluster") return;
  const btn = $("r-blacklist");
  btn.disabled = true;
  try {
    await api("/api/blacklist", { method: "POST", body: JSON.stringify({ cl_id: current.query, source: "внутренний список банка", reason: "внесён по итогам проверки" }) });
    btn.textContent = "В чёрном списке";
    $("query").value = current.query;
    runCheck();
  } catch (e) {
    btn.disabled = false;
    $("check-error").textContent = e.message;
  }
});

/* ---------- Граф ---------- */
async function loadGraph(ccId, focus) {
  const box = $("graph");
  if (!window.vis) { box.textContent = "Библиотека графа не загрузилась."; return; }
  try {
    const g = await api(`/api/graph/${encodeURIComponent(ccId)}${focus ? `?focus=${encodeURIComponent(focus)}` : ""}`);
    const nodes = g.nodes.map((n) => {
      const listed = n.blacklisted, isFocus = n.focus;
      return {
        id: n.id,
        label: n.label,
        title: `Кластер ${n.id}\nОтправляет: ${n.out} · получает: ${n.in}${listed ? "\nВ чёрном списке" : ""}`,
        shape: "dot",
        size: 12 + Math.min(n.out + n.in, 8) * 2,
        borderWidth: isFocus ? 3 : 2,
        shadow: isFocus ? { enabled: true, color: "rgba(240,180,41,0.55)", size: 18, x: 0, y: 0 } : false,
        color: {
          background: listed ? "#F6465D" : isFocus ? "#F0B429" : "#262B33",
          border: listed ? "#F6465D" : isFocus ? "#F0B429" : "#6B7584",
          highlight: { background: listed ? "#F6465D" : "#363C45", border: "#F0B429" },
          hover: { background: listed ? "#F6465D" : "#363C45", border: "#B7BDC6" },
        },
        font: { face: "SF Mono, Menlo, monospace", size: 11, color: "#B7BDC6", strokeWidth: 3, strokeColor: "#0B0E11" },
      };
    });
    const edges = g.edges.map((e) => ({ from: e.from, to: e.to, arrows: { to: { enabled: true, scaleFactor: 0.6 } } }));
    const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
    const options = {
      physics: { solver: "forceAtlas2Based", stabilization: { iterations: 200 }, forceAtlas2Based: { gravitationalConstant: -60, springLength: 110 } },
      edges: { color: { color: "#4A5260", highlight: "#F0B429", hover: "#848E9C" }, width: 1.4, smooth: { type: "continuous" } },
      interaction: { hover: true, tooltipDelay: 120, zoomView: true },
    };
    if (network) network.destroy();
    network = new vis.Network(box, data, options);
    graphData = data;
    baseNodes = nodes.map((n) => ({ id: n.id, color: n.color, borderWidth: n.borderWidth }));
    network.once("stabilizationIterationsDone", () => {
      network.fit({ animation: false });
      const first = current && (current.patterns || []).find((p) => (p.nodes || []).length);
      highlightPattern(first, false);
    });
  } catch (e) {
    box.textContent = e.message;
  }
}

/* ---------- Журнал ---------- */
async function loadHistory() {
  try {
    const rows = await api("/api/history?limit=15");
    if (!rows.length) return;
    $("history").innerHTML = rows.map((r) => `
      <tr>
        <td>${r.id}</td>
        <td>${new Date(r.created_at).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" })}</td>
        <td class="q" title="${esc(r.query)}">${esc(r.query)}</td>
        <td>${r.score}</td>
        <td class="lvl-${r.risk_level}">${esc(r.level_ru)}${r.blacklisted ? " · чёрный список" : ""}</td>
        <td>${r.signature_verified ? "подтверждена" : "—"}</td>
        <td><a class="link" href="/api/report/${r.id}" download>PDF</a></td>
      </tr>`).join("");
  } catch (_) { /* журнал необязателен для работы */ }
}

$("sig-message").value = defaultMessage();
loadStatus();
loadHistory();

/* ---------- Подсветка типовой схемы на графе ---------- */
const SCHEME_COLOR = "#8B7CF6";
function highlightPattern(p, scroll) {
  if (!graphData || !network) return;
  const set = new Set((p && p.nodes) || []);
  $("legend-scheme").hidden = !set.size;
  graphData.nodes.update(baseNodes.map((b) => {
    if (!set.has(b.id)) return b;
    const hub = p.hub === b.id;
    return {
      id: b.id,
      borderWidth: hub ? 5 : 3,
      color: { ...b.color, border: SCHEME_COLOR, background: hub ? SCHEME_COLOR : b.color.background,
               highlight: { background: SCHEME_COLOR, border: SCHEME_COLOR } },
    };
  }));
  graphData.edges.update(graphData.edges.get().map((e) => {
    const inSet = set.has(e.from) && set.has(e.to);
    const on = inSet && (p.code === "fan_out" ? e.from === p.hub
      : p.code === "fan_in" ? e.to === p.hub
      : p.hub ? (e.from === p.hub || e.to === p.hub) : true);
    return { id: e.id, color: on ? { color: SCHEME_COLOR } : { color: "#4A5260" }, width: on ? 2.6 : 1.4 };
  }));
  if (scroll) $("graph").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
}

/* ---------- Вкладки, подпись, горячая клавиша ---------- */
const TAB_GROUPS = { risk: ["risk", "mitigating"], mitigating: ["risk", "mitigating"], history: ["history", "model"], model: ["history", "model"] };
function selectTab(name) {
  (TAB_GROUPS[name] || []).forEach((n) => {
    const on = n === name;
    const t = $(`t-${n}`), panel = $(`tab-${n}`);
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", String(on));
    panel.hidden = !on;
  });
}
document.querySelectorAll(".tab[data-tab]").forEach((t) => t.addEventListener("click", () => selectTab(t.dataset.tab)));

function setSigState(ok) {
  const st = $("sig-state");
  st.textContent = ok ? "подтверждено" : "подпись не прошла";
  st.classList.toggle("ok", ok);
}

// «/» — к поиску, как в эксплорерах
document.addEventListener("keydown", (e) => {
  if (e.key !== "/" || e.metaKey || e.ctrlKey || e.altKey) return;
  const tag = (document.activeElement && document.activeElement.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA") return;
  e.preventDefault();
  $("query").focus();
  $("query").select();
});
