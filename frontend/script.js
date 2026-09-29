const API_BASE = "http://127.0.0.1:8000";
/* SarcasmLens AI: local FastAPI health, metrics, and paired comparison. */
(() => {
  "use strict";

  const CONFIG = {
    api: { health: "/health", metrics: "/metrics", compare: "/compare" },
    historyKey: "sarcasmlens.history.v1",
    historyLimit: 20,
  };

  const EXAMPLES = [
    { context: "I studied all night for the exam, but the paper covered a chapter we never touched. When I told my friend, he replied:", response: "Wow, great preparation strategy. Truly flawless." },
    { context: "My teammate asked me to move his entire project folder while I was already late for a meeting. I said I was busy, and he replied:", response: "Oh sure, take your time. I love waiting." },
    { context: "We planned a picnic all week. Right as we arrived, it started pouring. My sister said:", response: "Perfect weather for a picnic, isn't it?" },
  ];

  const $ = (id) => document.getElementById(id);
  const el = {
    form: $("analysis-form"), context: $("context"), response: $("response"),
    contextCount: $("context-count"), responseCount: $("response-count"),
    error: $("error"), button: $("analyze-button"), buttonText: $("button-text"),
    panel: $("result-panel"), empty: $("result-empty"), content: $("result-content"),
    label: $("prediction-label"), confidence: $("confidence"),
    bar: $("sarcasm-bar"), meter: $("probability-meter"),
    pSarcasm: $("sarcasm-probability"), pNot: $("not-sarcasm-probability"),
    connection: $("connection"),
    historyList: $("history-list"), historyEmpty: $("history-empty"), storageMsg: $("storage-message"),
  };

  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  const labelText = (isSarcasm) => (isSarcasm ? "Sarcasm detected" : "Not sarcastic");
  /* ---------- Tabs ---------- */
  const TABS = ["workspace", "performance", "history"];

  function showTab(name) {
    if (!TABS.includes(name)) name = "workspace";
    TABS.forEach((t) => {
      $(t).hidden = t !== name;
      const tab = $(`tab-${t}`);
      const active = t === name;
      tab.classList.toggle("nav-active", active);
      tab.setAttribute("aria-selected", active);
      tab.tabIndex = active ? 0 : -1;
    });
    if (name === "performance") { loadMetrics(); loadTypeMetrics(); }
    if (name === "history") renderHistory();
  }

  document.querySelector('nav[role="tablist"]').addEventListener("click", (e) => {
    const link = e.target.closest("a[role=tab]");
    if (!link) return;
    e.preventDefault();
    const name = link.id.replace("tab-", "");
    history.replaceState(null, "", `#${name}`);
    showTab(name);
  });

  document.querySelector('nav[role="tablist"]').addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    e.preventDefault();
    const current = TABS.findIndex((t) => $(`tab-${t}`).getAttribute("aria-selected") === "true");
    const next = (current + (e.key === "ArrowRight" ? 1 : -1) + TABS.length) % TABS.length;
    showTab(TABS[next]);
    history.replaceState(null, "", `#${TABS[next]}`);
    $(`tab-${TABS[next]}`).focus();
  });

  /* ---------- API helper ---------- */
  async function request(path, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 120000);
    try {
      const res = await fetch(`${API_BASE}${path}`, {
        ...options, headers: { "Content-Type": "application/json", ...options.headers }, signal: controller.signal,
      });
      const data = await res.json().catch(() => { throw new Error("The model server returned an unreadable response."); });
      if (!res.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map(item => `${item.loc.at(-1)}: ${item.msg}`).join(" ")
          : data.detail;
        throw new Error(detail || `Request failed (${res.status})`);
      }
      return data;
    } catch (err) {
      if (err.name === "AbortError") throw new Error("Analysis timed out. Please try again.");
      if (err instanceof TypeError) throw new Error("Cannot reach the backend at http://127.0.0.1:8000. Run start_backend.cmd and keep its window open.");
      throw err;
    } finally { clearTimeout(timer); }
  }

  function parsePrediction(data) {
    const { prediction, confidence, sarcasm_probability: sarcasm, not_sarcasm_probability: notSarcasm } = data;
    if (!["SARCASM", "NOT_SARCASM"].includes(prediction)
        || ![confidence, sarcasm, notSarcasm].every(v => Number.isFinite(v) && v >= 0 && v <= 1)
        || Math.abs(sarcasm + notSarcasm - 1) > 0.0001) {
      throw new Error("Unexpected response from the model server.");
    }
    return { isSarcasm: prediction === "SARCASM", confidence, sarcasm, notSarcasm, type: data.sarcasm_type };
  }

  /* ---------- Connection ---------- */
  async function checkConnection() {
    el.connection.className = "status";
    el.connection.lastChild.textContent = "Connecting";
    try {
      const health = await request(CONFIG.api.health);
      el.connection.classList.add(health.model_loaded ? "online" : "offline");
      el.connection.lastChild.textContent = health.model_loaded ? "Model online" : "Model not ready";
    } catch {
      el.connection.classList.add("offline");
      el.connection.lastChild.textContent = "Server offline";
    }
  }

  /* ---------- Inputs ---------- */
  const updateCounts = () => {
    el.contextCount.textContent = `${el.context.value.length.toLocaleString()} characters`;
    el.responseCount.textContent = `${el.response.value.length.toLocaleString()} characters`;
  };
  function inputsChanged() {
    updateCounts();
    el.content.hidden = true; el.empty.hidden = false;
    ["response-only", "context-aware"].forEach(prefix => {
      $(`${prefix}-result`).textContent = "Waiting for your conversation";
      $(`${prefix}-probability`).textContent = "Sarcasm probability appears after analysis.";
      $(`${prefix}-bar`).style.width = "0%";
    });
  }
  el.context.addEventListener("input", inputsChanged);
  el.response.addEventListener("input", inputsChanged);

  document.querySelectorAll(".example").forEach((btn) =>
    btn.addEventListener("click", () => {
      const ex = EXAMPLES[Number(btn.dataset.example)];
      el.context.value = ex.context;
      el.response.value = ex.response;
      inputsChanged();
      hideError();
      el.response.focus();
    })
  );

  const showError = (msg) => { el.error.textContent = msg; el.error.hidden = false; };
  const hideError = () => { el.error.hidden = true; };

  function setLoading(loading) {
    el.button.disabled = loading;
    el.buttonText.textContent = loading ? "Reading between the lines…" : "Analyze Conversation";
    el.panel.setAttribute("aria-busy", loading);
  }

  /* ---------- Rendering ---------- */
  function renderMain(r) {
    el.empty.hidden = true;
    el.content.hidden = false;
    el.label.textContent = labelText(r.isSarcasm);
    el.label.className = r.isSarcasm ? "is-sarcasm" : "is-literal";
    el.confidence.textContent = pct(r.confidence);
    const type = r.type;
    $("sarcasm-type-name").textContent = type?.status === "predicted" ? type.name : type?.status === "not_applicable" ? "Not applicable" : "Type model unavailable";
    $("sarcasm-type-confidence").textContent = type?.status === "predicted" ? `${pct(type.confidence)} type confidence · conditional on sarcasm` : "";
    $("sarcasm-type-note").textContent = type?.note || "Train the separate type model to enable this feature.";
    el.pSarcasm.textContent = pct(r.sarcasm);
    el.pNot.textContent = pct(r.notSarcasm);
    el.meter.setAttribute("aria-valuenow", Math.round(r.sarcasm * 100));
    requestAnimationFrame(() => { el.bar.style.width = pct(r.sarcasm); });
  }

  function renderComparison(prefix, r) {
    $(`${prefix}-result`).textContent = labelText(r.isSarcasm);
    $(`${prefix}-probability`).textContent = `Sarcasm probability: ${pct(r.sarcasm)}`;
    requestAnimationFrame(() => { $(`${prefix === "response-only" ? "response-only" : "context-aware"}-bar`).style.width = pct(r.sarcasm); });
  }

  /* ---------- Submit ---------- */
  el.form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError();
    const context = el.context.value.trim();
    const response = el.response.value.trim();
    if (!context || !response) {
      showError("Add both the conversation context and the target response.");
      return;
    }
    setLoading(true);
    el.content.hidden = true;
    el.empty.hidden = false;
    try {
      const comparison = await request(CONFIG.api.compare, { method: "POST", body: JSON.stringify({ context, response }) });
      const aware = parsePrediction(comparison.context_aware);
      const only = parsePrediction(comparison.response_only);
      if (el.context.value.trim() === context && el.response.value.trim() === response) {
        renderMain(aware);
        renderComparison("context-aware", aware);
        renderComparison("response-only", only);
      }
      saveHistory({ context, response, isSarcasm: aware.isSarcasm, confidence: aware.confidence, type: aware.type, time: Date.now() });
    } catch (err) {
      showError(`${err.message} Check that the model server is running, then try again.`);
      checkConnection();
    } finally {
      setLoading(false);
    }
  });

  /* ---------- Metrics ---------- */
  async function loadMetrics() {
    const msg = $("metrics-message"), grid = $("metrics-grid"), detail = $("metrics-detail");
    msg.hidden = false; msg.textContent = "Checking model performance…";
    grid.hidden = detail.hidden = true;
    try {
      const m = await request(CONFIG.api.metrics);
      ["macro_f1", "sarcasm_f1", "accuracy", "precision", "recall"].forEach((k) => {
        $(`metric-${k}`).textContent = typeof m[k] === "number" ? pct(m[k]) : "—";
      });
      msg.hidden = true; grid.hidden = false;
      if (m.dataset || m.validation_size) {
        detail.textContent = `Validation results for ${m.dataset ?? "held-out data"}${m.validation_size ? ` (${m.validation_size.toLocaleString()} examples)` : ""}.`;
        detail.hidden = false;
      }
    } catch {
      msg.textContent = "Metrics are unavailable. Train and evaluate the model, then refresh.";
    }
  }
  async function loadTypeMetrics() {
    try {
      const m = await request("/type-metrics");
      $("type-metrics-message").textContent = `Type Macro F1: ${pct(m.macro_f1)} · Accuracy: ${pct(m.accuracy)} · ${m.validation_size} labelled sarcastic validation examples. ${m.evaluation_note} ${m.note}`;
    } catch {
      $("type-metrics-message").textContent = "Train the separate sarcasm-type model to view its performance.";
    }
  }
  $("refresh-status").addEventListener("click", () => { checkConnection(); loadMetrics(); loadTypeMetrics(); });

  /* ---------- History (localStorage, fails gracefully) ---------- */
  function readHistory() {
    try { const items = JSON.parse(localStorage.getItem(CONFIG.historyKey)); return Array.isArray(items) ? items.filter(it => it && typeof it === "object") : []; } catch { return []; }
  }
  function writeHistory(items) {
    try { localStorage.setItem(CONFIG.historyKey, JSON.stringify(items)); el.storageMsg.hidden = true; return true; }
    catch { el.storageMsg.textContent = "Browser storage is unavailable, so history won't be saved."; el.storageMsg.hidden = false; return false; }
  }
  const saveHistory = (item) => writeHistory([item, ...readHistory()].slice(0, CONFIG.historyLimit));

  function renderHistory() {
    const items = readHistory();
    el.historyEmpty.hidden = items.length > 0;
    el.historyList.replaceChildren(...items.map((it) => {
      const card = document.createElement("article");
      card.className = "history-item";
      const head = document.createElement("header");
      const tag = document.createElement("span");
      tag.className = `tag ${it.isSarcasm ? "is-sarcasm" : "is-literal"}`;
      tag.textContent = it.manualOnly ? "Manual style review · No model prediction" : `${labelText(it.isSarcasm)} · ${pct(it.confidence)}`;
      const time = document.createElement("time");
      time.textContent = new Date(it.time).toLocaleString();
      head.append(tag, time);
      const ctx = document.createElement("p"); ctx.className = "ctx"; ctx.textContent = it.context;
      const res = document.createElement("p"); res.textContent = `“${it.response}”`;
      card.append(head, ctx, res);
      if (it.manualStyles?.length) {
        const styles = document.createElement("p"); styles.textContent = `Your labels: ${it.manualStyles.join(" · ")}`; card.append(styles);
      }
      if (it.type) {
        const type = document.createElement("p");
        type.textContent = it.type.status === "predicted" ? `Estimated type: ${it.type.name} (${pct(it.type.confidence)})` : "Type: " + (it.type.status === "not_applicable" ? "Not applicable" : "Unavailable");
        card.append(type);
      }
      return card;
    }));
  }

  $("clear-history").addEventListener("click", () => { writeHistory([]); renderHistory(); });

  window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

  /* ---------- Init ---------- */
  updateCounts();
  showTab(location.hash.slice(1));
  checkConnection();
})();
