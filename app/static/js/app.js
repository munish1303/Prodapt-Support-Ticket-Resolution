/* Resolve · agent console. Plain JS, no build step. All API text is escaped before it touches the DOM. */
(() => {
  "use strict";

  const API = "/api/v1";
  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const reduced =
    window.matchMedia("(prefers-reduced-motion: reduce)").matches || new URLSearchParams(location.search).has("static");
  if (reduced) document.documentElement.classList.add("no-motion");
  const esc = (v) =>
    String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const pretty = (s) => String(s ?? "").replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
  const pct = (x) => `${Math.round((x ?? 0) * 100)}%`;
  const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);
  const fmtN = (n) => Number(n ?? 0).toLocaleString("en-GB");

  const EXAMPLES = [
    { label: "Red LOS light on the fibre box", text: "The fibre box on the wall has a red LOS light and we have had no internet since this morning. I work from home." },
    { label: "Charged twice", text: "I was charged twice for this month's bill, the same amount left my bank account two times." },
    { label: "Wi-Fi drops every evening", text: "My broadband drops every evening around 8 and I've already restarted the router twice, I work from home and this is costing me." },
    { label: "No mobile data abroad", text: "I'm on holiday in Spain and my mobile data doesn't work at all, the phone shows a Spanish network." },
    { label: "Strict NAT on my console", text: "My PlayStation says NAT type strict and I cannot join online games with my friends." },
  ];
  const STAGES = ["embed", "understand_retrieve", "generate", "validate", "decide"];
  const STAGE_LABEL = { embed: "embed", understand_retrieve: "understand + retrieve", generate: "generate", validate: "validate", decide: "decide" };
  const FLAG = {
    unknown_intent: ["warn", "Unrecognised issue type"],
    possible_prompt_injection: ["bad", "Instruction-like text in complaint"],
    llm_unavailable_extractive_fallback: ["warn", "LLM unavailable: extractive draft"],
    generation_error: ["bad", "Generation error"],
    contradiction_detected: ["bad", "A step contradicts its source"],
    low_confidence: ["warn", "Low confidence"],
    retrieval_unavailable: ["bad", "Knowledge base unavailable"],
    embedding_unavailable: ["bad", "Complaint could not be processed"],
    understanding_unavailable: ["warn", "Understanding unavailable"],
    validation_unavailable: ["warn", "Steps not validated"],
  };
  const SUPPORT = {
    supported: ["✓", "Supported"],
    weakly_supported: ["≈", "Weak support"],
    unsupported: ["✕", "Unsupported"],
    contradicted: ["✕", "Contradicted"],
    uncited: ["?", "No citation"],
    unverified: ["?", "Not validated"],
  };
  const DECISION_ICON = { RESOLVE: "✓", REVIEW: "◐", ESCALATE: "▲" };

  let lastComplaint = "";
  let stageTimer = null;
  let progressTimer = null;

  async function api(path, opts = {}) {
    const res = await fetch(API + path, { headers: { "Content-Type": "application/json" }, ...opts });
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON */ }
    if (!res.ok) {
      const detail = body && body.detail;
      const msg = Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail || `HTTP ${res.status}`;
      throw new Error(msg);
    }
    return body;
  }

  // Grow bars/meters from 0 to their data-w width. Reading offsetWidth flushes layout so the 0 width is committed
  // first and the CSS transition always runs (deterministic, unlike chaining requestAnimationFrame/setTimeout).
  function fillBars(root) {
    const els = $$("[data-w]", root);
    els.forEach((el) => void el.offsetWidth);
    els.forEach((el) => (el.style.width = el.dataset.w));
  }

  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("is-on");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => t.classList.remove("is-on"), 4200);
  }

  /* ---------------- headline letter reveal ---------------- */
  function splitHeadlines(root = document) {
    $$("[data-split]", root).forEach((line, li) => {
      if (line.dataset.done) return;
      const text = line.textContent;
      line.textContent = "";
      [...text].forEach((ch, i) => {
        const s = document.createElement("span");
        s.className = ch === " " ? "char char--space" : "char";
        s.textContent = ch === " " ? " " : ch;
        s.style.animationDelay = `${0.15 + li * 0.14 + i * 0.028}s`;
        line.appendChild(s);
      });
      line.dataset.done = "1";
    });
  }

  function replayHeadline(root) {
    $$(".char", root).forEach((c) => { c.style.animation = "none"; void c.offsetWidth; c.style.animation = ""; });
  }

  /* ---------------- parallax on the line-work ---------------- */
  window.addEventListener("mousemove", (e) => {
    if (reduced) return;
    const x = (e.clientX / window.innerWidth - 0.5) * -14;
    const y = (e.clientY / window.innerHeight - 0.5) * -10;
    document.documentElement.style.setProperty("--px", `${x.toFixed(1)}px`);
    document.documentElement.style.setProperty("--py", `${y.toFixed(1)}px`);
  }, { passive: true });

  /* ---------------- health ---------------- */
  async function loadHealth() {
    const pill = $("#status");
    try {
      const h = await api("/health");
      const ok = h.status === "healthy";
      pill.classList.toggle("is-ok", ok);
      pill.classList.toggle("is-bad", !ok);
      const llm = h.llm_provider.startsWith("configured") ? h.llm_provider.replace("configured ", "LLM ") : "extractive mode";
      $(".status__text", pill).textContent = `${fmtN(h.corpus.tickets)} tickets · ${llm}`;
      $("#corpusLabel").textContent = `${fmtN(h.corpus.tickets)} tickets · ${h.corpus.kb_articles} KB articles`;
    } catch {
      pill.classList.add("is-bad");
      $(".status__text", pill).textContent = "API unreachable";
    }
  }

  /* ---------------- views & states ---------------- */
  async function setView(view) {
    if (document.body.dataset.view === view) return;
    document.body.dataset.view = view;
    $$("[data-view-link]").forEach((b) => b.classList.toggle("is-active", b.dataset.viewLink === view));
    const el = view === "kb" ? $("#viewKb") : $("#viewConsole");
    el.classList.remove("is-entering"); void el.offsetWidth; el.classList.add("is-entering");
    window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
    if (view === "kb") { replayHeadline(el); loadKb(); }
  }

  async function setState(next) {
    const body = document.body;
    const current = body.dataset.state;
    if (current === next) return;
    const outgoing = { idle: [$("#hero"), $("#inputCard")], searching: [$("#searchCard")], result: [$("#resultCard")] }[current] || [];
    if (!reduced) {
      outgoing.forEach((el) => el && el.classList.add("is-leaving"));
      await sleep(330);
      outgoing.forEach((el) => el && el.classList.remove("is-leaving"));
    }
    body.dataset.state = next;
    const incoming = { idle: [$("#inputCard")], searching: [$("#searchCard")], result: [$("#resultCard")] }[next] || [];
    incoming.forEach((el) => { el.classList.remove("is-arriving"); void el.offsetWidth; el.classList.add("is-arriving"); });
    if (next === "idle") replayHeadline($("#hero"));
    window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
  }

  /* ---------------- composer ---------------- */
  const ta = () => $("#complaint");
  function autosize() {
    const t = ta();
    t.style.height = "auto";
    t.style.height = `${Math.min(t.scrollHeight, 260)}px`;
    $("#charCount").textContent = `${t.value.length} / 5000`;
  }

  async function typeInto(text) {
    const t = ta();
    t.value = "";
    if (reduced) { t.value = text; autosize(); return; }
    const step = Math.max(1, Math.round(text.length / 60));
    for (let i = 0; i < text.length; i += step) {
      t.value = text.slice(0, i + step);
      autosize();
      await sleep(12);
    }
  }

  function renderExamples() {
    $("#examples").innerHTML = EXAMPLES.map(
      (e, i) => `<button type="button" class="example" data-i="${i}" style="animation-delay:${0.9 + i * 0.07}s">${esc(e.label)}</button>`
    ).join("");
    $$(".example").forEach((b) =>
      b.addEventListener("click", async () => {
        await typeInto(EXAMPLES[Number(b.dataset.i)].text);
        submit();
      })
    );
  }

  function ripple(btn, e) {
    const r = document.createElement("span");
    const rect = btn.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    r.className = "ripple";
    r.style.width = r.style.height = `${size}px`;
    r.style.left = `${(e?.clientX ?? rect.left + rect.width / 2) - rect.left - size / 2}px`;
    r.style.top = `${(e?.clientY ?? rect.top + rect.height / 2) - rect.top - size / 2}px`;
    btn.appendChild(r);
    setTimeout(() => r.remove(), 650);
  }

  /* ---------------- searching scene ---------------- */
  function startSearch() {
    const stack = $("#ticketStack");
    stack.innerHTML = [0, 1, 2, 3]
      .map((i) => `<div class="ticket is-shuffling" style="--y:${i * 30}px;transform:translateY(${i * 30}px);animation-delay:${i * 0.3}s">TKT-······</div>`)
      .join("");
    $("#linkLabel").textContent = "searching past queries";
    $$("#stages li").forEach((li) => { li.classList.remove("is-active", "is-done"); $(".stages__ms", li).textContent = ""; });
    let idx = 0;
    const activate = () => {
      $$("#stages li").forEach((li, i) => li.classList.toggle("is-active", i === idx));
    };
    activate();
    // While waiting, the highlighted stage is an estimate; real per-stage timings replace it on response.
    stageTimer = setInterval(() => { if (idx < 3) { idx += 1; activate(); } }, 1300);
    const bar = $("#progressBar");
    let p = 6;
    bar.style.width = `${p}%`;
    progressTimer = setInterval(() => { p += (88 - p) * 0.06; bar.style.width = `${p}%`; }, 250);
  }

  async function finishSearch(data) {
    clearInterval(stageTimer);
    clearInterval(progressTimer);
    $("#progressBar").style.width = "100%";
    $("#linkLabel").textContent = `found ${data.sources.length} relevant sources`;
    STAGES.forEach((s) => {
      const li = $(`#stages li[data-stage="${s}"]`);
      li.classList.remove("is-active");
      li.classList.add("is-done");
      const ms = data.stage_latency_ms?.[s];
      $(".stages__ms", li).textContent = ms != null ? fmtMs(ms) : "";
    });
    const stack = $("#ticketStack");
    stack.innerHTML = "";
    data.sources.slice(0, 4).forEach((s, i) => {
      const d = document.createElement("div");
      d.className = "ticket is-real";
      d.style.setProperty("--y", `${i * 30}px`);
      d.style.animationDelay = `${i * 0.12}s`;
      d.textContent = s.id;
      stack.appendChild(d);
    });
    await sleep(reduced ? 0 : 1100);
  }

  /* ---------------- submit ---------------- */
  async function submit(e) {
    if (e) e.preventDefault();
    const text = ta().value.trim();
    if (text.length < 10) { toast("Please paste a complaint of at least 10 characters."); ta().focus(); return; }
    $("#sendBtn").disabled = true;
    lastComplaint = text;
    await setState("searching");
    startSearch();
    try {
      const data = await api("/tickets/resolve", { method: "POST", body: JSON.stringify({ complaint: text }) });
      await finishSearch(data);
      renderResult(text, data);
      await setState("result");
      animateResult(data);
    } catch (err) {
      clearInterval(stageTimer);
      clearInterval(progressTimer);
      toast(`Could not resolve: ${err.message}`);
      await setState("idle");
    } finally {
      $("#sendBtn").disabled = false;
    }
  }

  /* ---------------- result ---------------- */
  const CITE_RE = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  function parseStep(step) {
    const nums = [];
    step.replace(CITE_RE, (_, g) => { g.split(",").forEach((n) => nums.push(Number(n.trim()))); return ""; });
    return { text: step.replace(CITE_RE, "").replace(/\s{2,}/g, " ").trim(), cites: [...new Set(nums)] };
  }

  function renderResult(complaint, d) {
    $("#rComplaint").textContent = complaint;
    const u = d.understanding;
    const chips = [
      `<span class="chip" style="animation-delay:.15s"><small>Intent</small><b>${esc(pretty(u.intent))}</b><span class="chip__bar" title="vote share ${pct(u.intent_confidence)}"><i data-w="${pct(u.intent_confidence)}"></i></span></span>`,
      ...u.products.map((p, i) => `<span class="chip" style="animation-delay:${0.22 + i * 0.06}s"><small>Product</small><b>${esc(pretty(p))}</b></span>`),
      `<span class="chip" style="animation-delay:.4s"><small>Severity</small><b>${esc(pretty(u.severity))}</b></span>`,
      `<span class="chip" style="animation-delay:.46s"><small>Sentiment</small><b>${esc(pretty(u.sentiment))}</b></span>`,
    ];
    $("#rChips").innerHTML = chips.join("");

    const dec = d.decision.action;
    const stamp = $("#rStamp");
    stamp.className = `stamp stamp--${esc(dec)}`;
    stamp.innerHTML = `<span aria-hidden="true">${DECISION_ICON[dec] || "•"}</span>${esc(dec)}`;
    $("#rReason").textContent = d.decision.reason;
    $("#rScore").textContent = "0";
    $("#rRingValue").style.strokeDashoffset = "100";

    $("#rFlags").innerHTML = d.flags
      .map((f, i) => {
        const [kind, label] = FLAG[f] || ["warn", pretty(f)];
        return `<span class="flag flag--${kind}" style="animation-delay:${0.5 + i * 0.08}s"><span aria-hidden="true">${kind === "bad" ? "▲" : "●"}</span>${esc(label)}</span>`;
      })
      .join("");

    const v = d.validation;
    const gen = d.resolution.generator.replace(/^llm:/, "");
    const genText = gen === "extractive" ? "extractive draft (no LLM)" : gen === "none" ? "no draft" : `drafted by ${gen}`;
    $("#rGenerator").textContent = `${genText} · groundedness ${pct(v.groundedness_score)} · citations ${pct(v.citation_accuracy)} valid`;

    const steps = d.resolution.steps;
    const list = $("#rSteps");
    list.innerHTML = "";
    if (!steps.length) {
      list.innerHTML = `<li class="declined"><b>The assistant declined to draft steps.</b><br />${esc(d.resolution.summary || "The retrieved sources do not address this complaint.")}</li>`;
    }
    steps.forEach((raw, i) => {
      const { text, cites } = parseStep(raw);
      const claim = v.claims[i] || {};
      const status = claim.support_status || "uncited";
      const [icon, label] = SUPPORT[status] || ["?", pretty(status)];
      const sig = claim.signals || {};
      const tip = sig.quoted
        ? "Quoted verbatim from the cited source"
        : `similarity ${(sig.semantic ?? 0).toFixed(2)} · entailment ${(sig.entailment ?? 0).toFixed(2)} · contradiction ${(sig.contradiction ?? 0).toFixed(2)} · word overlap ${(sig.lexical ?? 0).toFixed(2)}`;
      const li = document.createElement("li");
      li.className = "step";
      li.dataset.cites = cites.join(",");
      li.innerHTML = `<span class="step__n" aria-hidden="true"></span>
        <div class="step__text"><span class="step__typed" data-full="${esc(text)}"></span><span class="cites">${cites
          .map((n) => `<button type="button" class="cite" data-n="${n}" aria-label="Source ${n}">${n}</button>`)
          .join("")}</span></div>
        <span class="support support--${esc(status)}" title="${esc(tip)}"><span aria-hidden="true">${icon}</span>${esc(label)}</span>`;
      list.appendChild(li);
    });
    const est = d.resolution.estimated_time ? ` · estimated time ${d.resolution.estimated_time}` : "";
    $("#rSummary").textContent = steps.length ? `${d.resolution.summary || ""}${est}` : "";

    const comp = d.confidence.components;
    const names = { retrieval_quality: "Retrieval quality", evidence_quality: "Evidence quality", understanding_quality: "Understanding", sufficiency_score: "Evidence sufficiency" };
    $("#rBreakdown").innerHTML = Object.keys(names)
      .map((k) => `<div class="meter"><div class="meter__top"><span>${names[k]}</span><span>${(comp[k] ?? 0).toFixed(2)}</span></div><div class="meter__track"><div class="meter__fill" data-w="${pct(comp[k])}"></div></div></div>`)
      .join("");

    $("#rSourceCount").textContent = `${d.sources.length} sources · scroll for all`;
    const citedBy = {};
    steps.forEach((raw, i) => parseStep(raw).cites.forEach((n) => (citedBy[n] = [...(citedBy[n] || []), i + 1])));
    $("#rSources").innerHTML = d.sources
      .map((s, i) => {
        const ranks = [s.semantic_rank ? `semantic #${s.semantic_rank}` : null, s.lexical_rank ? `lexical #${s.lexical_rank}` : null].filter(Boolean).join(" · ") || "KB slot";
        const by = citedBy[s.source_number] ? `cited by step ${citedBy[s.source_number].join(", ")}` : "not cited";
        return `<article class="source" id="src-${s.source_number}" data-n="${s.source_number}" style="animation-delay:${0.25 + i * 0.07}s">
          <div class="source__head"><span class="badge">${s.source_number}</span><span class="tag tag--${esc(s.type)}">${s.type === "kb_article" ? "KB article" : "Ticket"}</span>
            <span class="mono">${esc(s.id)}</span>
            <span class="source__rel" title="cosine similarity to the complaint"><span class="relbar"><i data-w="${pct(Math.max(0, s.relevance_score))}"></i></span>${s.relevance_score.toFixed(2)}</span></div>
          ${s.title ? `<div class="mono" style="margin-top:8px;color:var(--ink)">${esc(s.title)}</div>` : ""}
          <p class="source__excerpt">${esc(s.excerpt)}</p>
          <div class="source__foot"><span>${esc(ranks)} · ${esc(by)}</span><button type="button" class="link-btn" data-open="${esc(s.type)}|${esc(s.id)}">Open in database ↗</button></div>
        </article>`;
      })
      .join("");

    const total = STAGES.reduce((a, s) => a + (d.stage_latency_ms?.[s] || 0), 0) || 1;
    $("#rTimeline").innerHTML = STAGES.map((s) => {
      const ms = d.stage_latency_ms?.[s] || 0;
      return `<span class="seg" data-w="${((ms / total) * 100).toFixed(1)}%" title="${STAGE_LABEL[s]}: ${fmtMs(ms)}">${ms / total > 0.14 ? `${STAGE_LABEL[s]} ${fmtMs(ms)}` : ""}</span>`;
    }).join("");
    $("#rTimeline").title = `total ${fmtMs(d.latency_ms)}`;

    const rate = $("#rFeedback .rate");
    rate.innerHTML = [1, 2, 3, 4, 5].map((n) => `<button type="button" role="radio" aria-label="${n}" data-r="${n}">${n}</button>`).join("");
    rate.dataset.request = d.request_id || "";
    wireResult();
  }

  async function animateResult(d) {
    await sleep(reduced ? 0 : 150);
    fillBars($("#resultCard"));
    const score = d.confidence.heuristic_score;
    $("#rRingValue").style.strokeDashoffset = String(100 - Math.round(score * 100));
    countUp($("#rScore"), Math.round(score * 100), 1300);
    setTimeout(() => $("#rStamp").classList.add("is-in"), reduced ? 0 : 900);
    const steps = $$(".step", $("#rSteps"));
    for (const [i, li] of steps.entries()) {
      await sleep(reduced ? 0 : i === 0 ? 350 : 120);
      li.classList.add("is-in");
      await typeStep($(".step__typed", li));
    }
  }

  async function typeStep(el) {
    if (!el) return;
    const full = el.dataset.full;
    if (reduced) { el.textContent = full; return; }
    const caret = document.createElement("span");
    caret.className = "caret";
    el.after(caret);
    const dur = Math.min(650, full.length * 9);
    const chunk = Math.max(1, Math.ceil(full.length / (dur / 16)));
    for (let i = 0; i <= full.length; i += chunk) { el.textContent = full.slice(0, i); await sleep(16); }
    el.textContent = full;
    caret.remove();
  }

  function countUp(el, to, ms) {
    if (reduced) { el.textContent = String(to); return; }
    const t0 = performance.now();
    const tick = (t) => {
      const k = Math.min(1, (t - t0) / ms);
      el.textContent = String(Math.round(to * (1 - Math.pow(1 - k, 3))));
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function highlightSource(n, scroll) {
    $$(".source").forEach((s) => s.classList.toggle("is-hot", s.dataset.n === String(n)));
    $$(".cite").forEach((c) => c.classList.toggle("is-hot", c.dataset.n === String(n)));
    const target = $(`#src-${n}`);
    if (scroll && target) {
      const box = $("#rSources");
      box.scrollTo({ top: target.offsetTop - box.offsetTop - 6, behavior: reduced ? "auto" : "smooth" });
    }
  }

  function wireResult() {
    $$(".cite").forEach((c) => {
      c.addEventListener("mouseenter", () => highlightSource(c.dataset.n, true));
      c.addEventListener("click", () => highlightSource(c.dataset.n, true));
    });
    $$(".source").forEach((s) => {
      s.addEventListener("mouseenter", () => $$(".step").forEach((st) => st.classList.toggle("is-linked", st.dataset.cites.split(",").includes(s.dataset.n))));
      s.addEventListener("mouseleave", () => $$(".step").forEach((st) => st.classList.remove("is-linked")));
    });
    $$("[data-open]", $("#resultCard")).forEach((b) => b.addEventListener("click", () => { const [t, id] = b.dataset.open.split("|"); openRecord(t, id); }));
    $$("#rFeedback .rate button").forEach((b) =>
      b.addEventListener("click", async () => {
        const n = Number(b.dataset.r);
        $$("#rFeedback .rate button").forEach((x) => x.classList.toggle("is-on", Number(x.dataset.r) <= n));
        const id = $("#rFeedback .rate").dataset.request;
        if (!id) { toast("This request was not logged, so feedback cannot be stored."); return; }
        try { await api(`/tickets/${id}/feedback`, { method: "POST", body: JSON.stringify({ rating: n }) }); toast(`Thanks: rated ${n}/5 and stored for confidence calibration.`); }
        catch (err) { toast(`Feedback failed: ${err.message}`); }
      })
    );
  }

  /* ---------------- drawers ---------------- */
  function openDrawer(id) { const d = $(`#${id}`); d.classList.add("is-open"); d.setAttribute("aria-hidden", "false"); }
  function closeDrawer(id) { const d = $(`#${id}`); d.classList.remove("is-open"); d.setAttribute("aria-hidden", "true"); }

  async function openRecord(type, id) {
    $("#recordTitle").textContent = id;
    $("#recordBody").innerHTML = `<p class="muted">Loading from PostgreSQL…</p>`;
    openDrawer("record");
    try {
      const r = await api(type === "kb_article" ? `/corpus/kb/${encodeURIComponent(id)}` : `/corpus/tickets/${encodeURIComponent(id)}`);
      const field = (label, html, delay) => `<div class="field" style="animation-delay:${delay}s"><div class="field__label">${label}</div>${html}</div>`;
      const kv = (obj) => `<dl class="kv">${Object.entries(obj).filter(([, v]) => v !== null && v !== undefined && v !== "").map(([k, v]) => `<dt>${esc(pretty(k))}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`;
      const parts = [];
      parts.push(`<p class="muted small" style="margin-top:0">Row read live from the <span class="mono">${type === "kb_article" ? "kb_articles" : "tickets"}</span> table.</p>`);
      if (r.type === "ticket") {
        parts.push(field("Complaint", `<div class="field__text">${esc(r.complaint)}</div>`, 0.05));
        parts.push(field("Resolution", `<div class="field__text">${esc(r.resolution || "—")}</div>`, 0.1));
        parts.push(field("Labels", kv({ category: pretty(r.category), product: pretty(r.product), severity: r.severity, sentiment: r.sentiment, created_at: r.created_at, resolved_at: r.resolved_at }), 0.15));
      } else {
        parts.push(field("Title", `<div class="field__text"><b>${esc(r.title)}</b></div>`, 0.05));
        parts.push(field("Content", `<div class="field__text">${esc(r.content)}</div>`, 0.1));
        parts.push(field("Labels", kv({ category: pretty(r.category), product: pretty(r.product), version: r.version, archived_versions: r.archived_versions, updated_at: r.updated_at }), 0.15));
      }
      parts.push(field("Provenance", kv(r.provenance || {}), 0.2));
      if (r.embedding) {
        const max = Math.max(...r.embedding.preview.map(Math.abs)) || 1;
        const bars = r.embedding.preview.map((x) => `<i class="${x >= 0 ? "pos" : "neg"}" style="--h:${(Math.abs(x) / max).toFixed(3)}" title="${x}"></i>`).join("");
        parts.push(field(`Stored embedding · ${r.embedding.dims}-dimensional · L2 norm ${r.embedding.l2_norm}`,
          `<div class="vec" aria-label="first ${r.embedding.preview.length} dimensions">${bars}</div><p class="muted small">First ${r.embedding.preview.length} of ${r.embedding.dims} values (red positive, dark negative). This vector is what pgvector compares against the complaint.</p>`, 0.25));
      }
      parts.push(field("Metadata (JSONB)", `<pre class="pre">${esc(JSON.stringify(r.metadata, null, 2))}</pre>`, 0.3));
      $("#recordBody").innerHTML = parts.join("");
    } catch (err) {
      $("#recordBody").innerHTML = `<p class="alert">${esc(err.message)}</p>`;
    }
  }

  async function openInsights() {
    openDrawer("insights");
    const body = $("#insightsBody");
    body.innerHTML = `<p class="muted">Loading…</p>`;
    try {
      const [m, d] = await Promise.all([api("/metrics"), api("/monitoring/drift")]);
      const total = m.requests_total || 0;
      const dec = ["RESOLVE", "REVIEW", "ESCALATE"].map((k) => ({ k, n: m.requests_by_decision?.[k] || 0 }));
      body.innerHTML = `
        <div class="metric-row">
          <div class="metric"><b>${fmtN(total)}</b><span>requests logged</span></div>
          <div class="metric"><b>${m.latency_ms?.p50 != null ? fmtMs(m.latency_ms.p50) : "–"}</b><span>p50 latency</span></div>
          <div class="metric"><b>${m.latency_ms?.p95 != null ? fmtMs(m.latency_ms.p95) : "–"}</b><span>p95 latency</span></div>
          <div class="metric"><b>${m.avg_confidence != null ? m.avg_confidence.toFixed(2) : "–"}</b><span>mean confidence</span></div>
          <div class="metric"><b>${m.avg_groundedness != null ? pct(m.avg_groundedness) : "–"}</b><span>mean groundedness</span></div>
          <div class="metric"><b>${m.llm_fallback_rate != null ? pct(m.llm_fallback_rate) : "–"}</b><span>LLM fallback rate</span></div>
        </div>
        <div class="field"><div class="field__label">Decision mix</div><div class="bars">${dec
          .map(({ k, n }) => `<div class="bar"><span class="bar__label">${DECISION_ICON[k]} ${k}</span><span class="bar__track"><span class="bar__fill" data-w="${total ? ((n / total) * 100).toFixed(1) : 0}%"></span></span><span class="bar__value">${n}</span><span class="tip">${k}: ${n} of ${total}</span></div>`)
          .join("")}</div></div>
        <div class="field"><div class="field__label">Drift (last ${d.window_hours} h vs before)</div>
          ${d.alerts.length ? d.alerts.map((a) => `<div class="alert">▲ ${esc(a)}</div>`).join("") : `<div class="ok-note">✓ No drift alerts</div>`}
          <dl class="kv" style="margin-top:12px">
            <dt>Recent requests</dt><dd>${d.recent.n ?? 0}</dd>
            <dt>Unknown-intent rate</dt><dd>${d.recent.unknown_intent_rate != null ? pct(d.recent.unknown_intent_rate) : "–"}</dd>
            <dt>Intent-mix shift (TVD)</dt><dd>${d.intent_tvd ?? "–"}${d.intent_tvd_alert_threshold ? ` (alert above ${d.intent_tvd_alert_threshold})` : ""}</dd>
            <dt>Mean nearest similarity</dt><dd>${d.recent.mean_nearest_similarity ?? "–"}</dd>
          </dl></div>`;
      fillBars(body);
    } catch (err) {
      body.innerHTML = `<p class="alert">${esc(err.message)}</p>`;
    }
  }

  /* ---------------- knowledge base ---------------- */
  let kbLoaded = false;
  let ticketPage = 0;
  const PAGE = 20;

  async function loadKb() {
    try {
      const s = await api("/corpus/stats");
      const ivf = s.indexes.filter((i) => /ivfflat|hnsw/i.test(i.definition)).length;
      const tiles = [
        [s.tickets.total, "historical tickets"],
        [s.kb_articles.total, "KB articles"],
        [s.embedding.dims, "embedding dimensions"],
        [ivf, "vector (ANN) indexes"],
        [s.requests_logged, "resolutions logged"],
      ];
      $("#kbTiles").innerHTML = tiles
        .map(([v, l], i) => `<div class="tile" style="animation-delay:${0.1 + i * 0.07}s"><div class="tile__value" data-count="${v}">0</div><div class="tile__label">${l}</div></div>`)
        .join("");
      $$("[data-count]").forEach((el) => countUp(el, Number(el.dataset.count), 1100));

      const max = Math.max(...s.categories.map((c) => c.count), 1);
      $("#kbCatMeta").textContent = `${s.categories.length} classes`;
      $("#kbCats").innerHTML = s.categories
        .map((c) => `<div class="bar"><span class="bar__label">${esc(pretty(c.category))}</span><span class="bar__track"><span class="bar__fill" data-w="${((c.count / max) * 100).toFixed(1)}%"></span></span><span class="bar__value">${c.count}</span><span class="tip">${esc(pretty(c.category))}: ${c.count} tickets (${((c.count / s.tickets.total) * 100).toFixed(1)}%)</span></div>`)
        .join("");
      fillBars($("#kbCats"));

      const rc = s.retrieval_config;
      const cfg = {
        "Embedding model": s.embedding.model.split("/").pop(),
        "Fusion": `RRF · semantic ${rc.rrf_weights.semantic} / lexical ${rc.rrf_weights.lexical}`,
        "Lexical query": rc.lexical_query_mode === "or" ? "OR-of-terms tsquery" : "plainto_tsquery (AND)",
        "Sources per answer": `${rc.top_k} (≥ ${rc.kb_min_slots} KB articles)`,
        "Reranking": rc.reranking ? "on" : "off (Experiment 2)",
        "Database": `PostgreSQL ${s.database.postgres?.split(" ")[0] ?? ""} · pgvector ${s.database.pgvector ?? ""}`,
      };
      $("#kbConfig").innerHTML = Object.entries(cfg).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
      $("#kbIndexes").innerHTML = s.indexes
        .map((i) => `<li>${esc(i.definition).replace(/(ivfflat|hnsw|gin)/i, "<b>$1</b>")}</li>`)
        .join("");

      if (!kbLoaded) {
        const sel = $("#ticketCategory");
        s.categories.forEach((c) => sel.insertAdjacentHTML("beforeend", `<option value="${esc(c.category)}">${esc(pretty(c.category))}</option>`));
        kbLoaded = true;
        if (!$("#playQuery").value) $("#playQuery").value = lastComplaint || EXAMPLES[0].text;
        moveInk();
      }
    } catch (err) {
      toast(`Could not load the knowledge base: ${err.message}`);
    }
  }

  function moveInk() {
    const active = $(".tab.is-active");
    const ink = $("#tabInk");
    if (active && ink) { ink.style.left = `${active.offsetLeft}px`; ink.style.width = `${active.offsetWidth}px`; }
  }

  function setTab(name) {
    $$(".tab").forEach((t) => t.classList.toggle("is-active", t.dataset.tab === name));
    $$(".tabpanel").forEach((p) => p.classList.toggle("is-active", p.dataset.panel === name));
    moveInk();
    if (name === "tickets") loadTickets();
    if (name === "kb") loadKbArticles();
    if (name === "requests") loadRequests();
  }

  async function runPlayground(e) {
    if (e) e.preventDefault();
    const q = $("#playQuery").value.trim();
    if (q.length < 3) { toast("Type at least 3 characters."); return; }
    const box = $("#playResults");
    box.innerHTML = `<div class="empty">Querying PostgreSQL…</div>`;
    try {
      const r = await api(`/corpus/search?q=${encodeURIComponent(q)}&k=8`);
      const titles = { semantic: "Semantic", lexical: "Lexical", hybrid: "Hybrid (deployed)" };
      box.innerHTML = ["semantic", "lexical", "hybrid"]
        .map((m, ci) => {
          const hits = r.results[m];
          const rows = hits.length
            ? hits.map((h, i) => `<div class="hit" data-id="${esc(h.id)}" data-type="${esc(h.type)}" style="animation-delay:${0.1 + ci * 0.08 + i * 0.04}s">
                <span class="hit__rank">${h.rank}</span>
                <div><div class="hit__top"><span class="tag tag--${esc(h.type)}">${h.type === "kb_article" ? "KB" : "Ticket"}</span><span class="mono">${esc(h.id)}</span>
                  <span class="hit__score" title="cosine similarity to your text">cos ${h.relevance.toFixed(2)}</span></div>
                  <div class="hit__x">${esc(h.title ? `${h.title} — ${h.excerpt}` : h.excerpt)}</div></div></div>`).join("")
            : `<div class="empty">No matches</div>`;
          return `<div class="col" style="animation-delay:${ci * 0.08}s"><div class="col__head"><h3>${titles[m]}</h3><p>${esc(r.methods[m])}</p></div>${rows}</div>`;
        })
        .join("");
      $$(".hit", box).forEach((h) => {
        h.addEventListener("mouseenter", () => $$(".hit", box).forEach((x) => x.classList.toggle("is-linked", x.dataset.id === h.dataset.id)));
        h.addEventListener("mouseleave", () => $$(".hit", box).forEach((x) => x.classList.remove("is-linked")));
        h.addEventListener("click", () => openRecord(h.dataset.type, h.dataset.id));
      });
    } catch (err) {
      box.innerHTML = `<div class="empty">${esc(err.message)}</div>`;
    }
  }

  async function loadTickets() {
    const q = $("#ticketQuery").value.trim();
    const cat = $("#ticketCategory").value;
    const params = new URLSearchParams({ limit: PAGE, offset: ticketPage * PAGE });
    if (q) params.set("q", q);
    if (cat) params.set("category", cat);
    try {
      const r = await api(`/corpus/tickets?${params}`);
      $("#ticketTable").innerHTML = `<thead><tr><th>Ticket</th><th>Complaint</th><th>Resolution</th><th>Category</th><th>Severity</th></tr></thead><tbody>${r.items
        .map((t, i) => `<tr data-id="${esc(t.ticket_id)}" style="animation-delay:${i * 0.02}s"><td class="mono">${esc(t.ticket_id)}</td><td><span class="clamp">${esc(t.complaint)}</span></td><td><span class="clamp">${esc(t.resolution || "")}</span></td><td>${esc(pretty(t.category))}</td><td>${esc(t.severity || "")}</td></tr>`)
        .join("")}</tbody>`;
      $$("#ticketTable tbody tr").forEach((tr) => tr.addEventListener("click", () => openRecord("ticket", tr.dataset.id)));
      const from = r.total ? ticketPage * PAGE + 1 : 0;
      const to = Math.min((ticketPage + 1) * PAGE, r.total);
      $("#ticketPager").innerHTML = `<span>${from}–${to} of ${fmtN(r.total)} tickets</span><div><button type="button" data-p="-1" ${ticketPage === 0 ? "disabled" : ""}>← Prev</button><button type="button" data-p="1" ${to >= r.total ? "disabled" : ""}>Next →</button></div>`;
      $$("#ticketPager button").forEach((b) => b.addEventListener("click", () => { ticketPage += Number(b.dataset.p); loadTickets(); }));
    } catch (err) {
      toast(`Could not load tickets: ${err.message}`);
    }
  }

  async function loadKbArticles() {
    const q = $("#kbQuery").value.trim();
    try {
      const r = await api(`/corpus/kb?limit=100${q ? `&q=${encodeURIComponent(q)}` : ""}`);
      $("#kbTable").innerHTML = `<thead><tr><th>Article</th><th>Title</th><th>Category</th><th>Version</th></tr></thead><tbody>${r.items
        .map((a, i) => `<tr data-id="${esc(a.article_id)}" style="animation-delay:${i * 0.015}s"><td class="mono">${esc(a.article_id)}</td><td>${esc(a.title)}</td><td>${esc(pretty(a.category))}</td><td>v${esc(a.version)}</td></tr>`)
        .join("")}</tbody>`;
      $$("#kbTable tbody tr").forEach((tr) => tr.addEventListener("click", () => openRecord("kb_article", tr.dataset.id)));
    } catch (err) {
      toast(`Could not load KB articles: ${err.message}`);
    }
  }

  async function loadRequests() {
    try {
      const rows = await api("/requests?limit=30");
      $("#reqTable").innerHTML = rows.length
        ? `<thead><tr><th>Time</th><th>Complaint (PII-redacted)</th><th>Intent</th><th>Decision</th><th>Conf.</th><th>Retrieved sources</th></tr></thead><tbody>${rows
            .map((r, i) => `<tr style="animation-delay:${i * 0.02}s;cursor:default"><td class="mono">${esc((r.created_at || "").replace("T", " ").slice(0, 19))}</td><td><span class="clamp">${esc(r.complaint)}</span></td><td>${esc(pretty(r.understanding.intent))}</td><td>${DECISION_ICON[r.decision] || ""} ${esc(r.decision)}</td><td class="mono">${r.confidence != null ? r.confidence.toFixed(2) : ""}</td><td>${(r.sources || [])
              .map((s) => `<button type="button" class="link-btn mono" data-open="${esc(s.type)}|${esc(s.id)}">${esc(s.id)}</button>`)
              .join(" ")}</td></tr>`)
            .join("")}</tbody>`
        : `<tbody><tr><td class="empty">No requests yet: resolve a complaint in the console first.</td></tr></tbody>`;
      $$("#reqTable [data-open]").forEach((b) => b.addEventListener("click", () => { const [t, id] = b.dataset.open.split("|"); openRecord(t, id); }));
    } catch (err) {
      toast(`Could not load the request log: ${err.message}`);
    }
  }

  function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

  /* ---------------- wiring ---------------- */
  function init() {
    splitHeadlines();
    renderExamples();
    loadHealth();
    ta().addEventListener("input", autosize);
    ta().addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit(e); });
    $("#complaintForm").addEventListener("submit", submit);
    $("#sendBtn").addEventListener("pointerdown", (e) => ripple(e.currentTarget, e));
    $("#newBtn").addEventListener("click", async () => { ta().value = ""; autosize(); await setState("idle"); ta().focus(); });
    $$("[data-view-link]").forEach((b) => b.addEventListener("click", () => setView(b.dataset.viewLink)));
    $(".brand").addEventListener("click", (e) => { e.preventDefault(); setView("console"); setState("idle"); });
    $("#openInsights").addEventListener("click", openInsights);
    $$("[data-close]").forEach((b) => b.addEventListener("click", () => closeDrawer(b.dataset.close)));
    $$(".drawer").forEach((d) => d.addEventListener("click", (e) => { if (e.target === d) closeDrawer(d.id); }));
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") $$(".drawer.is-open").forEach((d) => closeDrawer(d.id)); });
    $$(".tab").forEach((t) => t.addEventListener("click", () => setTab(t.dataset.tab)));
    $("#playForm").addEventListener("submit", runPlayground);
    $("#ticketQuery").addEventListener("input", debounce(() => { ticketPage = 0; loadTickets(); }, 300));
    $("#ticketCategory").addEventListener("change", () => { ticketPage = 0; loadTickets(); });
    $("#kbQuery").addEventListener("input", debounce(loadKbArticles, 300));
    window.addEventListener("resize", moveInk);

    // Deep links for demos: /?q=<complaint> runs it; /?view=kb opens the knowledge base, with optional
    // &tab=tickets|kb|requests, &play=<text> (runs the retrieval playground), &record=ticket:<id> | kb_article:<id>.
    const params = new URLSearchParams(location.search);
    if (params.get("view") === "kb") {
      setView("kb");
      if (params.get("tab")) setTab(params.get("tab"));
      if (params.get("play")) { $("#playQuery").value = params.get("play"); runPlayground(); }
    }
    if (params.has("insights")) openInsights();
    if (params.get("record")) { const [t, id] = params.get("record").split(":"); openRecord(t, id); }
    if (params.get("q")) { ta().value = params.get("q"); autosize(); setTimeout(() => submit(), 900); }
    // Design preview of the searching scene without calling the API (/?preview=searching).
    if (params.get("preview") === "searching") { document.body.dataset.state = "searching"; startSearch(); }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
