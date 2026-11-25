"use strict";

window.createEvidencePanel = function ({ getJSON, esc, context, openContact }) {
  const { t } = window.brainlensI18n;
  const el = (id) => document.getElementById(id);
  const panel = el("evidencePanel");
  const result = el("evidenceResult");
  const status = el("evidenceStatus");
  const mapButton = el("loadEvidenceMap");
  const eiButton = el("loadEpileptogenicity");
  const onset = el("eiOnsetSeconds");
  let contextKey = "";
  let request = 0;
  let hasRun = false;

  // Keep the source response so switching language never reruns EI or changes the recording.
  let rendered = null;

  function setStatusKey(key) {
    status.dataset.i18n = key;
    status.textContent = t(key);
  }

  function clear() {
    rendered = null;
    delete status.dataset.i18n;
    ++request;
    contextKey = "";
    result.replaceChildren();
    status.textContent = "";
    onset.value = "";
    panel.hidden = true;
  }

  function sync(availableRun) {
    const current = context();
    if (!current) {
      clear();
      return;
    }
    const key = JSON.stringify(current);
    if (key !== contextKey) {
      clear();
      contextKey = key;
      if (!availableRun) setStatusKey("A saved HFO run is required for the evidence map.");
      eiButton.disabled = false;
    }
    panel.hidden = false;
    hasRun = availableRun;
    mapButton.disabled = !hasRun;
  }

  function format(value) {
    return Number.isFinite(value) ? value.toLocaleString("en-US", { maximumSignificantDigits: 4 }) : "—";
  }

  function sourceLine(label, source) {
    if (!source) return "";
    const p = source.provenance || {};
    const origin = p.is_synthetic ? t("synthetic engineering data") : (p.dataset_name || p.origin_kind?.replaceAll("_", " "));
    const detail = [source.recording_id, origin, p.summary].filter(Boolean).join(" · ");
    return `<p class="hint"><strong>${esc(label)}</strong> ${esc(detail)}</p>`;
  }

  function showMap(doc) {
    const reasons = new Set();
    const heads = doc.columns.map((col) =>
      `<th scope="col">${esc(t(col.label))} <small>${esc(col.unit || "")} · ${esc(col.recording_id || t("no source"))}</small></th>`
    ).join("");
    const rows = doc.rows.map((row) => {
      const cells = doc.columns.map(({ key }) => {
        const cell = row.biomarkers[key];
        if (cell.value == null) {
          const reason = cell.reason || cell.state;
          reasons.add(reason);
          return `<td class="evidence-missing" title="${esc(reason)}">${esc(t(cell.state))}</td>`;
        }
        const pct = Math.min(100, Math.max(0, Number(cell.percentile) || 0));
        const tooltip = t("{state} · percentile {percentile} · rank {rank}", { state: t(cell.state), percentile: format(pct), rank: cell.rank ?? "—" });
        return `<td class="evidence-measure" style="--shade:${0.45 * pct}%" title="${esc(tooltip)}">${format(cell.value)}</td>`;
      }).join("");
      const firstIED = row.biomarkers.ied.event_times?.[0];
      const time = Number.isFinite(firstIED) ? ` data-time="${firstIED}"` : "";
      const tags = (row.external_tags || []).map(tag => t(tag)).join(", ") || (doc.external_overlay_available ? "—" : t("No overlay"));
      return `<tr><th scope="row"><button type="button" class="evidence-contact" data-evidence-contact="${esc(row.channel)}"${time}>${esc(row.channel)}</button>
        <small>${esc(t(row.state.replaceAll("_", " ")))}</small></th>
        ${cells}<td>${format(row.convergence_count)} / ${format(row.available_count)}</td><td>${esc(tags)}</td></tr>`;
    }).join("");
    const legend = reasons.size ? `<details><summary>${t("Unavailable measurements")}</summary><ul>${[...reasons].map(
      (r) => `<li>${esc(r)}</li>`).join("")}</ul></details>` : "";
    status.textContent = t("{recording} · {run} · Colour shows percentile within the source recording. Select a contact to open the reader.", { recording: doc.recording_id, run: doc.run_name });
    result.innerHTML = `<div class="evidence-scroll"><table class="clin-table"><thead><tr><th scope="col">${t("Contact")}</th>
      ${heads}<th scope="col">${t("Convergence / available")}<small>${t("Descriptive, not a validated predictor")}</small></th>
      <th scope="col">${t("External tags")}<small>${esc(doc.external_overlay_source)}</small></th></tr></thead>
      <tbody>${rows}</tbody></table></div>${legend}<p class="hint">${esc(t(doc.top_tier_rule))}</p>
      ${sourceLine(t("Rate source"), doc)}${sourceLine(t("Ictal EI source"), doc.ei_source)}<p class="hint">${esc(t(doc.scope))}</p>`;
  }

  // Missing frames break the trace. Each channel has its own vertical scale and shares the recording-time axis.
  function plotER(canvas, times, row) {
    if (!times.length || !Array.isArray(row.er)) return;
    const values = row.er.filter(Number.isFinite);
    const t0 = times[0], span = times[times.length - 1] - t0;
    if (!values.length || !(span > 0)) return;
    const peak = Math.max(...values, Number.EPSILON);
    const ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
    const x = (t) => 1 + (w - 2) * (t - t0) / span;
    ctx.strokeStyle = "#008680";
    ctx.beginPath();
    let connected = false;
    row.er.forEach((v, i) => {
      if (!Number.isFinite(v) || !Number.isFinite(times[i])) { connected = false; return; }
      const y = h - 1 - (h - 2) * v / peak;
      if (connected) ctx.lineTo(x(times[i]), y);
      else ctx.moveTo(x(times[i]), y);
      connected = true;
    });
    ctx.stroke();
    if (Number.isFinite(row.detection_seconds)) {
      ctx.strokeStyle = "#dd6600";
      ctx.beginPath();
      ctx.moveTo(x(row.detection_seconds), 0);
      ctx.lineTo(x(row.detection_seconds), h);
      ctx.stroke();
    }
    canvas.title = t("ER range 0–{peak}, {start}–{end} s", { peak: format(peak), start: format(t0), end: format(t0 + span) });
  }

  function showEI(doc) {
    const channels = [...doc.channels].sort((a, b) => (b.ei ?? -1) - (a.ei ?? -1));
    const times = doc.times_seconds;
    const rows = channels.map((row, i) => {
      const score = row.ei == null ? "—" : `<meter value="${Number(row.ei)}" min="0" max="1"></meter> ${format(row.ei)}`;
      return `<tr><th scope="row">${esc(row.channel)}</th><td>${score}</td><td>${format(row.detection_seconds)}</td>
        <td><canvas data-er-row="${i}" width="220" height="28" aria-label="${esc(t("Energy ratio over time for {channel}", { channel: row.channel }))}"></canvas></td>
        <td>${format(row.gamma_upper_hz)}</td><td>${esc(row.reason || t(row.state))}</td></tr>`;
    }).join("");
    status.textContent = t("Ictal EI · {recording} · onset {time} s ({source})", { recording: doc.recording_id, time: format(doc.onset_seconds), source: doc.onset_source === "explicit" ? t("entered") : t("global marker") });
    result.innerHTML = `<div class="evidence-scroll"><table class="clin-table"><thead><tr><th scope="col">${t("Contact")}</th>
      <th scope="col">${t("Normalized EI")}</th><th scope="col">${t("Detection (s)")}</th>
      <th scope="col">${t("Energy ratio")}<small>${format(times[0])}–${format(times[times.length - 1])} s · ${t("per-contact scale · orange line marks detection")}</small></th>
      <th scope="col">${t("Gamma upper limit (Hz)")}</th><th scope="col">${t("Assessment")}</th></tr></thead><tbody>${rows}</tbody></table></div>
      ${sourceLine(t("Ictal EI source"), doc)}<p class="hint">${esc(t(doc.scope))}</p>`;
    result.querySelectorAll("[data-er-row]").forEach((canvas) => plotER(canvas, times, channels[Number(canvas.dataset.erRow)]));
  }

  async function load(ictal) {
    const params = context();
    if (!params || (ictal && !onset.reportValidity())) return;
    const sequence = ++request;
    const key = JSON.stringify(params);
    const current = () => sequence === request && key === JSON.stringify(context());
    const query = { case: params.case, recording: params.recording };
    if (ictal) query.onset_seconds = onset.value;
    else query.run_name = params.run_name;
    result.replaceChildren();
    rendered = null;
    setStatusKey(ictal ? "Loading ictal EI…" : "Loading evidence map…");
    eiButton.disabled = ictal;
    mapButton.disabled = !hasRun || !ictal;
    try {
      let data = await getJSON(ictal ? "/api/epileptogenicity" : "/api/evidence-map", query);
      while (data.job_id && current()) {
        if (data.status === "failed") throw new Error(data.error?.message || data.detail);
        if (data.status === "complete") { data = data.result; break; }
        delete status.dataset.i18n;
        status.textContent = data.detail;
        await new Promise((resolve) => setTimeout(resolve, 350));
        if (!current()) return;
        data = await getJSON(`/api/jobs/${encodeURIComponent(data.job_id)}`);
      }
      if (!current()) return;
      if ((params.fingerprint && data.recording_fingerprint !== params.fingerprint) ||
          (!ictal && params.generation && data.generation_id !== params.generation)) {
        setStatusKey("The recording or run changed. Refresh the overview to load the current evidence.");
        return;
      }
      delete status.dataset.i18n;
      rendered = { ictal, data };
      (ictal ? showEI : showMap)(data);
    } catch (error) {
      if (current()) {
        delete status.dataset.i18n;
        status.textContent = error.message;
      }
    } finally {
      if (current()) { eiButton.disabled = false; mapButton.disabled = !hasRun; }
    }
  }

  document.addEventListener("brainlens-languagechange", () => {
    if (rendered && contextKey === JSON.stringify(context())) {
      (rendered.ictal ? showEI : showMap)(rendered.data);
    }
  });
  mapButton.addEventListener("click", () => load(false));
  eiButton.addEventListener("click", () => load(true));
  result.addEventListener("click", (e) => {
    const button = e.target.closest("[data-evidence-contact]");
    if (button && contextKey === JSON.stringify(context())) {
      openContact(button.dataset.evidenceContact, Number.parseFloat(button.dataset.time));
    }
  });
  return { clear, sync };
};
