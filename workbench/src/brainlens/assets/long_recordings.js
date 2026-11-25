"use strict";

window.createLongRecordingPanels = function ({ context, getJSON, postJSON, jump, analyze, palette, questions }) {
  const { t } = window.brainlensI18n;
  const el = (id) => document.getElementById(id);
  const admission = el("admissionPanel");
  const trends = el("trendPanel");
  const canvas = el("trendHeatmap");
  const viewport = el("trendViewport");
  const left = 88, rowHeight = 20;
  const colors = palette.map(rgb => `rgb(${rgb.join(",")})`);
  const num = (value) => Number(value.toPrecision(5)).toString();
  const roundTime = (value) => Number(value.toFixed(6));
  const between = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  let key = "", timeline, trend, trendKey, timelineLoading, trendRequest = 0;
  let segments = [], shownEvents = 100, selection, keyboardCell, drag, lo, hi;
  let questionMarker = null;
  const askBox = el("admissionAskAgent");
  const markerTarget = document.createElement("div");
  markerTarget.hidden = true;
  askBox.before(markerTarget);
  const markerAnchor = ({ label, start_seconds, end_seconds }) => ({ label, start_seconds, end_seconds });
  questions.mount(askBox, () => {
    const c = context();
    if (!c || !key || currentKey() !== key) return null;
    const scope = { case: c.case, source: "admission_timeline" };
    if (!questionMarker) return scope;
    const { recording, start_seconds, end_seconds, channels } = questionMarker;
    return { ...scope, recording, start_seconds, end_seconds,
      channel: channels?.[0], marker: markerAnchor(questionMarker) };
  });
  el("trendMeasure").value = "ripple";

  function clock(seconds) {
    const ms = Math.round(seconds * 1000);
    const h = Math.floor(ms / 3600000), m = Math.floor(ms / 60000) % 60;
    const s = String(Math.floor(ms / 1000) % 60).padStart(2, "0");
    const sub = ms % 1000 ? "." + String(ms % 1000).padStart(3, "0").replace(/0+$/, "") : "";
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${s}${sub}`;
  }

  function dayClock(seconds) {
    const ms = Math.round(seconds * 1000);
    return t("Day {day} · {time}", { day: 1 + Math.floor(ms / 86400000), time: clock((ms % 86400000) / 1000) });
  }

  function status(id, message = "") { el(id).textContent = message; }

  function currentKey() {
    const c = context();
    return c ? JSON.stringify([c.case, c.recording, c.run_name, c.fingerprint, c.generation]) : "";
  }

  function clear() {
    key = "";
    timeline = trend = trendKey = timelineLoading = selection = keyboardCell = drag = null;
    questionMarker = null;
    markerTarget.hidden = true;
    markerTarget.replaceChildren();
    trendRequest++;
    segments = [];
    shownEvents = 100;
    ["admissionStatus", "admissionFiles", "admissionGaps", "admissionEvents", "admissionEventCount", "segmentList",
      "trendStatus", "trendReadout", "segmentStatus"].forEach(id => el(id).replaceChildren());
    ["segmentStart", "segmentEnd", "segmentLabel", "segmentPattern", "admissionSearch"].forEach(id => { el(id).value = ""; });
    el("addMarkerSegments").disabled = false;
    el("trendPlot").hidden = el("moreMarkers").hidden = admission.hidden = trends.hidden = true;
    el("analyzeSegments").disabled = true;
  }

  function bar(track, className, start, end, span, title = "") {
    const part = document.createElement("span");
    part.className = className;
    part.style.left = `${100 * start / span}%`;
    part.style.width = `${100 * (end - start) / span}%`;
    if (title) part.title = title;
    track.append(part);
  }

  function renderTimeline() {
    if (!timeline) return;
    const rows = timeline.recordings || [];
    const duration = Math.max(1, ...rows.filter(r => Number.isFinite(r.offset_seconds)).map(r => r.offset_seconds + r.duration_seconds));
    const files = el("admissionFiles");
    files.replaceChildren();
    const gapBox = el("admissionGaps");
    const wasOpen = gapBox.querySelector("details")?.open || false;
    gapBox.replaceChildren();
    const gapList = document.createElement("ul");
    // The server ends each inter-file gap at the next dated file's offset.
    const preceding = new Map(timeline.gaps.map(gap => [gap.end_seconds, gap]));
    function gapBar(track, start, end, scale, label) {
      bar(track, "admission-gap", start, end, scale, label);
      const row = document.createElement("li");
      row.textContent = label;
      gapList.append(row);
    }
    for (const rec of rows) {
      const known = Number.isFinite(rec.offset_seconds);
      const origin = known ? rec.offset_seconds : 0;
      const scale = known ? duration : Math.max(1, rec.duration_seconds);
      const file = document.createElement("button"), text = document.createElement("span");
      const name = document.createElement("strong"), detail = document.createElement("small");
      const track = document.createElement("span");
      file.type = "button";
      file.className = "admission-file";
      name.textContent = rec.recording;
      detail.textContent = [
        known ? `${dayClock(origin)} – ${dayClock(origin + rec.duration_seconds)}` : t("Unknown offset · file order"),
        known ? rec.absolute_start : null, clock(rec.duration_seconds),
        rec.header_date_unreliable ? t("header date unreliable") : null,
      ].filter(Boolean).join(" · ");
      text.append(name, detail);
      track.className = "admission-track";
      const gapBefore = known && preceding.get(origin);
      if (gapBefore) {
        preceding.delete(origin);
        const { start_seconds: a, end_seconds: b } = gapBefore;
        gapBar(track, a, b, scale, `${t("Inter-file gap")} · ${dayClock(a)} – ${dayClock(b)} (${clock(b - a)})`);
      }
      const pieces = rec.pieces.length ? rec.pieces : [{ start_seconds: 0, end_seconds: rec.duration_seconds }];
      for (const p of pieces) bar(track, "admission-piece", origin + p.start_seconds, origin + p.end_seconds, scale);
      for (const gap of rec.gaps) {
        gapBar(track, origin + gap.start_seconds, origin + gap.end_seconds, scale,
          `${rec.recording} · ${t("Within-file gap")} · ${clock(gap.start_seconds)} – ${clock(gap.end_seconds)}`);
      }
      file.append(text, track);
      file.addEventListener("click", () => jump(rec.recording, 0));
      files.append(file);
    }
    const count = gapList.childElementCount;
    if (count) {
      const details = document.createElement("details"), summary = document.createElement("summary");
      details.open = wasOpen;
      const updateLabel = () => {
        summary.textContent = `${count === 1 ? t("1 gap") : t("{count} gaps", { count })} · ${details.open ? t("hide") : t("show")}`;
      };
      updateLabel();
      details.addEventListener("toggle", updateLabel);
      details.append(summary, gapList);
      gapBox.append(details);
    }
    renderEvents();
  }

  function renderEvents() {
    if (!timeline) return;
    const needle = el("admissionSearch").value.toLocaleLowerCase().trim();
    const events = timeline.events.filter(m => !needle || m.label.toLocaleLowerCase().includes(needle));
    const list = el("admissionEvents");
    status("admissionEventCount", t("Markers ({count})", { count: events.length }));
    list.replaceChildren();
    markerTarget.replaceChildren();
    markerTarget.hidden = !questionMarker;
    if (questionMarker) {
      const m = questionMarker, remove = document.createElement("button");
      remove.type = "button";
      remove.className = "btn btn-xs";
      remove.textContent = t("Clear marker");
      remove.addEventListener("click", () => {
        questionMarker = null;
        renderEvents();
        askBox.querySelector("input, textarea")?.focus();
      });
      markerTarget.append(`${m.recording} · ${clock(m.start_seconds)} · ${m.label} `, remove);
    }
    for (const m of events.slice(0, shownEvents)) {
      const button = document.createElement("button");
      const when = document.createElement("time"), file = document.createElement("strong");
      button.type = "button";
      when.textContent = Number.isFinite(m.admission_seconds) ? dayClock(m.admission_seconds) : clock(m.start_seconds);
      file.textContent = m.recording;
      button.append(when, file, document.createTextNode(m.label));
      button.addEventListener("click", () => jump(m.recording, m.start_seconds, m.channels?.[0], true, markerAnchor(m)));
      const row = document.createElement("div"), ask = document.createElement("button");
      row.className = "admission-event";
      ask.type = "button";
      ask.className = "marker-ask";
      ask.textContent = t("Ask agent");
      ask.title = t("Ask about marker");
      ask.setAttribute("aria-label", `${ask.title} · ${m.recording} · ${clock(m.start_seconds)} · ${m.label}`);
      ask.setAttribute("aria-pressed", String(m === questionMarker));
      ask.addEventListener("click", () => {
        questionMarker = m;
        renderEvents();
        askBox.querySelector("input, textarea")?.focus();
      });
      row.append(button, ask);
      list.append(row);
    }
    if (!events.length) list.textContent = t("No matching markers.");
    el("moreMarkers").hidden = events.length <= shownEvents;
  }

  async function loadTimeline() {
    if (!key || !admission.open || timeline || timelineLoading) return;
    const k = key;
    timelineLoading = true;
    status("admissionStatus", t("Loading timeline…"));
    try {
      const doc = await getJSON("/api/timeline", { case: context().case });
      if (k !== currentKey() || k !== key) return;
      timeline = doc;
      status("admissionStatus");
      renderTimeline();
    } catch (e) {
      if (k === key) status("admissionStatus", t("Navigation error: {message}", { message: e.message }));
    } finally {
      if (k === key) timelineLoading = false;
    }
  }

  function renderScale() {
    const scale = el("trendScale");
    scale.replaceChildren();
    if (Number.isFinite(lo)) {
      const gradient = document.createElement("i");
      gradient.className = "trend-scale-gradient";
      scale.append(document.createTextNode(num(lo)), gradient, document.createTextNode(`${num(hi)} ${t(trend.units)}`));
    } else {
      scale.textContent = t("No values available for this measure.");
    }
  }

  function drawHeatmap() {
    if (!trend || !trends.open || trends.hidden || !canvas.clientWidth) return;
    const width = canvas.clientWidth, plotWidth = width - left;
    const height = Math.min(340, rowHeight * trend.channels.length);
    if (height <= 0 || plotWidth < 1) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const style = getComputedStyle(canvas);
    const ink = style.getPropertyValue("--ink-1").trim(), surface = style.getPropertyValue("--bg-surface").trim();
    ctx.fillStyle = surface;
    ctx.fillRect(0, 0, width, height);
    const x = s => left + s / trend.duration_seconds * plotWidth;
    const scroll = viewport.scrollTop;
    const first = Math.floor(scroll / rowHeight), last = Math.min(trend.channels.length, Math.ceil((scroll + height) / rowHeight));
    ctx.font = "11px system-ui";
    ctx.textBaseline = "middle";
    for (let r = first; r < last; r++) {
      const y = r * rowHeight - scroll, values = trend.matrix[r];
      ctx.fillStyle = ink;
      ctx.fillText(trend.channels[r].name, 2, y + rowHeight / 2, left - 6);
      ctx.fillStyle = "#a8b3c2";
      ctx.fillRect(left, y, plotWidth, rowHeight - 1);
      // Preserve narrow peaks when multiple acquired epochs share a display pixel.
      let start = -1, end = 0, peak = -Infinity;
      function paint() {
        if (!Number.isFinite(peak)) return;
        ctx.fillStyle = colors[hi > lo ? between(Math.round(255 * (peak - lo) / (hi - lo)), 0, 255) : 127];
        ctx.fillRect(start, y, Math.max(1, end - start), rowHeight - 1);
      }
      for (let c = 0; c < values.length; c++) {
        if (!Number.isFinite(values[c])) continue;
        const a = Math.floor(x(trend.times_seconds[c]));
        const b = Math.min(width, Math.ceil(x(trend.times_seconds[c] + trend.epoch_seconds)));
        if (a !== start) { paint(); start = a; peak = -Infinity; }
        end = b;
        peak = Math.max(peak, values[c]);
      }
      paint();
    }
    for (const gap of trend.gaps) {
      const a = x(gap.start_seconds), b = x(gap.end_seconds);
      ctx.save();
      ctx.beginPath(); ctx.rect(a, 0, Math.max(1, b-a), height); ctx.clip();
      ctx.fillStyle = "#cbd5e1"; ctx.fillRect(a, 0, Math.max(1, b-a), height);
      ctx.strokeStyle = "#64748b"; ctx.beginPath();
      for (let i = a - height; i < b; i += 8) { ctx.moveTo(i, height); ctx.lineTo(i + height, 0); }
      ctx.stroke(); ctx.restore();
    }
    ctx.fillStyle = "#f59e0b";
    for (const m of trend.markers || []) ctx.fillRect(x(m.start), 0, 1, 8);
    ctx.fillStyle = ink;
    for (const s of trend.file_boundaries) ctx.fillRect(between(x(s), left, width - 2), 0, 2, height);
    if (selection) {
      const a = x(selection.start_seconds), b = x(selection.end_seconds);
      ctx.fillStyle = "rgba(75, 186, 255, .25)"; ctx.fillRect(a, 0, b-a, height);
      ctx.strokeStyle = "#38bdf8"; ctx.strokeRect(a + .5, .5, b-a, height - 1);
    }
    if (keyboardCell && document.activeElement === canvas) {
      const [r,c] = keyboardCell;
      const a = x(trend.times_seconds[c]), b = Math.min(width, x(trend.times_seconds[c] + trend.epoch_seconds));
      ctx.strokeStyle = "#fff"; ctx.strokeRect(a, r * rowHeight - scroll, Math.max(2, b-a), rowHeight - 1);
    }
  }

  let frame;
  function scheduleDraw() {
    if (!frame) frame = requestAnimationFrame(() => { frame = 0; drawHeatmap(); });
  }

  function renderTrend() {
    if (!trend) return;
    status("trendStatus");
    el("trendPlot").hidden = false;
    el("trendRows").style.height = `${rowHeight * trend.channels.length}px`;
    el("trendAxis").replaceChildren(...[0,.25,.5,.75,1].map(f => {
      const label = document.createElement("span");
      label.textContent = clock(trend.duration_seconds * f);
      return label;
    }));
    renderScale();
    scheduleDraw();
  }

  async function loadTrend() {
    if (!key || !trends.open || !el("trendEpoch").reportValidity()) return;
    const c = context(), measure = el("trendMeasure").value, epoch_seconds = el("trendEpoch").valueAsNumber;
    const nextKey = JSON.stringify([key, measure, epoch_seconds]);
    if (trendKey === nextKey) return;
    const request = ++trendRequest;
    trendKey = nextKey;
    trend = selection = drag = keyboardCell = null;
    viewport.scrollTop = 0;
    el("trendPlot").hidden = true;
    status("trendReadout");
    status("trendStatus", t("Loading quantitative trends…"));
    try {
      const doc = await getJSON("/api/trends", { case: c.case, recording: c.recording, run_name: c.run_name, measure, epoch_seconds });
      if (request !== trendRequest || key !== currentKey()) return;
      lo = Infinity; hi = -Infinity;
      for (const row of doc.matrix) {
        for (const v of row) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      }
      trend = doc;
      renderTrend();
    } catch (e) {
      if (request !== trendRequest) return;
      trendKey = "";
      status("trendStatus", t("Navigation error: {message}", { message: e.message }));
    }
  }

  function point(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      row: between(Math.floor((event.clientY - rect.top + viewport.scrollTop) / rowHeight), 0, trend.channels.length - 1),
      time: between((event.clientX - rect.left - left) / (rect.width - left), 0, 1) * trend.duration_seconds,
    };
  }

  function readout(r, c, time) {
    const channel = trend.channels[r].name, val = trend.matrix[r][c];
    const inGap = trend.gaps.some(g => time >= g.start_seconds && time < g.end_seconds);
    const value = inGap ? t("Gap") : Number.isFinite(val) ? `${num(val)} ${t(trend.units)}` : t("Unavailable");
    status("trendReadout", t("{channel} · {time} · {value}", { channel, time: clock(time), value }));
  }

  canvas.addEventListener("pointerdown", e => {
    if (!trend || e.button || e.clientX < canvas.getBoundingClientRect().left + left) return;
    const p = point(e);
    drag = { start: p.time, row: p.row, x: e.clientX };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", e => {
    if (!trend) return;
    const p = point(e), col = Math.min(trend.times_seconds.length - 1, Math.floor(p.time / trend.epoch_seconds));
    readout(p.row, col, p.time);
    if (drag && Math.abs(e.clientX - drag.x) >= 3) {
      selection = { start_seconds: Math.min(drag.start, p.time), end_seconds: Math.max(drag.start, p.time) };
      scheduleDraw();
    }
  });
  canvas.addEventListener("pointercancel", () => { drag = selection = null; scheduleDraw(); });
  canvas.addEventListener("pointerup", e => {
    if (!trend || !drag) return;
    const p = point(e), initial = drag;
    drag = null;
    if (Math.abs(e.clientX - initial.x) < 3) { jump(context().recording, p.time, trend.channels[initial.row].name, trend.measure !== "hfo_rate"); return; }
    selection = { start_seconds: roundTime(Math.min(initial.start, p.time)), end_seconds: roundTime(Math.max(initial.start, p.time)) };
    el("segmentStart").value = selection.start_seconds;
    el("segmentEnd").value = selection.end_seconds;
    el("segmentLabel").focus({ preventScroll: true });
    scheduleDraw();
  });
  canvas.addEventListener("keydown", e => {
    if (!trend || !["ArrowUp","ArrowDown","ArrowLeft","ArrowRight","Enter"].includes(e.key)) return;
    e.preventDefault(); e.stopPropagation();
    let [r,c] = keyboardCell || [0,0];
    if (e.key === "Enter") { jump(context().recording, trend.times_seconds[c], trend.channels[r].name, trend.measure !== "hfo_rate"); return; }
    r = between(r + Number(e.key === "ArrowDown") - Number(e.key === "ArrowUp"), 0, trend.channels.length - 1);
    c = between(c + Number(e.key === "ArrowRight") - Number(e.key === "ArrowLeft"), 0, trend.times_seconds.length - 1);
    keyboardCell = [r,c];
    viewport.scrollTop = between(viewport.scrollTop, (r + 1) * rowHeight - viewport.clientHeight, r * rowHeight);
    readout(r,c,trend.times_seconds[c]); scheduleDraw();
  });
  viewport.addEventListener("scroll", scheduleDraw, { passive: true });
  new ResizeObserver(scheduleDraw).observe(viewport);

  function renderSegments() {
    const list = el("segmentList");
    list.replaceChildren();
    segments.forEach((s,i) => {
      const row = document.createElement("li"), remove = document.createElement("button");
      remove.className = "btn btn-xs"; remove.type = "button"; remove.textContent = t("Remove");
      remove.addEventListener("click", () => { segments.splice(i, 1); renderSegments(); });
      row.append(document.createTextNode(`${s.label} · ${clock(s.start_seconds)} – ${clock(s.end_seconds)}`), remove);
      list.append(row);
    });
    el("analyzeSegments").disabled = segments.length === 0;
  }

  function addSegments(items) {
    const candidates = [...segments, ...items].sort((a,b) => a.start_seconds - b.start_seconds);
    const duration = context().duration;
    if (candidates.some(s => !Number.isFinite(s.start_seconds) || !Number.isFinite(s.end_seconds)
          || s.start_seconds < 0 || s.start_seconds >= s.end_seconds || s.end_seconds > duration + 1e-7)) {
      status("segmentStatus", t("Choose a start and end within this recording.")); return;
    }
    if (candidates.some((s,i) => i && s.start_seconds < candidates[i-1].end_seconds)) {
      status("segmentStatus", t("Analysis segments must not overlap.")); return;
    }
    segments = candidates;
    status("segmentStatus"); renderSegments();
  }
  el("segmentForm").addEventListener("submit", e => {
    e.preventDefault();
    if (!key || currentKey() !== key) return;
    addSegments([{ start_seconds: el("segmentStart").valueAsNumber, end_seconds: el("segmentEnd").valueAsNumber,
      label: el("segmentLabel").value.trim() || t("Segment {number}", { number: segments.length + 1 }) }]);
  });
  el("markerSegmentForm").addEventListener("submit", async e => {
    e.preventDefault();
    if (!key || key !== currentKey()) return;
    const c = context(), k = key;
    el("addMarkerSegments").disabled = true;
    try {
      const data = await postJSON("/api/marker-segments", { case: c.case, recording: c.recording,
        pattern: el("segmentPattern").value, before_seconds: el("segmentBefore").valueAsNumber, after_seconds: el("segmentAfter").valueAsNumber });
      if (k === key && k === currentKey()) addSegments(data.segments);
    } catch (err) {
      if (k === key) status("segmentStatus", t("Navigation error: {message}", { message: err.message }));
    } finally {
      if (k === key) el("addMarkerSegments").disabled = false;
    }
  });
  el("analyzeSegments").addEventListener("click", () => {
    if (key && key === currentKey() && segments.length) analyze([...segments]);
  });
  el("admissionSearch").addEventListener("input", () => { shownEvents = 100; renderEvents(); });
  el("moreMarkers").addEventListener("click", () => { shownEvents += 100; renderEvents(); });
  admission.addEventListener("toggle", loadTimeline);
  trends.addEventListener("toggle", () => { if (trend) scheduleDraw(); else loadTrend(); });
  el("trendMeasure").addEventListener("change", loadTrend);
  el("trendEpoch").addEventListener("change", loadTrend);
  el("loadTrends").addEventListener("click", loadTrend);

  function sync(languageChanged) {
    const next = currentKey();
    if (!next) { clear(); return; }
    if (next !== key) { clear(); key = next; }
    admission.hidden = trends.hidden = false;
    el("trendMeasure").querySelector('[value="hfo_rate"]').disabled = !context().run_name;
    if (!context().run_name && el("trendMeasure").value === "hfo_rate") el("trendMeasure").value = "ripple";
    if (languageChanged) {
      renderTimeline(); renderTrend(); renderSegments();
      if (timelineLoading) status("admissionStatus", t("Loading timeline…"));
      if (trendKey && !trend) status("trendStatus", t("Loading quantitative trends…"));
    }
    loadTimeline(); loadTrend();
  }
  return { clear, sync, redraw: scheduleDraw };
};
