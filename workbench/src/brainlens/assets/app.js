(() => {
  "use strict";
  const { t, modelScoreLabel } = window.brainlensI18n;

  const urlParams = new URLSearchParams(window.location.search);
  let initialTheme = urlParams.get("theme");
  if (!initialTheme) {
    try { initialTheme = localStorage.getItem("brainlens_theme") || localStorage.getItem("btk_theme"); } catch (_) {}
  }
  initialTheme = initialTheme === "dark" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", initialTheme);

  const MC_PAD_L = 84;
  const MC_PAD_B = 40;
  const mcPlotWidth = (width) => Math.max(20, Math.floor(width) - MC_PAD_L - 16);
  const readerScales = createReaderScales();
  const FILTER_EDGES = {
    broadband: [null, null], eeg_1_70: [1, 70],
    ripple_80_250: [80, 250], fast_ripple_250_450: [250, 450],
  };
  const AUTH_TOKEN = (document.querySelector('meta[name="brainlens-token"]') || document.querySelector('meta[name="btk-token"]') || {}).content || "";
  const WORKSPACE_ID = (document.querySelector('meta[name="brainlens-workspace-id"]') || document.querySelector('meta[name="btk-workspace-id"]') || {}).content || "";

  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function anatomyLine(row) {
    const label = row?.anatomy?.display_label;
    const source = row?.anatomy?.source;
    const citation = source ? ` title="${esc(t("External source: {source}", { source }))}"` : "";
    return label ? `<span class="anatomy-label"${citation}>${esc(label)}</span>` : "";
  }

  function shaftBadge(row) {
    return row.display_shaft ? ` <span class="shaft-badge">${esc(row.display_shaft)}</span>` : "";
  }

  function channelCaption(row) {
    const channel = row.display_label || row.channel_name;
    return row.anatomy?.display_label ? `${channel} · ${row.anatomy.display_label}` : channel;
  }

  function channelTooltip(row) {
    const source = row.anatomy?.source;
    return source ? `${channelCaption(row)}\n${t("External source: {source}", { source })}` : channelCaption(row);
  }

  function genOpId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") return window.crypto.randomUUID();
    return `op_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
  }

  const st = {
    theme: initialTheme,
    cases: [],
    availInputs: [],
    protocols: [],
    cohorts: [],
    active: null,
    caseStatus: null,
    caseSlug: urlParams.get("case") || null,
    recording: urlParams.get("recording") || null,
    runName: urlParams.get("run") || "baseline",
    compareRun: urlParams.get("compare_run") || null,
    view: urlParams.get("view") || "overview",
    overviewMetric: "focal",
    shaft: urlParams.get("shaft") || "PAGE:0",
    page: 0,
    channel: urlParams.get("channel") || null,
    queueMode: urlParams.get("qmode") || "focal",
    reviewSet: urlParams.get("review_set") || "",
    blinded: false,
    identityRevision: 0,
    agreementSeq: 0,
    agreementResult: null,
    queueSort: urlParams.get("qsort") || "time",
    filterChannelOnly: false,
    queueItems: [],
    queueLimit: 150,
    reviewSetStatus: null,
    reviewProgress: null,
    queueTotal: 0,
    queueHasMore: false,
    selectedItemId: urlParams.get("event_id") || null,
    startSec: parseFloat(urlParams.get("start_seconds") || urlParams.get("start") || "0") || 0,
    viewStartSec: parseFloat(urlParams.get("start_seconds") || urlParams.get("start") || "0") || 0,
    winSec: parseFloat(urlParams.get("dur") || "8") || 8,
    gain: 1,
    rowHeightMode: "fit", // "fit" | 28 | 40 | 56 | 80
    scrollY: 0,
    maxScrollY: 0,
    curRowH: 32,
    vScrollDrag: null,
    mcScrollRect: null,
    calKey: "",
    baseUvLane: null,
    montage: urlParams.get("montage") || "acquired",
    dispFilter: urlParams.get("band") || "broadband",
    customLff: null,
    customHff: null,
    dispNotch: urlParams.get("notch") === "1",
    triadSpan: 0.6,
    sideBySide: urlParams.get("sbs") === "1",
    sourceOnly: false,
    mcData: null,
    mcBaseCanvas: null,
    mcBaseKey: "",
    mcBaseFirstRow: 0,
    mcBaseLastRow: -1,
    mcBaseMeta: null,
    mcRafId: 0,
    triadData: null,
    ctrlTriadData: null,
    pendingTriadTarget: null,
    manualCursorTarget: null,
    cmpData: null,
    mcBoxes: [],
    mcLabelHits: [],
    hiddenChannels: new Set(),
    channelGain: {},
    chMenuChannel: null,
    mcTool: "pan", // "pan" | "measure" | "exclude"
    multiShaftMode: false,
    triadCollapsed: false,
    mcSplitFrac: (() => {
      try {
        const v = parseFloat(localStorage.getItem("brainlens_reader_split") || localStorage.getItem("btk_reader_split") || "");
        if (Number.isFinite(v) && v >= 0.16 && v <= 0.88) return v;
      } catch (_) {}
      return 0.54;
    })(),
    mcHover: null,
    panDrag: null,
    measureDrag: null,
    minimapDrag: false,
    wheelPanTimer: null,
    wheelPageLastMs: 0,
    wheelZoomAcc: 0,
    wheelZoomLastMs: 0,
    wheelGainAcc: 0,
    dragSel: null, // { active, x0, x1, t0, t1, channel }
    justDragged: false,
    userEditedRole: false,
    preferredRealRole: (() => {
      try {
        const saved = localStorage.getItem("brainlens_reviewer_role") || localStorage.getItem("btk_reviewer_role");
        if (saved && saved !== "engineering_rehearsal") return saved;
      } catch (_) {}
      return null;
    })(),
    lastLocalVerdictAt: 0,
    palette: {},
    loadedSheetKey: "",
    sheetStaleRetries: 0,
    orbit: { yaw: -0.62, pitch: 0.34, dragging: false, moved: false, lx: 0, ly: 0, pts: [] },
    uiSeq: 0,
    activeJobId: null,
    bootReqSeq: 0,
    mcReqSeq: 0,
    mcRenderedSeq: -1,
    navigationMarker: null,
    triadReqSeq: 0,
    queueReqSeq: 0,
    overviewCmpSeq: 0,
    sheetBuildSeq: 0,
    triadLoading: false,
    contextSwitching: false,
    savingTargets: new Set(),
    pendingViewRefresh: false,
    pendingActionCount: 0,
    sseConnected: false,
    sseEverOpened: false,
    needsRefreshOnReconnect: false,
    sseSource: null,
    ssePageListenersBound: false,
    sessionExpired: false,
    iframeComposerOpen: false,
    sidebarCollapsed: false,
    deferredUiSync: null,
    recentLocalOpIds: new Set(),
    loadedCaseSlug: "",
    loadedRecSlug: "",
    loadedRunName: "",
    draftNotes: {},
    draftQuestions: {},
    lastDraftKey: "",
    lastQuestionCase: "",
  };

  function draftKeyFor(caseSlug, rec, run, itemId) {
    return `${caseSlug || ""}|${rec || ""}|${run || ""}|${itemId || "window"}`;
  }

  function windowDraftToken(tr) {
    if (!tr) return "window";
    const ch = tr.channel_name || st.channel || "";
    const mt = st.manualCursorTarget;
    if (
      mt &&
      mt.caseSlug === (tr.case || st.caseSlug) &&
      mt.recording === (tr.recording || st.recording) &&
      mt.runName === (tr.run_name || st.runName || "baseline") &&
      mt.channel === ch &&
      Number.isFinite(Number(mt.centerSeconds))
    ) {
      return `win:${ch}:${Number(mt.centerSeconds).toFixed(2)}`;
    }
    const w = tr.window_seconds;
    const mid = (w && w.length === 2 && Number.isFinite(Number(w[0])) && Number.isFinite(Number(w[1])))
      ? (0.5 * (Number(w[0]) + Number(w[1]))).toFixed(2)
      : "";
    return `win:${ch}:${mid}`;
  }

  function activeNoteItemKey() {
    if (st.triadData) {
      if (st.triadData.focus_event && st.triadData.focus_event.id) {
        return st.triadData.focus_event.id;
      }
      return windowDraftToken(st.triadData);
    }
    return st.selectedItemId || "window";
  }

  function currentDraftKey() {
    return draftKeyFor(st.caseSlug, st.recording, st.runName, activeNoteItemKey());
  }

  function swapNoteDraftForItem(caseSlug, rec, run, itemId) {
    const noteEl = $("noteInput");
    if (!noteEl) return;
    if (st.lastDraftKey) {
      st.draftNotes[st.lastDraftKey] = noteEl.value;
    }
    const nextKey = draftKeyFor(caseSlug, rec, run, itemId);
    if (nextKey !== st.lastDraftKey) {
      noteEl.value = st.draftNotes[nextKey] || "";
    }
    st.lastDraftKey = nextKey;
  }

  function isActiveContextCommitted(expectedCase = st.caseSlug) {
    return Boolean(
      st.active &&
      expectedCase &&
      st.loadedCaseSlug === expectedCase &&
      st.active.case === expectedCase &&
      (!st.recording || st.loadedRecSlug === st.recording) &&
      st.loadedRunName === (st.runName || "baseline")
    );
  }

  function isActiveCaseLoaded(expectedCase = st.caseSlug) {
    return !st.contextSwitching && isActiveContextCommitted(expectedCase);
  }

  function renderCaseStatus() {
    const statusMsg = st.caseStatus();
    $("triadHeaderText").textContent = statusMsg;
    $("verdictEventMeta").textContent = statusMsg;
    $("caseHeroBanner").innerHTML = `<div class="hero-sub">${esc(statusMsg)}</div>`;
  }

  function clearCaseDisplayForSwitch(isNewCase, status) {
    st.caseStatus = status;
    renderCaseStatus();
    $("agreementResults").replaceChildren();
    st.reviewSetStatus = null;
    st.reviewProgress = null;
    $("reviewSetProgress").textContent = "";
    evidencePanel.clear();
    ["mcCanvas", "triadCanvas", "readerMinimapCanvas", "spatialCanvas", "overviewTimelineCanvas"].forEach((cid) => {
      const c = $(cid);
      if (!c) return;
      const ctx = c.getContext && c.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, c.width, c.height);
      if (cid === "mcCanvas" || cid === "spatialCanvas") c.removeAttribute("title");
    });
    st.mcData = null;
    $("readerTimebaseReadout").textContent = "";
    $("readerFilterReadout").textContent = "";
    $("montageNotice").classList.add("hidden");
    st.orbit.pts = [];
    st.queueItems = [];
    st.triadData = null;
    st.ctrlTriadData = null;
    if ($("queueList")) $("queueList").innerHTML = "";
    if ($("queueCount")) $("queueCount").textContent = "(0)";
    if ($("shaftPills")) $("shaftPills").innerHTML = "";
    if ($("selectionBar")) $("selectionBar").classList.add("hidden");
    if ($("minimapReadout")) $("minimapReadout").textContent = "";
    if ($("syncClassBadge")) {
      $("syncClassBadge").textContent = "—";
      $("syncClassBadge").className = "sync-tag";
    }
    if ($("synchronyAlert")) {
      $("synchronyAlert").innerHTML = "";
      $("synchronyAlert").classList.add("hidden");
    }
    if ($("timelineMeta")) $("timelineMeta").textContent = "";
    if ($("kpiGrid")) $("kpiGrid").innerHTML = "";
    if ($("activeChannelsTableWrap")) $("activeChannelsTableWrap").innerHTML = "";
    if ($("branchTableWrap")) $("branchTableWrap").innerHTML = "";
    if ($("overviewDiagnosticsBox")) $("overviewDiagnosticsBox").innerHTML = "";
    if ($("overviewProvenanceBox")) $("overviewProvenanceBox").innerHTML = "";
    if ($("ovCmpRunA")) $("ovCmpRunA").innerHTML = "";
    if ($("ovCmpRunB")) $("ovCmpRunB").innerHTML = "";
    if ($("overlaySummaryBar")) {
      $("overlaySummaryBar").innerHTML = "";
      $("overlaySummaryBar").classList.add("hidden");
    }
    if ($("spatialBadge")) $("spatialBadge").textContent = "";
    if ($("sheetFrame")) {
      $("sheetFrame").src = "about:blank";
      st.loadedSheetKey = "";
    }
    if ($("printSheetLink")) {
      $("printSheetLink").classList.add("hidden");
      $("printSheetLink").href = "#";
    }
    if (isNewCase && $("provBadge")) {
      $("provBadge").textContent = "—";
      $("provBadge").classList.remove("synthetic");
    }
  }

  function hasSavingForCase(caseSlug) {
    if (!caseSlug || !st.savingTargets.size) return false;
    const prefix = `${caseSlug}|`;
    for (const k of st.savingTargets) {
      if (k.startsWith(prefix)) return true;
    }
    return false;
  }

  function currentTriadSaveKey(tr) {
    if (!tr) return null;
    const ev = tr.focus_event;
    const targetCase = tr.case || st.caseSlug || "";
    const targetRec = tr.recording || st.recording || "";
    const targetRun = tr.run_name || st.runName || "baseline";
    const targetEvId = ev ? ev.id : "win";
    return `${targetCase}|${targetRec}|${targetRun}|${targetEvId}`;
  }

  function closeAndUnbindCaseModals() {
    ["overlayModal", "protocolModal", "branchModal"].forEach((id) => {
      const m = $(id);
      if (!m) return;
      m.classList.add("hidden");
      delete m.dataset.boundCase;
      delete m.dataset.boundRecording;
      delete m.dataset.boundRun;
      delete m.dataset.boundEpoch;
    });
  }

  function stashAndClearDraftsForSwitch(isNewCase = false) {
    const noteEl = $("noteInput");
    if (noteEl && st.lastDraftKey) {
      st.draftNotes[st.lastDraftKey] = noteEl.value;
    }
    if (noteEl) {
      noteEl.value = "";
    }
    st.lastDraftKey = "";
    if (isNewCase) {
      const qEl = $("questionInput");
      if (qEl && st.lastQuestionCase) {
        st.draftQuestions[st.lastQuestionCase] = qEl.value;
      }
      if (qEl) {
        qEl.value = "";
      }
      st.lastQuestionCase = "";
    }
  }

  function stashAndSwapDrafts(nextCaseSlug, nextRec, nextRun) {
    const noteEl = $("noteInput");
    if (noteEl && st.lastDraftKey) {
      st.draftNotes[st.lastDraftKey] = noteEl.value;
    }
    const qEl = $("questionInput");
    if (qEl && st.lastQuestionCase) {
      st.draftQuestions[st.lastQuestionCase] = qEl.value;
    }
    const prefix = `${nextCaseSlug || ""}|${nextRec || ""}|${nextRun || ""}|`;
    const nextKey =
      st.lastDraftKey && st.lastDraftKey.startsWith(prefix)
        ? st.lastDraftKey
        : draftKeyFor(nextCaseSlug, nextRec, nextRun, activeNoteItemKey());
    if (noteEl && nextKey !== st.lastDraftKey) {
      noteEl.value = st.draftNotes[nextKey] || "";
    }
    if (qEl && (nextCaseSlug || "") !== st.lastQuestionCase) {
      qEl.value = st.draftQuestions[nextCaseSlug || ""] || "";
    }
    st.lastDraftKey = nextKey;
    st.lastQuestionCase = nextCaseSlug || "";
  }

  function resetTransientState(isNewCase = false) {
    st.selectedItemId = null;
    st.dragSel = null;
    st.panDrag = null;
    st.measureDrag = null;
    st.mcHover = null;
    st.justDragged = false;
    st.triadData = null;
    st.ctrlTriadData = null;
    st.pendingTriadTarget = null;
    st.manualCursorTarget = null;
    st.mcBaseKey = "";
    st.calKey = "";
    st.baseUvLane = null;
    st.scrollY = 0;
    st.maxScrollY = 0;
    st.vScrollDrag = null;
    if (st.wheelPanTimer) { clearTimeout(st.wheelPanTimer); st.wheelPanTimer = null; }
    if ($("selectionBar")) $("selectionBar").classList.add("hidden");
    if ($("chMenu")) $("chMenu").classList.add("hidden");
    if ($("mcReadoutHud")) $("mcReadoutHud").classList.add("hidden");
    if (isNewCase) {
      st.recording = null;
      st.runName = "baseline";
      st.compareRun = null;
      st.channel = null;
      st.shaft = "PAGE:0";
      st.page = 0;
      st.startSec = 0;
      st.viewStartSec = 0;
      st.hiddenChannels.clear();
      st.channelGain = {};
      st.loadedSheetKey = "";
      st.montage = "acquired";
      if (st.queueMode === "ehfo_fr") st.queueMode = "focal";
      if (st.queueSort === "ehfo") st.queueSort = "time";
    }
  }

  function syncSheetContext(locked = false) {
    const fr = $("sheetFrame");
    if (fr && fr.contentWindow) {
      try {
        fr.contentWindow.postMessage({
          type: "brainlens_sheet_context",
          locked: Boolean(locked),
          case: st.caseSlug || "",
          run_name: st.runName || "",
        }, window.location.origin);
      } catch (_) {}
    }
  }

  function lockSheetFrame(clearContent = false) {
    const fr = $("sheetFrame");
    if (!fr) return;
    fr.style.pointerEvents = "none";
    fr.style.opacity = "0.45";
    fr.setAttribute("aria-busy", "true");
    syncSheetContext(true);
    if (clearContent) {
      st.loadedSheetKey = "";
      fr.onload = null;
      try { fr.src = "about:blank"; } catch (_) {}
      if ($("printSheetLink")) $("printSheetLink").classList.add("hidden");
    }
  }

  function unlockSheetFrame() {
    const fr = $("sheetFrame");
    if (!fr) return;
    fr.style.pointerEvents = "";
    fr.style.opacity = "";
    fr.removeAttribute("aria-busy");
    syncSheetContext(false);
  }

  function beginContextSwitch(isNewCase = false) {
    questions.clear();
    st.mcRenderedSeq = -1;
    st.navigationMarker = null;
    longPanels.clear();
    setSourceNavigation(false);
    st.caseStatus = null;
    st.agreementSeq++;
    st.agreementResult = null;
    st.cmpData = null;
    $("agreementResults").replaceChildren();
    $("agreementSetSelect").replaceChildren(reviewSetPlaceholder("All events in run"));
    $("reviewSetSelect").replaceChildren();
    st.reviewSetStatus = null;
    renderReviewProgress(null);
    st.contextEpoch = (st.contextEpoch || 0) + 1;
    clearExportDownloads();
    if (isNewCase) $("exportPseudonym").value = "";
    $("exportEdfBtn").disabled = false;
    st.contextSwitching = true;
    st.pendingViewRefresh = false;
    st.sheetStaleRetries = 0;
    // Always invalidate loadedRunName & loadedRecSlug on any switch so a failed run/recording switch NEVER exposes or exports the old context!
    st.loadedRunName = "";
    st.loadedRecSlug = "";
    if (isNewCase) {
      // Clear stale previous-case metadata & rendered pixels immediately so a failed bootstrap NEVER exposes Case A under Case B!
      st.active = null;
      st.loadedCaseSlug = "";
    }
    stashAndClearDraftsForSwitch(isNewCase);
    closeAndUnbindCaseModals();
    resetTransientState(isNewCase);
    if (isNewCase) {
      clearCaseDisplayForSwitch(true, () => t("Loading case…"));
    }
    lockSheetFrame(true);
    setVerdictButtonsBusy(true);
  }

  function isRenderedWaveformSynced() {
    return Boolean(
      !st.contextSwitching &&
      st.mcData &&
      st.caseSlug &&
      st.recording &&
      st.mcData.case === st.caseSlug &&
      st.mcData.recording === st.recording &&
      (st.mcData.run_name || "baseline") === (st.runName || "baseline")
    );
  }

  function isRenderedTriadSynced() {
    return Boolean(
      !st.contextSwitching &&
      !st.triadLoading &&
      st.triadData &&
      st.caseSlug &&
      st.recording &&
      st.triadData.case === st.caseSlug &&
      st.triadData.recording === st.recording &&
      (st.triadData.run_name || "baseline") === (st.runName || "baseline")
    );
  }

  function hasOpenModal() {
    return !!document.querySelector(".modal:not(.hidden), dialog[open]");
  }

  function syncThemeToSheet() {
    const fr = $("sheetFrame");
    if (fr && fr.contentWindow) {
      try { fr.contentWindow.postMessage({ type: "brainlens_theme", theme: st.theme }, window.location.origin); } catch (_) {}
    }
  }

  function currentReviewerIdentity() {
    const roleEl = $("authorRoleSelect");
    const authEl = $("authorInput");
    const role = (roleEl && roleEl.value) || "clinician";
    const author = (authEl && authEl.value.trim()) || (role === "engineering_rehearsal" ? "rehearsal" : role);
    return { author, role };
  }

  function syncReviewerToSheet() {
    const fr = $("sheetFrame");
    if (fr && fr.contentWindow) {
      const { author, role } = currentReviewerIdentity();
      try { fr.contentWindow.postMessage({ type: "brainlens_reviewer", author, role }, window.location.origin); } catch (_) {}
    }
  }

  function sheetReviewerParams() {
    const { author, role } = currentReviewerIdentity();
    return `&role=${encodeURIComponent(role)}&author=${encodeURIComponent(author)}`;
  }

  function renderSyncBadge() {
    const syncEl = $("agentSyncBadge");
    if (!syncEl) return;
    if (st.blinded) {
      syncEl.textContent = t("Blinded");
      syncEl.title = t("Only this reader’s judgments are visible. Agent focus updates are paused.");
      return;
    }
    if (st.sessionExpired) {
      syncEl.innerHTML = `<span class="sync-led offline"></span> ${t("Reload Needed")}`;
      syncEl.title = t("Workbench server restarted — reload this page to refresh session token");
      return;
    }
    if (!st.sseConnected) {
      syncEl.innerHTML = `<span class="sync-led offline"></span> ${t("Reconnecting…")}`;
      syncEl.title = t("Live event stream disconnected — attempting automatic reconnection");
      return;
    }
    const pCnt = Number(st.pendingActionCount || 0);
    syncEl.innerHTML = pCnt > 0
      ? `<span class="sync-led"></span> ${t("Linked")} · <strong>${pCnt}</strong>`
      : `<span class="sync-led"></span> ${t("Linked")}`;
    syncEl.title = pCnt > 0
      ? t("{count} pending report annotation(s) queued for Agent review", { count: pCnt })
      : t("Synchronized live via SSE with Claude / Codex / Antigravity MCP");
  }

  function updateBlindingControls() {
    $("blindedToggle").checked = st.blinded;
    document.querySelector('#viewNav [data-view="sheet"]').disabled = st.blinded;
    $("jumpToSheetBtn").disabled = st.blinded;
    $("agreementPanel").classList.toggle("hidden", st.blinded || !st.active?.run_summary);
    renderSyncBadge();
  }

  let identityRefreshTimer;
  function reloadReaderIdentity(identityChanged = false) {
    clearTimeout(identityRefreshTimer);
    ++st.identityRevision;
    ++st.bootReqSeq;
    ++st.queueReqSeq;
    ++st.mcReqSeq;
    ++st.triadReqSeq;
    ++st.overviewCmpSeq;
    ++st.sheetBuildSeq;
    beginContextSwitch();
    st.cmpData = null;
    st.active = null;
    st.deferredUiSync = null;
    st.pendingActionCount = 0;
    if (identityChanged || st.blinded) {
      st.draftNotes = {};
      st.draftQuestions = {};
      st.lastQuestionCase = "";
      $("questionInput").value = "";
    }
    clearCaseDisplayForSwitch(false, () => t("Loading reader…"));
    $("toast").classList.add("hidden");
    for (const button of document.querySelectorAll(".v-btn.active")) button.classList.remove("active");
    updateBlindingControls();
    identityRefreshTimer = setTimeout(() => bootstrap(), identityChanged ? 200 : 0);
  }

  const $ = (id) => document.getElementById(id);
  function refreshPalette() {
    const cs = getComputedStyle(document.documentElement);
    const keys = [
      "--bg-app", "--bg-surface", "--bg-elevated", "--bg-subtle", "--bg-hover",
      "--ink-0", "--ink-1", "--ink-2", "--ink-3", "--border", "--border-strong",
      "--accent", "--accent-soft", "--focal", "--focal-soft", "--burst", "--burst-soft",
      "--warn", "--warn-soft", "--clip", "--clip-soft",
      "--canvas-bg", "--canvas-alt", "--canvas-grid", "--trace",
    ];
    const next = {};
    keys.forEach((k) => { next[k] = cs.getPropertyValue(k).trim(); });
    st.palette = next;
  }
  const css = (v) => (st.palette && st.palette[v]) || getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  refreshPalette();

  function syncReviewerSummary(opts = {}) {
    const el = $("reviewerSummaryText");
    const { author, role } = currentReviewerIdentity();
    if (el) {
      const roleOpt = $("authorRoleSelect") && $("authorRoleSelect").selectedOptions[0];
      const roleLbl = roleOpt ? roleOpt.textContent : t("Clinician");
      el.innerHTML = `${t("Reviewer:")} <strong>${esc(author)}</strong> (${esc(roleLbl)})`;
    }
    if (opts.persist) {
      try {
        if (role !== "engineering_rehearsal") {
          st.preferredRealRole = role;
          localStorage.setItem("brainlens_reviewer_role", role);
        }
        localStorage.setItem("brainlens_reviewer_author", author);
      } catch (_) {}
    }
    syncReviewerToSheet();
  }

  let formatToast = null;
  function toast(message, ms = 2500) {
    const el = $("toast");
    formatToast = typeof message === "function" ? message : null;
    el.textContent = formatToast ? formatToast() : message;
    el.classList.remove("hidden");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => {
      el.classList.add("hidden");
      formatToast = null;
    }, ms);
  }

  async function getJSON(path, params = {}) {
    const revision = st.identityRevision;
    const u = new URL(path, window.location.origin);
    if (st.blinded) {
      const { author, role } = currentReviewerIdentity();
      u.searchParams.set("blind_for", author);
      u.searchParams.set("blind_role", role);
    }
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") u.searchParams.set(k, String(v));
    });
    const r = await fetch(u.toString(), { cache: "no-store" });
    const d = await r.json();
    if (revision !== st.identityRevision) throw new Error(t("Reader changed."));
    if (!r.ok) {
      const err = new Error((d.error && d.error.message) || d.error || `HTTP ${r.status}`);
      err.httpStatus = r.status;
      throw err;
    }
    return d;
  }

  async function postJSON(path, body = {}) {
    const headers = { "Content-Type": "application/json" };
    if (AUTH_TOKEN) headers["X-BrainLens-Token"] = AUTH_TOKEN;
    const payload = (WORKSPACE_ID && body && typeof body === "object" && body.workspace_id === undefined)
      ? { workspace_id: WORKSPACE_ID, ...body }
      : body;
    const r = await fetch(path, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) {
      if (r.status === 401) {
        st.sessionExpired = true;
        st.sseConnected = false;
        renderSyncBadge();
        toast(() => t("Workbench server restarted — reload page to reconnect"), 5000);
      }
      const err = new Error((d.error && d.error.message) || d.error || `HTTP ${r.status}`);
      err.httpStatus = r.status;
      throw err;
    }
    return d;
  }

  async function syncToAgent(patch = {}) {
    if (st.blinded || st.queueMode === "review_set") return;
    try {
      const d = await postJSON("/api/ui-state", {
        case: st.caseSlug,
        recording: st.recording,
        run_name: st.runName,
        compare_run: st.compareRun,
        view: st.view,
        channel: st.channel,
        shaft: st.shaft,
        start_seconds: st.startSec,
        window_seconds: st.winSec,
        event_id: st.selectedItemId,
        queue_mode: st.queueMode,
        updated_by: "workbench",
        ...patch,
      });
      st.uiSeq = d.seq || st.uiSeq;
    } catch (_) {}
  }

  const MAGMA_STOPS = [
    [0.0, [12, 9, 38]],
    [0.25, [62, 15, 114]],
    [0.5, [140, 41, 129]],
    [0.75, [222, 73, 104]],
    [0.9, [254, 159, 109]],
    [1.0, [252, 253, 191]],
  ];
  const LUT = Array.from({ length: 256 }, (_, idx) => {
    const t = idx / 255;
    for (let i = 0; i < MAGMA_STOPS.length - 1; i++) {
      const [a, ca] = MAGMA_STOPS[i];
      const [b, cb] = MAGMA_STOPS[i + 1];
      if (t >= a && t <= b) {
        const u = (t - a) / (b - a || 1);
        return [
          Math.round(ca[0] + u * (cb[0] - ca[0])),
          Math.round(ca[1] + u * (cb[1] - ca[1])),
          Math.round(ca[2] + u * (cb[2] - ca[2])),
        ];
      }
    }
    return [252, 253, 191];
  });

  function hiDPI(canvas, fixedH = null) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(20, Math.floor(rect.width));
    const h = Math.max(20, fixedH || Math.floor(rect.height));
    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  // --- Bootstrap ---

  let readerRecordingKey = "";
  let depthReaderSettings = null;

  function setTriadCollapsed(collapsed) {
    st.triadCollapsed = collapsed;
    document.querySelector(".reader-main").classList.toggle("triad-collapsed", collapsed);
    $("toggleTriadBtn").classList.toggle("active", collapsed);
    $("toggleTriadBtn").setAttribute("aria-expanded", String(!collapsed));
  }

  function applyRecordingDefaults(active) {
    const key = JSON.stringify([active.case, active.active_recording]);
    if (key === readerRecordingKey) return;
    const first = !readerRecordingKey;
    readerRecordingKey = key;
    const spec = active.recording_spec;
    if (spec?.modality === "scalp") {
      // Remember the depth reader's settings across a visit to a scalp recording.
      if (!depthReaderSettings) {
        depthReaderSettings = {
          dispFilter: st.dispFilter, dispNotch: st.dispNotch, winSec: st.winSec,
          customLff: st.customLff, customHff: st.customHff, triadCollapsed: st.triadCollapsed,
          timebase: $("readerTimebaseSelect").value,
        };
      }
      st.shaft = first && urlParams.get("shaft")?.startsWith("PAGE:") ? urlParams.get("shaft") : "ALL";
      st.multiShaftMode = false;
      st.hiddenChannels.clear();
      st.scrollY = 0;
      st.dispFilter = first && urlParams.has("band") ? st.dispFilter : "eeg_1_70";
      st.dispNotch = first && urlParams.has("notch") ? st.dispNotch : Number(spec.line_freq) > 0;
      st.winSec = first && urlParams.has("dur") ? st.winSec : 10;
      st.customLff = st.customHff = null;
      $("readerTimebaseSelect").value = "fit";
      $("readerDurSelect").classList.remove("hidden");
      setTriadCollapsed(true);
    } else if (depthReaderSettings) {
      const { timebase, ...settings } = depthReaderSettings;
      Object.assign(st, settings);
      $("readerTimebaseSelect").value = timebase;
      $("readerDurSelect").classList.toggle("hidden", timebase !== "fit");
      st.shaft = "PAGE:0";
      setTriadCollapsed(settings.triadCollapsed);
      depthReaderSettings = null;
    }
  }

  async function bootstrap(opts = {}) {
    const reqSeq = ++st.bootReqSeq;
    const reqCase = opts.caseSlug !== undefined ? opts.caseSlug : st.caseSlug;
    const reqRec = opts.recording !== undefined ? opts.recording : st.recording;
    const reqRun = opts.runName !== undefined ? opts.runName : st.runName;
    const switchingContext = Boolean(
      st.contextSwitching ||
      (reqCase && st.caseSlug && reqCase !== st.caseSlug) ||
      (reqRec && st.recording && reqRec !== st.recording) ||
      (reqRun && st.runName && reqRun !== st.runName)
    );
    if (switchingContext) st.contextSwitching = true;
    try {
      const boot = await getJSON("/api/bootstrap", {
        case: reqCase,
        recording: reqRec,
        run_name: reqRun,
        lite: opts.lite ? "1" : "",
      });
      if (reqSeq !== st.bootReqSeq) return;
      st.cases = boot.cases || [];
      if (!opts.lite || (boot.available_inputs && boot.available_inputs.length)) {
        st.availInputs = boot.available_inputs || [];
      }
      st.protocols = boot.protocols || [];
      st.cohorts = boot.cohorts || [];
      if (boot.ui_state) {
        if (boot.ui_state.seq && !st.deferredUiSync) st.uiSeq = Math.max(st.uiSeq, boot.ui_state.seq);
        if (boot.ui_state.pending_action_count !== undefined) {
          st.pendingActionCount = Number(boot.ui_state.pending_action_count || 0);
          renderSyncBadge();
        }
      }

      if (!opts.lite) renderOpenModal();
      renderProtocolOptions();
      renderCohortModal();

      if (!boot.active) {
        st.active = null;
        st.loadedCaseSlug = "";
        st.loadedRecSlug = "";
        st.loadedRunName = "";
        renderCaseSelect();
        $("openModal").classList.remove("hidden");
        return;
      }

      st.caseStatus = null;
      st.active = boot.active;
      st.loadedCaseSlug = boot.active.case || "";
      st.loadedRecSlug = boot.active.active_recording || "";
      st.loadedRunName = boot.active.active_run || "baseline";
      st.caseSlug = boot.active.case;
      st.recording = boot.active.active_recording;
      st.runName = boot.active.active_run || "baseline";
      applyRecordingDefaults(boot.active);
      renderCaseSelect();
      if (opts.targetCompareRun !== undefined) st.compareRun = opts.targetCompareRun;
      if (opts.targetChannel) st.channel = opts.targetChannel;
      if (opts.targetShaft) st.shaft = opts.targetShaft;
      if (opts.targetQueueMode) st.queueMode = opts.targetQueueMode;
      if (opts.targetView) st.view = opts.targetView;

      const rsum = boot.active.run_summary;
      // Validate st.channel exists in this recording/run; otherwise pick focal leader
      const validNames = new Set([
        ...((rsum && rsum.channels) || []).map((c) => c.channel_name),
        ...((boot.active.recording_spec && boot.active.recording_spec.channels) || []).map((c) => c.name),
      ]);
      if (Number.isFinite(opts.focusStart)) {
        const contact = boot.active.recording_spec?.channels?.find(ch => ch.id === st.channel);
        if (contact && !validNames.has(st.channel)) st.channel = contact.name;
      }
      if (!st.channel || (validNames.size && !validNames.has(st.channel))) {
        const lead = rsum?.leading_focal_channels?.[0] || rsum?.leading_raw_channels?.[0];
        st.channel = lead?.channel_name || (!rsum && boot.active.recording_spec?.channels?.find(
          ch => ch.unit === "V" && !ch.is_auxiliary
        )?.name) || null;
      }

      if (Number.isFinite(opts.focusStart) && st.view === "reader") {
        st.selectedItemId = st.manualCursorTarget = null;
        const ref = rsum?.effective_reference?.mode;
        const inRun = rsum?.channels?.some(ch => ch.channel_name === st.channel);
        const rawChannel = boot.active.recording_spec?.channels?.find(ch => ch.name === st.channel || ch.id === st.channel);
        if (rawChannel && !inRun) st.channel = rawChannel.name;
        setSourceNavigation(Boolean(rawChannel?.is_auxiliary || (ref && ref !== "original" && rawChannel && !inRun)));
        if (!st.channel || !validNames.has(st.channel)) {
          st.channel = rsum?.channels?.[0]?.channel_name
            || boot.active.recording_spec?.channels?.find(ch => ch.unit === "V" && !ch.is_auxiliary)?.name || null;
        }
        st.shaft = resolveChannelShaft(st.channel) || "ALL";
        st.hiddenChannels.delete(st.channel);
        st.startSec = betweenReaderBounds(opts.focusStart);
        st.viewStartSec = st.startSec;
      }
      stashAndSwapDrafts(st.caseSlug, st.recording, st.runName);
      renderHeaderControls();
      switchView(st.view, false);
      if (!opts.skipRender) {
        await renderActiveView(Number.isFinite(opts.focusStart)
          ? { ...opts, keepWindow: true, manualTime: opts.focusStart } : opts);
      }
    } catch (err) {
      if (reqSeq === st.bootReqSeq) {
        const failedNewCase = !st.active || st.loadedCaseSlug !== st.caseSlug;
        if (failedNewCase || st.loadedRunName !== (st.runName || "baseline") || st.loadedRecSlug !== (st.recording || "")) {
          st.active = null;
          st.loadedRecSlug = "";
          st.loadedRunName = "";
          if (failedNewCase) st.loadedCaseSlug = "";
          document.querySelectorAll("#recPills button").forEach((btn) => btn.classList.remove("active"));
          closeAndUnbindCaseModals();
          const { caseSlug, recording, runName } = st;
          clearCaseDisplayForSwitch(failedNewCase, () => t("Failed to load {context} — previous display cleared; retry switching to reload.", {
            context: `${caseSlug || t("case")}${recording ? ` / ${recording}` : ""}${runName ? ` (${runName})` : ""}`,
          }));
          lockSheetFrame(true);
          setVerdictButtonsBusy(true);
        }
        toast(() => t("Failed to load workspace: {message}", { message: err.message }), 4500);
      }
    } finally {
      if (reqSeq === st.bootReqSeq && !opts.keepSwitching) {
        st.contextSwitching = false;
        questions.refresh(false);
        if (isActiveCaseLoaded() && !st.triadLoading && st.triadData && isRenderedTriadSynced()) {
          updateVerdictPanel(st.triadData);
        } else if (!isActiveCaseLoaded()) {
          setVerdictButtonsBusy(true);
        }
      }
    }
  }

  function renderCaseSelect() {
    const sel = $("caseSelect");
    sel.innerHTML = "";
    st.cases.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c.case;
      opt.textContent = c.case;
      opt.title = c.recording_count === 1
        ? t("{caseName} (1 recording)", { caseName: c.case })
        : t("{caseName} ({count} recordings)", { caseName: c.case, count: c.recording_count });
      if (c.case === st.caseSlug) opt.selected = true;
      sel.appendChild(opt);
    });
  }

  function renderRecordingSwitcher(act) {
    const pills = $("recPills");
    pills.replaceChildren();
    const recordings = act.recordings || [];
    async function select(recording) {
      if (recording === st.recording && isActiveCaseLoaded()) return;
      const runs = act.runs_by_recording?.[recording] || [];
      const runName = runs.length && !runs.some(r => r.run_name === st.runName)
        ? (runs.find(r => r.run_name === "baseline") || runs[0]).run_name : st.runName;
      beginContextSwitch(false);
      const epoch = st.contextEpoch;
      st.recording = recording;
      st.runName = runName;
      await bootstrap({ caseSlug: st.caseSlug, recording, runName });
      if (st.contextEpoch === epoch && isActiveCaseLoaded()) syncToAgent();
    }
    function button(label, recording, title = label) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = label;
      btn.title = title;
      btn.disabled = !recording;
      if (recording === st.recording) btn.classList.add("active");
      btn.addEventListener("click", () => select(recording));
      return btn;
    }
    if (recordings.length <= 3) {
      pills.append(...recordings.map(r => button(r, r)));
      return;
    }
    const index = recordings.indexOf(st.recording);
    const picker = document.createElement("details");
    picker.className = "recording-picker";
    const current = document.createElement("summary");
    current.textContent = current.title = st.recording;
    const menu = document.createElement("div");
    menu.className = "recording-menu";
    const search = document.createElement("input");
    search.type = "search";
    search.placeholder = t("Search recordings");
    search.setAttribute("aria-label", search.placeholder);
    const list = document.createElement("div");
    function filter() {
      const matches = recordings.filter(r => r.toLowerCase().includes(search.value.trim().toLowerCase()));
      list.replaceChildren(...matches.map(r => button(r, r)));
      if (!matches.length) list.textContent = t("No matching recordings.");
    }
    search.addEventListener("input", filter);
    search.addEventListener("keydown", e => {
      if (e.key === "Enter") list.querySelector("button")?.click();
      if (e.key === "ArrowDown") { e.preventDefault(); list.querySelector("button")?.focus(); }
    });
    picker.addEventListener("toggle", () => {
      if (picker.open) { search.value = ""; filter(); search.focus(); }
    });
    picker.addEventListener("keydown", e => {
      if (e.key === "Escape") { picker.open = false; current.focus(); }
    });
    picker.addEventListener("focusout", e => {
      if (!picker.contains(e.relatedTarget)) picker.open = false;
    });
    menu.append(search, list);
    picker.append(current, menu);
    const prev = button("‹", recordings[index - 1], t("Previous recording"));
    const next = button("›", recordings[index + 1], t("Next recording"));
    for (const btn of [prev, next]) btn.setAttribute("aria-label", btn.title);
    pills.append(prev, picker, next);
  }

  function renderHeaderControls() {
    const act = st.active;
    if (!act) return;

    renderRecordingSwitcher(act);

    const runs = (act.runs_by_recording && act.runs_by_recording[st.recording]) || [];
    const rsel = $("runSelect");
    rsel.innerHTML = "";
    runs.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.run_name;
      const fcnt = m.counts ? m.counts.non_burst_events : 0;
      const rcnt = m.counts ? m.counts.total_raw_events : 0;
      opt.textContent = t("{run} · {count} focal", { run: m.run_name, count: fcnt });
      opt.title = t("{run} ({focal} focal / {raw} raw)", { run: m.run_name, focal: fcnt, raw: rcnt });
      if (m.run_name === st.runName) opt.selected = true;
      rsel.appendChild(opt);
    });

    // Overview branch compare selectors (default Run A to active run, Run B to a distinct branch)
    const runNames = runs.map((m) => m.run_name);
    const defaultA = runNames.includes(st.runName) ? st.runName : (runNames.includes("baseline") ? "baseline" : (runNames[0] || ""));
    if (!st.compareRun || st.compareRun === defaultA || !runNames.includes(st.compareRun)) {
      st.compareRun = runNames.find((rn) => rn !== defaultA) || null;
    }
    ["ovCmpRunA", "ovCmpRunB"].forEach((id, idx) => {
      const el = $(id);
      el.innerHTML = "";
      if (idx === 1) {
        const auto = document.createElement("option");
        auto.value = "";
        auto.textContent = t("(auto branch)");
        el.appendChild(auto);
      }
      runs.forEach((m) => {
        const o = document.createElement("option");
        o.value = m.run_name;
        o.textContent = m.run_name;
        if (idx === 0 && m.run_name === defaultA) o.selected = true;
        if (idx === 1 && m.run_name === st.compareRun) o.selected = true;
        el.appendChild(o);
      });
    });

    const spec = act.recording_spec || {};
    const prov = spec.provenance || {};
    const p = $("provBadge");
    if (prov.is_synthetic) {
      p.textContent = t("Synthetic Demo (CC0)");
      p.classList.add("synthetic");
      const curRole = $("authorRoleSelect").value;
      if (curRole && curRole !== "engineering_rehearsal" && !st.preferredRealRole) {
        st.preferredRealRole = curRole;
      }
      $("authorRoleSelect").value = "engineering_rehearsal";
      syncReviewerSummary({ persist: false });
    } else {
      const ds = (prov.dataset_name || "").trim();
      const shortDs = /^chb-mit\b/i.test(ds)
        ? `CHB-MIT · ${prov.license || t("Public")}`
        : /zurich/i.test(ds)
        ? "Zurich iEEG HFO · CC0"
        : /hup/i.test(ds)
        ? "HUP iEEG · CC0"
        : ds
        ? `${ds.slice(0, 22)} · ${prov.license || t("Public")}`
        : t("Local Recording");
      p.textContent = shortDs;
      p.classList.remove("synthetic");
      if ($("authorRoleSelect").value === "engineering_rehearsal") {
        $("authorRoleSelect").value = st.preferredRealRole || "clinician";
        syncReviewerSummary({ persist: false });
      } else if (!st.userEditedRole && st.preferredRealRole && $("authorRoleSelect").value !== st.preferredRealRole) {
        $("authorRoleSelect").value = st.preferredRealRole;
        syncReviewerSummary({ persist: false });
      }
    }
    p.title = prov.summary || "";

    // Populate state/sleep-stage slice options in Branch Modal
    const stSel = $("brState");
    if (stSel) {
      stSel.innerHTML = `<option value="">${t("(Full recording / All states)")}</option>`;
      (act.available_states || []).forEach((sLbl) => {
        const o = document.createElement("option");
        o.value = sLbl;
        o.textContent = sLbl;
        stSel.appendChild(o);
      });
    }
    // Sync notch label and dynamic montage wording for derived runs
    const lf = spec.line_freq || 60;
    if ($("notchLbl")) $("notchLbl").textContent = t("Notch {freq} Hz", { freq: lf });
    const rsum = act.run_summary;
    const effMode = (rsum && rsum.effective_reference && rsum.effective_reference.mode) || "original";
    const msel = $("montageSelect");
    if (msel && msel.options.length >= 2) {
      msel.options[0].textContent = effMode !== "original" ? t("Run ({mode})", { mode: t(effMode) }) : t("Acquired");
      msel.options[1].disabled = effMode !== "original";
      if (effMode !== "original" && st.montage === "adjacent_bipolar") {
        st.montage = "acquired";
        msel.value = "acquired";
      }
    }
    renderOverlayBar();
  }

  function switchView(vname, triggerRender = true, opts = {}) {
    if (st.blinded && vname === "sheet") vname = "reader";
    if (vname !== "reader") setSourceNavigation(false);
    st.view = vname;
    document.documentElement.removeAttribute("data-init-view");
    document.querySelectorAll(".view-tab").forEach((b) => {
      b.classList.toggle("active", b.getAttribute("data-view") === vname);
    });
    document.querySelectorAll(".view-pane").forEach((sec) => {
      sec.classList.toggle("active", sec.id === `view-${vname}`);
    });
    questions.render();
    if (triggerRender) {
      renderActiveView(opts);
      if (isActiveCaseLoaded()) syncToAgent();
    }
  }

  async function renderActiveView(opts = {}) {
    if (!isActiveContextCommitted()) return;
    if (st.view === "overview") {
      await renderOverview();
    } else if (st.view === "reader") {
      renderShaftPills();
      await loadReaderQueueAndCanvases(opts);
    } else if (st.view === "sheet") {
      drawSpatialMap();
      const targetKey = `${st.caseSlug}:${st.runName}:${st.active.sheet_mtime_ns || 0}`;
      const fr = $("sheetFrame");
      if (st.active.sheet_url && st.loadedSheetKey !== targetKey) {
        const expectedCase = st.caseSlug;
        const expectedRun = st.runName;
        const sep = st.active.sheet_url.includes("?") ? "&" : "?";
        const bust = st.active.sheet_mtime_ns ? `&v=${encodeURIComponent(String(st.active.sheet_mtime_ns))}` : "";
        lockSheetFrame(false);
        st.loadedSheetKey = targetKey;
        fr.onload = () => {
          if (st.caseSlug === expectedCase && st.runName === expectedRun && st.loadedSheetKey === targetKey) {
            unlockSheetFrame();
            syncThemeToSheet();
            syncReviewerToSheet();
          }
        };
        fr.src = `${st.active.sheet_url}${sep}theme=${encodeURIComponent(st.theme)}${bust}${sheetReviewerParams()}`;
      } else if (!st.active.sheet_url && st.loadedSheetKey !== targetKey) {
        lockSheetFrame(true);
        await buildCaseSheet({ auto: true });
      } else {
        if (!st.contextSwitching) unlockSheetFrame();
        syncThemeToSheet();
        syncReviewerToSheet();
      }
      if (st.active.sheet_url || (fr.getAttribute("src") && fr.getAttribute("src") !== "about:blank")) {
        $("printSheetLink").href = st.active.sheet_url || fr.getAttribute("src");
        $("printSheetLink").classList.remove("hidden");
      }
    }
  }

  // --- VIEW 1: CASE TRIAGE OVERVIEW ---

  function referenceLabel(label) {
    const original = /^original \((.+)\)$/.exec(label);
    if (original) return t("original ({reference})", { reference: t(original[1]) });
    const pairs = /^bipolar \((\d+) pairs\)$/.exec(label);
    if (pairs) return t("bipolar ({count} pairs)", { count: pairs[1] });
    const average = /^common average \((\d+) ch\)$/.exec(label);
    if (average) return t("common average ({count} ch)", { count: average[1] });
    return label ? t(label) : "";
  }

  async function renderOverview({ languageChanged = false } = {}) {
    const act = st.active;
    const rsum = act.run_summary;
    const overlay = act.anatomy_labels;
    const anatomySource = overlay && Object.keys(overlay.labels || {}).length
      ? `<div class="hero-sub anatomy-source"><strong>${t("Anatomy labels:")}</strong> ${t("external, source={source}, patient-native={native}, atlas={atlas}", { source: esc(overlay.source), native: overlay.patient_native ? t("yes") : t("no"), atlas: esc(overlay.atlas || t("unspecified")) })}</div>`
      : "";
    $("agreementPanel").classList.toggle("hidden", st.blinded || !rsum);
    if (rsum && !st.blinded && $("agreementPanel").open) {
      if (languageChanged) renderAgreement();
      else loadAgreement();
    }
    evidencePanel.sync(Boolean(rsum));
    longPanels.sync(languageChanged);
    questions.render();
    if (!rsum) {
      const screening = languageChanged && $("runBaselineNowBtn")?.disabled;
      $("synchronyAlert").classList.add("hidden");
      $("caseHeroBanner").innerHTML = `
        <div>
          <div class="hero-title-row"><h1 class="hero-title">${esc(act.case)}</h1><span class="hero-chip">${esc(act.active_recording)}</span></div>
          <div class="hero-sub">${t("Recording inspected and source units verified. No HFO screening run exists yet.")}</div>
          ${anatomySource}
        </div>
        <button id="runBaselineNowBtn" class="btn btn-accent">${t("Run Baseline Screening")}</button>
      `;
      $("kpiGrid").innerHTML = "";
      $("activeChannelsTableWrap").innerHTML = `<div class="quiet-footer">${t("Run baseline screening to populate candidate rankings.")}</div>`;
      $("branchTableWrap").innerHTML = `<div class="quiet-footer">${t("No runs yet.")}</div>`;
      const rb = $("runBaselineNowBtn");
      if (rb) {
        if (screening) {
          rb.disabled = true;
          rb.textContent = t("Running QC + PyHFO...");
        }
        rb.addEventListener("click", async () => {
          if (st.contextSwitching) return;
          const targetCase = act.case;
          const targetRec = act.active_recording;
          const startEpoch = st.contextEpoch || 0;
          rb.disabled = true;
          rb.textContent = t("Running QC + PyHFO...");
          await submitAsyncJob("/api/jobs/branch", { target: `${targetCase}/${targetRec}`, run_name: "baseline" }, async () => {
            if ((st.contextEpoch || 0) === startEpoch && st.caseSlug === targetCase && st.recording === targetRec) {
              await bootstrap({ caseSlug: targetCase, recording: targetRec, runName: "baseline" });
            } else {
              toast(() => t("Completed baseline screening on {caseName}/{recording}", { caseName: targetCase, recording: targetRec }));
            }
          });
        });
      }
      return;
    }
    const cnt = rsum.counts;
    const spec = act.recording_spec || {};
    const prov = spec.provenance || {};
    const effRefLbl = rsum.effective_reference_label || rsum.acquired_reference;
    const qs = (!st.blinded && act.case_doc && act.case_doc.open_questions) || [];

    // Hero Case Banner
    $("caseHeroBanner").innerHTML = `
      <div>
        <div class="hero-title-row">
          <h1 class="hero-title">${esc(act.case)}</h1>
          <span class="hero-chip">${esc((rsum.modality || "seeg").toUpperCase())}</span>
          <span class="hero-chip">${esc(rsum.selection.sfreq_hz)} Hz</span>
          <span class="hero-chip">${esc(referenceLabel(effRefLbl))}</span>
          <span class="hero-chip">${esc(act.active_recording)} · ${esc(act.active_run)}</span>
        </div>
        <div class="hero-sub">${qs.length ? `<strong>${t("Open Clinical Question:")}</strong> ${esc(qs[0])}` : esc(prov.summary || t("Evidence-grounded interictal HFO screening and signal QC review."))}</div>
        ${anatomySource}
      </div>
      <div class="row-inline">
        <button id="heroOpenReaderBtn" class="btn btn-accent">${t("Inspect Waveforms →")}</button>
      </div>
    `;
    const hrBtn = $("heroOpenReaderBtn");
    if (hrBtn) hrBtn.addEventListener("click", () => switchView("reader"));

    // Synchrony alert banner
    const alertEl = $("synchronyAlert");
    const effSh = spec.channels?.some(c => c.display_shaft) && rsum.synchrony_thresholds?.effective_min_burst_shafts;
    if (cnt.widespread_burst_events > 0) {
      const pct = (100 * cnt.widespread_burst_fraction).toFixed(1);
      const winMs = (rsum.synchrony_thresholds && rsum.synchrony_thresholds.window_ms) || 30;
      alertEl.innerHTML = `
        <div class="alert-icon">!</div>
        <div>
          <strong>${t("Synchronous bursts")}</strong> · ${esc(act.active_recording)} · ${esc(act.active_run)}<br>
          ${t("{count} of {total} detections ({pct}%) occurred {where} within ±{window} ms. These {episodes} bursts may reflect a shared reference or a global transient.", { count: cnt.widespread_burst_events, total: cnt.total_raw_events, pct, where: effSh ? t("across ≥{shafts} shafts", { shafts: effSh }) : t("across channels"), window: winMs, episodes: cnt.widespread_burst_episodes })}
          ${t("The focal ranking includes {count} candidates across {channels} channels.", { count: cnt.non_burst_events, channels: cnt.active_focal_channels })}
        </div>
      `;
      alertEl.classList.remove("hidden");
    } else {
      alertEl.classList.add("hidden");
    }

    // KPI boxes
    const lead0 = rsum.leading_focal_channels[0];
    const ehfoHint = cnt.ehfo_events != null ? ` · ONNX eHFO: ${cnt.ehfo_events}` : "";
    const frHint = cnt.fast_ripple_focal_events != null ? t(" · FR-dom: {count}", { count: cnt.fast_ripple_focal_events }) : "";
    $("kpiGrid").innerHTML = `
      <div class="kpi-box">
        <div class="kpi-top-rule" style="background:var(--focal)"></div>
        <div class="kpi-lbl">${effSh ? t("Focal candidates (<{shafts} shafts)", { shafts: effSh }) : t("Focal candidates")}</div>
        <div class="kpi-val focal">${cnt.non_burst_events}</div>
        <div class="kpi-sub">${t("Active on {active} of {measured} measured ch", { active: cnt.active_focal_channels, measured: cnt.measured_channels })}${frHint}${ehfoHint}</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-top-rule" style="background:${cnt.widespread_burst_events ? "var(--burst)" : "var(--border-strong)"}"></div>
        <div class="kpi-lbl">${effSh ? t("Burst detections (≥{shafts} shafts)", { shafts: effSh }) : t("Burst detections")}</div>
        <div class="kpi-val ${cnt.widespread_burst_events ? "burst" : ""}">${cnt.widespread_burst_events}</div>
        <div class="kpi-sub">${t("{episodes} burst episode(s) · Total raw: {total}", { episodes: cnt.widespread_burst_episodes, total: cnt.total_raw_events })}</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-top-rule" style="background:var(--accent)"></div>
        <div class="kpi-lbl">${t("Leading Focal Contact ({recording})", { recording: esc(act.active_recording) })}</div>
        <div class="kpi-val">${lead0 ? esc(lead0.channel_name) : t("Quiet")}</div>
        ${anatomyLine(lead0)}
        <div class="kpi-sub">${lead0 ? t("{events} events / {seconds}s ({rate}/min)", { events: lead0.focal_count, seconds: lead0.detector_analyzed_seconds, rate: lead0.focal_rate_per_min.toFixed(2) }) : t("0 focal events")}</div>
      </div>
      <div class="kpi-box ${st.blinded ? "hidden" : ""}">
        <div class="kpi-top-rule" style="background:var(--clip)"></div>
        <div class="kpi-lbl">${t("Physician Verification")}</div>
        <div class="kpi-val">${cnt.clinician_confirmed} <span style="font-size:13px;font-weight:600;color:var(--ink-2)">${t("/ {count} confirmed", { count: cnt.non_burst_events })}</span></div>
        <div class="kpi-sub">${t("Clipped: {count} · Unassessed: {channels}", { count: cnt.clipped_for_discussion || 0, channels: esc(rsum.unassessed_channel_names.join(", ") || t("none")) })}</div>
      </div>
    `;

    drawOverviewTimeline(rsum);

    // Populate bottom Diagnostics & Provenance cards (zero dead empty floor)
    const qcOv = rsum.qc_overview || {};
    const clfSt = (rsum.classification && rsum.classification.stages) || {};
    const clfStr = Object.keys(clfSt).length
      ? Object.entries(clfSt).map(([k, v]) => `${t(k)}: ${t(v.status)}`).join(" · ")
      : t("Detector only (enable in Branch)");
    if ($("overviewDiagnosticsBox")) {
      $("overviewDiagnosticsBox").innerHTML = `
        <div class="qc-grid">
          <div class="qc-cell"><div class="lbl">${t("Line Noise ({freq} Hz)", { freq: Math.round(qcOv.line_freq_hz || 60) })}</div><div class="val">${qcOv.median_line_noise_snr_db != null ? Number(qcOv.median_line_noise_snr_db).toFixed(1) + " dB" : t("Clean / N/A")}</div></div>
          <div class="qc-cell"><div class="lbl">${t("Flat / Amp Flagged")}</div><div class="val">${t("{flat} / {amp} ch", { flat: (qcOv.flat_window_channels || []).length, amp: (qcOv.amplitude_warning_channels || []).length })}</div></div>
          <div class="qc-cell"><div class="lbl">${t("Spatial Frame")}</div><div class="val">${rsum.spatial && rsum.spatial.available ? `${esc(rsum.spatial.coordinate_system)}${rsum.spatial.template ? t(" (tpl)") : ""}` : t("None")}</div></div>
        </div>
        <div class="quiet-footer" style="margin-top:10px"><strong>${t("ONNX Models:")}</strong> ${esc(clfStr)}</div>
      `;
    }
    if ($("overviewProvenanceBox")) {
      const lims = (prov.limitations || []).slice(0, 2);
      $("overviewProvenanceBox").innerHTML = `
        <div style="font-size:12px; color:var(--ink-1); margin-bottom:6px"><strong>${t("Origin:")}</strong> ${esc(prov.summary || t(prov.origin_kind || "Local recording"))}</div>
        <div style="font-size:11.4px; color:var(--ink-2); line-height:1.45">${lims.length ? esc(lims.join(" · ")) : t("Preserve acquisition reference and channel coverage limits.")}</div>
        <div class="quiet-footer" style="margin-top:8px">${t("BrainLens is not a medical device. Quantitative neurophysiology review support only — not a standalone surgical resection boundary.")}</div>
      `;
    }

    if (languageChanged) {
      if (st.cmpData) renderOverviewTables(rsum, st.cmpData);
      return;
    }
    const cmpSeq = ++st.overviewCmpSeq;
    const snapCase = st.caseSlug;
    const snapRec = st.recording;
    const snapRun = st.runName;
    const cmp = await getJSON("/api/comparisons", {
      case: snapCase,
      recording: snapRec,
      run_a: $("ovCmpRunA").value || snapRun,
      run_b: $("ovCmpRunB").value || "",
    });
    if (cmpSeq !== st.overviewCmpSeq || st.caseSlug !== snapCase || st.recording !== snapRec || st.runName !== snapRun) return;
    st.cmpData = cmp;
    renderOverviewTables(rsum, cmp);
  }

  function drawOverviewTimeline(rsum) {
    const canvas = $("overviewTimelineCanvas");
    const { ctx, w, h } = hiDPI(canvas);
    ctx.clearRect(0, 0, w, h);

    const bins = rsum.timeline_bins || [];
    if (!bins.length) return;
    const totalDur = bins[bins.length - 1].end_seconds || 300;
    const qcOv = rsum.qc_overview || {};
    const lnTxt = qcOv.median_line_noise_snr_db != null
      ? ` · ${Math.round(qcOv.line_freq_hz || 60)} Hz SNR: ${Number(qcOv.median_line_noise_snr_db).toFixed(1)} dB`
      : "";
    $("timelineMeta").textContent = t("0 – {duration}s · 10s epochs{lineNoise}", { duration: totalDur, lineNoise: lnTxt });

    const padL = 42, padR = 12, padT = 8, qcH = 12, padB = 18;
    const pw = w - padL - padR, ph = h - padT - qcH - padB - 4;
    const qcY = padT + ph + 4;
    const maxFoc = Math.max(1, ...bins.map((b) => b.focal_events));
    const maxBst = Math.max(1, ...bins.map((b) => b.widespread_burst_events));

    ctx.strokeStyle = css("--canvas-grid");
    ctx.strokeRect(padL, padT, pw, ph);
    ctx.strokeRect(padL, qcY, pw, qcH);

    bins.forEach((b) => {
      const x = padL + (b.start_seconds / totalDur) * pw;
      const bw = Math.max(3, ((b.end_seconds - b.start_seconds) / totalDur) * pw - 1.5);
      if (b.widespread_burst_events > 0) {
        const bh = (b.widespread_burst_events / maxBst) * (ph * 0.92);
        ctx.fillStyle = "rgba(220, 38, 38, 0.28)";
        ctx.fillRect(x, padT + ph - bh, bw, bh);
      }
      if (b.focal_events > 0) {
        const fh = Math.max(5, (b.focal_events / maxFoc) * (ph * 0.85));
        ctx.fillStyle = "#059669";
        ctx.fillRect(x + 1, padT + ph - fh, Math.max(2, bw - 2), fh);
      }
      // QC Bad-Segment Strip
      const inv = Number(b.invalid_channel_fraction || 0);
      if (inv > 0.001) {
        ctx.fillStyle = inv > 0.25 ? "rgba(220, 38, 38, 0.72)" : "rgba(217, 119, 6, 0.58)";
        ctx.fillRect(x, qcY + 1, bw, qcH - 2);
      } else {
        ctx.fillStyle = "rgba(16, 185, 129, 0.22)";
        ctx.fillRect(x, qcY + 1, bw, qcH - 2);
      }
    });

    ctx.fillStyle = css("--ink-2");
    ctx.font = "10px sans-serif";
    ctx.fillText(t("Events"), 4, padT + 12);
    ctx.fillText(t("QC"), 16, qcY + 9);
    for (let i = 0; i <= 6; i++) {
      const t = (i / 6) * totalDur;
      const x = padL + (i / 6) * pw;
      ctx.fillText(`${Math.round(t)}s`, x - 10, h - 4);
    }
  }

  function renderOverviewTables(rsum, cmp) {
    const isFocal = st.overviewMetric === "focal";
    const wrapA = $("activeChannelsTableWrap");

    if (cmp.cross_recording && cmp.cross_recording.recordings.length > 1) {
      const cr = cmp.cross_recording;
      const recs = cr.recordings;
      $("activeTableTitle").textContent = t("Consistency across recordings · {run}", { run: cr.run_name });
      const rows = cr.aligned_active_channels || [];
      let maxCrossRt = 0.05;
      rows.forEach((row) => {
        recs.forEach((r) => {
          const c = row.by_recording[r];
          if (c) {
            const rt = (isFocal ? c.focal_rate_per_min : c.raw_rate_per_min) || 0;
            if (rt > maxCrossRt) maxCrossRt = rt;
          }
        });
      });
      wrapA.innerHTML = `
        <table class="clin-table">
          <thead><tr><th>${t("Channel")}</th>${recs.map((r) => `<th>${esc(r)}</th>`).join("")}</tr></thead>
          <tbody>
            ${rows.map((row) => {
              const activeRecCnt = recs.filter((r) => {
                const c = row.by_recording[r];
                return c && (isFocal ? c.focal_rank : c.raw_rank);
              }).length;
              const consBadge = activeRecCnt >= 2 ? `<span class="consistent-pill">${t("Consistent")}</span>` : "";
              return `
              <tr class="row-link" data-ch="${esc(row.channel)}">
                <td><strong>${esc(row.channel)}</strong>${shaftBadge(row)}${consBadge}</td>
                ${recs.map((r) => {
                  const c = row.by_recording[r];
                  if (!c) return `<td style="color:var(--ink-2)">${t("unassessed")}</td>`;
                  const rk = isFocal ? c.focal_rank : c.raw_rank;
                  const cnt = isFocal ? c.focal_count : c.raw_count;
                  const rt = isFocal ? c.focal_rate_per_min : c.raw_rate_per_min;
                  const pct = Math.min(100, Math.round((rt / maxCrossRt) * 100));
                  return rk
                    ? `<td><div class="rate-bar-wrap"><div class="rate-bar-track" style="width:42px"><div class="rate-bar-fill" style="width:${pct}%"></div></div><span><strong>#${rk}</strong> · ${t(isFocal ? "{count} focal in {seconds} s ({rate}/min)" : "{count} total in {seconds} s ({rate}/min)", { count: cnt, seconds: c.analyzed_seconds, rate: rt.toFixed(2) })}</span></div>${anatomyLine(c)}</td>`
                    : `<td style="color:var(--ink-2)">${t("0 / {seconds}s (quiet)", { seconds: c.analyzed_seconds })}${anatomyLine(c)}</td>`;
                }).join("")}
              </tr>`;
            }).join("")}
          </tbody>
        </table>
        <div class="quiet-footer">${t("Quiet measured channels (0 {kind} events in {recording}):", { kind: isFocal ? t("focal") : t("raw"), recording: esc(st.recording) })} <strong>${t("{quiet} of {measured} channels", { quiet: isFocal ? rsum.counts.measured_channels - rsum.counts.active_focal_channels : rsum.counts.quiet_measured_channels, measured: rsum.counts.measured_channels })}</strong>${t(" · Unassessed: {channels}", { channels: esc(rsum.unassessed_channel_names.join(", ") || t("none")) })}</div>
      `;
    } else {
      $("activeTableTitle").textContent = t("Active Candidate Channels ({recording} · {run})", { recording: st.recording, run: st.runName });
      const list = isFocal ? rsum.leading_focal_channels : rsum.leading_raw_channels;
      const maxRt = Math.max(0.05, ...list.slice(0, 12).map((r) => (isFocal ? r.focal_rate_per_min : r.raw_rate_per_min) || 0));
      wrapA.innerHTML = list.length ? `
        <table class="clin-table">
          <thead><tr><th>${t("Rank")}</th><th>${t("Channel")}</th><th>${t("Events")}</th><th>${t("Analyzed")}</th><th>${t("Rate (/min)")}</th></tr></thead>
          <tbody>
            ${list.slice(0, 12).map((r) => {
              const rt = (isFocal ? r.focal_rate_per_min : r.raw_rate_per_min) || 0;
              const pct = Math.min(100, Math.round((rt / maxRt) * 100));
              return `
              <tr class="row-link" data-ch="${esc(r.channel_name)}">
                <td class="mono-meta"><strong>#${isFocal ? r.focal_rank : r.raw_rank}</strong></td>
                <td><strong>${esc(r.channel_name)}</strong>${shaftBadge(r)}${anatomyLine(r)}</td>
                <td class="mono-meta">${isFocal ? r.focal_count : r.raw_count}</td>
                <td class="mono-meta">${r.detector_analyzed_seconds}s</td>
                <td><div class="rate-bar-wrap"><div class="rate-bar-track"><div class="rate-bar-fill" style="width:${pct}%"></div></div><strong class="mono-meta">${rt.toFixed(2)}/m</strong></div></td>
              </tr>`;
            }).join("")}
          </tbody>
        </table>
        <div class="quiet-footer">${t("Quiet measured channels (0 {kind} events):", { kind: isFocal ? t("focal") : t("raw") })} <strong>${t("{quiet} of {measured}", { quiet: isFocal ? rsum.counts.measured_channels - rsum.counts.active_focal_channels : rsum.counts.quiet_measured_channels, measured: rsum.counts.measured_channels })}</strong></div>
      ` : `<div class="quiet-footer">${t("0 channels had events in this mode ({quiet} measured channels quiet).", { quiet: rsum.counts.quiet_measured_channels })}</div>`;
    }

    // Branch comparison table
    const wrapB = $("branchTableWrap");
    if (!cmp.branch) {
      wrapB.innerHTML = `<div class="quiet-footer">${t("No secondary branch on {recording} yet. Click {button} above (e.g. exclude [0, 60)s or enable ONNX classifier) to test sensitivity while keeping baseline unchanged.", { recording: `<strong>${esc(st.recording)}</strong>`, button: `<strong>${t("+ Modify / Branch")}</strong>` })}</div>`;
    } else if (!cmp.branch.directly_comparable_rates) {
      const br = cmp.branch;
      const la = br.run_a_leading || [];
      const lb = br.run_b_leading || [];
      const nMax = Math.max(la.length, lb.length, 1);
      const rowsHtml = Array.from({ length: Math.min(8, nMax) }, (_, i) => {
        const a = la[i];
        const b = lb[i];
        return `<tr>
          <td>${i + 1}</td>
          <td>${a ? `<strong>${esc(a.channel_name)}</strong> (${a.focal_count} · ${a.focal_rate_per_min.toFixed(2)}/m)${anatomyLine(a)}` : "—"}</td>
          <td>${b ? `<strong>${esc(b.channel_name)}</strong> (${b.focal_count} · ${b.focal_rate_per_min.toFixed(2)}/m)${anatomyLine(b)}` : "—"}</td>
        </tr>`;
      }).join("");
      wrapB.innerHTML = `
        <div class="hint" style="margin:0 0 6px 0">${esc(br.reason)}</div>
        <table class="clin-table">
          <thead><tr><th>#</th><th>${esc(br.run_a)} (${esc(t((br.run_a_reference || {}).mode || "a"))})</th><th>${esc(br.run_b)} (${esc(t((br.run_b_reference || {}).mode || "b"))})</th></tr></thead>
          <tbody>${rowsHtml}</tbody>
        </table>
      `;
    } else {
      const br = cmp.branch;
      const sm = br.summary_metrics;
      const ef = br.event_fate;
      wrapB.innerHTML = `
        <div class="hint" style="margin:0 0 6px 0">
          ${t("Focal events: {focal} (Raw: {raw})", { focal: `<strong>${sm.focal_events_a} → ${sm.focal_events_b}</strong>`, raw: `${sm.raw_events_a} → ${sm.raw_events_b}` })} ·
          ${t("Excluded by mask: {excluded} · Lost in analyzed span: {lost}", { excluded: ef.excluded_by_changed_time_or_channel_mask, lost: ef.disappeared_within_analyzed_coverage })}
        </div>
        <table class="clin-table">
          <thead><tr><th>${t("Channel")}</th><th>${t("Rank Shift")}</th><th>${esc(br.run_a)}</th><th>${esc(br.run_b)}</th><th>Δ /min</th></tr></thead>
          <tbody>
            ${(br.channels || []).slice(0, 11).map((c) => `
              <tr class="row-link" data-ch="${esc(c.channel)}">
                <td><strong>${esc(c.channel)}</strong> <span class="shaft-badge">${esc(c.shaft)}</span>${anatomyLine(c)}</td>
                <td>${c.rank_a ? "#" + c.rank_a : t("quiet")} → <strong>${c.rank_b ? "#" + c.rank_b : t("quiet")}</strong></td>
                <td>${c.count_a}/${c.analyzed_seconds_a}s (${c.rate_a.toFixed(2)})</td>
                <td>${c.count_b}/${c.analyzed_seconds_b}s (${c.rate_b.toFixed(2)})</td>
                <td><strong>${c.rate_delta >= 0 ? "+" : ""}${c.rate_delta.toFixed(2)}</strong></td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    }

    document.querySelectorAll(".row-link[data-ch]").forEach((tr) => {
      tr.addEventListener("click", async () => {
        const ch = tr.getAttribute("data-ch");
        st.channel = ch;
        const sh = resolveChannelShaft(ch);
        if (sh) st.shaft = sh;
        st.selectedItemId = null;
        switchView("reader");
      });
    });
  }

  // --- VIEW 2: SYNCHRONIZED WAVEFORM & EVENT READER ---

  function readerChannels() {
    const rsum = st.active?.run_summary;
    const isDerived = !st.sourceOnly && rsum && rsum.effective_reference && rsum.effective_reference.mode !== "original" && st.montage === "acquired";
    return isDerived
      ? (rsum.channels || []).filter((c) => !c.is_auxiliary && c.unit === "V")
      : (st.active?.recording_spec?.channels || []).filter(c => c.unit === "V" && (!c.is_auxiliary || (st.sourceOnly && c.name === st.channel)));
  }

  function renderShaftPills() {
    const box = $("shaftPills");
    box.innerHTML = "";
    if (!st.active?.recording_spec) return;
    const rsum = st.active.run_summary;
    const chs = readerChannels();
    const nPages = Math.max(1, Math.ceil(chs.length / 16));

    const activeShafts = new Set((rsum?.leading_focal_channels || []).map(r => r.display_shaft));
    const scalp = st.active.recording_spec.modality === "scalp";
    const defaultGroup = scalp ? "ALL" : "PAGE:0";
    const shafts = scalp ? []
      : [...new Set(chs.map(c => c.display_shaft).filter(Boolean))];
    $("multiShaftBtn").hidden = shafts.length < 2;
    const shaftSet = new Set(shafts);
    const validPages = new Set(Array.from({ length: nPages }, (_, p) => `PAGE:${p}`));

    // Validate ALL, PAGE:k, or single/comma-separated multi-shaft selection against current recording
    const curRaw = String(st.shaft || "");
    if (curRaw === "ALL") {
      // Valid full-recording vertically scrollable view
    } else if (curRaw.startsWith("PAGE:")) {
      if (!validPages.has(curRaw)) {
        st.shaft = defaultGroup;
        st.page = 0;
      }
    } else {
      const kept = curRaw.split(",").map((s) => s.trim()).filter((s) => shaftSet.has(s));
      if (kept.length) {
        st.shaft = kept.join(",");
      } else {
        const resolved = st.channel ? resolveChannelShaft(st.channel) : null;
        st.shaft = (resolved && shaftSet.has(resolved)) ? resolved : defaultGroup;
        st.page = 0;
      }
    }
    const activeMultiSet = new Set(
      (String(st.shaft) === "ALL" || String(st.shaft).startsWith("PAGE:"))
        ? []
        : String(st.shaft).split(",").map((s) => s.trim()).filter(Boolean)
    );
    if ($("multiShaftBtn")) $("multiShaftBtn").classList.toggle("active", st.multiShaftMode || activeMultiSet.size > 1);

    // "All" pill: load all channels in one smooth vertically-scrollable view!
    const allBtn = document.createElement("button");
    allBtn.textContent = t("All");
    allBtn.title = t("All {count} channels (vertically scrollable)", { count: chs.length });
    if (st.shaft === "ALL") allBtn.classList.add("active");
    allBtn.addEventListener("click", async () => {
      st.multiShaftMode = false;
      st.shaft = "ALL";
      st.scrollY = 0;
      renderShaftPills();
      await loadMultichannelCanvas(true);
    });
    box.appendChild(allBtn);

    for (let p = 0; p < nPages; p++) {
      const btn = document.createElement("button");
      const key = `PAGE:${p}`;
      btn.textContent = t("P{page}", { page: p + 1 });
      btn.title = t("Channels {first}–{last}", { first: p * 16 + 1, last: Math.min(chs.length, (p + 1) * 16) });
      if (st.shaft === key) btn.classList.add("active");
      btn.addEventListener("click", async () => {
        st.multiShaftMode = false;
        st.shaft = key;
        st.page = p;
        st.scrollY = 0;
        renderShaftPills();
        await loadMultichannelCanvas();
      });
      box.appendChild(btn);
    }

    shafts.forEach((sh) => {
      const btn = document.createElement("button");
      const hasHot = activeShafts.has(sh);
      btn.innerHTML = hasHot ? `${esc(sh)}<span class="shaft-dot" title="${t("Contains focal HFO candidates")}"></span>` : esc(sh);
      btn.title = t("Shaft {shaft} (⌘/Ctrl+Click to combine multiple shafts)", { shaft: sh });
      if (activeMultiSet.has(sh)) btn.classList.add("active");
      btn.addEventListener("click", async (e) => {
        if (st.multiShaftMode || e.metaKey || e.ctrlKey) {
          const next = new Set(activeMultiSet);
          if (next.has(sh)) {
            if (next.size > 1) next.delete(sh);
          } else {
            next.add(sh);
          }
          st.shaft = [...next].join(",");
        } else {
          st.shaft = sh;
          st.scrollY = 0;
        }
        renderShaftPills();
        await loadMultichannelCanvas(true);
      });
      box.appendChild(btn);
    });
  }

  function updateQueueCountLabel() {
    const shown = st.queueItems.length;
    const tot = st.queueTotal;
    $("queueCount").textContent = st.queueHasMore || shown < tot
      ? t("Showing {shown} / {total}", { shown, total: tot })
      : `(${tot})`;
  }

  function reviewScope() {
    return { case: st.caseSlug, recording: st.recording, run_name: st.runName };
  }

  function reviewSetPlaceholder(label) {
    const option = new Option(t(label), "");
    option.dataset.i18n = label;
    return option;
  }

  function populateSetSelect(select, sets, preferred, allEvents = false) {
    select.replaceChildren();
    if (allEvents) select.add(reviewSetPlaceholder("All events in run"));
    for (const set of sets) select.add(new Option(set.name, set.name));
    if (!select.options.length) select.add(reviewSetPlaceholder("No review sets available"));
    if (Array.from(select.options).some((opt) => opt.value === preferred)) select.value = preferred;
  }

  function showReviewSetStatus() {
    const { message, localize } = st.reviewSetStatus;
    const text = localize ? t(message) : message;
    $("queueList").textContent = text;
    $("verdictEventMeta").textContent = text;
    $("triadHeaderText").textContent = text;
  }

  function refreshQueueLocale() {
    if (!isActiveContextCommitted()) return;
    updateQueueCountLabel();
    if (st.reviewSetStatus) showReviewSetStatus();
    else renderQueueList();
    renderReviewProgress(st.reviewProgress);
  }

  function resetSetTarget(message, localize = true) {
    ++st.mcReqSeq;
    ++st.triadReqSeq;
    resetTransientState();
    st.triadLoading = false;
    st.queueItems = [];
    st.queueHasMore = false;
    st.queueTotal = 0;
    st.mcData = null;
    st.reviewSetStatus = { message, localize };
    showReviewSetStatus();
    renderReviewProgress(null);
    $("syncClassBadge").textContent = "—";
    for (const button of document.querySelectorAll(".v-btn.active")) button.classList.remove("active");
    ["mcCanvas", "triadCanvas", "readerMinimapCanvas"].forEach((id) => {
      const canvas = $(id);
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    });
    updateQueueCountLabel();
    setVerdictButtonsBusy(true);
  }

  function reviewQueueParams() {
    if (st.queueMode !== "review_set") return { sort: st.queueSort, channel: st.filterChannelOnly ? st.channel : "" };
    const { author, role } = currentReviewerIdentity();
    return { review_set: st.reviewSet, reader: author, reader_role: role };
  }

  function renderReviewProgress(progress) {
    st.reviewProgress = progress;
    $("reviewSetProgress").textContent = progress && st.queueMode === "review_set"
      ? t("{judged} / {total} judged", { judged: progress.judged, total: progress.total }) : "";
  }

  function formatKappa(stat) {
    if (stat.value == null) return t("Unavailable ({reason})", { reason: esc(stat.reason || t("insufficient data")) });
    const estimate = Number(stat.value).toFixed(3);
    return stat.ci95
      ? t("{estimate} (95% CI {lower} to {upper})", { estimate, lower: Number(stat.ci95[0]).toFixed(3), upper: Number(stat.ci95[1]).toFixed(3) })
      : t("{estimate} (95% CI unavailable)", { estimate });
  }

  async function loadAgreement() {
    const seq = ++st.agreementSeq;
    st.agreementResult = null;
    if (st.blinded || !isActiveContextCommitted() || !st.active.run_summary || !$("agreementPanel").open) return;
    const scope = reviewScope();
    const current = () => seq === st.agreementSeq && !st.blinded && isActiveContextCommitted(scope.case) &&
      st.recording === scope.recording && st.runName === scope.run_name;
    const result = $("agreementResults");
    result.dataset.i18n = "Calculating agreement…";
    result.textContent = t("Calculating agreement…");
    try {
      const list = await getJSON("/api/review-sets", scope);
      if (!current()) return;
      const select = $("agreementSetSelect");
      populateSetSelect(select, list.review_sets || [], select.value, true);
      const setName = select.value;
      const report = await getJSON("/api/agreement", {
        ...scope, review_set: setName, readers: $("agreementReaders").value.trim(),
      });
      if (!current()) return;
      st.agreementResult = { report, setName, current };
      delete result.dataset.i18n;
      renderAgreement();
    } catch (err) {
      if (current()) {
        delete result.dataset.i18n;
        result.textContent = err.message;
      }
    }
  }

  function renderAgreement() {
    if (!st.agreementResult?.current()) return;
    const { report, setName, current } = st.agreementResult;
    const result = $("agreementResults");
    const categories = report.categories;
    const pairs = report.pairwise.map((pair) => {
      const [a, b] = pair.readers;
      const rows = pair.confusion_matrix.map((counts, i) => `<tr><th scope="row">${esc(t(categories[i]))}</th>${counts.map((n) => `<td>${esc(n)}</td>`).join("")}</tr>`);
      const percent = pair.percent_agreement == null ? t("Unavailable") : `${Number(pair.percent_agreement).toFixed(1)}%`;
      return `<h3>${t("{first} and {second}", { first: esc(a), second: esc(b) })}</h3>
        <p>${t("{shared} shared events · Agreement {percent}", { shared: esc(pair.n_shared), percent })}<br>
        ${t("Cohen’s κ {value}", { value: formatKappa(pair.kappa) })}<br>${t("Confirm vs. other κ {value}", { value: formatKappa(pair.binary_kappa) })}</p>
        <table class="clin-table"><caption>${t("Rows: {first} · Columns: {second}", { first: esc(a), second: esc(b) })}</caption><thead><tr><th>${t("Verdict")}</th>
        ${categories.map((c) => `<th scope="col">${esc(t(c))}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
    });
    const disagreements = report.disagreements;
    const discordRows = disagreements.map((event, i) => `<tr class="row-link" data-event-index="${i}">
      <td><button type="button" class="btn btn-xs">${esc(event.event_id)}</button></td><td>${esc(event.channel)}</td>
      <td>${esc(event.time_seconds)} s</td>${report.readers.map((reader) => `<td>${esc(t(event.verdicts[reader] || "—"))}</td>`).join("")}</tr>`);
    const fleiss = report.readers.length >= 3 ? `<p>${t("Fleiss’ κ {value} · {count} complete events", { value: formatKappa(report.fleiss), count: esc(report.fleiss.n_shared) })}</p>` : "";
    const ambiguous = Object.entries(report.excluded_ambiguous_readers || {});
    const roleHint = ambiguous.length ? `<p>${t("These readers submitted judgments in multiple roles. To include a reader, enter one role-qualified name in Readers.")}</p><ul>${ambiguous.map(([author, roles]) =>
        `<li>${esc(author)} · ${roles.map((role) => `<code>${esc(role)}:${esc(author)}</code>`).join(t(" or "))}</li>`).join("")}</ul>` : "";
    result.innerHTML = `<p>${t("{count} events · {recording} · {run}", { count: esc(report.n_events), recording: esc(report.recording), run: esc(report.run_name) })}</p>
      ${roleHint}${pairs.join("") || `<p>${t("Two readers are required for a pairwise comparison.")}</p>`}${fleiss}
      <h3>${t("Disagreements ({count})", { count: disagreements.length })}</h3>
      ${disagreements.length ? `<div class="agreement-disagreements"><table class="clin-table"><thead><tr><th>${t("Event")}</th><th>${t("Channel")}</th><th>${t("Onset")}</th>
      ${report.readers.map((reader) => `<th>${esc(reader)}</th>`).join("")}</tr></thead><tbody>${discordRows.join("")}</tbody></table></div>` : ""}
      <p class="quiet-footer">${esc(t(report.scope))}</p>`;
    result.querySelectorAll("[data-event-index]").forEach((row) => row.addEventListener("click", async () => {
      if (!current()) return;
      const event = disagreements[Number(row.dataset.eventIndex)];
      st.queueMode = setName ? "review_set" : "all";
      st.reviewSet = setName;
      st.selectedItemId = event.event_id;
      st.channel = event.channel;
      st.shaft = resolveChannelShaft(event.channel) || "ALL";
      st.startSec = Math.max(0, Number(event.time_seconds) - st.winSec * 0.5);
      st.viewStartSec = st.startSec;
      switchView("reader", true, { keepWindow: true });
    }));
  }

  async function loadReaderQueueAndCanvases(opts = {}) {
    if (!st.caseSlug || !st.recording) return;
    // Auto-fallback from eHFO/FR mode or ONNX sort when switching to a run without ONNX/FR scores
    const rsum = st.active && st.active.run_summary;
    const rcnt = (rsum && rsum.counts) || {};
    if (!opts.explicitTab) {
      if (!rsum && st.queueMode === "review_set" && !st.blinded) st.queueMode = "focal";
      if (
        st.queueMode === "ehfo_fr" &&
        !(rcnt.ehfo_events > 0 || rcnt.spike_hfo_events > 0 || rcnt.fast_ripple_focal_events > 0)
      ) {
        st.queueMode = "focal";
      }
      if (st.queueSort === "ehfo" && rcnt.ehfo_events == null) {
        st.queueSort = "time";
      }
    }
    if ($("montageSelect")) $("montageSelect").value = st.montage;
    syncFilterControls();
    if ($("readerTimebaseSelect").value === "fit") $("readerDurSelect").value = String(st.winSec);
    if ($("displayNotchChk")) $("displayNotchChk").checked = st.dispNotch;
    if ($("queueSortSelect")) $("queueSortSelect").value = st.queueSort;
    if ($("sideBySideBtn")) $("sideBySideBtn").classList.toggle("active", st.sideBySide);
    document.querySelectorAll("#queueTabs button").forEach((b) => {
      b.classList.toggle("active", b.getAttribute("data-qmode") === st.queueMode);
    });
    const qSeq = ++st.queueReqSeq;
    const isSet = st.queueMode === "review_set";
    $("reviewSetControls").classList.toggle("hidden", !isSet);
    $("filterChOnlyChk").closest("label").classList.toggle("hidden", isSet);
    $("queueSortControls").classList.toggle("hidden", isSet);
    if (isSet) {
      try {
        const response = await getJSON("/api/review-sets", reviewScope());
        if (qSeq !== st.queueReqSeq) return;
        populateSetSelect($("reviewSetSelect"), response.review_sets || [], st.reviewSet);
        st.reviewSet = $("reviewSetSelect").value;
        if (!st.reviewSet) {
          resetSetTarget("Create a review set to start.");
          return;
        }
      } catch (err) {
        if (qSeq === st.queueReqSeq) resetSetTarget(err.message, false);
        return;
      }
    }
    const effLimit = (opts.keepWindow && st.queueItems.length > st.queueLimit)
      ? Math.min(500, st.queueItems.length)
      : st.queueLimit;
    let qResp;
    try {
      qResp = await getJSON("/api/queue", {
        ...reviewScope(), mode: st.queueMode, ...reviewQueueParams(), offset: 0, limit: effLimit,
      });
    } catch (err) {
      if (!isSet) throw err;
      if (qSeq === st.queueReqSeq) resetSetTarget(err.message, false);
      return;
    }
    if (qSeq !== st.queueReqSeq) return;
    st.queueItems = qResp.items || [];
    st.queueTotal = Number(qResp.total ?? st.queueItems.length);
    st.queueHasMore = Boolean(qResp.has_more);
    updateQueueCountLabel();
    renderReviewProgress(qResp.progress);
    if (isSet && !st.queueItems.length) {
      resetSetTarget("This review set has no events.");
      renderReviewProgress(qResp.progress);
      return;
    }

    if (opts.manualTime !== undefined) {
      const span = opts.manualSpan && opts.manualSpan > 0 ? opts.manualSpan : st.triadSpan;
      st.pendingTriadTarget = {
        channel: st.channel,
        w0: opts.manualTime - span * 0.5,
        w1: opts.manualTime + span * 0.5,
      };
      renderQueueList();
      await Promise.all([
        loadMultichannelCanvas(true),
        loadTriadCanvas({
          eventId: "",
          centerSeconds: opts.manualTime,
          windowSeconds: opts.manualSpan && opts.manualSpan > 0 ? Math.max(0.6, opts.manualSpan) : undefined,
        }),
      ]);
      return;
    }

    if (!st.selectedItemId && st.queueItems.length && (!opts.keepWindow || isSet)) {
      const preferred = (!isSet && st.channel && st.queueItems.find((x) => x.channel_name === st.channel)) || st.queueItems[0];
      await focusQueueItem(preferred, false);
      renderQueueList();
      return;
    }

    renderQueueList();
    await Promise.all([loadMultichannelCanvas(), loadTriadCanvas()]);
  }

  async function loadMoreQueue() {
    if (!st.caseSlug || !st.recording || !st.queueHasMore) return;
    const btn = $("queueLoadMoreBtn");
    if (btn) { btn.disabled = true; btn.textContent = t("Loading…"); }
    const qSeq = st.queueReqSeq;
    try {
      const qResp = await getJSON("/api/queue", {
        case: st.caseSlug,
        recording: st.recording,
        run_name: st.runName,
        mode: st.queueMode,
        ...reviewQueueParams(),
        offset: st.queueItems.length,
        limit: st.queueLimit,
      });
      if (qSeq !== st.queueReqSeq) return;
      const seen = new Set(st.queueItems.map((x) => x.id));
      (qResp.items || []).forEach((it) => {
        if (!seen.has(it.id)) {
          st.queueItems.push(it);
          seen.add(it.id);
        }
      });
      st.queueTotal = Number(qResp.total ?? st.queueItems.length);
      st.queueHasMore = Boolean(qResp.has_more);
      updateQueueCountLabel();
      renderReviewProgress(qResp.progress);
      renderQueueList();
    } catch (err) {
      if (qSeq !== st.queueReqSeq) return;
      toast(() => t("Queue load failed: {message}", { message: err.message }), 3500);
      if (btn) { btn.disabled = false; btn.textContent = t("Retry Load More"); }
    }
  }

  function renderQueueList() {
    st.reviewSetStatus = null;
    const box = $("queueList");
    box.innerHTML = "";
    if (!st.queueItems.length) {
      box.innerHTML = `<div class="hint" style="padding:10px">${t("No items in '{mode}' queue. Switch queue tabs above (e.g. Controls or Bursts).", { mode: esc(t(st.queueMode)) })}</div>`;
      return;
    }
    const allowedDot = new Set(["control", "focal", "widespread_burst"]);
    st.queueItems.forEach((item) => {
      const div = document.createElement("div");
      div.className = "q-item" + (item.id === st.selectedItemId ? " active" : "");
      const isCtrl = item.kind === "background_control";
      const rawDot = isCtrl ? "control" : (item.synchrony_class || "focal");
      const dotCls = allowedDot.has(rawDot) ? rawDot : "focal";
      const star = item.clipped ? "★ " : (item.clinician_verdict === "confirm" ? "✓ " : "");
      const durMs = isCtrl ? item.duration_ms : ((item.end_seconds - item.start_seconds) * 1000);
      let badge = "";
      if (!isCtrl) {
        if (item.scores && item.scores.ehfo != null && (item.labels && item.labels.ehfo === 1)) {
          badge = `<span class="q-badge">${esc(modelScoreLabel(item, "ehfo"))}</span>`;
        } else if (item.dominant_band === "fast_ripple") {
          badge = `<span class="q-badge fr">${t("FR")}</span>`;
        }
      }
      div.innerHTML = `
        <span><span class="q-dot ${dotCls}"></span><strong>${star}${esc(item.channel_name)}</strong>${badge}${anatomyLine(item)}</span>
        <span class="mono-meta">${Number(item.start_seconds).toFixed(2)}s · ${Number(durMs).toFixed(0)}ms</span>
      `;
      div.addEventListener("click", () => focusQueueItem(item, true));
      box.appendChild(div);
    });
    if (st.queueHasMore) {
      const rem = Math.max(0, st.queueTotal - st.queueItems.length);
      const moreBtn = document.createElement("button");
      moreBtn.type = "button";
      moreBtn.id = "queueLoadMoreBtn";
      moreBtn.className = "queue-more-btn";
      moreBtn.textContent = t("Load More ({count} remaining)", { count: rem });
      moreBtn.addEventListener("click", loadMoreQueue);
      box.appendChild(moreBtn);
    }
  }

  function resolveChannelShaft(chName) {
    if (st.active?.recording_spec?.modality === "scalp") return null;
    const rsum = st.active && st.active.run_summary;
    const runChs = (rsum && rsum.channels) || [];
    const hitRun = !st.sourceOnly && runChs.find((x) => x.channel_name === chName);
    if (hitRun) return hitRun.display_shaft;
    const rawChs = (st.active && st.active.recording_spec && st.active.recording_spec.channels) || [];
    const hitRaw = rawChs.find((x) => x.name === chName);
    return hitRaw?.display_shaft;
  }

  async function focusQueueItem(item, sync = true) {
    st.navigationMarker = null;
    if (!st.caseSlug || !st.recording) return;
    setSourceNavigation(false);
    const startEpoch = st.contextEpoch || 0;
    swapNoteDraftForItem(st.caseSlug, st.recording, st.runName || "baseline", item.id);
    st.selectedItemId = item.id;
    st.channel = item.channel_name;
    const channels = readerChannels();
    const index = channels.findIndex(c => (c.name || c.channel_name) === item.channel_name);
    const group = resolveChannelShaft(item.channel_name) || (index >= 0 ? `PAGE:${Math.floor(index / 16)}` : "ALL");
    if (st.shaft !== "ALL" && !st.shaft.split(",").includes(group)) {
      st.shaft = group;
      st.scrollY = 0;
      renderShaftPills();
    }
    const mid = item.center_seconds != null ? Number(item.center_seconds) : 0.5 * (item.start_seconds + item.end_seconds);
    const dur = totalRecordingDuration();
    st.startSec = Math.max(0, Math.min(Math.max(0, dur - st.winSec), mid - st.winSec * 0.5));
    st.viewStartSec = st.startSec;
    st.pendingTriadTarget = { channel: item.channel_name, w0: mid - st.triadSpan * 0.5, w1: mid + st.triadSpan * 0.5 };
    $("readerStartInput").value = st.startSec.toFixed(1);
    renderQueueList();

    if (item.kind === "background_control") {
      await Promise.all([
        loadMultichannelCanvas(true),
        loadTriadCanvas({ centerSeconds: mid, eventId: "control" }),
      ]);
    } else {
      await Promise.all([
        loadMultichannelCanvas(true),
        loadTriadCanvas({ eventId: item.id }),
      ]);
    }
    if (sync && (st.contextEpoch || 0) === startEpoch && !st.contextSwitching) syncToAgent();
  }

  function totalRecordingDuration() {
    if (st.mcData?.case === st.caseSlug && st.mcData?.recording === st.recording && st.mcData.total_duration_seconds) {
      return Number(st.mcData.total_duration_seconds);
    }
    const spec = st.active && st.active.recording_spec;
    if (spec && spec.duration) return Number(spec.duration);
    if (spec?.sfreq && spec.n_samples) {
      const last = spec.pieces?.at(-1);
      return last ? last.onset_seconds + (last.stop_sample - last.start_sample) / spec.sfreq : spec.n_samples / spec.sfreq;
    }
    const rsum = st.active && st.active.run_summary;
    const bins = (rsum && rsum.timeline_bins) || [];
    if (bins.length) return Number(bins[bins.length - 1].end_seconds || 300);
    return 300;
  }

  function applyReaderSplitLayout() {
    const topWrap = $("mcCanvasWrap");
    const dock = $("triadDock");
    if (!topWrap || !dock || st.triadCollapsed) return;
    const pct = Math.round(Math.max(16, Math.min(88, st.mcSplitFrac * 100)) * 10) / 10;
    const targetTop = `${pct} 1 0`;
    const targetDock = `${(100 - pct).toFixed(1)} 1 0`;
    if (topWrap.style.flex !== targetTop) topWrap.style.flex = targetTop;
    if (dock.style.flex !== targetDock) dock.style.flex = targetDock;
  }

  function drawReaderMinimap() {
    const canvas = $("readerMinimapCanvas");
    if (!canvas) return;
    const { ctx, w, h } = hiDPI(canvas);
    ctx.clearRect(0, 0, w, h);

    const dur = Math.max(1, totalRecordingDuration());
    const rsum = st.active && st.active.run_summary;
    const bins = (rsum && rsum.timeline_bins) || [];
    const screenedStart = bins.length ? Number(bins[0].start_seconds || 0) : 0;
    const screenedEnd = bins.length ? Number(bins[bins.length - 1].end_seconds || dur) : dur;
    const partialHead = screenedStart > 1;
    const partialTail = screenedEnd < dur - 1;
    const partialScreen = partialHead || partialTail;
    const padL = 8, padR = partialScreen ? 182 : 106;
    const pw = Math.max(40, w - padL - padR);
    const xOf = (t) => padL + (Math.max(0, Math.min(dur, t)) / dur) * pw;

    // Subtle track
    ctx.fillStyle = css("--bg-subtle");
    ctx.fillRect(padL, 4, pw, h - 9);

    // Gently shade unscreened head/tail when run only analyzed a sub-interval
    if (partialHead) {
      const hx = xOf(screenedStart);
      ctx.fillStyle = st.theme === "dark" ? "rgba(255,255,255,0.035)" : "rgba(15,23,42,0.045)";
      ctx.fillRect(padL, 4, Math.max(0, hx - padL), h - 9);
      ctx.strokeStyle = css("--border-strong");
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(hx, 3);
      ctx.lineTo(hx, h - 3);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (partialTail) {
      const sx = xOf(screenedEnd);
      ctx.fillStyle = st.theme === "dark" ? "rgba(255,255,255,0.035)" : "rgba(15,23,42,0.045)";
      ctx.fillRect(sx, 4, padL + pw - sx, h - 9);
      ctx.strokeStyle = css("--border-strong");
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(sx, 3);
      ctx.lineTo(sx, h - 3);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (bins.length) {
      const maxFoc = Math.max(1, ...bins.map((b) => b.focal_events || 0));
      const maxBst = Math.max(1, ...bins.map((b) => b.widespread_burst_events || 0));
      bins.forEach((b) => {
        const bx = xOf(b.start_seconds);
        const bw = Math.max(2, xOf(b.end_seconds) - bx - 0.8);
        if (b.widespread_burst_events > 0) {
          const bh = Math.max(3, (b.widespread_burst_events / maxBst) * (h - 12));
          ctx.fillStyle = "rgba(220, 38, 38, 0.28)";
          ctx.fillRect(bx, h - 5 - bh, bw, bh);
        }
        if (b.focal_events > 0) {
          const fh = Math.max(4, (b.focal_events / maxFoc) * (h - 11));
          ctx.fillStyle = "#059669";
          ctx.fillRect(bx, h - 5 - fh, Math.max(2.5, bw), fh);
        }
        const inv = Number(b.invalid_channel_fraction || 0);
        ctx.fillStyle = inv > 0.25 ? "rgba(220,38,38,0.68)" : (inv > 0.001 ? "rgba(217,119,6,0.5)" : "rgba(16,185,129,0.22)");
        ctx.fillRect(bx, h - 4, bw, 2.5);
      });
    }

    // Active Triad event pin on minimap
    const pinWin = (st.triadData && st.triadData.window_seconds && st.triadData.recording === st.recording)
      ? st.triadData.window_seconds
      : (st.pendingTriadTarget ? [st.pendingTriadTarget.w0, st.pendingTriadTarget.w1] : null);
    if (pinWin) {
      const mid = 0.5 * (pinWin[0] + pinWin[1]);
      const px = xOf(mid);
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, 2);
      ctx.lineTo(px, h - 3);
      ctx.stroke();
    }

    // Shaded current viewport window (always synced smoothly to st.viewStartSec!)
    const vStart = Math.max(0, Math.min(Math.max(0, dur - st.winSec), st.viewStartSec));
    const vEnd = Math.min(dur, vStart + st.winSec);
    const vx0 = xOf(vStart);
    const vx1 = Math.max(vx0 + 6, xOf(vEnd));
    ctx.fillStyle = st.theme === "dark" ? "rgba(56, 189, 248, 0.22)" : "rgba(29, 78, 216, 0.16)";
    ctx.fillRect(vx0, 2, vx1 - vx0, h - 4);
    ctx.strokeStyle = css("--accent");
    ctx.lineWidth = 1.6;
    ctx.strokeRect(vx0, 2, vx1 - vx0, h - 4);

    const ro = $("minimapReadout");
    if (ro) {
      const screenTag = partialScreen ? t(" ({start}–{end}s screened)", { start: Math.round(screenedStart), end: Math.round(screenedEnd) }) : "";
      ro.textContent = `${vStart.toFixed(1)}–${vEnd.toFixed(1)}s / ${Math.round(dur)}s${screenTag}`;
    }
  }

  function sampleTraceAtTime(tr, tSec) {
    const ts = tr.times || [];
    const vs = tr.values_uv || [];
    if (!ts.length) return null;
    let lo = 0, hi = ts.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (ts[mid] < tSec) lo = mid + 1;
      else hi = mid;
    }
    const idx = Math.max(0, Math.min(ts.length - 1, lo));
    return vs[idx];
  }

  function computeChannelSpanMetrics(tr, tA, tB) {
    const loT = Math.min(tA, tB), hiT = Math.max(tA, tB);
    const ts = tr.times || [];
    const vs = tr.values_uv || [];
    let vmin = Infinity, vmax = -Infinity, count = 0;
    for (let k = 0; k < ts.length; k++) {
      if (ts[k] >= loT && ts[k] <= hiT && vs[k] !== null && Number.isFinite(vs[k])) {
        if (vs[k] < vmin) vmin = vs[k];
        if (vs[k] > vmax) vmax = vs[k];
        count++;
      }
    }
    const dtSec = Math.max(1e-4, hiT - loT);
    return {
      dtMs: dtSec * 1000,
      freqHz: 1.0 / dtSec,
      p2pUv: count >= 2 ? (vmax - vmin) : null,
      vmin: count >= 1 ? vmin : null,
      vmax: count >= 1 ? vmax : null,
    };
  }

  function updateMeasureReadout(traces) {
    const measure = st.measureDrag;
    const trace = traces.find(tr => tr.channel_name === measure.channel);
    if (!trace) return;
    const metrics = computeChannelSpanMetrics(trace, measure.t0, measure.t1);
    const amplitude = metrics.p2pUv === null ? "" : t(" · disp P2P {value} µV", { value: metrics.p2pUv.toFixed(1) });
    const frequency = metrics.dtMs >= 2 ? ` (${metrics.freqHz.toFixed(1)} Hz)` : "";
    updateReadoutHud(`Δ ${trace.display_label} · Δt ${metrics.dtMs.toFixed(1)} ms${frequency}${amplitude}`);
  }

  function updateReadoutHud(text) {
    const hud = $("mcReadoutHud");
    if (!hud) return;
    if (!text) {
      hud.classList.add("hidden");
      hud.textContent = "";
    } else {
      hud.textContent = text;
      hud.classList.remove("hidden");
    }
  }

  function scheduleMcDraw(forceRebuild = false) {
    if (forceRebuild) st.mcBaseKey = "";
    if (st.mcRafId) return;
    st.mcRafId = requestAnimationFrame(() => {
      st.mcRafId = 0;
      drawMultichannel();
    });
  }

  function computeMedianP95Uv(traces) {
    const p95s = traces.map((tr) => {
      const fin = tr.values_uv.filter((v) => v !== null && Number.isFinite(v)).sort((a, b) => a - b);
      if (!fin.length) return 80;
      return Math.max(12, fin[Math.floor(fin.length * 0.95)] - fin[Math.floor(fin.length * 0.05)]);
    }).sort((a, b) => a - b);
    return (p95s[Math.floor(p95s.length / 2)] || 100) * 1.45;
  }

  function syncFilterControls() {
    const edges = st.dispFilter === "custom" ? [st.customLff, st.customHff] : FILTER_EDGES[st.dispFilter];
    $("dispFilterSelect").value = st.dispFilter;
    $("displayLff").value = edges?.[0] ?? "";
    $("displayHff").value = edges?.[1] ?? "";
  }

  function describeDisplayFilter(data) {
    const traces = data.traces || [];
    if (!traces.length) return "";
    const label = (values) => {
      const unique = [...new Set(values)];
      if (unique.length > 1) return t("varies");
      return unique[0] == null ? t("off") : `${Number(unique[0].toPrecision(6))} Hz`;
    };
    const edge = (i) => label(traces.map(tr => tr.display_filter_band_hz?.[i]));
    const notch = label(traces.map(tr => tr.effective_notch_hz?.[0]));
    return t("LFF {lff} · HFF {hff} · Notch {notch}", { lff: edge(0), hff: edge(1), notch });
  }

  function setReaderWindow(seconds) {
    const center = st.viewStartSec + st.winSec / 2;
    st.winSec = seconds;
    st.startSec = Math.max(0, Math.min(totalRecordingDuration() - seconds, center - seconds / 2));
    st.viewStartSec = st.startSec;
    $("readerStartInput").value = st.startSec.toFixed(1);
  }

  function currentSpeedWindow() {
    const width = $("mcCanvas").getBoundingClientRect().width;
    return width > 0 ? readerScales.windowAtSpeed(mcPlotWidth(width), totalRecordingDuration()) : null;
  }

  function computeMcVerticalLayout(traces, ph) {
    const n = Math.max(1, traces.length);
    const minFitRowHeight = st.active?.recording_spec?.modality === "scalp" ? 18 : 28;
    const rowH = st.rowHeightMode === "fit"
      ? Math.max(minFitRowHeight, ph / n)
      : Math.max(18, Number(st.rowHeightMode) || 40);
    const totalH = n * rowH;
    const maxScrollY = Math.max(0, totalH - ph);
    st.scrollY = Math.max(0, Math.min(maxScrollY, st.scrollY));
    st.maxScrollY = maxScrollY;
    st.curRowH = rowH;
    return { rowH, totalH, maxScrollY };
  }

  function scrollFocusChannelIntoView(center = false) {
    const canvas = $("mcCanvas");
    const d = st.mcData;
    if (!canvas || !d || !st.channel) return;
    const traces = (d.traces || []).filter((tr) => !st.hiddenChannels.has(tr.channel_name));
    if (!traces.length) return;
    const ph = Math.max(20, canvas.getBoundingClientRect().height - 22 - MC_PAD_B);
    const { rowH, maxScrollY } = computeMcVerticalLayout(traces, ph);
    if (maxScrollY <= 0) { st.scrollY = 0; return; }
    const idx = traces.findIndex((tr) => tr.channel_name === st.channel);
    if (idx < 0) return;
    const rowTop = idx * rowH;
    const rowBot = rowTop + rowH;
    if (center || rowTop < st.scrollY || rowBot > st.scrollY + ph) {
      st.scrollY = Math.max(0, Math.min(maxScrollY, rowTop + rowH * 0.5 - ph * 0.5));
    }
  }

  function renderMultichannelStatus(data) {
    const shaftKey = String(st.shaft || "PAGE:0");
    const isPage = shaftKey.startsWith("PAGE:");
    const isAll = shaftKey === "ALL";
    const pageNum = isPage ? parseInt(shaftKey.split(":")[1], 10) || 0 : 0;
    const nTr = (data.traces || []).length;
    $("pageLabel").textContent = isAll
      ? t("All ({count} ch)", { count: nTr })
      : isPage
      ? t("Page {page}/{total}", { page: pageNum + 1, total: data.total_pages })
      : (shaftKey.includes(",") ? shaftKey.replace(/,/g, " + ") : t("Shaft {name}", { name: shaftKey }));
    const mb = $("montageNotice");
    $("readerFilterReadout").textContent = describeDisplayFilter(data);
    if (data.montage_notice) {
      mb.textContent = data.montage_notice;
      mb.classList.remove("hidden");
    } else {
      mb.classList.add("hidden");
    }
  }

  async function loadMultichannelCanvas(ensureFocusVisible = false) {
    if (!st.caseSlug || !st.recording) return;
    const seconds = currentSpeedWindow();
    if (seconds !== null && seconds !== st.winSec) setReaderWindow(seconds);
    const seq = ++st.mcReqSeq;
    questions.render();
    const reqCase = st.caseSlug;
    const reqRec = st.recording;
    const reqRun = st.runName;
    const shaftKey = String(st.shaft || "PAGE:0");
    const isPage = shaftKey.startsWith("PAGE:");
    const isAll = shaftKey === "ALL";
    const pageNum = isPage ? parseInt(shaftKey.split(":")[1], 10) || 0 : 0;
    try {
      const data = await getJSON("/api/multichannel", {
        source_only: st.sourceOnly ? 1 : null,
        focus_channel: st.sourceOnly ? st.channel : null,
        case: reqCase,
        recording: reqRec,
        run_name: reqRun,
        shaft: isPage ? "" : shaftKey,
        page: pageNum,
        page_size: isAll ? 256 : 16,
        start_seconds: st.startSec,
        duration_seconds: st.winSec,
        montage: st.montage,
        display_filter: st.dispFilter === "custom" ? "broadband" : st.dispFilter,
        display_lff: st.dispFilter === "custom" ? st.customLff : null,
        display_hff: st.dispFilter === "custom" ? st.customHff : null,
        display_notch: st.dispNotch ? "1" : "0",
      });
      if (seq !== st.mcReqSeq || reqCase !== st.caseSlug || reqRec !== st.recording || reqRun !== st.runName) return;
      st.mcData = data;
      st.mcRenderedSeq = seq;
      const marker = st.navigationMarker;
      if (marker && (marker.end_seconds < data.start_seconds || marker.start_seconds >= data.end_seconds)) {
        st.navigationMarker = null;
      }
      questions.render();
      $("mcCanvas").removeAttribute("title");
      st.mcBaseKey = "";
      st.startSec = Number(data.start_seconds ?? st.startSec);
      // Only snap viewStartSec to loaded startSec if user is not actively dragging or wheel-scrubbing right now
      if (!(st.panDrag && st.panDrag.active) && !st.minimapDrag && !st.wheelPanTimer) {
        st.viewStartSec = st.startSec;
      }
      if ($("readerStartInput")) $("readerStartInput").value = st.viewStartSec.toFixed(1);
      renderMultichannelStatus(data);
      applyReaderSplitLayout();
      if (ensureFocusVisible) scrollFocusChannelIntoView(isAll);
      drawMultichannel(true);
      drawReaderMinimap();
    } catch (err) {
      if (seq === st.mcReqSeq) toast(() => t("Waveform load error: {message}", { message: err.message }), 4000);
    }
  }

  function drawGapBands(ctx, gaps, xOf, left, right, top, height) {
    ctx.save();
    ctx.textAlign = "center";
    ctx.font = "bold 11px sans-serif";
    for (const gap of gaps || []) {
      const a = Math.max(left, xOf(gap.start_seconds));
      const b = Math.min(right, xOf(gap.end_seconds));
      if (b <= a) continue;
      ctx.fillStyle = "rgba(100,116,139,0.24)";
      ctx.fillRect(a, top, Math.max(1, b - a), height);
      ctx.fillStyle = css("--ink-1");
      const label = t("Gap {duration} s", { duration: +gap.duration_seconds.toFixed(6) });
      ctx.fillText(label, Math.max(left + 24, Math.min(right - 24, (a + b) / 2)), top + 12);
    }
    ctx.restore();
  }

  function ensureMcBaseCanvas(d, traces, pw, ph, rowH, dpr, uvLane) {
    const firstVis = Math.max(0, Math.floor(st.scrollY / rowH) - 1);
    const lastVis = Math.min(traces.length - 1, Math.ceil((st.scrollY + ph) / rowH) + 1);
    const chGainSig = Object.entries(st.channelGain).filter(([, v]) => v !== 1).map(([k, v]) => `${k}:${v}`).join(",");
    const hiddenSig = [...st.hiddenChannels].sort().join(",");
    const baseKey = `${st.mcReqSeq}|${pw}|${rowH.toFixed(2)}|${dpr}|${st.theme}|${uvLane.toFixed(2)}|${st.channel || ""}|${chGainSig}|${hiddenSig}`;
    if (
      st.mcBaseCanvas &&
      st.mcBaseKey === baseKey &&
      firstVis >= st.mcBaseFirstRow &&
      lastVis <= st.mcBaseLastRow
    ) {
      return st.mcBaseCanvas;
    }

    // Virtualized offscreen buffer: only render visible rows +/- 3-row padding (stays fast & within GPU limits even at 83ch x 80px!)
    const bufFirst = Math.max(0, firstVis - 3);
    const bufLast = Math.min(traces.length - 1, lastVis + 3);
    const bufRows = Math.max(1, bufLast - bufFirst + 1);
    const bufH = bufRows * rowH;

    if (!st.mcBaseCanvas) st.mcBaseCanvas = document.createElement("canvas");
    const off = st.mcBaseCanvas;
    const bw = Math.max(1, Math.floor(pw * dpr));
    const bh = Math.max(1, Math.floor(bufH * dpr));
    if (off.width !== bw || off.height !== bh) {
      off.width = bw;
      off.height = bh;
    }
    const octx = off.getContext("2d");
    octx.setTransform(dpr, 0, 0, dpr, 0, 0);
    octx.clearRect(0, 0, pw, bufH);

    const t0 = d.start_seconds, t1 = d.end_seconds, span = Math.max(1e-6, t1 - t0);
    const oxOf = (t) => ((t - t0) / span) * pw;

    const colAccent = css("--accent");
    const colAccentSoft = css("--accent-soft");
    const colCanvasAlt = css("--canvas-alt");
    const colBorderStrong = css("--border-strong");
    const colTrace = css("--trace");

    // Pass 1: lane backgrounds, QC shading, and dashed shaft dividers
    for (let i = bufFirst; i <= bufLast; i++) {
      const tr = traces[i];
      const ly = (i - bufFirst) * rowH;
      const isFocus = tr.channel_name === st.channel;
      if (isFocus) {
        octx.fillStyle = colAccentSoft;
        octx.fillRect(0, ly, pw, rowH);
      } else if (i % 2 === 1) {
        octx.fillStyle = colCanvasAlt;
        octx.fillRect(0, ly, pw, rowH);
      }
      if (i > 0 && traces[i - 1].display_shaft !== tr.display_shaft) {
        octx.save();
        octx.strokeStyle = colBorderStrong;
        octx.setLineDash([4, 3]);
        octx.lineWidth = 1;
        octx.beginPath();
        octx.moveTo(0, ly);
        octx.lineTo(pw, ly);
        octx.stroke();
        octx.restore();
      }
      (tr.invalid_seconds || []).forEach(([ia, ib]) => {
        octx.fillStyle = "rgba(220, 38, 38, 0.11)";
        octx.fillRect(oxOf(ia), ly, Math.max(2, oxOf(ib) - oxOf(ia)), rowH);
      });
    }

    // Pass 2: Unclipped clinical SEEG traces (natural vertical excursions bounded only by buffer height for GPU safety)
    for (let i = bufFirst; i <= bufLast; i++) {
      const tr = traces[i];
      const ym = (i - bufFirst + 0.5) * rowH;
      const isFocus = tr.channel_name === st.channel;
      const chMult = readerScales.fixed() ? 1 : (st.channelGain[tr.channel_name] || 1);
      const effUvLane = uvLane / chMult;

      octx.strokeStyle = isFocus ? colAccent : colTrace;
      octx.lineWidth = isFocus ? 1.25 : 0.92;
      octx.beginPath();
      let down = false;
      const ts = tr.times, vs = tr.values_uv;
      const maxExcursion = Math.max(bufH, ph);
      for (let k = 0; k < ts.length; k++) {
        const v = vs[k];
        if (v === null) { down = false; continue; }
        const x = oxOf(ts[k]);
        const dy = Math.max(-maxExcursion, Math.min(maxExcursion, (v / effUvLane) * rowH));
        if (!down) { octx.moveTo(x, ym - dy); down = true; }
        else octx.lineTo(x, ym - dy);
      }
      octx.stroke();
    }

    st.mcBaseKey = baseKey;
    st.mcBaseFirstRow = bufFirst;
    st.mcBaseLastRow = bufLast;
    return off;
  }

  function drawMultichannel(forceRebuild = false) {
    if (forceRebuild) st.mcBaseKey = "";
    const canvas = $("mcCanvas");
    const d = st.mcData;
    if (!canvas || !d) return;
    const dpr = window.devicePixelRatio || 1;
    const { ctx, w, h } = hiDPI(canvas);
    ctx.clearRect(0, 0, w, h);

    const allTraces = d.traces || [];
    const traces = allTraces.filter((tr) => !st.hiddenChannels.has(tr.channel_name));
    if ($("unhideAllBtn")) $("unhideAllBtn").classList.toggle("hidden", st.hiddenChannels.size === 0);
    const padL = MC_PAD_L, padT = 22, padB = MC_PAD_B;
    const pw = mcPlotWidth(w);
    const t0 = d.start_seconds, t1 = d.end_seconds, span = Math.max(1e-6, t1 - t0);
    const speedLabel = readerScales.timebaseLabel(pw, span);
    const speedReadout = $("readerTimebaseReadout");
    if (speedReadout.textContent !== speedLabel) speedReadout.textContent = speedLabel;
    if (!traces.length) return;

    const ph = Math.max(20, h - padT - padB);
    const { rowH, totalH, maxScrollY } = computeMcVerticalLayout(traces, ph);
    const scrollY = st.scrollY;

    // Only the displayed response can establish the Auto baseline during a pending filter change.
    const calKey = [d.case, d.recording, d.run_name, d.requested_montage, d.requested_display_filter,
      d.requested_display_lff, d.requested_display_hff, d.display_notch].join("|");
    if (st.calKey !== calKey || !st.baseUvLane) {
      st.baseUvLane = computeMedianP95Uv(traces);
      st.calKey = calKey;
    }
    const uvLane = readerScales.fixed() ? readerScales.uvPerRow(rowH) : st.baseUvLane / st.gain;

    // Continuous visual offset: ZERO snap-back while pan/wheel HTTP request is in flight!
    const previewShiftSec = st.viewStartSec - t0;
    const shiftPx = -(previewShiftSec / span) * pw;
    const xOf = (t) => padL + ((t - t0) / span) * pw + shiftPx;
    const yOfRow = (idx) => padT + idx * rowH - scrollY;

    const colGrid = css("--canvas-grid");
    const colInk1 = css("--ink-1");
    const colInk2 = css("--ink-2");
    const colAccent = css("--accent");
    const colAccentSoft = css("--accent-soft");
    const colBorderStrong = css("--border-strong");

    // 1) Blit cached high-DPI virtualized waveform layer in <0.3ms!
    const baseCanvas = ensureMcBaseCanvas(d, traces, pw, ph, rowH, dpr, uvLane);
    const bufH = (st.mcBaseLastRow - st.mcBaseFirstRow + 1) * rowH;
    const baseTopY = padT + st.mcBaseFirstRow * rowH - scrollY;

    ctx.save();
    ctx.beginPath();
    ctx.rect(padL, padT, pw, ph);
    ctx.clip();
    ctx.drawImage(baseCanvas, padL + shiftPx, baseTopY, pw, bufH);
    drawGapBands(ctx, d.gaps, xOf, padL, padL + pw, padT, ph);

    // Subtle hover lane tint (lightweight single rect, no trace redraw!)
    if (st.mcHover && st.mcHover.chIdx >= 0 && st.mcHover.chIdx < traces.length && traces[st.mcHover.chIdx].channel_name !== st.channel) {
      ctx.fillStyle = st.theme === "dark" ? "rgba(255,255,255,0.038)" : "rgba(15,23,42,0.032)";
      ctx.fillRect(padL, yOfRow(st.mcHover.chIdx), pw, rowH);
    }

    // Vertical time gridlines
    ctx.strokeStyle = colGrid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 8; i++) {
      const gx = padL + (i / 8) * pw;
      ctx.moveTo(gx, padT);
      ctx.lineTo(gx, padT + ph);
    }
    ctx.stroke();

    // Synchronized Triad Viewfinder Band (immediate target bracket while loading, then exact Triad window!)
    let triadWin = null;
    const trFocus = st.triadData;
    if (st.pendingTriadTarget) {
      const tx0 = xOf(st.pendingTriadTarget.w0), tx1 = xOf(st.pendingTriadTarget.w1);
      if (tx1 >= padL && tx0 <= padL + pw) {
        triadWin = { x0: Math.max(padL, tx0), x1: Math.min(padL + pw, Math.max(tx0 + 4, tx1)), ch: st.pendingTriadTarget.channel };
      }
    } else if (trFocus && trFocus.window_seconds && trFocus.recording === st.recording && trFocus.run_name === st.runName) {
      const [tw0, tw1] = trFocus.window_seconds;
      const tx0 = xOf(tw0), tx1 = xOf(tw1);
      if (tx1 >= padL && tx0 <= padL + pw) {
        triadWin = { x0: Math.max(padL, tx0), x1: Math.min(padL + pw, Math.max(tx0 + 4, tx1)), ch: trFocus.channel_name };
      }
    }
    if (triadWin) {
      ctx.fillStyle = st.theme === "dark" ? "rgba(56, 189, 248, 0.08)" : "rgba(29, 78, 216, 0.06)";
      ctx.fillRect(triadWin.x0, padT, triadWin.x1 - triadWin.x0, ph);
      const focusIdx = traces.findIndex((tr) => tr.channel_name === triadWin.ch);
      if (focusIdx >= 0) {
        const fy = yOfRow(focusIdx);
        if (fy + rowH >= padT && fy <= padT + ph) {
          ctx.fillStyle = st.theme === "dark" ? "rgba(56, 189, 248, 0.16)" : "rgba(29, 78, 216, 0.13)";
          ctx.fillRect(triadWin.x0, fy, triadWin.x1 - triadWin.x0, rowH);
          ctx.strokeStyle = colAccent;
          ctx.lineWidth = 1.6;
          ctx.strokeRect(triadWin.x0, fy + 0.5, triadWin.x1 - triadWin.x0, rowH - 1);
        }
      }
      ctx.strokeStyle = st.theme === "dark" ? "rgba(56, 189, 248, 0.45)" : "rgba(29, 78, 216, 0.35)";
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(triadWin.x0, padT); ctx.lineTo(triadWin.x0, padT + ph);
      ctx.moveTo(triadWin.x1, padT); ctx.lineTo(triadWin.x1, padT + ph);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Candidate event overlays (green = focal, red = widespread burst)
    const idxByCid = {};
    traces.forEach((tr, i) => { idxByCid[tr.channel_id] = i; });
    st.mcBoxes = [];
    const evRows = new Set();
    (d.events || []).forEach((ev) => {
      const i = idxByCid[ev.channel_id];
      if (i === undefined) return;
      evRows.add(i);
      const y0 = yOfRow(i) + 2, bh = rowH - 4;
      if (y0 + bh < padT || y0 > padT + ph) return;
      const xa = xOf(ev.start_seconds), xb = Math.max(xa + 5, xOf(ev.end_seconds));
      const isBurst = ev.synchrony_class === "widespread_burst";
      ctx.fillStyle = isBurst ? "rgba(220, 38, 38, 0.16)" : "rgba(5, 150, 105, 0.20)";
      ctx.fillRect(xa, y0, xb - xa, bh);
      ctx.strokeStyle = isBurst ? "#dc2626" : "#059669";
      ctx.lineWidth = ev.id === st.selectedItemId ? 2.0 : 1.1;
      ctx.strokeRect(xa, y0, xb - xa, bh);
      st.mcBoxes.push({ x0: xa - 3, x1: xb + 3, y0, y1: y0 + bh, ev });
    });

    // Exclude / Clip interval selection overlay (strictly anchored in time & active context!)
    if (
      st.dragSel &&
      st.dragSel.caseSlug === st.caseSlug &&
      st.dragSel.recording === st.recording &&
      st.dragSel.runName === st.runName
    ) {
      const lx = st.dragSel.t0 !== undefined ? xOf(Math.min(st.dragSel.t0, st.dragSel.t1)) : Math.min(st.dragSel.x0, st.dragSel.x1);
      const rx = st.dragSel.t0 !== undefined ? xOf(Math.max(st.dragSel.t0, st.dragSel.t1)) : Math.max(st.dragSel.x0, st.dragSel.x1);
      if (rx >= padL && lx <= padL + pw) {
        ctx.fillStyle = "rgba(2, 132, 199, 0.16)";
        ctx.fillRect(lx, padT, Math.max(1, rx - lx), ph);
        ctx.strokeStyle = "#0284c7";
        ctx.lineWidth = 1.4;
        ctx.strokeRect(lx, padT, Math.max(1, rx - lx), ph);
      }
    }

    // Quantitative Measurement Caliper Overlay (anchored to time + channel name so pan/scroll/shaft stay exact!)
    if (
      st.measureDrag &&
      st.measureDrag.caseSlug === st.caseSlug &&
      st.measureDrag.recording === st.recording &&
      st.measureDrag.t0 !== undefined &&
      st.measureDrag.t1 !== undefined
    ) {
      const ci = st.measureDrag.channel ? traces.findIndex((tr) => tr.channel_name === st.measureDrag.channel) : -1;
      if (!st.measureDrag.channel || ci >= 0) {
        const lx = xOf(Math.min(st.measureDrag.t0, st.measureDrag.t1));
        const rx = xOf(Math.max(st.measureDrag.t0, st.measureDrag.t1));
        if (rx >= padL && lx <= padL + pw) {
          const cy0 = ci >= 0 ? yOfRow(ci) : padT;
          const chH = ci >= 0 ? rowH : ph;
          ctx.fillStyle = "rgba(14, 165, 233, 0.18)";
          ctx.fillRect(lx, cy0, Math.max(1, rx - lx), chH);
          ctx.strokeStyle = "#0ea5e9";
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(lx, padT); ctx.lineTo(lx, padT + ph);
          ctx.moveTo(rx, padT); ctx.lineTo(rx, padT + ph);
          const midY = cy0 + chH * 0.5;
          ctx.moveTo(lx, midY); ctx.lineTo(rx, midY);
          ctx.stroke();
        }
      }
    }

    // Real-time Hover Vertical Crosshair
    if (st.mcHover && !(st.panDrag && st.panDrag.active)) {
      ctx.save();
      ctx.strokeStyle = st.theme === "dark" ? "rgba(226, 232, 240, 0.45)" : "rgba(15, 23, 42, 0.35)";
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(st.mcHover.x, padT);
      ctx.lineTo(st.mcHover.x, padT + ph);
      ctx.stroke();
      ctx.restore();
    }
    ctx.restore();

    // 2) Left Channel Gutter Labels + Shaft Dividers (clipped strictly to visible plot height!)
    st.mcLabelHits = [];
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, padT, padL, ph);
    ctx.clip();
    const firstVis = Math.max(0, Math.floor(scrollY / rowH));
    const lastVis = Math.min(traces.length - 1, Math.ceil((scrollY + ph) / rowH));
    for (let i = firstVis; i <= lastVis; i++) {
      const tr = traces[i];
      const ry = yOfRow(i);
      const ym = ry + 0.5 * rowH;
      const isFocus = tr.channel_name === st.channel;
      const chMult = readerScales.fixed() ? 1 : (st.channelGain[tr.channel_name] || 1);
      if (i > 0 && traces[i - 1].display_shaft !== tr.display_shaft) {
        ctx.save();
        ctx.strokeStyle = colBorderStrong;
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(8, ry);
        ctx.lineTo(padL, ry);
        ctx.stroke();
        ctx.restore();
      }
      const lblTxt = tr.display_label.slice(0, 10) + (chMult !== 1 ? ` ×${chMult}` : "");
      if (isFocus) {
        ctx.fillStyle = colAccentSoft;
        ctx.beginPath();
        ctx.roundRect(5, ym - 9, padL - 12, 18, 5);
        ctx.fill();
      }
      ctx.fillStyle = isFocus ? colAccent : colInk1;
      ctx.font = isFocus ? "700 10.8px ui-monospace, Menlo, monospace" : "500 10.5px ui-monospace, Menlo, monospace";
      ctx.textAlign = "left";
      ctx.fillText(lblTxt, 10, ym + 3.5);
      st.mcLabelHits.push({
        y0: Math.max(padT, ry), y1: Math.min(padT + ph, ry + rowH),
        channel: tr.channel_name, tooltip: channelTooltip(tr),
      });
    }
    ctx.restore();

    // 2b) Interactive Right-Edge Vertical Scrollbar (appears whenever channels exceed viewport height!)
    if (maxScrollY > 0.5) {
      const sbX = w - 11, sbW = 7, sbY = padT + 2, sbH = Math.max(24, ph - 4);
      ctx.fillStyle = st.theme === "dark" ? "rgba(255,255,255,0.06)" : "rgba(15,23,42,0.06)";
      ctx.beginPath();
      ctx.roundRect(sbX, sbY, sbW, sbH, 4);
      ctx.fill();

      // Event & Focus-channel position markers on scrollbar track
      evRows.forEach((ri) => {
        const my = sbY + ((ri + 0.5) / traces.length) * sbH;
        ctx.fillStyle = "#10b981";
        ctx.fillRect(sbX + 1, my - 1.5, sbW - 2, 3);
      });
      const focusIdx = traces.findIndex((tr) => tr.channel_name === st.channel);
      if (focusIdx >= 0) {
        const my = sbY + ((focusIdx + 0.5) / traces.length) * sbH;
        ctx.fillStyle = colAccent;
        ctx.fillRect(sbX, my - 2, sbW, 4);
      }

      const thumbH = Math.max(26, (ph / totalH) * sbH);
      const thumbY = sbY + (scrollY / maxScrollY) * (sbH - thumbH);
      ctx.fillStyle = st.vScrollDrag ? colAccent : (st.theme === "dark" ? "rgba(203,213,225,0.45)" : "rgba(71,85,105,0.42)");
      ctx.beginPath();
      ctx.roundRect(sbX, thumbY, sbW, thumbH, 4);
      ctx.fill();
      st.mcScrollRect = { x: sbX - 4, y: sbY, w: sbW + 8, h: sbH, thumbY, thumbH, maxScrollY };
    } else {
      st.mcScrollRect = null;
    }

    // Sensitivity and visible channel range, above the plot.
    ctx.textAlign = "left";
    ctx.fillStyle = colAccent;
    ctx.font = "600 10px ui-monospace, Menlo, monospace";
    const rowTag = maxScrollY > 0.5 ? ` · ${firstVis + 1}–${Math.min(traces.length, lastVis + 1)}/${traces.length}` : "";
    ctx.fillText(readerScales.scaleLabel(uvLane) + rowTag, 10, 14);
    ctx.strokeStyle = colGrid;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL + pw, padT);
    ctx.moveTo(padL, padT + ph);
    ctx.lineTo(padL + pw, padT + ph);
    ctx.stroke();

    // 4) Bottom Time Axis + Bottom-Left Non-Occluding Calibration Scale
    ctx.fillStyle = colInk2;
    ctx.font = "10.2px ui-monospace, Menlo, monospace";
    for (let i = 0; i <= 8; i++) {
      const t = t0 + previewShiftSec + (i / 8) * span;
      const x = padL + (i / 8) * pw;
      ctx.textAlign = i === 0 ? "left" : (i === 8 ? "right" : "center");
      const lx = i === 0 ? x + 2 : (i === 8 ? x - 2 : x);
      ctx.fillText(`${Math.max(0, t).toFixed(2)}s`, lx, h - 5);
    }
    ctx.strokeStyle = colInk2;
    ctx.lineWidth = 1;
    ctx.font = "600 9.4px ui-monospace, Menlo, monospace";
    readerScales.drawPulse(ctx, padL, h - 17, pw, span, rowH / uvLane);
  }

  function setVerdictButtonsBusy(busy) {
    ["vConfirmBtn", "vArtifactBtn", "vUncertainBtn", "vClipBtn"].forEach((id) => {
      const el = $(id);
      if (el) el.disabled = busy;
    });
  }

  async function loadTriadCanvas(opts = {}) {
    if (!st.caseSlug || !st.recording) return;
    const seq = ++st.triadReqSeq;
    const reqCase = st.caseSlug;
    const reqRec = st.recording;
    const reqRun = st.runName;
    st.triadLoading = true;
    // Keep previous Triad & buttons steady during fast (<350ms) loads — saveVerdict() is already guarded by st.triadLoading!
    const slowDimTimer = setTimeout(() => {
      if (st.triadLoading && seq === st.triadReqSeq) setVerdictButtonsBusy(true);
    }, 350);

    const params = {
      source_only: st.sourceOnly ? 1 : null,
      case: reqCase,
      recording: reqRec,
      run_name: reqRun,
      channel: st.channel || "",
      window_seconds: opts.windowSeconds !== undefined ? opts.windowSeconds : st.triadSpan,
    };
    const mt = st.manualCursorTarget;
    const mtMatches = Boolean(
      mt &&
      mt.caseSlug === reqCase &&
      mt.recording === reqRec &&
      mt.runName === (reqRun || "baseline")
    );
    if (opts.eventId !== undefined) {
      params.event_id = opts.eventId;
      if (opts.centerSeconds !== undefined) params.center_seconds = opts.centerSeconds;
      // A concurrent save/SSE refresh must follow the click even while its fetch is in flight.
      if (opts.eventId === "" && Number.isFinite(opts.centerSeconds)) {
        st.manualCursorTarget = {
          caseSlug: reqCase, recording: reqRec, runName: reqRun || "baseline", channel: params.channel,
          eventId: "", centerSeconds: opts.centerSeconds, windowSeconds: params.window_seconds,
        };
      } else if (opts.eventId !== "" && opts.eventId !== "control") {
        st.manualCursorTarget = null;
      }
    } else if (st.selectedItemId && !String(st.selectedItemId).startsWith("ctrl_")) {
      params.event_id = st.selectedItemId;
      st.manualCursorTarget = null;
    } else if (mtMatches) {
      // Preserve exact original manual cursor / background control target across co-review SSE refreshes (zero endpoint drift!)
      params.channel = mt.channel || params.channel;
      params.event_id = mt.eventId;
      params.center_seconds = mt.centerSeconds;
      if (opts.windowSeconds === undefined && Number.isFinite(Number(mt.windowSeconds)) && Number(mt.windowSeconds) >= 0.2) {
        params.window_seconds = Number(mt.windowSeconds);
      }
    } else if (
      st.triadData &&
      !st.triadData.focus_event &&
      st.triadData.case === reqCase &&
      st.triadData.recording === reqRec &&
      (st.triadData.run_name || "baseline") === (reqRun || "baseline") &&
      Array.isArray(st.triadData.window_seconds) &&
      st.triadData.window_seconds.length === 2
    ) {
      // Preserve active manual cursor / background control window across lite refreshes (NEVER jump back to top_focal!)
      const w0 = Number(st.triadData.window_seconds[0]);
      const w1 = Number(st.triadData.window_seconds[1]);
      params.channel = st.triadData.channel_name || params.channel;
      params.event_id = st.triadData.view_mode === "background_control" ? "control" : "";
      params.center_seconds = 0.5 * (w0 + w1);
      if (opts.windowSeconds === undefined && Number.isFinite(w1 - w0) && w1 - w0 >= 0.2) {
        params.window_seconds = +(w1 - w0).toFixed(5);
      }
    } else {
      params.event_id = "top_focal";
    }

    const curEvId = st.triadData && st.triadData.focus_event ? st.triadData.focus_event.id : null;
    const sameLoadedTarget = Boolean(
      st.triadData &&
      st.triadData.case === reqCase &&
      st.triadData.recording === reqRec &&
      (st.triadData.run_name || "baseline") === (reqRun || "baseline") &&
      (
        (params.event_id && params.event_id === curEvId) ||
        (!curEvId &&
          !params.event_id &&
          params.center_seconds !== undefined &&
          Array.isArray(st.triadData.window_seconds) &&
          st.triadData.channel_name === params.channel &&
          Math.abs(0.5 * (st.triadData.window_seconds[0] + st.triadData.window_seconds[1]) - Number(params.center_seconds)) < 1e-3)
      )
    );
    if (!sameLoadedTarget) {
      setVerdictButtonsBusy(true);
    }
    try {
      const canParallelCtrl = Boolean(
        st.sideBySide && params.event_id !== "control" && params.event_id !== "" && params.channel
      );
      const triadPromise = getJSON("/api/triad", params);
      const parallelCtrlPromise = canParallelCtrl
        ? getJSON("/api/triad", {
            case: params.case,
            recording: params.recording,
            run_name: params.run_name,
            channel: params.channel,
            event_id: "control",
            window_seconds: st.triadSpan,
          }).catch(() => null)
        : Promise.resolve(null);

      const [tr, prefetchedCtrl] = await Promise.all([triadPromise, parallelCtrlPromise]);
      clearTimeout(slowDimTimer);
      if (seq !== st.triadReqSeq || st.caseSlug !== reqCase || st.recording !== reqRec || st.runName !== reqRun) return;

      let ctrlTr = null;
      const allowSideBySide = st.sideBySide && tr.view_mode !== "background_control" && Boolean(tr.focus_event);
      if (allowSideBySide) {
        if (prefetchedCtrl && prefetchedCtrl.channel_name === tr.channel_name) {
          ctrlTr = prefetchedCtrl;
        } else {
          try {
            ctrlTr = await getJSON("/api/triad", {
              case: params.case,
              recording: params.recording,
              run_name: params.run_name,
              channel: tr.channel_name,
              event_id: "control",
              window_seconds: st.triadSpan,
            });
          } catch (_) {
            ctrlTr = null;
          }
          if (seq !== st.triadReqSeq || st.caseSlug !== reqCase || st.recording !== reqRec || st.runName !== reqRun) return;
        }
      }
      st.triadLoading = false;
      st.pendingTriadTarget = null;
      st.triadData = tr;
      st.ctrlTriadData = ctrlTr;
      st.channel = tr.channel_name;
      if (tr.focus_event) {
        st.manualCursorTarget = null;
      } else if (Array.isArray(tr.window_seconds) && tr.window_seconds.length === 2) {
        const w0 = Number(tr.window_seconds[0]);
        const w1 = Number(tr.window_seconds[1]);
        const reqWinVal = Number(params.window_seconds);
        const curMt = st.manualCursorTarget;
        const keepExistingMt = Boolean(
          curMt &&
          opts.eventId === undefined &&
          opts.windowSeconds === undefined &&
          curMt.caseSlug === (tr.case || reqCase) &&
          curMt.recording === (tr.recording || reqRec) &&
          curMt.runName === (tr.run_name || reqRun || "baseline") &&
          curMt.channel === tr.channel_name
        );
        if (!keepExistingMt) {
          const effCenter = opts.centerSeconds !== undefined
            ? Number(opts.centerSeconds)
            : (0.5 * (w0 + w1));
          const effSpan = (Number.isFinite(reqWinVal) && reqWinVal >= 0.2)
            ? reqWinVal
            : Math.max(0.2, +(w1 - w0).toFixed(5));
          st.manualCursorTarget = {
            caseSlug: tr.case || reqCase,
            recording: tr.recording || reqRec,
            runName: tr.run_name || reqRun || "baseline",
            channel: tr.channel_name,
            eventId: tr.view_mode === "background_control" ? "control" : "",
            centerSeconds: effCenter,
            windowSeconds: effSpan,
          };
        }
      }
      const trItemKey = (tr.focus_event && tr.focus_event.id) || windowDraftToken(tr);
      swapNoteDraftForItem(tr.case || reqCase, tr.recording || reqRec, tr.run_name || reqRun || "baseline", trItemKey);
      updateVerdictPanel(tr);
      drawTriad();
      drawMultichannel();
      drawReaderMinimap();
    } catch (e) {
      clearTimeout(slowDimTimer);
      if (seq === st.triadReqSeq) {
        st.triadLoading = false;
        st.pendingTriadTarget = null;
        st.triadData = null;
        st.ctrlTriadData = null;
        setVerdictButtonsBusy(true);
        const header = $("triadHeaderText");
        header.dataset.i18n = "Failed to load event — select another candidate";
        header.textContent = t("Failed to load event — select another candidate");
        toast(e.message);
      }
    }
  }

  function updateVerdictPanel(tr) {
    const ev = tr.focus_event;
    const sm = tr.signal_metrics || {};
    const channel = channelCaption(tr);
    if (st.sideBySide && st.ctrlTriadData && tr.view_mode !== "background_control" && ev) {
      const csm = st.ctrlTriadData.signal_metrics || {};
      const ratio = (sm.ripple_rms_uv && csm.ripple_rms_uv > 0) ? t(" ({ratio}× floor)", { ratio: (sm.ripple_rms_uv / csm.ripple_rms_uv).toFixed(1) }) : "";
      $("triadHeaderText").textContent = t("{channel} · Ripple RMS: candidate {candidate} µV, control {control} µV{ratio}", {
        channel, candidate: sm.ripple_rms_uv ?? "—", control: csm.ripple_rms_uv ?? "—", ratio,
      });
    } else if (tr.view_mode === "background_control") {
      $("triadHeaderText").textContent = t("{channel} · {recording} · Quiet control: ripple RMS {rms} µV, P95 {p95} µV", {
        channel, recording: tr.recording, rms: sm.ripple_rms_uv ?? "—", p95: sm.ripple_p95_uv ?? "—",
      });
    } else {
      const peak = tr.peak_hfo_frequency_hz ? t(" · Peak ~{peak} Hz (+{level} dB)", { peak: tr.peak_hfo_frequency_hz, level: tr.peak_island_db }) : "";
      $("triadHeaderText").textContent = `${channel} (${tr.recording})${peak}`;
    }

    const badge = $("syncClassBadge");
    ["vConfirmBtn", "vArtifactBtn", "vUncertainBtn", "vClipBtn"].forEach((id) => $(id).classList.remove("active"));
    const isNonEvent = tr.view_mode === "background_control" || !ev;
    const isThisTargetSaving = st.savingTargets.has(currentTriadSaveKey(tr));
    const blockButtons = isThisTargetSaving || st.contextSwitching || st.sourceOnly || !isActiveCaseLoaded();
    ["vConfirmBtn", "vArtifactBtn", "vUncertainBtn"].forEach((id) => { $(id).disabled = isNonEvent || blockButtons; });
    $("vClipBtn").disabled = blockButtons;

    if (isNonEvent) {
      badge.textContent = tr.view_mode === "background_control" ? t("QUIET CONTROL") : t("WINDOW");
      badge.className = "sync-tag ctrl";
      const windowLabel = tr.gaps?.length && tr.tfr?.pieces?.length === 0
        ? t("Recording gap. No samples in this window.")
        : st.sourceOnly || !st.active?.run_summary ? t("Acquired signal · manual cursor window.")
        : t("Candidate-free background / manual cursor window.");
      $("verdictEventMeta").innerHTML = `<strong>${esc(channel)}</strong> · <code>[${tr.window_seconds[0].toFixed(2)}, ${tr.window_seconds[1].toFixed(2)}]s</code><br/>${windowLabel}`;
      return;
    }

    const isBurst = ev.synchrony_class === "widespread_burst";
    const kind = isBurst ? t("Synchronous burst") : ev.synchrony_class === "regional" ? t("Regional") : t("Focal");
    const coactive = [t("{count} ch", { count: ev.coactive_channel_count })];
    if (tr.display_shaft) coactive.push(`${ev.coactive_shaft_count} ${ev.coactive_shaft_count === 1 ? t("shaft") : t("shafts")}`);
    badge.textContent = `${kind} · ${coactive.join(" / ")}`;
    badge.className = "sync-tag " + (isBurst ? "burst" : "focal");

    const durMs = ((ev.end_seconds - ev.start_seconds) * 1000).toFixed(1);
    const scoreLines = ["artifact", "spike", "ehfo"].map(stage => modelScoreLabel(ev, stage))
      .filter(Boolean).map(line => `<div>${esc(line)}</div>`).join("");
    const domStr = ev.dominant_band === "fast_ripple" ? ` · <strong>${t("FR-dom")}</strong>` : "";
    const effVerdict = ev.clinician_verdict || ev.researcher_verdict || ev.rehearsal_verdict || null;
    const verdictLabel = ev.clinician_verdict
      ? t(ev.clinician_verdict)
      : (ev.researcher_verdict
        ? t("{verdict} (researcher)", { verdict: t(ev.researcher_verdict) })
        : (ev.rehearsal_verdict ? t("{verdict} (rehearsal)", { verdict: t(ev.rehearsal_verdict) }) : t("unreviewed")));
    const extraRoles = [];
    if (ev.clinician_verdict && ev.researcher_verdict) extraRoles.push(t("Researcher: {verdict}", { verdict: esc(t(ev.researcher_verdict)) }));
    if (ev.agent_verdict) extraRoles.push(t("Agent: {verdict}", { verdict: esc(t(ev.agent_verdict)) }));
    const agentTag = extraRoles.length ? ` · ${extraRoles.join(" · ")}` : "";
    const clipTag = ev.clipped
      ? t("★ Clipped")
      : (ev.researcher_clipped ? t("★ Clipped (researcher)") : (ev.rehearsal_clipped ? t("★ Clipped (rehearsal)") : ""));
    $("verdictEventMeta").innerHTML = `
      <strong>${esc(channel)}</strong> · <code>${ev.start_seconds.toFixed(3)}–${ev.end_seconds.toFixed(3)} s</code> · ${durMs} ms${domStr}<br/>
      ${scoreLines}
      ${t("Verdict:")} <strong>${esc(verdictLabel)}</strong>${agentTag} ${clipTag}
    `;
    if (effVerdict === "confirm") $("vConfirmBtn").classList.add("active");
    if (effVerdict === "artifact") $("vArtifactBtn").classList.add("active");
    if (effVerdict === "uncertain") $("vUncertainBtn").classList.add("active");
    if (ev.clipped || ev.researcher_clipped || ev.rehearsal_clipped) $("vClipBtn").classList.add("active");
  }

  function _signalRange(vals) {
    const fin = (vals || []).filter((v) => v !== null && Number.isFinite(v));
    if (!fin.length) return { mid: 0, rng: 20 };
    const lo = Math.min(...fin), hi = Math.max(...fin);
    return { mid: 0.5 * (lo + hi), rng: Math.max(1e-3, hi - lo) * 1.15 };
  }

  function _drawTriadColumn(ctx, tr, x0, colW, h, locks = null, colTitle = null) {
    const padL = locks ? 54 : 62, padR = 12, padT = colTitle ? 18 : 8, padB = 20, gap = 7;
    const pw = Math.max(20, colW - padL - padR);
    const ah = Math.max(40, h - padT - padB - gap * 2);
    const h1 = ah * 0.31, h2 = ah * 0.30, h3 = ah * 0.39;
    const y1 = padT, y2 = y1 + h1 + gap, y3 = y2 + h2 + gap;
    const leftX = x0 + padL;

    if (colTitle) {
      ctx.fillStyle = tr.view_mode === "background_control" ? css("--accent") : css("--focal");
      ctx.font = "bold 10.5px sans-serif";
      ctx.fillText(colTitle, leftX, 13);
    }

    const times = tr.times || [];
    if (!times.length) return;
    const [t0, t1] = tr.window_seconds;
    const span = Math.max(1e-6, t1 - t0);
    const xOf = (t) => leftX + ((t - t0) / span) * pw;

    const plotLine = (vals, top, bh, color, dash = false, fixedRng = null) => {
      const sr = _signalRange(vals);
      const mid = sr.mid;
      const rng = fixedRng || sr.rng;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = dash ? 0.95 : 1.2;
      if (dash) ctx.setLineDash([4, 3]);
      ctx.beginPath();
      let stp = false;
      for (let i = 0; i < vals.length; i++) {
        if (vals[i] === null) { stp = false; continue; }
        const x = xOf(times[i]);
        const y = Math.max(top + 1, Math.min(top + bh - 1, top + bh * 0.5 - ((vals[i] - mid) / rng) * bh));
        if (!stp) { ctx.moveTo(x, y); stp = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
      return rng;
    };

    (tr.overlapping_events || []).forEach((e) => {
      const xa = xOf(e.start_seconds), xb = Math.max(xa + 3, xOf(e.end_seconds));
      const isBurst = e.synchrony_class === "widespread_burst";
      ctx.fillStyle = isBurst ? "rgba(220,38,38,0.15)" : "rgba(5,150,105,0.18)";
      ctx.fillRect(xa, y1, xb - xa, h1);
      ctx.fillRect(xa, y2, xb - xa, h2);
    });

    ctx.strokeStyle = css("--canvas-grid");
    ctx.strokeRect(leftX, y1, pw, h1);
    const rRng = plotLine(tr.broadband_uv, y1, h1, css("--trace"), false, locks && locks.rawRng);
    if (tr.adjacent_bipolar) plotLine(tr.adjacent_bipolar.values_uv, y1, h1, "#64748b", true, locks && locks.rawRng);
    ctx.fillStyle = css("--ink-2");
    ctx.font = "9.8px sans-serif";
    ctx.fillText(t("Raw µV"), x0 + 4, y1 + 12);
    ctx.fillText(`±${Math.round(rRng * 0.5)}`, x0 + 4, y1 + h1 - 4);

    ctx.strokeStyle = css("--canvas-grid");
    ctx.strokeRect(leftX, y2, pw, h2);
    // Subtle zero-baseline for bandpass oscillation clarity
    ctx.save();
    ctx.strokeStyle = css("--canvas-grid");
    ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(leftX, y2 + h2 * 0.5); ctx.lineTo(leftX + pw, y2 + h2 * 0.5); ctx.stroke();
    ctx.restore();
    if (tr.bands.ripple && tr.bands.ripple.available) {
      const bRng = plotLine(tr.bands.ripple.values_uv, y2, h2, "#0284c7", false, locks && locks.bandRng);
      ctx.fillStyle = "#0284c7";
      ctx.fillText("80–250", x0 + 4, y2 + 12);
      ctx.fillStyle = css("--ink-2");
      ctx.fillText(`±${(bRng * 0.5).toFixed(0)}µV`, x0 + 4, y2 + h2 - 4);
    }
    if (tr.bands.fast_ripple && tr.bands.fast_ripple.available) {
      plotLine(tr.bands.fast_ripple.values_uv, y2, h2, "#dc2626", true, locks && locks.bandRng);
    }

    // Morlet Time-Frequency Island
    const tfr = tr.tfr;
    if (tfr && tfr.times?.length && tfr.db_above_median?.length) {
      const nF = tfr.frequencies_hz.length, nT = tfr.times.length;
      const off = document.createElement("canvas");
      off.width = nT; off.height = nF;
      const octx = off.getContext("2d");
      const img = octx.createImageData(nT, nF);
      let hi = (locks && locks.tfrMax) || 6;
      if (!locks) {
        for (let r = 0; r < nF; r++) for (let c = 0; c < nT; c++) if (tfr.db_above_median[r][c] > hi) hi = tfr.db_above_median[r][c];
        hi = Math.min(18, Math.max(6, hi * 0.92));
      }
      for (let r = 0; r < nF; r++) {
        const dr = nF - 1 - r;
        for (let c = 0; c < nT; c++) {
          const u = Math.max(0, Math.min(1, (tfr.db_above_median[r][c] + 2) / (hi + 2)));
          const rgb = LUT[Math.round(u * 255)];
          const p = (dr * nT + c) * 4;
          img.data[p] = rgb[0]; img.data[p + 1] = rgb[1]; img.data[p + 2] = rgb[2]; img.data[p + 3] = 255;
        }
      }
      octx.putImageData(img, 0, 0);
      for (const part of tfr.pieces) {
        const x = xOf(part.start_seconds);
        ctx.drawImage(off, part.first_column, 0, part.end_column - part.first_column, nF,
          x, y3, Math.max(1, xOf(part.end_seconds) - x), h3);
      }

      // Compact Magma dB Colorbar Strip on right edge of TFR
      const cbX = leftX + pw - 7;
      for (let py = 0; py < Math.floor(h3); py++) {
        const u = 1 - py / Math.max(1, h3 - 1);
        const rgb = LUT[Math.round(u * 255)];
        ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
        ctx.fillRect(cbX, y3 + py, 6, 1.5);
      }
      ctx.fillStyle = css("--ink-2");
      ctx.font = "8.5px ui-monospace, Menlo, monospace";
      ctx.textAlign = "right";
      ctx.fillText("dB", leftX + pw, y3 - 2);
      ctx.textAlign = "left";

      const f0 = tfr.frequencies_hz[0], f1 = tfr.frequencies_hz[nF - 1];
      [80, 250].forEach((fl) => {
        if (fl > f0 && fl < f1) {
          const yf = y3 + h3 - ((fl - f0) / (f1 - f0)) * h3;
          ctx.save(); ctx.strokeStyle = "rgba(255,255,255,0.5)"; ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.moveTo(leftX, yf); ctx.lineTo(leftX + pw - 10, yf); ctx.stroke(); ctx.restore();
        }
      });
      if (tr.focus_event) {
        [tr.focus_event.start_seconds, tr.focus_event.end_seconds].forEach((ts) => {
          const xe = xOf(ts);
          ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 1.2;
          ctx.beginPath(); ctx.moveTo(xe, y3); ctx.lineTo(xe, y3 + h3); ctx.stroke();
        });
      }
      ctx.font = "9.8px sans-serif";
      ctx.fillStyle = css("--ink-2");
      ctx.fillText(`${Math.round(f1)}Hz`, x0 + 4, y3 + 11);
      ctx.fillText("Morlet", x0 + 4, y3 + h3 * 0.54);
      ctx.fillText(`${Math.round(f0)}Hz`, x0 + 4, y3 + h3 - 2);
    }

    ctx.fillStyle = css("--ink-2");
    ctx.font = "9.8px sans-serif";
    drawGapBands(ctx, tr.gaps, xOf, leftX, leftX + pw, y1, y3 + h3 - y1);
    const nTicks = locks ? 2 : 4;
    for (let i = 0; i <= nTicks; i++) {
      const t = t0 + (i / nTicks) * span;
      ctx.textAlign = i === 0 ? "left" : (i === nTicks ? "right" : "center");
      const lx = i === 0 ? leftX : (i === nTicks ? leftX + pw - 10 : xOf(t));
      ctx.fillText(`${t.toFixed(2)}s`, lx, h - 4);
    }
    ctx.textAlign = "left";
  }

  function drawTriad() {
    const canvas = $("triadCanvas");
    const tr = st.triadData;
    if (!canvas || !tr) return;
    const { ctx, w, h } = hiDPI(canvas);
    ctx.clearRect(0, 0, w, h);

    if (st.sideBySide && st.ctrlTriadData && tr.view_mode !== "background_control" && tr.focus_event) {
      const ctrl = st.ctrlTriadData;
      const rawRng = Math.max(_signalRange(tr.broadband_uv).rng, _signalRange(ctrl.broadband_uv).rng);
      const ripA = (tr.bands.ripple && tr.bands.ripple.values_uv) || [];
      const ripB = (ctrl.bands.ripple && ctrl.bands.ripple.values_uv) || [];
      const bandRng = Math.max(_signalRange(ripA).rng, _signalRange(ripB).rng);
      let tfrMax = 6;
      const dbA = (tr.tfr && tr.tfr.db_above_median) || [];
      for (let r = 0; r < dbA.length; r++) for (let c = 0; c < dbA[r].length; c++) if (dbA[r][c] > tfrMax) tfrMax = dbA[r][c];
      tfrMax = Math.min(18, Math.max(8, tfrMax * 0.92));
      const locks = { rawRng, bandRng, tfrMax };
      const halfW = Math.floor(w * 0.5);
      _drawTriadColumn(ctx, tr, 0, halfW - 4, h, locks, t("CANDIDATE ({channel} · {start}s)", { channel: tr.channel_name, start: tr.window_seconds[0].toFixed(2) }));
      ctx.strokeStyle = css("--border");
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(halfW, 4); ctx.lineTo(halfW, h - 4); ctx.stroke();
      _drawTriadColumn(ctx, ctrl, halfW + 2, w - halfW - 2, h, locks, t("QUIET CONTROL (0 candidates · Locked ±µV & dB scale)"));
    } else {
      _drawTriadColumn(ctx, tr, 0, w, h, null, null);
    }
  }

  // --- Verdicts & Interval Actions ---

  async function saveVerdict(patch = {}, stepNext = false, ctxOverride = null) {
    if (!st.caseSlug || !st.recording || st.sourceOnly) return false;
    if (!isActiveCaseLoaded()) {
      toast(
        st.contextSwitching
          ? () => t("Switching case/recording — please wait until waveforms finish loading")
          : () => t("Active case failed to load — retry switching cases before saving"),
        2500
      );
      return false;
    }
    // Selection-bar interval clips pass explicit ctxOverride (dragged sel) and do NOT consume #noteInput;
    // Triad-dock cursor clips pass interval_seconds WITHOUT ctxOverride and DO bind to st.triadData + #noteInput!
    const isIntervalClip = Boolean(ctxOverride && Array.isArray(patch.interval_seconds));
    if (isIntervalClip) {
      if (!isRenderedWaveformSynced()) {
        toast(() => t("Waveform view is updating — wait before clipping interval"), 2200);
        return false;
      }
      if (
        ctxOverride &&
        (ctxOverride.caseSlug !== st.mcData.case ||
          ctxOverride.recording !== st.mcData.recording ||
          (ctxOverride.runName || "baseline") !== (st.mcData.run_name || "baseline"))
      ) {
        toast(() => t("Cleared stale interval selection from another case/recording"), 2500);
        return false;
      }
    } else if (st.triadLoading) {
      toast(() => t("Still loading selected event — try again in a moment"), 1800);
      return false;
    }
    // Capture event metadata atomically from rendered Triad/Waveform so rapid switches never mix Case A with Case B
    const tr = isIntervalClip ? null : st.triadData;
    const ev = tr && tr.focus_event;
    if (patch.verdict && !ev) {
      toast(() => t("Select a candidate event to record Confirm / Artifact / Uncertain"), 2200);
      return false;
    }
    if (!isIntervalClip && !tr && patch.clipped !== undefined) {
      toast(() => t("No event or window loaded to clip"), 2000);
      return false;
    }
    if (!isIntervalClip && !isRenderedTriadSynced()) {
      toast(() => t("Active view changed — reload event before saving verdict"), 2500);
      return false;
    }
    const targetCase = isIntervalClip ? st.mcData.case : tr.case;
    const targetRec = isIntervalClip ? st.mcData.recording : tr.recording;
    const targetRun = isIntervalClip ? (st.mcData.run_name || "baseline") : (tr.run_name || "baseline");
    const targetRecFp = isIntervalClip
      ? ((st.mcData && st.mcData.recording_fingerprint) || (st.active && st.active.recording_spec && st.active.recording_spec.fingerprint) || null)
      : ((tr && tr.recording_fingerprint) || (st.active && st.active.recording_spec && st.active.recording_spec.fingerprint) || null);
    const targetAsig = isIntervalClip
      ? ((st.mcData && st.mcData.analysis_signature) || (st.active && st.active.run_analysis_signature) || null)
      : ((tr && tr.analysis_signature) || (st.active && st.active.run_analysis_signature) || null);
    const targetAgen = isIntervalClip
      ? ((st.mcData && st.mcData.analysis_generation_id) || (st.active && st.active.run_analysis_generation_id) || null)
      : ((tr && tr.analysis_generation_id) || (st.active && st.active.run_analysis_generation_id) || null);
    const targetCh = isIntervalClip
      ? ((ctxOverride && ctxOverride.channel) ||
          (st.mcData.traces && st.mcData.traces.some((t) => t.channel_name === st.channel) ? st.channel : (st.mcData.traces && st.mcData.traces[0] && st.mcData.traces[0].channel_name)) ||
          st.channel)
      : ((ev && ev.channel_name) || (tr && tr.channel_name) || st.channel);
    const targetEvId = isIntervalClip ? null : (ev ? ev.id : null);
    const targetCenter = patch.center_seconds !== undefined
      ? patch.center_seconds
      : (tr && tr.window_seconds ? 0.5 * (tr.window_seconds[0] + tr.window_seconds[1]) : null);

    // Scope in-flight deduplication to this exact (case, recording, run, event/interval) so
    // switching to Case B while Case A's POST is still in flight never drops Case B's clicks!
    const targetItemKey = isIntervalClip ? "clip" : ((ev && ev.id) || windowDraftToken(tr));
    const saveKey = `${targetCase}|${targetRec}|${targetRun}|${
      isIntervalClip
        ? `clip:${patch.interval_seconds[0]}:${patch.interval_seconds[1]}:${targetCh || ""}`
        : targetItemKey
    }`;
    if (st.savingTargets.has(saveKey)) return false;

    const curIdx = st.queueItems.findIndex((x) => ev && x.id === ev.id);
    const nextTargetId = (curIdx >= 0 && curIdx + 1 < st.queueItems.length) ? st.queueItems[curIdx + 1].id : null;
    const setName = st.queueMode === "review_set" ? st.reviewSet : "";
    const nextPage = Boolean(setName && curIdx >= 0 && curIdx + 1 === st.queueItems.length && st.queueHasMore);
    const identityRevision = st.identityRevision;
    const reviewer = currentReviewerIdentity();
    // Only read & clear #noteInput when this save actually consumes the clinician's note box (NOT interval clips with patch.note)!
    const usesNoteInput = !isIntervalClip && patch.note === undefined;
    const expectedDraftKey = draftKeyFor(targetCase, targetRec, targetRun, targetItemKey);
    const draftOwned = usesNoteInput && st.lastDraftKey === expectedDraftKey;
    const submittedNote = draftOwned && $("noteInput") ? $("noteInput").value.trim() : "";
    const startEpoch = st.contextEpoch || 0;
    const startBootSeq = st.bootReqSeq;
    st.savingTargets.add(saveKey);
    if (!isIntervalClip) setVerdictButtonsBusy(true);
    const opId = genOpId();
    st.recentLocalOpIds.add(opId);
    setTimeout(() => st.recentLocalOpIds.delete(opId), 30000);
    const _contextMoved = () => (
      (st.contextEpoch || 0) !== startEpoch ||
      st.bootReqSeq !== startBootSeq ||
      st.contextSwitching ||
      st.caseSlug !== targetCase ||
      st.recording !== targetRec ||
      (st.runName || "baseline") !== targetRun
    );
    try {
      await postJSON("/api/judgment", {
        operation_id: opId,
        case: targetCase,
        recording: targetRec,
        recording_fingerprint: targetRecFp,
        analysis_signature: targetAsig,
        analysis_generation_id: targetAgen,
        run_name: targetRun,
        channel: targetCh,
        event_id: targetEvId,
        center_seconds: targetCenter,
        author: st.blinded || setName ? reviewer.author : ($("authorInput").value.trim() || "clinician"),
        role: reviewer.role,
        ...(st.blinded ? { blind_for: reviewer.author } : {}),
        ...(setName ? { review_set: setName } : {}),
        note: submittedNote,
        ...patch,
      });
      if (st.identityRevision !== identityRevision) return true;
      if (usesNoteInput && st.draftNotes[expectedDraftKey] !== undefined && st.draftNotes[expectedDraftKey].trim() === submittedNote) {
        st.draftNotes[expectedDraftKey] = "";
      }
      if (_contextMoved()) {
        // User switched case/recording/run while this POST was in flight:
        // NEVER overwrite the new context's note, queue, screen, or UI state!
        toast(
          patch.verdict
            ? () => t("Saved verdict on {case}: {verdict}", { case: targetCase, verdict: t(patch.verdict) })
            : patch.clipped !== undefined
            ? () => t("Updated Case Sheet clip on {case}", { case: targetCase })
            : () => t("Saved observation on {case}", { case: targetCase }),
          1500
        );
        return true;
      }
      // Only clear #noteInput when THIS save consumed #noteInput AND the clinician is STILL focused on the exact same event/window!
      const stillOnSameTarget = st.lastDraftKey === expectedDraftKey;
      if (usesNoteInput && stillOnSameTarget && $("noteInput") && $("noteInput").value.trim() === submittedNote) {
        $("noteInput").value = "";
        st.draftNotes[expectedDraftKey] = "";
      }
      toast(() => patch.verdict
        ? t("Saved verdict: {verdict}", { verdict: t(patch.verdict) })
        : patch.clipped !== undefined ? t("Updated Case Sheet clip") : t("Saved observation"), 1500);

      const movedToOtherEvent = !isIntervalClip && !stillOnSameTarget;
      const willAdvance = Boolean(!movedToOtherEvent && stepNext && $("autoNextChk").checked && (nextTargetId || nextPage));
      await bootstrap({
        caseSlug: targetCase,
        recording: targetRec,
        runName: targetRun,
        keepWindow: true,
        lite: true,
        skipRender: Boolean(setName) || willAdvance || movedToOtherEvent,
      });
      if ((st.contextEpoch || 0) !== startEpoch || st.contextSwitching || st.caseSlug !== targetCase) {
        return true;
      }
      if (setName) {
        const seq = st.queueReqSeq;
        const current = () => seq === st.queueReqSeq && isActiveCaseLoaded(targetCase) &&
          (st.contextEpoch || 0) === startEpoch && st.queueMode === "review_set" && st.reviewSet === setName;
        if (!current()) return true;
        // Refresh this row and progress while keeping every loaded page in its reader-specific order.
        const page = await getJSON("/api/queue", {
          ...reviewScope(), ...reviewQueueParams(), mode: "review_set", offset: Math.max(0, curIdx), limit: 1,
        });
        if (!current()) return true;
        if (curIdx >= 0 && page.items[0]?.id === targetEvId) st.queueItems[curIdx] = page.items[0];
        renderReviewProgress(page.progress);
        renderQueueList();
        if (willAdvance && nextPage) await loadMoreQueue();
        if (!current()) return true;
        if (!willAdvance && !movedToOtherEvent && st.lastDraftKey === expectedDraftKey) await loadTriadCanvas();
      }
      if (willAdvance && st.lastDraftKey === expectedDraftKey) {
        const nextItem = nextPage ? st.queueItems[curIdx + 1] : st.queueItems.find((x) => x.id === nextTargetId);
        if (nextItem) await focusQueueItem(nextItem, true);
      }
      flushDeferredUiSync();
      return true;
    } catch (err) {
      if (st.identityRevision !== identityRevision) return false;
      toast(() => t("Failed to save judgment: {message}", { message: err.message }), 4000);
      if ((st.contextEpoch || 0) === startEpoch && !st.contextSwitching && st.triadData) {
        updateVerdictPanel(st.triadData);
      }
      return false;
    } finally {
      st.savingTargets.delete(saveKey);
      if ((st.contextEpoch || 0) === startEpoch && !st.contextSwitching) {
        if (st.pendingViewRefresh && !hasSavingForCase(st.caseSlug)) {
          st.pendingViewRefresh = false;
          bootstrap({ caseSlug: st.caseSlug, recording: st.recording, runName: st.runName, keepWindow: true, lite: true });
        } else if (!st.triadLoading && st.triadData) {
          updateVerdictPanel(st.triadData);
        }
      } else if (
        !st.contextSwitching &&
        st.caseSlug === targetCase &&
        st.recording === targetRec &&
        (st.runName || "baseline") === targetRun &&
        !hasSavingForCase(targetCase)
      ) {
        // User navigated A -> B -> A before A's POST finished: re-sync A's badges & buttons now!
        bootstrap({ caseSlug: st.caseSlug, recording: st.recording, runName: st.runName, keepWindow: true, lite: true });
      }
    }
  }

  async function stepQueue(delta) {
    if (!st.queueItems.length) return;
    const idx = st.queueItems.findIndex((x) => x.id === st.selectedItemId);
    if (st.queueMode === "review_set" && delta > 0 && idx === st.queueItems.length - 1 && st.queueHasMore) {
      const seq = st.queueReqSeq;
      await loadMoreQueue();
      if (seq !== st.queueReqSeq || !st.queueItems.length) return;
    }
    const next = Math.max(0, Math.min(st.queueItems.length - 1, (idx < 0 ? 0 : idx) + delta));
    await focusQueueItem(st.queueItems[next], true);
  }

  function renderOverlayBar() {
    const bar = $("overlaySummaryBar");
    if (!bar) return;
    if (!st.active) {
      bar.innerHTML = "";
      bar.classList.add("hidden");
      return;
    }
    const conc = st.active.overlay_concordance;
    const ov = st.active.clinical_overlay;
    if (!ov) {
      bar.innerHTML = "";
      bar.classList.add("hidden");
      return;
    }
    const mt = (conc && conc.available && conc.metrics) || null;
    const p5 = mt && mt.soz_precision_at_top_k != null ? t("Precision@{k}: {percent}%", { k: mt.top_k, percent: (100 * mt.soz_precision_at_top_k).toFixed(0) }) : "";
    const rec = mt && mt.soz_recall_all_focal != null ? t(" · Recall: {percent}%", { percent: (100 * mt.soz_recall_all_focal).toFixed(0) }) : "";
    const jac = mt && mt.soz_focal_jaccard != null ? t(" · Jaccard: {index}", { index: mt.soz_focal_jaccard.toFixed(2) }) : "";
    const sozTxt = (ov.soz_channels || []).map(esc).join(", ") || "—";
    const resTxt = (ov.resected_channels || []).map(esc).join(", ") || "—";
    bar.innerHTML = `<strong>${t("External Reference Overlay")}</strong> (${esc(ov.source)}): ${t("SOZ")} [${sozTxt}] · ${t("Resected")} [${resTxt}] ${p5 ? `· <strong>${esc(p5)}${esc(rec)}${esc(jac)}</strong>` : ""}`;
    bar.classList.remove("hidden");
  }

  function renderCohortModal() {
    const sel = $("cohProtocolSelect");
    if (sel) {
      sel.innerHTML = `<option value="">${t("(Default Modality Baseline)")}</option>`;
      st.protocols.forEach((p) => {
        const o = document.createElement("option");
        o.value = p.slug;
        o.textContent = `${p.name} (${p.modality} · ${t(p.reference_mode)})`;
        sel.appendChild(o);
      });
    }
    const chkBox = $("cohCasesChecks");
    if (chkBox) {
      chkBox.innerHTML = "";
      st.cases.forEach((c) => {
        const lbl = document.createElement("label");
        lbl.className = "chk-mini";
        lbl.innerHTML = `<input type="checkbox" value="${esc(c.case)}" checked /> ${esc(c.case)} (${t("{count} rec", { count: Number(c.recording_count || 0) })})`;
        chkBox.appendChild(lbl);
      });
    }
    const box = $("savedCohortsBox");
    if (box) {
      box.innerHTML = st.cohorts.length ? "" : `<div class="hint" style="margin:0">${t("No cohort batches saved yet.")}</div>`;
      st.cohorts.forEach((ch) => {
        const div = document.createElement("div");
        div.className = "inp-row";
        const encSlug = encodeURIComponent(String(ch.cohort_slug || ""));
        const progress = t("{done}/{total} recordings complete · protocol: {protocol}", {
          done: Number(ch.completed_recordings || 0),
          total: Number(ch.total_recordings || 0),
          protocol: ch.protocol || t("default"),
        });
        div.innerHTML = `
          <div>
            <strong>${esc(ch.cohort_slug)}</strong>
            <div class="hint" style="margin:0">${esc(progress)}</div>
          </div>
          <div class="row-inline">
            <a class="btn btn-xs" href="/cohorts/${encSlug}/cohort_summary.csv" target="_blank">${t("Summary CSV")}</a>
            <a class="btn btn-xs" href="/cohorts/${encSlug}/cohort_channels.csv" target="_blank">${t("Channels CSV")}</a>
          </div>
        `;
        box.appendChild(div);
      });
    }
  }

  async function submitAsyncJob(endpoint, body, onComplete) {
    try {
      const job = await postJSON(endpoint, body);
      st.activeJobId = job.job_id;
      const bar = $("jobBar");
      bar.classList.remove("hidden");
      $("jobTitle").textContent = job.label || t("Running job...");
      $("jobStagePct").textContent = "5%";
      $("jobFill").style.width = "5%";
      $("jobDetail").textContent = job.detail || "";

      const poll = async () => {
        if (!st.activeJobId) return;
        try {
          const cur = await getJSON(`/api/jobs/${st.activeJobId}`);
          const pct = Math.round((cur.progress || 0) * 100);
          $("jobTitle").textContent = `${cur.label} [${cur.stage}]`;
          $("jobStagePct").textContent = `${pct}%`;
          $("jobFill").style.width = `${Math.max(4, pct)}%`;
          $("jobDetail").textContent = cur.detail || "";
          if (cur.status === "complete") {
            st.activeJobId = null;
            setTimeout(() => bar.classList.add("hidden"), 1400);
            if (onComplete) await onComplete(cur.result);
            return;
          }
          if (cur.status === "failed") {
            st.activeJobId = null;
            toast(() => t("Job failed: {message}", { message: (cur.error && cur.error.message) || cur.detail }), 5000);
            setTimeout(() => bar.classList.add("hidden"), 2500);
            return;
          }
          setTimeout(poll, 320);
        } catch (err) {
          st.activeJobId = null;
          bar.classList.add("hidden");
          toast(() => t("Job status error: {message}", { message: err.message }), 4000);
        }
      };
      setTimeout(poll, 220);
    } catch (err) {
      toast(() => t("Failed to start job: {message}", { message: err.message }), 4500);
    }
  }

  // --- VIEW 3: 3D ELECTRODE MAP & CASE SHEET ---

  function drawSpatialMap() {
    const canvas = $("spatialCanvas");
    const rsum = st.active && st.active.run_summary;
    st.orbit.pts = [];
    if (canvas) canvas.removeAttribute("title");
    if (!canvas || !rsum) return;
    const { ctx, w, h } = hiDPI(canvas);
    ctx.clearRect(0, 0, w, h);

    const sp = rsum.spatial;
    const floatBar = document.querySelector(".spatial-float-bar");
    if (!sp || !sp.available || !sp.contacts.length) {
      if (floatBar) floatBar.classList.add("hidden");
      $("spatialBadge").textContent = t("No 3D coordinates provided for this case");
      ctx.fillStyle = css("--ink-2");
      ctx.font = "13px sans-serif";
      ctx.fillText(t("No electrode coordinates supplied in dataset metadata."), 28, 46);
      return;
    }
    if (floatBar) floatBar.classList.remove("hidden");
    $("spatialBadge").textContent = `${sp.coordinate_system} (${sp.units})${sp.template ? ` · ${t("TEMPLATE (not patient MRI)")}` : ""}`;

    const pts = sp.contacts;
    const xs = pts.map((c) => c.xyz[0]), ys = pts.map((c) => c.xyz[1]), zs = pts.map((c) => c.xyz[2]);
    const cx = 0.5 * (Math.min(...xs) + Math.max(...xs));
    const cy = 0.5 * (Math.min(...ys) + Math.max(...ys));
    const cz = 0.5 * (Math.min(...zs) + Math.max(...zs));
    const maxR = Math.max(1, ...pts.map((c) => Math.hypot(c.xyz[0] - cx, c.xyz[1] - cy, c.xyz[2] - cz)));
    const scale = (Math.min(w, h) * 0.40) / maxR;

    const cyw = Math.cos(st.orbit.yaw), syw = Math.sin(st.orbit.yaw);
    const cpt = Math.cos(st.orbit.pitch), spt = Math.sin(st.orbit.pitch);
    const proj = ([x, y, z]) => {
      const dx = x - cx, dy = y - cy, dz = z - cz;
      const rx = dx * cyw - dy * syw, ry = dx * syw + dy * cyw;
      return { sx: w * 0.5 + rx * scale, sy: h * 0.5 + (ry * spt - dz * cpt) * scale, z: ry * cpt + dz * spt };
    };

    // Subtle 3D stereotactic reference equatorial ring & centroid crosshair
    ctx.save();
    ctx.strokeStyle = st.theme === "dark" ? "rgba(255,255,255,0.065)" : "rgba(15,23,42,0.06)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const ringR = maxR * 0.92;
    for (let a = 0; a <= 64; a++) {
      const th = (a / 64) * Math.PI * 2;
      const rp = proj([cx + Math.cos(th) * ringR, cy + Math.sin(th) * ringR, cz]);
      if (a === 0) ctx.moveTo(rp.sx, rp.sy); else ctx.lineTo(rp.sx, rp.sy);
    }
    ctx.stroke();
    ctx.setLineDash([3, 4]);
    const pXm = proj([cx - ringR, cy, cz]), pXp = proj([cx + ringR, cy, cz]);
    const pYm = proj([cx, cy - ringR, cz]), pYp = proj([cx, cy + ringR, cz]);
    ctx.beginPath();
    ctx.moveTo(pXm.sx, pXm.sy); ctx.lineTo(pXp.sx, pXp.sy);
    ctx.moveTo(pYm.sx, pYm.sy); ctx.lineTo(pYp.sx, pYp.sy);
    ctx.stroke();
    ctx.restore();

    const byShaft = {};
    pts.filter(c => c.display_shaft).forEach(c => {
      (byShaft[c.display_shaft] ||= []).push({ ...c, p: proj(c.xyz) });
    });
    Object.values(byShaft).forEach((arr) => {
      if (arr.length > 1) {
        ctx.strokeStyle = "rgba(100, 116, 139, 0.42)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        arr.forEach((it, idx) => (idx === 0 ? ctx.moveTo(it.p.sx, it.p.sy) : ctx.lineTo(it.p.sx, it.p.sy)));
        ctx.stroke();
      }
    });

    const maxFoc = Math.max(0.1, ...pts.map((c) => c.focal_rate_per_min || 0));
    const mapped = pts.map((c) => ({ ...c, p: proj(c.xyz) })).sort((a, b) => a.p.z - b.p.z);

    const ov = st.active && st.active.clinical_overlay;
    const sozSet = new Set(((ov && ov.soz_channels) || []).map((x) => x.toUpperCase()));
    const resSet = new Set(((ov && ov.resected_channels) || []).map((x) => x.toUpperCase()));

    mapped.forEach((c) => {
      const val = c.focal_rate_per_min || 0;
      const u = val > 0 ? Math.min(1, 0.25 + 0.75 * (val / maxFoc)) : 0;
      const rad = val > 0 ? 5.5 + 7 * (val / maxFoc) : 3.8;
      const isSel = c.channel_name === st.channel;
      const isSoz = sozSet.has(c.channel_name.toUpperCase());
      const isRes = resSet.has(c.channel_name.toUpperCase());
      c._rad = rad;
      c._isSel = isSel;
      c._isOverlay = isSoz || isRes;
      if (val > 0) {
        // Subtle glowing depth halo around active focal contacts
        const rgb = LUT[Math.min(255, Math.round(70 + u * 175))];
        const halo = ctx.createRadialGradient(c.p.sx, c.p.sy, rad * 0.4, c.p.sx, c.p.sy, rad * 2.4);
        halo.addColorStop(0, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.36)`);
        halo.addColorStop(1, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0)`);
        ctx.fillStyle = halo;
        ctx.beginPath();
        ctx.arc(c.p.sx, c.p.sy, rad * 2.4, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(c.p.sx, c.p.sy, isSel ? rad + 2.5 : rad, 0, Math.PI * 2);
      if (val > 0) {
        const rgb = LUT[Math.min(255, Math.round(70 + u * 175))];
        ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      } else {
        ctx.fillStyle = "rgba(148, 163, 184, 0.58)";
      }
      ctx.fill();
      ctx.strokeStyle = isSel ? css("--accent") : "rgba(255,255,255,0.88)";
      ctx.lineWidth = isSel ? 2.2 : 1.0;
      ctx.stroke();

      if (isSoz) {
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([3, 2]);
        ctx.arc(c.p.sx, c.p.sy, rad + 5.5, 0, Math.PI * 2);
        ctx.strokeStyle = "#0284c7";
        ctx.lineWidth = 2.0;
        ctx.stroke();
        ctx.restore();
      }
      if (isRes) {
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([2, 2]);
        ctx.arc(c.p.sx, c.p.sy, rad + 8.5, 0, Math.PI * 2);
        ctx.strokeStyle = "#059669";
        ctx.lineWidth = 1.6;
        ctx.stroke();
        ctx.restore();
      }
    });

    // Collision-free 2D label placement prioritizing active focal & external overlay contacts first
    const labelCands = mapped
      .filter((c) => (c.focal_rate_per_min || 0) > 0 || c.contact_index === 1 || c._isSel || c._isOverlay)
      .sort((a, b) => ((b._isSel ? 100 : 0) + (b._isOverlay ? 20 : 0) + (b.focal_rate_per_min || 0)) - ((a._isSel ? 100 : 0) + (a._isOverlay ? 20 : 0) + (a.focal_rate_per_min || 0)));
    const placedBoxes = [];
    labelCands.forEach((c) => {
      const isHot = (c.focal_rate_per_min || 0) > 0 || c._isSel;
      ctx.font = isHot ? "bold 11px sans-serif" : "10px sans-serif";
      const tw = ctx.measureText(c.channel_name).width + 4;
      const th = 12;
      const offsets = [
        [c._rad + 4, 3],
        [c._rad + 4, -10],
        [c._rad + 4, 15],
        [-c._rad - tw - 2, 3],
      ];
      for (const [dx, dy] of offsets) {
        const lx = c.p.sx + dx;
        const ly = c.p.sy + dy;
        const box = { x0: lx - 1, y0: ly - th + 2, x1: lx + tw, y1: ly + 3 };
        const hit = placedBoxes.some((b) => !(box.x1 < b.x0 || box.x0 > b.x1 || box.y1 < b.y0 || box.y0 > b.y1));
        if (!hit) {
          placedBoxes.push(box);
          if (isHot) {
            ctx.fillStyle = css("--bg-surface");
            ctx.globalAlpha = 0.88;
            ctx.beginPath();
            ctx.roundRect(lx - 3, ly - 10, tw + 2, 14, 4);
            ctx.fill();
            ctx.globalAlpha = 1;
          }
          ctx.fillStyle = isHot ? css("--ink-0") : css("--ink-2");
          ctx.fillText(c.channel_name, lx, ly);
          break;
        }
      }
    });

    // Live-rotating 3D Anatomical Orientation Triad (R / A / S) in bottom-right corner
    const gx = w - 48, gy = h - 46, glen = 22;
    const axes = [
      { vec: [1, 0, 0], lbl: "R", col: "#ef4444" },
      { vec: [0, 1, 0], lbl: "A", col: "#10b981" },
      { vec: [0, 0, 1], lbl: "S", col: "#3b82f6" },
    ];
    axes.forEach(({ vec, lbl, col }) => {
      const rx = vec[0] * cyw - vec[1] * syw;
      const ry = vec[0] * syw + vec[1] * cyw;
      const ex = gx + rx * glen;
      const ey = gy + (ry * spt - vec[2] * cpt) * glen;
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.fillStyle = col;
      ctx.font = "bold 9.5px ui-monospace, Menlo, monospace";
      ctx.fillText(lbl, ex + (rx >= 0 ? 3 : -9), ey + 3);
    });
    st.orbit.pts = mapped;
  }

  function exportOptions() {
    const deidentify = $("deidentifyExport").checked;
    return { deidentify, pseudonym: deidentify ? $("exportPseudonym").value.trim() || null : null };
  }

  function clearExportDownloads() {
    $("exportDownloads").replaceChildren();
    $("exportDownloads").classList.add("hidden");
  }

  function showExportDownloads(files) {
    const downloads = $("exportDownloads");
    downloads.replaceChildren();
    for (const [url, label] of files) {
      if (!url) continue;
      const link = document.createElement("a");
      link.href = url;
      link.download = "";
      link.dataset.i18n = label;
      link.textContent = t(label);
      link.className = "btn btn-xs";
      downloads.appendChild(link);
    }
    downloads.classList.remove("hidden");
  }

  function exportContextMatches(epoch, options) {
    const current = exportOptions();
    return st.contextEpoch === epoch && isActiveCaseLoaded() &&
      options.deidentify === current.deidentify && options.pseudonym === current.pseudonym;
  }

  async function buildCaseSheet(opts = {}) {
    if (st.blinded) return;
    const isAuto = Boolean(opts && opts.auto);
    if (
      !st.caseSlug ||
      !st.active ||
      st.active.case !== st.caseSlug ||
      st.loadedCaseSlug !== st.caseSlug ||
      st.loadedRunName !== (st.runName || "baseline")
    ) {
      if (!isAuto) toast(() => t("Active case/run is not loaded yet — retry switching cases"), 2500);
      return;
    }
    if (!isAuto && st.contextSwitching) {
      toast(() => t("Switching case — wait until loaded before rebuilding report"), 2200);
      return;
    }
    if (!isAuto && $("deidentifyExport").checked) {
      const epoch = st.contextEpoch;
      const options = exportOptions();
      clearExportDownloads();
      try {
        const res = await postJSON("/api/export-sheet", {
          case: st.caseSlug, primary_run: st.runName, comparison_run: st.compareRun,
          clinical_question: st.lastQuestionCase === st.caseSlug ? $("questionInput").value.trim() || null : null,
          ...options,
        });
        if (exportContextMatches(epoch, options)) {
          showExportDownloads([[res.sheet_url, "Download de-identified case sheet"]]);
        }
      } catch (err) {
        if (exportContextMatches(epoch, options)) toast(() => t("Case sheet export failed: {message}", { message: err.message }), 4000);
      }
      return;
    }
    const bSeq = ++st.sheetBuildSeq;
    const bEpoch = st.contextEpoch || 0;
    const bCase = st.caseSlug;
    const bRun = st.runName;
    // Only attach #questionInput when it is explicitly owned by bCase (never carry Case A's draft into Case B!)
    const submittedQ =
      !isAuto && !st.contextSwitching && st.lastQuestionCase === bCase && $("questionInput")
        ? $("questionInput").value.trim()
        : "";
    lockSheetFrame(true);
    toast(() => t("Building Case Discussion Sheet..."));
    try {
      const res = await postJSON("/api/export-sheet", {
        case: bCase,
        primary_run: bRun,
        comparison_run: st.compareRun,
        clinical_question: submittedQ || null,
      });
      // Drop stale response if clinician switched cases or runs while export ran!
      if (bSeq !== st.sheetBuildSeq || (st.contextEpoch || 0) !== bEpoch || st.caseSlug !== bCase || st.runName !== bRun) return;
      if (submittedQ && st.lastQuestionCase === bCase && $("questionInput") && $("questionInput").value.trim() === submittedQ) {
        $("questionInput").value = "";
        st.draftQuestions[bCase] = "";
      }
      if (st.active) {
        st.active.sheet_url = res.sheet_url;
        if (res.sheet_mtime_ns) st.active.sheet_mtime_ns = res.sheet_mtime_ns;
      }
      const nextKey = `${bCase}:${bRun}:${res.sheet_mtime_ns || Date.now()}`;
      st.loadedSheetKey = nextKey;
      const fr = $("sheetFrame");
      fr.onload = () => {
        if (st.caseSlug === bCase && st.runName === bRun && st.loadedSheetKey === nextKey) {
          unlockSheetFrame();
          syncThemeToSheet();
          syncReviewerToSheet();
        }
      };
      fr.src = `${res.sheet_url}?t=${Date.now()}&theme=${encodeURIComponent(st.theme)}${sheetReviewerParams()}`;
      const a = $("printSheetLink");
      a.href = res.sheet_url;
      a.classList.remove("hidden");
      toast(() => t("Case Discussion Sheet updated"));
      flushDeferredUiSync();
    } catch (err) {
      if (bSeq === st.sheetBuildSeq && (st.contextEpoch || 0) === bEpoch) toast(() => t("Failed to build report: {message}", { message: err.message }), 4500);
    }
  }

  // --- Modals ---

  function renderOpenModal() {
    const box = $("availInputsBox");
    box.innerHTML = "";
    st.availInputs.forEach((it) => {
      const div = document.createElement("div");
      div.className = "inp-row";
      const files = t("{count} file(s): {files}", {
        count: Number(it.recording_count || 0), files: (it.files || []).slice(0, 3).join(", "),
      });
      div.innerHTML = `
        <div><strong>${esc(it.label)}</strong><div class="hint" style="margin:0">${esc(files)}</div></div>
        <div class="row-inline"><button class="btn btn-xs" data-inspect>${t("Browse recording")}</button>
        <button class="btn btn-xs btn-accent" data-analyze>${t("Open & Analyze")}</button></div>
      `;
      div.querySelector("[data-inspect]").addEventListener("click", e => browseDataset(it.target, e.currentTarget));
      div.querySelector("[data-analyze]").addEventListener("click", () => openDataset(it.target));
      box.appendChild(div);
    });
  }

  function renderProtocolOptions() {
    const sel = $("brProtocol");
    sel.innerHTML = `<option value="">${t("(Current / Standard SEEG)")}</option>`;
    st.protocols.forEach((p) => {
      const o = document.createElement("option");
      o.value = p.slug;
      o.textContent = `${p.name} (${p.modality})`;
      sel.appendChild(o);
    });
  }

  async function browseDataset(target, button) {
    if (!target || button.disabled) return;
    button.disabled = true;
    button.textContent = t("Reading recording metadata…");
    try {
      const { result } = await postJSON("/api/open-case", { target, analyze: false });
      $("openModal").classList.add("hidden");
      beginContextSwitch(true);
      await bootstrap({ caseSlug: result.case, recording: result.recordings[0].recording, targetView: "overview" });
    } catch (e) {
      toast(() => t("Navigation error: {message}", { message: e.message }));
    } finally {
      button.disabled = false;
      button.textContent = t("Browse recording");
    }
  }

  async function openDataset(target) {
    if (!target) return;
    $("openModal").classList.add("hidden");
    await submitAsyncJob("/api/jobs/open-case", { target }, async (out) => {
      beginContextSwitch(true);
      const leadCh = (out.runs[0].leading_focal_channels[0] && out.runs[0].leading_focal_channels[0].channel) || null;
      await bootstrap({
        caseSlug: out.case,
        recording: out.runs[0].recording,
        runName: out.run_name || "baseline",
        targetChannel: leadCh,
      });
      toast(() => t("Opened {case}", { case: out.case }));
    });
  }

  // --- Wire Listeners ---

  function setSourceNavigation(enabled) {
    if (st.sourceOnly === enabled) return;
    st.sourceOnly = enabled;
    $("sourceNavigation").hidden = !enabled;
    $("montageSelect").hidden = enabled;
    $("saveNoteOnlyBtn").disabled = enabled;
    $("clipSelectionBtn").disabled = enabled;
    $("showControlBtn").disabled = enabled;
    $("sideBySideBtn").disabled = enabled;
  }

  // Open the recording first. Contact IDs in annotations belong to that recording.
  async function jumpFromNavigation(recording, time, channel = null, sourceSignal = false, marker = null) {
    if (!isActiveCaseLoaded() || st.blinded || !Number.isFinite(time)) return;
    const expected = st.caseSlug;
    if (recording !== st.recording) {
      beginContextSwitch();
      const epoch = st.contextEpoch;
      await bootstrap({ caseSlug: expected, recording, runName: st.runName, skipRender: true, lite: true });
      if (st.contextEpoch !== epoch || !isActiveCaseLoaded(expected) || st.recording !== recording) return;
    }
    st.navigationMarker = marker;
    const raw = st.active.recording_spec?.channels || [];
    const rawCh = raw.find(ch => ch.id === channel || ch.name === channel);
    const runChannels = st.active.run_summary?.channels || [];
    const derived = st.active.run_summary?.effective_reference?.mode;
    setSourceNavigation(Boolean(sourceSignal && rawCh && (rawCh.is_auxiliary || (derived && derived !== "original"))));
    const ch = st.sourceOnly ? rawCh.name : (runChannels.find(c => c.channel_name === channel)?.channel_name
      || rawCh?.name || runChannels[0]?.channel_name || raw.find(c => c.unit === "V" && !c.is_auxiliary)?.name);
    st.channel = ch || null;
    st.selectedItemId = null;
    st.manualCursorTarget = null;
    st.montage = "acquired";
    st.shaft = resolveChannelShaft(st.channel) || "ALL";
    st.hiddenChannels.delete(st.channel);
    st.scrollY = 0;
    st.startSec = betweenReaderBounds(time - st.winSec / 2);
    st.viewStartSec = st.startSec;
    $("readerStartInput").value = st.startSec.toFixed(1);
    switchView("reader", true, { keepWindow: true, manualTime: time });
  }

  function betweenReaderBounds(start) {
    return Math.max(0, Math.min(Math.max(0, totalRecordingDuration() - st.winSec), start));
  }

  const questions = createAgentQuestions({
    getCase: () => !st.blinded && isActiveCaseLoaded() ? st.caseSlug : null,
    getIdentity: currentReviewerIdentity, getJSON, postJSON, operationId: genOpId, onSubmitted: flushDeferredUiSync,
  });
  questions.mount($("readerAskAgent"), () => {
    const data = st.mcData;
    if (st.blinded || !isRenderedWaveformSynced() || st.mcRenderedSeq !== st.mcReqSeq ||
        st.panDrag?.active || st.minimapDrag || st.wheelPanTimer) return null;
    return {
      case: data.case, source: "waveform_reader", recording: data.recording,
      run_name: st.active.run_summary ? data.run_name : null,
      channel: st.channel || null, start_seconds: data.start_seconds, end_seconds: data.end_seconds,
      marker: st.navigationMarker,
    };
  });

  const longPanels = window.createLongRecordingPanels({
    questions, getJSON, postJSON, palette: LUT, jump: jumpFromNavigation,
    context: () => !st.blinded && isActiveContextCommitted() && ({
      case: st.caseSlug, recording: st.recording, run_name: st.active.run_summary ? st.runName : null,
      fingerprint: st.active.recording_spec?.fingerprint, generation: st.active.run_analysis_generation_id,
      duration: totalRecordingDuration(),
    }),
    analyze: async segments => {
      if (!isActiveCaseLoaded() || st.blinded || st.activeJobId) return;
      const c = st.caseSlug, r = st.recording, epoch = st.contextEpoch;
      await submitAsyncJob("/api/jobs/branch", {
        target: `${c}/${r}`, parent_run: st.active.run_summary ? st.runName : null, overrides: { segments },
      }, async result => {
        if (st.contextEpoch === epoch && st.caseSlug === c && st.recording === r) {
          beginContextSwitch();
          await bootstrap({ caseSlug: c, recording: r, runName: result.run_name });
        }
      });
    },
  });

  const evidencePanel = window.createEvidencePanel({
    getJSON, esc,
    context: () => isActiveContextCommitted() && ({
      case: st.caseSlug,
      recording: st.recording,
      run_name: st.runName,
      generation: st.active.run_summary?.generation_id,
      fingerprint: st.active.recording_spec?.fingerprint,
      overlay: st.active.clinical_overlay,
    }),
    openContact: (channel, firstIED) => {
      if (!isActiveCaseLoaded()) return;
      const selection = st.active.run_summary?.selection || {};
      const center = Number.isFinite(firstIED) ? firstIED : ((selection.start_seconds || 0) + (selection.end_seconds || 0)) / 2;
      st.channel = channel;
      st.selectedItemId = null;
      st.manualCursorTarget = null;
      st.montage = "acquired";
      st.shaft = resolveChannelShaft(channel) || "ALL";
      st.hiddenChannels.delete(channel);
      st.startSec = Math.max(0, center - st.winSec / 2);
      st.viewStartSec = st.startSec;
      $("readerStartInput").value = st.startSec.toFixed(1);
      switchView("reader", true, { keepWindow: true, manualTime: center });
    },
  });

  function renderSelectionText() {
    const sel = st.dragSel;
    if (!sel || sel.t0 == null || sel.t1 == null) return;
    $("selectionText").textContent = t("Selected interval [{start}s, {end}s) on {caseName}/{recording}", {
      start: sel.t0, end: sel.t1, caseName: sel.caseSlug, recording: sel.recording,
    });
  }

  function renderLanguageChange() {
    questions.render();
    if (formatToast) $("toast").textContent = formatToast();
    if (st.caseStatus) renderCaseStatus();
    renderSyncBadge();
    syncReviewerSummary();
    renderCaseSelect();
    renderOpenModal();

    // The header and modal renderers rebuild their options. Preserve in-progress choices.
    const selects = ["ovCmpRunA", "ovCmpRunB", "brState", "brProtocol", "cohProtocolSelect"];
    const values = selects.map(id => $(id).value);
    const unchecked = new Set([...$("cohCasesChecks").querySelectorAll("input:not(:checked)")].map(el => el.value));
    const compareRun = st.compareRun;
    renderProtocolOptions();
    renderCohortModal();
    if (isActiveContextCommitted()) renderHeaderControls();
    st.compareRun = compareRun;
    selects.forEach((id, i) => { $(id).value = values[i]; });
    $("cohCasesChecks").querySelectorAll("input").forEach(el => { el.checked = !unchecked.has(el.value); });

    $("editReviewerBtn").textContent = $("reviewerEditRow").classList.contains("hidden") ? t("Edit") : t("Done");
    $("rebuildSheetBtn").textContent = $("deidentifyExport").checked ? t("Export sheet") : t("Rebuild");
    $("toggleSidebarBtn").title = st.sidebarCollapsed ? t("Expand candidate queue sidebar") : t("Collapse candidate queue sidebar");
    if (!isActiveContextCommitted()) return;

    if (st.view === "overview") {
      renderOverview({ languageChanged: true }).catch(err => toast(err.message));
    } else if (st.view === "reader") {
      renderShaftPills();
      refreshQueueLocale();
      if (st.mcData) {
        renderMultichannelStatus(st.mcData);
        if (st.measureDrag) updateMeasureReadout(st.mcData.traces || []);
        drawMultichannel(true);
        drawReaderMinimap();
      }
      if (isRenderedTriadSynced()) {
        updateVerdictPanel(st.triadData);
        drawTriad();
      }
      renderSelectionText();
    } else if (st.view === "sheet") {
      drawSpatialMap();
    }
  }

  function wireEvents() {
    $("returnToRun").addEventListener("click", () => {
      setSourceNavigation(false);
      st.selectedItemId = st.manualCursorTarget = null;
      st.channel = st.active?.run_summary?.channels?.[0]?.channel_name || null;
      st.shaft = resolveChannelShaft(st.channel) || "ALL";
      switchView("reader", true, { keepWindow: true, manualTime: st.startSec + st.winSec / 2 });
    });
    $("languageBtn").addEventListener("click", () => {
      const i18n = window.brainlensI18n;
      i18n.setLanguage(i18n.language === "en" ? "zh-CN" : "en");
    });
    document.addEventListener("brainlens-languagechange", renderLanguageChange);
    try { st.blinded = localStorage.getItem("brainlens_review_blinded") === "1"; } catch (_) {}
    updateBlindingControls();
    $("blindedToggle").addEventListener("change", (evt) => {
      st.blinded = evt.target.checked;
      try { localStorage.setItem("brainlens_review_blinded", st.blinded ? "1" : "0"); } catch (_) {}
      if (st.sseSource) st.sseSource.close();
      st.sseSource = null;
      st.sseConnected = false;
      reloadReaderIdentity();
      if (!st.blinded) connectSSE();
    });
    $("agreementPanel").addEventListener("toggle", loadAgreement);
    $("agreementRefreshBtn").addEventListener("click", loadAgreement);
    $("agreementSetSelect").addEventListener("change", loadAgreement);
    $("reviewSetSelect").addEventListener("change", () => {
      st.reviewSet = $("reviewSetSelect").value;
      resetSetTarget("Loading review set…");
      loadReaderQueueAndCanvases({ explicitTab: true });
    });
    $("reviewSetForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!isActiveCaseLoaded()) return;
      const form = e.currentTarget;
      const data = new FormData(form);
      const epoch = st.contextEpoch || 0;
      const btn = form.querySelector("button");
      btn.disabled = true;
      try {
        const result = await postJSON("/api/review-sets", {
          ...reviewScope(), name: data.get("name").trim(), sample_size: data.get("sample_size") ? Number(data.get("sample_size")) : null,
          seed: data.get("seed") ? Number(data.get("seed")) : null,
        });
        if ((st.contextEpoch || 0) !== epoch || st.queueMode !== "review_set") return;
        st.reviewSet = result.name;
        form.reset();
        form.closest("details").open = false;
        resetSetTarget("Loading review set…");
        await loadReaderQueueAndCanvases({ explicitTab: true });
      } catch (err) {
        if ((st.contextEpoch || 0) === epoch) toast(err.message, 4000);
      } finally {
        btn.disabled = false;
      }
    });
    $("caseSelect").addEventListener("change", async (e) => {
      const nextCase = e.target.value;
      beginContextSwitch(true);
      const ep = st.contextEpoch;
      st.caseSlug = nextCase;
      await bootstrap({ caseSlug: nextCase });
      if (st.contextEpoch === ep && isActiveCaseLoaded()) syncToAgent();
    });
    $("runSelect").addEventListener("change", async (e) => {
      const nextRun = e.target.value;
      beginContextSwitch(false);
      const ep = st.contextEpoch;
      st.runName = nextRun;
      await bootstrap({ caseSlug: st.caseSlug, recording: st.recording, runName: nextRun, keepWindow: true });
      if (st.contextEpoch === ep && isActiveCaseLoaded()) syncToAgent();
    });
    $("viewNav").querySelectorAll(".view-tab").forEach((b) => {
      b.addEventListener("click", () => switchView(b.getAttribute("data-view")));
    });
    $("themeBtn").addEventListener("click", () => {
      st.theme = st.theme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", st.theme);
      try { localStorage.setItem("brainlens_theme", st.theme); } catch (_) {}
      refreshPalette();
      longPanels.redraw();
      syncThemeToSheet();
      // Instant single-frame canvas repaint with ZERO HTTP re-fetching!
      if (st.view === "overview" && st.active && st.active.run_summary) {
        drawOverviewTimeline(st.active.run_summary);
      } else if (st.view === "reader") {
        drawMultichannel(true);
        drawReaderMinimap();
        if (!st.triadCollapsed) drawTriad();
      } else if (st.view === "sheet") {
        drawSpatialMap();
      }
    });

    // Overview interactions
    $("metricFocalBtn").addEventListener("click", () => {
      st.overviewMetric = "focal";
      $("metricFocalBtn").classList.add("active");
      $("metricRawBtn").classList.remove("active");
      if (st.active && st.cmpData) renderOverviewTables(st.active.run_summary, st.cmpData);
    });
    $("metricRawBtn").addEventListener("click", () => {
      st.overviewMetric = "raw";
      $("metricRawBtn").classList.add("active");
      $("metricFocalBtn").classList.remove("active");
      if (st.active && st.cmpData) renderOverviewTables(st.active.run_summary, st.cmpData);
    });
    $("ovCmpRunA").addEventListener("change", renderOverview);
    $("ovCmpRunB").addEventListener("change", () => {
      st.compareRun = $("ovCmpRunB").value || null;
      renderOverview();
    });
    if ($("jumpToSheetBtn")) $("jumpToSheetBtn").addEventListener("click", () => switchView("sheet"));
    $("overviewTimelineCanvas").addEventListener("click", (e) => {
      const rsum = st.active && st.active.run_summary;
      if (!rsum || !rsum.timeline_bins.length) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const padL = 42, padR = 12;
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left - padL) / (rect.width - padL - padR)));
      const dur = rsum.timeline_bins[rsum.timeline_bins.length - 1].end_seconds || 300;
      const targetT = frac * dur;
      st.startSec = Math.max(0, Math.min(Math.max(0, dur - st.winSec), targetT - st.winSec * 0.5));
      st.viewStartSec = st.startSec;
      st.selectedItemId = null;
      $("readerStartInput").value = st.startSec.toFixed(1);
      switchView("reader", true, { keepWindow: true, manualTime: targetT });
    });

    // Reader controls
    $("queueTabs").querySelectorAll("button").forEach((b) => {
      b.addEventListener("click", async () => {
        st.queueMode = b.getAttribute("data-qmode");
        if (st.queueMode === "review_set") resetSetTarget("Loading review set…");
        $("queueTabs").querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
        st.selectedItemId = null;
        await loadReaderQueueAndCanvases({ explicitTab: true });
      });
    });
    $("filterChOnlyChk").addEventListener("change", async (e) => {
      st.filterChannelOnly = e.target.checked;
      await loadReaderQueueAndCanvases({ keepWindow: true, explicitTab: true });
    });
    if ($("queueSortSelect")) {
      $("queueSortSelect").addEventListener("change", async (e) => {
        st.queueSort = e.target.value;
        await loadReaderQueueAndCanvases({ keepWindow: true, explicitTab: true });
      });
    }

    const seekTime = async (start) => {
      const dur = totalRecordingDuration();
      st.startSec = Math.max(0, Math.min(Math.max(0, dur - st.winSec), start));
      st.viewStartSec = st.startSec;
      $("readerStartInput").value = st.startSec.toFixed(1);
      await loadMultichannelCanvas();
      syncToAgent();
    };
    const moveTime = (dir) => seekTime(st.startSec + dir * st.winSec * Number($("readerPageStep").value));
    $("btnStart").addEventListener("click", () => seekTime(0));
    $("btnPrevPage").addEventListener("click", () => moveTime(-1));
    $("btnNextPage").addEventListener("click", () => moveTime(1));
    $("readerStartInput").addEventListener("change", async (e) => {
      const dur = totalRecordingDuration();
      const maxStart = Math.max(0, dur - st.winSec);
      const val = parseFloat(e.target.value);
      st.startSec = Number.isFinite(val) ? Math.max(0, Math.min(maxStart, val)) : 0;
      st.viewStartSec = st.startSec;
      e.target.value = st.startSec.toFixed(1);
      await loadMultichannelCanvas();
    });
    $("readerDurSelect").addEventListener("change", async (e) => {
      setReaderWindow(Number(e.target.value) || 8);
      await loadMultichannelCanvas();
    });
    $("readerTimebaseSelect").addEventListener("change", () => {
      const fit = $("readerTimebaseSelect").value === "fit";
      $("readerDurSelect").classList.toggle("hidden", !fit);
      if (fit) setReaderWindow(Number($("readerDurSelect").value) || 8);
      loadMultichannelCanvas();
    });
    const refreshPhysicalScale = () => {
      drawMultichannel(true);
      const seconds = currentSpeedWindow();
      if (seconds !== null && seconds !== st.winSec) loadMultichannelCanvas();
    };
    readerScales.bindCalibration(refreshPhysicalScale);
    let speedResizeTimer;
    new ResizeObserver(() => {
      clearTimeout(speedResizeTimer);
      if (st.view !== "reader" || $("readerTimebaseSelect").value === "fit") return;
      speedResizeTimer = setTimeout(() => {
        if (st.view === "reader" && st.mcData) refreshPhysicalScale();
      }, 60);
    }).observe($("mcCanvas"));
    $("readerSensitivitySelect").addEventListener("change", () => {
      $("readerGainSelect").disabled = readerScales.fixed();
      $("chBoostBtn").disabled = readerScales.fixed();
      drawMultichannel(true);
    });
    $("readerShortcutBtn").addEventListener("click", () => $("readerShortcutDialog").showModal());
    for (const id of ["readerShortcutDialog", "screenCalibrationDialog"]) {
      $(id).addEventListener("close", flushDeferredUiSync);
    }
    $("montageSelect").addEventListener("change", async (e) => { st.montage = e.target.value; st.calKey = ""; await loadMultichannelCanvas(); });
    $("dispFilterSelect").addEventListener("change", async (e) => {
      st.dispFilter = e.target.value;
      st.customLff = st.customHff = null;
      syncFilterControls();
      await loadMultichannelCanvas();
    });
    for (const id of ["displayLff", "displayHff"]) {
      $(id).addEventListener("change", async () => {
        const inputs = [$("displayLff"), $("displayHff")];
        const [low, high] = inputs.map(el => el.value === "" ? null : el.valueAsNumber);
        const invalid = inputs.some(el => el.validity.badInput || (el.value !== "" && !(el.valueAsNumber > 0)));
        if (invalid || (low !== null && high !== null && low >= high)) {
          toast(() => invalid ? t("Enter a positive cutoff.") : t("LFF must be below HFF."));
          syncFilterControls();
          return;
        }
        st.customLff = low;
        st.customHff = high;
        st.dispFilter = low === null && high === null ? "broadband" : "custom";
        syncFilterControls();
        await loadMultichannelCanvas();
      });
    }
    if ($("displayNotchChk")) {
      $("displayNotchChk").addEventListener("change", async (e) => { st.dispNotch = e.target.checked; st.calKey = ""; await loadMultichannelCanvas(); });
    }
    $("readerGainSelect").addEventListener("change", (e) => { st.gain = parseFloat(e.target.value) || 1; drawMultichannel(true); });
    const GAIN_STEPS = [0.25, 0.5, 1, 2, 4, 8, 16];
    const stepGain = (delta) => {
      if (readerScales.fixed()) {
        readerScales.stepSensitivity(delta);
        drawMultichannel(true);
        return;
      }
      const idx = Math.max(0, GAIN_STEPS.indexOf(st.gain));
      const nextIdx = Math.max(0, Math.min(GAIN_STEPS.length - 1, idx + delta));
      st.gain = GAIN_STEPS[nextIdx];
      $("readerGainSelect").value = String(st.gain);
      drawMultichannel(true);
    };
    const ROW_HEIGHT_STEPS = ["fit", 28, 40, 56, 80];
    const setRowHeightMode = (val) => {
      st.rowHeightMode = val === "fit" ? "fit" : (parseInt(val, 10) || "fit");
      if ($("rowHeightSelect")) $("rowHeightSelect").value = String(st.rowHeightMode);
      scrollFocusChannelIntoView(false);
      drawMultichannel(true);
    };
    const stepRowHeight = (delta) => {
      const idx = Math.max(0, ROW_HEIGHT_STEPS.indexOf(st.rowHeightMode));
      const nextIdx = Math.max(0, Math.min(ROW_HEIGHT_STEPS.length - 1, idx + delta));
      setRowHeightMode(ROW_HEIGHT_STEPS[nextIdx]);
    };
    if ($("rowHeightSelect")) {
      $("rowHeightSelect").addEventListener("change", (e) => setRowHeightMode(e.target.value));
    }
    if ($("gainDownBtn")) $("gainDownBtn").addEventListener("click", () => stepGain(-1));
    if ($("gainUpBtn")) $("gainUpBtn").addEventListener("click", () => stepGain(1));
    if ($("unhideAllBtn")) {
      $("unhideAllBtn").addEventListener("click", () => { st.hiddenChannels.clear(); drawMultichannel(true); toast(() => t("Restored all hidden channels")); });
    }
    if ($("multiShaftBtn")) {
      $("multiShaftBtn").addEventListener("click", () => {
        st.multiShaftMode = !st.multiShaftMode;
        $("multiShaftBtn").classList.toggle("active", st.multiShaftMode);
        if (st.multiShaftMode) toast(() => t("Multi-shaft mode: click electrode shafts to combine (e.g. LB + LC)"), 2400);
      });
    }
    $("triadWinSelect").addEventListener("change", async (e) => {
      st.triadSpan = parseFloat(e.target.value) || 0.6;
      const cur = st.triadData;
      if (cur && cur.view_mode === "background_control") {
        if (cur.window_seconds) {
          const mid = 0.5 * (cur.window_seconds[0] + cur.window_seconds[1]);
          await loadTriadCanvas({ eventId: "control", centerSeconds: mid });
          // If widening the span overlapped an event, fall back to a clean control window at the new span
          if (!st.triadData) {
            await loadTriadCanvas({ eventId: "control" });
          }
        } else {
          await loadTriadCanvas({ eventId: "control" });
        }
      } else if (cur && cur.view_mode === "cursor_window" && cur.window_seconds) {
        const mid = 0.5 * (cur.window_seconds[0] + cur.window_seconds[1]);
        await loadTriadCanvas({ eventId: "", centerSeconds: mid });
      } else {
        await loadTriadCanvas();
      }
    });
    $("showControlBtn").addEventListener("click", async () => {
      st.sideBySide = false;
      if ($("sideBySideBtn")) $("sideBySideBtn").classList.remove("active");
      await loadTriadCanvas({ eventId: "control" });
    });
    if ($("sideBySideBtn")) {
      $("sideBySideBtn").addEventListener("click", async () => {
        st.sideBySide = !st.sideBySide;
        $("sideBySideBtn").classList.toggle("active", st.sideBySide);
        if (st.sideBySide && st.triadData && st.triadData.view_mode === "background_control") {
          await loadTriadCanvas({ eventId: "top_focal" });
        } else {
          await loadTriadCanvas();
        }
      });
    }
    if ($("toggleSidebarBtn")) {
      $("toggleSidebarBtn").addEventListener("click", () => {
        st.sidebarCollapsed = !st.sidebarCollapsed;
        const lay = document.querySelector(".reader-layout");
        if (lay) lay.classList.toggle("sidebar-collapsed", st.sidebarCollapsed);
        $("toggleSidebarBtn").classList.toggle("active", st.sidebarCollapsed);
        $("toggleSidebarBtn").setAttribute("aria-expanded", String(!st.sidebarCollapsed));
        $("toggleSidebarBtn").title = st.sidebarCollapsed ? t("Expand candidate queue sidebar") : t("Collapse candidate queue sidebar");
        requestAnimationFrame(() => {
          drawMultichannel();
          drawReaderMinimap();
          drawTriad();
        });
      });
    }
    const toggleTriadDock = () => {
      setTriadCollapsed(!st.triadCollapsed);
      if (!st.triadCollapsed) applyReaderSplitLayout();
      requestAnimationFrame(() => {
        drawMultichannel();
        drawReaderMinimap();
        if (!st.triadCollapsed) drawTriad();
      });
    };
    if ($("toggleTriadBtn")) $("toggleTriadBtn").addEventListener("click", toggleTriadDock);

    // Explicit Waveform Interaction Tool Mode (Pan / Measure / Exclude)
    const mc = $("mcCanvas");
    const setMcTool = (mode) => {
      if (!["pan", "measure", "exclude"].includes(mode)) return;
      st.mcTool = mode;
      mc.classList.remove("tool-pan", "tool-measure", "tool-exclude", "panning");
      mc.classList.add(`tool-${mode}`);
      if ($("mcToolModes")) {
        $("mcToolModes").querySelectorAll("button[data-tool]").forEach((b) => {
          b.classList.toggle("active", b.getAttribute("data-tool") === mode);
        });
      }
      if (mode !== "measure") {
        st.measureDrag = null;
        if (!st.mcHover) updateReadoutHud("");
      }
      drawMultichannel();
    };
    if ($("mcToolModes")) {
      $("mcToolModes").querySelectorAll("button[data-tool]").forEach((b) => {
        b.addEventListener("click", () => setMcTool(b.getAttribute("data-tool")));
      });
    }

    // Resizable Horizontal Split Divider (Multi-channel EEG vs Lower Triad Dock)
    const splitter = $("readerSplitter");
    if (splitter) {
      let splitDragging = false;
      let splitRaf = 0;
      splitter.addEventListener("mousedown", (e) => {
        e.preventDefault();
        splitDragging = true;
        splitter.classList.add("dragging");
      });
      window.addEventListener("mousemove", (e) => {
        if (!splitDragging) return;
        const rm = document.querySelector(".reader-main");
        const topWrap = $("mcCanvasWrap");
        if (!rm || !topWrap) return;
        const rmRect = rm.getBoundingClientRect();
        const topRect = topWrap.getBoundingClientRect();
        const availH = Math.max(240, rmRect.bottom - topRect.top - 10);
        const newTopH = Math.max(110, Math.min(availH - 120, e.clientY - topRect.top));
        st.mcSplitFrac = Math.max(0.16, Math.min(0.88, newTopH / availH));
        applyReaderSplitLayout();
        if (!splitRaf) {
          splitRaf = requestAnimationFrame(() => {
            splitRaf = 0;
            drawMultichannel();
            drawTriad();
          });
        }
      });
      window.addEventListener("mouseup", () => {
        if (!splitDragging) return;
        splitDragging = false;
        splitter.classList.remove("dragging");
        try { localStorage.setItem("brainlens_reader_split", String(st.mcSplitFrac.toFixed(3))); } catch (_) {}
        drawMultichannel();
        drawTriad();
      });
      splitter.addEventListener("dblclick", () => {
        st.mcSplitFrac = 0.54;
        applyReaderSplitLayout();
        try { localStorage.setItem("brainlens_reader_split", "0.54"); } catch (_) {}
        drawMultichannel();
        drawTriad();
      });
    }

    // Global Recording Event Minimap Scrubber (Click & Drag anywhere in recording!)
    const miniCanvas = $("readerMinimapCanvas");
    if (miniCanvas) {
      let miniFetchTimer = null;
      const scrubMinimap = (clientX, commitNow = false) => {
        const rect = miniCanvas.getBoundingClientRect();
        const dur = Math.max(1, totalRecordingDuration());
        const rsum = st.active && st.active.run_summary;
        const bins = (rsum && rsum.timeline_bins) || [];
        const screenedStart = bins.length ? Number(bins[0].start_seconds || 0) : 0;
        const screenedEnd = bins.length ? Number(bins[bins.length - 1].end_seconds || dur) : dur;
        const partialScreen = screenedStart > 1 || screenedEnd < dur - 1;
        const padL = 8, padR = partialScreen ? 182 : 106;
        const pw = Math.max(40, rect.width - padL - padR);
        const frac = Math.max(0, Math.min(1, (clientX - rect.left - padL) / pw));
        const targetStart = Math.max(0, Math.min(Math.max(0, dur - st.winSec), frac * dur - st.winSec * 0.5));
        st.startSec = +targetStart.toFixed(2);
        st.viewStartSec = st.startSec;
        $("readerStartInput").value = st.viewStartSec.toFixed(1);
        drawReaderMinimap();
        scheduleMcDraw();
        clearTimeout(miniFetchTimer);
        if (commitNow) {
          loadMultichannelCanvas().then(() => syncToAgent());
        } else {
          miniFetchTimer = setTimeout(() => loadMultichannelCanvas(), 80);
        }
      };
      miniCanvas.addEventListener("mousedown", (e) => {
        e.preventDefault();
        st.minimapDrag = true;
        scrubMinimap(e.clientX, false);
      });
      window.addEventListener("mousemove", (e) => {
        if (st.minimapDrag) scrubMinimap(e.clientX, false);
      });
      window.addEventListener("mouseup", (e) => {
        if (!st.minimapDrag) return;
        st.minimapDrag = false;
        scrubMinimap(e.clientX, true);
      });
    }

    // Unified Multi-Channel Canvas Mouse, 2D Drag, Scrollbar, Crosshair, Caliper Measure & Trackpad/Wheel Controls
    const getVisibleTraces = () => (st.mcData && st.mcData.traces ? st.mcData.traces.filter((tr) => !st.hiddenChannels.has(tr.channel_name)) : []);
    const canvasCoords = (e) => {
      const rect = mc.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const padT = 22, padB = MC_PAD_B;
      const pw = mcPlotWidth(rect.width);
      const ph = Math.max(20, rect.height - padT - padB);
      const span = st.mcData ? Math.max(1e-6, st.mcData.end_seconds - st.mcData.start_seconds) : st.winSec;
      const clampedX = Math.max(MC_PAD_L, Math.min(MC_PAD_L + pw, x));
      const t = st.viewStartSec + ((clampedX - MC_PAD_L) / pw) * span;
      const traces = getVisibleTraces();
      const { rowH } = computeMcVerticalLayout(traces, ph);
      const chIdx = (traces.length && y >= padT && y <= padT + ph)
        ? Math.max(0, Math.min(traces.length - 1, Math.floor((y - padT + st.scrollY) / rowH)))
        : -1;
      return { rect, x, y, clampedX, padT, pw, ph, span, t, traces, chIdx, rowH };
    };

    mc.addEventListener("mousedown", (e) => {
      if (e.button !== 0 || !st.mcData) return;
      const c = canvasCoords(e);
      // Right-edge vertical scrollbar drag / track jump!
      if (st.mcScrollRect && c.x >= st.mcScrollRect.x && c.y >= st.mcScrollRect.y && c.y <= st.mcScrollRect.y + st.mcScrollRect.h) {
        e.preventDefault();
        const sb = st.mcScrollRect;
        if (c.y >= sb.thumbY && c.y <= sb.thumbY + sb.thumbH) {
          st.vScrollDrag = { startY: c.y, origScrollY: st.scrollY, trackSpan: Math.max(1, sb.h - sb.thumbH), maxScrollY: sb.maxScrollY };
        } else {
          const frac = Math.max(0, Math.min(1, (c.y - sb.y - sb.thumbH * 0.5) / Math.max(1, sb.h - sb.thumbH)));
          st.scrollY = frac * sb.maxScrollY;
          st.vScrollDrag = { startY: c.y, origScrollY: st.scrollY, trackSpan: Math.max(1, sb.h - sb.thumbH), maxScrollY: sb.maxScrollY };
          scheduleMcDraw();
        }
        st.justDragged = true;
        return;
      }
      if (c.x < MC_PAD_L) return; // Left gutter opens channel menu on click
      st.justDragged = false;
      if (e.shiftKey || st.mcTool === "exclude") {
        if (!isRenderedWaveformSynced()) {
          toast(() => t("Switching case/recording — wait for waveforms to load"), 2000);
          return;
        }
        st.dragSel = {
          active: true,
          x0: c.clampedX,
          x1: c.clampedX,
          caseSlug: st.mcData.case,
          recording: st.mcData.recording,
          runName: st.mcData.run_name || "baseline",
          channel: c.chIdx >= 0 && c.traces[c.chIdx] ? c.traces[c.chIdx].channel_name : null,
        };
        return;
      }
      if (st.mcTool === "measure") {
        if (!isRenderedWaveformSynced()) return;
        st.measureDrag = {
          active: true,
          x0: c.clampedX,
          x1: c.clampedX,
          t0: c.t,
          t1: c.t,
          chIdx: c.chIdx,
          channel: c.chIdx >= 0 ? c.traces[c.chIdx].channel_name : null,
          caseSlug: st.mcData.case,
          recording: st.mcData.recording,
          runName: st.mcData.run_name || "baseline",
        };
        scheduleMcDraw();
        return;
      }
      // Pan mode: live 60fps 2D drag pan (horizontal = time, vertical = channels when scrollable!)
      st.panDrag = {
        active: true,
        startX: c.clampedX,
        curX: c.clampedX,
        startY: c.y,
        curY: c.y,
        origViewStart: st.viewStartSec,
        origScrollY: st.scrollY,
        movedX: false,
        moved: false,
      };
      mc.classList.add("panning");
    });

    mc.addEventListener("mousemove", (e) => {
      if (!st.mcData) return;
      if (st.vScrollDrag || (st.panDrag && st.panDrag.active) || (st.dragSel && st.dragSel.active) || (st.measureDrag && st.measureDrag.active)) return;
      const c = canvasCoords(e);
      const gutterHit = c.x < MC_PAD_L && st.mcLabelHits.find((b) => c.y >= b.y0 && c.y < b.y1);
      mc.title = gutterHit ? gutterHit.tooltip : "";
      if (c.x >= MC_PAD_L && c.x <= MC_PAD_L + c.pw && c.chIdx >= 0) {
        const tr = c.traces[c.chIdx];
        const uv = sampleTraceAtTime(tr, c.t);
        st.mcHover = { x: c.x, y: c.y, t: c.t, chIdx: c.chIdx, channel: tr.display_label, uv };
        if (!st.measureDrag) {
          const uvStr = (uv !== null && Number.isFinite(uv)) ? ` · ${uv >= 0 ? "+" : ""}${uv.toFixed(1)} µV` : "";
          updateReadoutHud(`${c.t.toFixed(3)}s · ${channelCaption(tr)}${uvStr}`);
        }
        scheduleMcDraw();
      } else if (st.mcHover) {
        st.mcHover = null;
        if (!st.measureDrag) updateReadoutHud("");
        scheduleMcDraw();
      }
    });

    mc.addEventListener("mouseleave", () => {
      mc.removeAttribute("title");
      if (st.mcHover) {
        st.mcHover = null;
        if (!st.measureDrag) updateReadoutHud("");
        scheduleMcDraw();
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (!st.mcData) return;
      if (st.vScrollDrag) {
        const c = canvasCoords(e);
        const dy = c.y - st.vScrollDrag.startY;
        st.scrollY = Math.max(0, Math.min(st.vScrollDrag.maxScrollY, st.vScrollDrag.origScrollY + (dy / st.vScrollDrag.trackSpan) * st.vScrollDrag.maxScrollY));
        scheduleMcDraw();
        return;
      }
      if (st.dragSel && st.dragSel.active) {
        const c = canvasCoords(e);
        st.dragSel.x1 = c.clampedX;
        scheduleMcDraw();
        return;
      }
      if (st.measureDrag && st.measureDrag.active) {
        const c = canvasCoords(e);
        st.measureDrag.x1 = c.clampedX;
        st.measureDrag.t1 = c.t;
        updateMeasureReadout(c.traces);
        scheduleMcDraw();
        return;
      }
      if (st.panDrag && st.panDrag.active) {
        const c = canvasCoords(e);
        st.panDrag.curX = c.clampedX;
        st.panDrag.curY = c.y;
        const absDx = Math.abs(st.panDrag.curX - st.panDrag.startX);
        const absDy = Math.abs(st.panDrag.curY - st.panDrag.startY);
        if (absDx > 4) { st.panDrag.moved = true; st.panDrag.movedX = true; }
        if (absDy > 4 && st.maxScrollY > 0) st.panDrag.moved = true;
        if (st.maxScrollY > 0) {
          st.scrollY = Math.max(0, Math.min(st.maxScrollY, st.panDrag.origScrollY - (st.panDrag.curY - st.panDrag.startY)));
        }
        const dxSec = -((st.panDrag.curX - st.panDrag.startX) / c.pw) * c.span;
        const dur = totalRecordingDuration();
        st.viewStartSec = Math.max(0, Math.min(Math.max(0, dur - st.winSec), st.panDrag.origViewStart + dxSec));
        $("readerStartInput").value = st.viewStartSec.toFixed(1);
        scheduleMcDraw();
        drawReaderMinimap();
      }
    });

    window.addEventListener("mouseup", (e) => {
      if (!st.mcData) return;
      if (st.vScrollDrag) {
        st.vScrollDrag = null;
        setTimeout(() => { st.justDragged = false; }, 120);
        scheduleMcDraw();
        return;
      }
      if (st.dragSel && st.dragSel.active) {
        st.dragSel.active = false;
        if (
          !isRenderedWaveformSynced() ||
          st.dragSel.caseSlug !== st.mcData.case ||
          st.dragSel.recording !== st.mcData.recording ||
          st.dragSel.runName !== (st.mcData.run_name || "baseline")
        ) {
          st.dragSel = null;
          $("selectionBar").classList.add("hidden");
          scheduleMcDraw();
          return;
        }
        const cUp = canvasCoords(e);
        st.dragSel.x1 = cUp.clampedX;
        const rect = mc.getBoundingClientRect();
        const pw = Math.max(20, rect.width - MC_PAD_L - 16);
        const lx = Math.max(MC_PAD_L, Math.min(st.dragSel.x0, st.dragSel.x1));
        const rx = Math.min(MC_PAD_L + pw, Math.max(st.dragSel.x0, st.dragSel.x1));
        if (rx - lx < 6) { st.dragSel = null; scheduleMcDraw(); return; }
        st.justDragged = true;
        setTimeout(() => { st.justDragged = false; }, 120);
        const span = st.mcData.end_seconds - st.mcData.start_seconds;
        const originStart = Number(st.mcData.start_seconds ?? st.viewStartSec);
        st.dragSel.t0 = +(originStart + ((lx - MC_PAD_L) / pw) * span).toFixed(2);
        st.dragSel.t1 = +(originStart + ((rx - MC_PAD_L) / pw) * span).toFixed(2);
        st.dragSel.caseSlug = st.mcData.case;
        st.dragSel.recording = st.mcData.recording;
        st.dragSel.runName = st.mcData.run_name || "baseline";
        renderSelectionText();
        $("selectionBar").classList.remove("hidden");
        return;
      }
      if (st.measureDrag && st.measureDrag.active) {
        const cUp = canvasCoords(e);
        st.measureDrag.x1 = cUp.clampedX;
        st.measureDrag.t1 = cUp.t;
        st.measureDrag.active = false;
        if (Math.abs(st.measureDrag.x1 - st.measureDrag.x0) < 4) {
          st.measureDrag = null;
          updateReadoutHud("");
          scheduleMcDraw();
        } else {
          updateMeasureReadout(cUp.traces);
          st.justDragged = true;
          setTimeout(() => { st.justDragged = false; }, 120);
          scheduleMcDraw();
        }
        return;
      }
      if (st.panDrag && st.panDrag.active) {
        const pd = st.panDrag;
        st.panDrag = null;
        mc.classList.remove("panning");
        if (!pd.moved) return;
        st.justDragged = true;
        setTimeout(() => { st.justDragged = false; }, 120);
        if (pd.movedX) {
          // Keep st.viewStartSec locked at the new target until loadMultichannelCanvas() swaps in fresh traces!
          st.startSec = +st.viewStartSec.toFixed(2);
          $("readerStartInput").value = st.startSec.toFixed(1);
          loadMultichannelCanvas().then(() => syncToAgent());
        }
      }
    });

    // Native Trackpad & Scroll Wheel Gestures (Zoom, Unclipped Gain, Row Height, Vertical Channel Scroll, Smooth Time Scrub!)
    const WIN_STEPS = [0.5, 1, 2, 4, 5, 8, 10, 15, 30];
    mc.addEventListener("wheel", (e) => {
      if (!st.mcData) return;
      e.preventDefault();
      // 1) Cmd/Ctrl + Wheel (or macOS trackpad pinch): horizontal time zoom centered on cursor!
      if (e.ctrlKey || e.metaKey) {
        const now = Date.now();
        if (now - st.wheelZoomLastMs < 110) return;
        st.wheelZoomAcc += e.deltaY;
        if (Math.abs(st.wheelZoomAcc) < 28) return;
        const dir = st.wheelZoomAcc > 0 ? 1 : -1;
        st.wheelZoomAcc = 0;
        st.wheelZoomLastMs = now;
        const c = canvasCoords(e);
        const curIdx = WIN_STEPS.reduce((best, val, idx) => (Math.abs(val - st.winSec) < Math.abs(WIN_STEPS[best] - st.winSec) ? idx : best), 0);
        const nextWin = WIN_STEPS[Math.max(0, Math.min(WIN_STEPS.length - 1, curIdx + dir))];
        if (nextWin !== st.winSec || $("readerTimebaseSelect").value !== "fit") {
          $("readerTimebaseSelect").value = "fit";
          $("readerDurSelect").classList.remove("hidden");
          const anchorFrac = Math.max(0, Math.min(1, (c.clampedX - MC_PAD_L) / c.pw));
          const anchorT = st.viewStartSec + anchorFrac * c.span;
          const dur = totalRecordingDuration();
          st.winSec = nextWin;
          st.startSec = +Math.max(0, Math.min(Math.max(0, dur - st.winSec), anchorT - anchorFrac * st.winSec)).toFixed(2);
          st.viewStartSec = st.startSec;
          $("readerDurSelect").value = String(st.winSec);
          $("readerStartInput").value = st.startSec.toFixed(1);
          loadMultichannelCanvas();
        }
        return;
      }
      // 2) Shift + Alt + Wheel -> step vertical channel row height; Alt + Wheel -> vertical amplitude gain!
      if (e.altKey) {
        st.wheelGainAcc += e.deltaY;
        if (Math.abs(st.wheelGainAcc) >= 36) {
          const dir = st.wheelGainAcc < 0 ? 1 : -1;
          st.wheelGainAcc = 0;
          if (e.shiftKey) stepRowHeight(dir);
          else stepGain(dir);
        }
        return;
      }
      // 3) Horizontal swipe or Shift+Wheel -> continuous zero-snap horizontal time scrub!
      const isHoriz = Math.abs(e.deltaX) > Math.abs(e.deltaY) * 0.65 || e.shiftKey;
      if (isHoriz) {
        const rawDelta = Math.abs(e.deltaX) >= Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        const rect = mc.getBoundingClientRect();
        const pw = Math.max(40, rect.width - MC_PAD_L - 16);
        const dSec = (rawDelta / pw) * st.winSec * 0.9;
        const dur = totalRecordingDuration();
        st.viewStartSec = Math.max(0, Math.min(Math.max(0, dur - st.winSec), st.viewStartSec + dSec));
        $("readerStartInput").value = st.viewStartSec.toFixed(1);
        scheduleMcDraw();
        drawReaderMinimap();
        clearTimeout(st.wheelPanTimer);
        st.wheelPanTimer = setTimeout(() => {
          st.wheelPanTimer = null;
          st.startSec = +st.viewStartSec.toFixed(2);
          loadMultichannelCanvas();
        }, 120);
        return;
      }
      // 4) Vertical wheel when channels overflow viewport -> smooth 60fps vertical channel scroll!
      if (st.maxScrollY > 0.5) {
        st.scrollY = Math.max(0, Math.min(st.maxScrollY, st.scrollY + e.deltaY));
        scheduleMcDraw();
        return;
      }
      // 5) Vertical wheel in PAGE mode when fitting -> step 16-channel pages!
      const isPageMode = String(st.shaft || "").startsWith("PAGE:");
      if (isPageMode) {
        const now = Date.now();
        if (now - st.wheelPageLastMs < 260) return;
        st.wheelPageLastMs = now;
        const curP = parseInt(String(st.shaft).split(":")[1], 10) || 0;
        const maxP = Math.max(0, (st.mcData.total_pages || 1) - 1);
        const nextP = Math.max(0, Math.min(maxP, curP + (e.deltaY > 0 ? 1 : -1)));
        if (nextP !== curP) {
          st.shaft = `PAGE:${nextP}`;
          st.page = nextP;
          st.scrollY = 0;
          renderShaftPills();
          loadMultichannelCanvas();
        }
        return;
      }
      // 6) Fallback vertical wheel on short single-shaft fit -> smooth horizontal time scrub
      const rect = mc.getBoundingClientRect();
      const pw = Math.max(40, rect.width - MC_PAD_L - 16);
      const dSec = (e.deltaY / pw) * st.winSec * 0.9;
      const dur = totalRecordingDuration();
      st.viewStartSec = Math.max(0, Math.min(Math.max(0, dur - st.winSec), st.viewStartSec + dSec));
      $("readerStartInput").value = st.viewStartSec.toFixed(1);
      scheduleMcDraw();
      drawReaderMinimap();
      clearTimeout(st.wheelPanTimer);
      st.wheelPanTimer = setTimeout(() => {
        st.wheelPanTimer = null;
        st.startSec = +st.viewStartSec.toFixed(2);
        loadMultichannelCanvas();
      }, 120);
    }, { passive: false });

    // Compact Reviewer Summary Toggle
    if ($("editReviewerBtn")) {
      $("editReviewerBtn").addEventListener("click", () => {
        const row = $("reviewerEditRow");
        if (!row) return;
        const hidden = row.classList.toggle("hidden");
        $("editReviewerBtn").textContent = hidden ? t("Edit") : t("Done");
      });
    }
    if ($("authorInput")) {
      try {
        const savedAuthor = localStorage.getItem("brainlens_reviewer_author") || localStorage.getItem("btk_reviewer_author");
        if (savedAuthor) $("authorInput").value = savedAuthor;
      } catch (_) {}
      $("authorInput").addEventListener("input", () => {
        syncReviewerSummary({ persist: true });
        if (st.blinded || st.queueMode === "review_set") reloadReaderIdentity(true);
      });
    }
    if ($("authorRoleSelect")) {
      if (st.preferredRealRole) $("authorRoleSelect").value = st.preferredRealRole;
      $("authorRoleSelect").addEventListener("change", () => {
        st.userEditedRole = true;
        if ($("authorRoleSelect").value !== "engineering_rehearsal") {
          st.preferredRealRole = $("authorRoleSelect").value;
        }
        syncReviewerSummary({ persist: true });
        if (st.blinded || st.queueMode === "review_set") reloadReaderIdentity(true);
      });
    }
    syncReviewerSummary({ persist: false });

    $("clearSelectionBtn").addEventListener("click", () => {
      st.dragSel = null;
      $("selectionBar").classList.add("hidden");
      drawMultichannel();
    });
    const runExcludeSelection = async (focusOnly) => {
      if (!st.dragSel) return;
      const sel = { ...st.dragSel };
      $("selectionBar").classList.add("hidden");
      st.dragSel = null;
      drawMultichannel();
      if (
        !isRenderedWaveformSynced() ||
        sel.caseSlug !== st.mcData.case ||
        sel.recording !== st.mcData.recording ||
        sel.runName !== (st.mcData.run_name || "baseline")
      ) {
        toast(() => t("Cleared stale interval selection from another case/recording"), 2500);
        return;
      }
      const { t0, t1 } = sel;
      const focusCh = sel.channel || st.channel;
      const exSpec = {
        start: t0,
        end: t1,
        reason: `Clinician waveform exclusion [${t0}, ${t1})s${focusOnly && focusCh ? ` on ${focusCh}` : ""}`,
      };
      if (focusOnly && focusCh) exSpec.channels = [focusCh];
      const startEpoch = st.contextEpoch || 0;
      await submitAsyncJob(
        "/api/jobs/branch",
        {
          target: `${sel.caseSlug}/${sel.recording}`,
          overrides: { exclude_intervals: [exSpec] },
          parent_run: sel.runName,
        },
        async (out) => {
          if ((st.contextEpoch || 0) !== startEpoch || st.caseSlug !== sel.caseSlug || st.recording !== sel.recording) {
            toast(() => t("Completed exclusion branch '{run}' on {caseName}/{recording}", { run: out.run_name, caseName: sel.caseSlug, recording: sel.recording }));
            return;
          }
          st.runName = out.run_name;
          st.compareRun = st.runName;
          await bootstrap({ caseSlug: sel.caseSlug, recording: sel.recording, runName: st.runName });
          switchView("overview");
          const run = st.runName;
          toast(() => t("Completed exclusion branch '{run}'", { run }));
        },
      );
    };
    $("excludeSelectionBtn").addEventListener("click", () => runExcludeSelection(false));
    if ($("excludeSelChBtn")) $("excludeSelChBtn").addEventListener("click", () => runExcludeSelection(true));

    // Channel quick-action popover (label click or right-click)
    const openChMenu = (chName, yPx) => {
      st.chMenuChannel = chName;
      const menu = $("chMenu");
      const trace = getVisibleTraces().find((tr) => tr.channel_name === chName);
      $("chMenuTitle").textContent = t("Channel {channel}", { channel: trace?.anatomy ? channelCaption(trace) : chName });
      $("chBoostBtn").textContent = (st.channelGain[chName] || 1) > 1 ? t("Reset Channel Gain (1×)") : t("Boost Channel Gain (2×)");
      menu.style.top = `${Math.max(24, Math.min(mc.clientHeight - 115, yPx))}px`;
      menu.classList.remove("hidden");
    };
    mc.addEventListener("contextmenu", (e) => {
      if (!st.mcData) return;
      e.preventDefault();
      const rect = mc.getBoundingClientRect();
      const y = e.clientY - rect.top;
      const hit = st.mcLabelHits.find((b) => y >= b.y0 && y <= b.y1);
      if (hit) openChMenu(hit.channel, y);
    });
    $("chHideBtn").addEventListener("click", () => {
      const channel = st.chMenuChannel;
      if (channel) {
        st.hiddenChannels.add(channel);
        drawMultichannel();
        toast(() => t("Hidden {channel} in display", { channel }));
      }
      $("chMenu").classList.add("hidden");
    });
    $("chBoostBtn").addEventListener("click", () => {
      if (st.chMenuChannel) {
        st.channelGain[st.chMenuChannel] = (st.channelGain[st.chMenuChannel] || 1) > 1 ? 1 : 2;
        drawMultichannel();
      }
      $("chMenu").classList.add("hidden");
    });
    $("chMarkBadBtn").addEventListener("click", async () => {
      const badCh = st.chMenuChannel;
      $("chMenu").classList.add("hidden");
      if (!badCh || !isRenderedWaveformSynced()) return;
      const targetCase = st.mcData.case;
      const targetRec = st.mcData.recording;
      const startEpoch = st.contextEpoch || 0;
      await submitAsyncJob(
        "/api/jobs/branch",
        {
          target: `${targetCase}/${targetRec}`,
          run_name: `excl_${badCh}`,
          overrides: { exclude_channels: [badCh], reason: `Clinician marked bad channel ${badCh} in Reader` },
          parent_run: st.mcData.run_name || "baseline",
        },
        async (out) => {
          if ((st.contextEpoch || 0) !== startEpoch || st.caseSlug !== targetCase || st.recording !== targetRec) {
            toast(() => t("Excluded bad channel {badCh} -> branch '{run}' on {caseName}/{recording}", { badCh, run: out.run_name, caseName: targetCase, recording: targetRec }));
            return;
          }
          st.runName = out.run_name;
          st.compareRun = st.runName;
          await bootstrap({ caseSlug: targetCase, recording: targetRec, runName: st.runName });
          const run = st.runName;
          toast(() => t("Excluded bad channel {badCh} -> branch '{run}'", { badCh, run }));
        },
      );
    });
    $("clipSelectionBtn").addEventListener("click", async () => {
      if (!st.dragSel) return;
      const sel = { ...st.dragSel };
      if (
        !isRenderedWaveformSynced() ||
        sel.caseSlug !== st.mcData.case ||
        sel.recording !== st.mcData.recording ||
        sel.runName !== (st.mcData.run_name || "baseline")
      ) {
        st.dragSel = null;
        $("selectionBar").classList.add("hidden");
        drawMultichannel();
        toast(() => t("Cleared stale interval selection from another case/recording"), 2500);
        return;
      }
      const mid = 0.5 * (sel.t0 + sel.t1);
      const ok = await saveVerdict(
        { clipped: true, center_seconds: mid, interval_seconds: [sel.t0, sel.t1], note: `Selected interval [${sel.t0}, ${sel.t1})s` },
        false,
        sel,
      );
      if (ok) {
        st.dragSel = null;
        $("selectionBar").classList.add("hidden");
        drawMultichannel();
      }
    });

    // Dismiss channel popover when clicking anywhere outside it or mcCanvas
    document.addEventListener("click", (e) => {
      const menu = $("chMenu");
      if (!menu || menu.classList.contains("hidden")) return;
      if (!menu.contains(e.target) && e.target !== mc) {
        menu.classList.add("hidden");
      }
    });

    mc.addEventListener("click", async (e) => {
      $("chMenu").classList.add("hidden");
      if (e.shiftKey || st.justDragged || !isRenderedWaveformSynced()) return;
      const rect = mc.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;
      if (x < MC_PAD_L && y < 22) {
        if (readerScales.fixed()) return;
        st.calKey = "";
        st.baseUvLane = null;
        drawMultichannel(true);
        toast(() => t("Recalibrated trace scale to current window"), 1600);
        return;
      }
      if (x < MC_PAD_L) {
        const lhit = st.mcLabelHits.find((b) => y >= b.y0 && y <= b.y1);
        if (lhit) { openChMenu(lhit.channel, y); return; }
      }
      if (st.mcTool !== "pan") return;
      const hit = st.mcBoxes.find((b) => x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1);
      if (hit) {
        const mid = 0.5 * (hit.ev.start_seconds + hit.ev.end_seconds);
        st.selectedItemId = hit.ev.id;
        st.channel = hit.ev.channel_name;
        st.pendingTriadTarget = { channel: hit.ev.channel_name, w0: mid - st.triadSpan * 0.5, w1: mid + st.triadSpan * 0.5 };
        renderQueueList();
        drawMultichannel();
        await loadTriadCanvas({ eventId: hit.ev.id });
        return;
      }
      const traces = (st.mcData.traces || []).filter((tr) => !st.hiddenChannels.has(tr.channel_name));
      const padT = 22, pw = mcPlotWidth(rect.width), ph = rect.height - padT - MC_PAD_B;
      if (x >= MC_PAD_L && x <= MC_PAD_L + pw && y >= padT && y <= padT + ph && traces.length) {
        const { rowH } = computeMcVerticalLayout(traces, ph);
        const idx = Math.max(0, Math.min(traces.length - 1, Math.floor((y - padT + st.scrollY) / rowH)));
        const t = st.viewStartSec + ((x - MC_PAD_L) / pw) * (st.mcData.end_seconds - st.mcData.start_seconds);
        st.channel = traces[idx].channel_name;
        st.navigationMarker = null;
        st.selectedItemId = null;
        st.pendingTriadTarget = { channel: traces[idx].channel_name, w0: t - st.triadSpan * 0.5, w1: t + st.triadSpan * 0.5 };
        renderQueueList();
        drawMultichannel();
        await loadTriadCanvas({ eventId: "", centerSeconds: t });
      }
    });

    // Verdict buttons & keyboard shortcuts
    $("vConfirmBtn").addEventListener("click", () => saveVerdict({ verdict: "confirm" }, true));
    $("vArtifactBtn").addEventListener("click", () => saveVerdict({ verdict: "artifact" }, true));
    $("vUncertainBtn").addEventListener("click", () => saveVerdict({ verdict: "uncertain" }, true));
    function clipActiveTriad() {
      const tr = st.triadData;
      if (!tr) return;
      const ev = tr.focus_event;
      const cur = Boolean(ev && ev.clipped);
      const patch = { clipped: !cur };
      if (!ev && Array.isArray(tr.window_seconds) && tr.window_seconds.length === 2) {
        const w0 = Number(tr.window_seconds[0]);
        const w1 = Number(tr.window_seconds[1]);
        if (Number.isFinite(w0) && Number.isFinite(w1) && w1 > w0) {
          patch.interval_seconds = [+w0.toFixed(5), +w1.toFixed(5)];
          patch.center_seconds = +(0.5 * (w0 + w1)).toFixed(5);
        }
      }
      saveVerdict(patch, false);
    }
    $("vClipBtn").addEventListener("click", () => clipActiveTriad());
    $("prevEvBtn").addEventListener("click", () => stepQueue(-1));
    $("nextEvBtn").addEventListener("click", () => stepQueue(1));
    $("saveNoteOnlyBtn").addEventListener("click", () => saveVerdict({}, false));
    if ($("noteInput")) {
      $("noteInput").addEventListener("blur", () => flushDeferredUiSync());
    }

    window.addEventListener("keydown", (e) => {
      if (document.querySelector("dialog[open]")) return;
      const tag = (e.target && e.target.tagName) || "";
      if (e.key === "Escape") {
        $("chMenu").classList.add("hidden");
        closeAndUnbindCaseModals();
        ["openModal", "cohortModal"].forEach((id) => $(id) && $(id).classList.add("hidden"));
        if (st.measureDrag) {
          st.measureDrag = null;
          updateReadoutHud("");
          drawMultichannel();
        }
        flushDeferredUiSync();
        return;
      }
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || e.target?.isContentEditable) return;
      if (hasOpenModal() || st.view !== "reader") return;
      if (e.key === "?") { e.preventDefault(); $("readerShortcutDialog").showModal(); }
      else if (e.key === "Home") { e.preventDefault(); seekTime(0); }
      else if (e.key === "End") { e.preventDefault(); seekTime(totalRecordingDuration()); }
      else if (e.key === "1") { e.preventDefault(); saveVerdict({ verdict: "confirm" }, true); }
      else if (e.key === "2") { e.preventDefault(); saveVerdict({ verdict: "artifact" }, true); }
      else if (e.key === "3") { e.preventDefault(); saveVerdict({ verdict: "uncertain" }, true); }
      else if (e.key === "c" || e.key === "C") {
        e.preventDefault();
        clipActiveTriad();
      } else if (e.key === "s" || e.key === "S") {
        e.preventDefault();
        $("sideBySideBtn").click();
      } else if (e.key === "f" || e.key === "F") {
        e.preventDefault();
        toggleTriadDock();
      } else if (e.key === "v" || e.key === "V") {
        e.preventDefault();
        setMcTool("pan");
      } else if (e.key === "m" || e.key === "M") {
        e.preventDefault();
        setMcTool("measure");
      } else if (e.key === "x" || e.key === "X") {
        e.preventDefault();
        setMcTool("exclude");
      } else if (e.key === "a" || e.key === "A" || e.key === "[" || e.key === "PageUp") { e.preventDefault(); moveTime(-1); }
      else if (e.key === "d" || e.key === "D" || e.key === "]" || e.key === "PageDown") { e.preventDefault(); moveTime(1); }
      else if (e.key === "+" || e.key === "=") {
        e.preventDefault(); stepGain(1);
      } else if (e.key === "-") {
        e.preventDefault(); stepGain(-1);
      } else if (e.key === "ArrowLeft") { e.preventDefault(); stepQueue(-1); }
      else if (e.key === "ArrowRight") { e.preventDefault(); stepQueue(1); }
      else if (e.key === "ArrowUp") {
        e.preventDefault();
        if (e.shiftKey) { stepRowHeight(1); }
        else if (st.maxScrollY > 0) { st.scrollY = Math.max(0, st.scrollY - st.curRowH * 2); scheduleMcDraw(); }
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        if (e.shiftKey) { stepRowHeight(-1); }
        else if (st.maxScrollY > 0) { st.scrollY = Math.min(st.maxScrollY, st.scrollY + st.curRowH * 2); scheduleMcDraw(); }
      }
    });

    // View 3: 3D orbit & Case Sheet & Protocol
    document.querySelectorAll("[data-rot]").forEach((b) => {
      b.addEventListener("click", () => {
        const m = b.getAttribute("data-rot");
        document.querySelectorAll("[data-rot]").forEach((x) => x.classList.toggle("active", x === b));
        if (m === "iso") { st.orbit.yaw = -0.62; st.orbit.pitch = 0.34; }
        else if (m === "axial") { st.orbit.yaw = 0; st.orbit.pitch = Math.PI / 2; }
        else if (m === "coronal") { st.orbit.yaw = 0; st.orbit.pitch = 0; }
        else if (m === "sagittal") { st.orbit.yaw = Math.PI / 2; st.orbit.pitch = 0; }
        drawSpatialMap();
      });
    });
    const spc = $("spatialCanvas");
    spc.addEventListener("mousedown", (e) => { st.orbit.dragging = true; st.orbit.moved = false; st.orbit.lx = e.clientX; st.orbit.ly = e.clientY; });
    window.addEventListener("mousemove", (e) => {
      if (!st.orbit.dragging) return;
      const dx = e.clientX - st.orbit.lx, dy = e.clientY - st.orbit.ly;
      if (Math.abs(dx) + Math.abs(dy) > 2) {
        st.orbit.moved = true;
        document.querySelectorAll("[data-rot]").forEach((x) => x.classList.remove("active"));
      }
      st.orbit.yaw += dx * 0.01;
      st.orbit.pitch = Math.max(-1.5, Math.min(1.5, st.orbit.pitch + dy * 0.01));
      st.orbit.lx = e.clientX; st.orbit.ly = e.clientY;
      drawSpatialMap();
    });
    window.addEventListener("mouseup", () => { st.orbit.dragging = false; });
    const spatialPointAt = (e) => {
      const rect = spc.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;
      let hit = null, best = 16;
      st.orbit.pts.forEach((c) => { const d = Math.hypot(c.p.sx - x, c.p.sy - y); if (d < best) { best = d; hit = c; } });
      return hit;
    };
    spc.addEventListener("mousemove", (e) => {
      const hit = !st.orbit.dragging && spatialPointAt(e);
      spc.title = hit ? channelTooltip(hit) : "";
    });
    spc.addEventListener("mouseleave", () => spc.removeAttribute("title"));
    spc.addEventListener("click", (e) => {
      if (st.orbit.moved || !st.orbit.pts.length) return;
      const hit = spatialPointAt(e);
      if (hit) { st.channel = hit.channel_name; st.shaft = hit.display_shaft || "ALL"; st.selectedItemId = null; switchView("reader"); }
    });

    $("deidentifyExport").addEventListener("change", () => {
      const sharing = $("deidentifyExport").checked;
      $("deidentifyExport").closest(".sheet-actions").classList.toggle("sharing", sharing);
      $("exportPseudonym").classList.toggle("hidden", !sharing);
      $("rebuildSheetBtn").textContent = sharing ? t("Export sheet") : t("Rebuild");
      clearExportDownloads();
    });
    $("exportPseudonym").addEventListener("input", clearExportDownloads);
    $("rebuildSheetBtn").addEventListener("click", buildCaseSheet);
    $("exportSlidesBtn").addEventListener("click", async (event) => {
      if (st.blinded || !isActiveCaseLoaded()) return;
      const button = event.currentTarget;
      const epoch = st.contextEpoch;
      const options = exportOptions();
      button.disabled = true;
      clearExportDownloads();
      try {
        const job = await postJSON("/api/export-slides", {
          case: st.caseSlug, primary_run: st.runName, comparison_run: st.compareRun,
          clinical_question: st.lastQuestionCase === st.caseSlug ? $("questionInput").value.trim() || null : null,
          ...options,
        });
        toast(() => t("Building MDT slides…"));
        while (!st.blinded && exportContextMatches(epoch, options)) {
          const status = await getJSON(`/api/jobs/${job.job_id}`);
          if (st.blinded || !exportContextMatches(epoch, options)) return;
          if (status.status === "failed") throw new Error(status.error?.message || status.detail);
          if (status.status === "complete") {
            showExportDownloads([[status.result.download_url, "Download MDT slides"]]);
            return;
          }
          await new Promise((resolve) => setTimeout(resolve, 500));
        }
      } catch (err) {
        if (!st.blinded && exportContextMatches(epoch, options)) toast(() => t("Slide export failed: {message}", { message: err.message }), 4000);
      } finally {
        button.disabled = false;
      }
    });
    if ($("exportBidsBtn")) {
      $("exportBidsBtn").addEventListener("click", async () => {
        if (st.blinded || !isActiveCaseLoaded() || !st.recording) return;
        const epoch = st.contextEpoch;
        const options = exportOptions();
        clearExportDownloads();
        try {
          const res = await postJSON("/api/export-bids", { case: st.caseSlug, recording: st.recording, run_name: st.runName, ...options });
          if (st.blinded || !exportContextMatches(epoch, options)) return;
          toast(() => t("Exported {count} events to BIDS TSV", { count: res.events_exported }));
          if (options.deidentify) {
            showExportDownloads([[res.tsv_url, "De-identified BIDS TSV"], [res.json_url, "BIDS sidecar JSON"]]);
          } else {
            window.open(res.tsv_url, "_blank");
          }
        } catch (err) {
          toast(() => t("BIDS export failed: {message}", { message: err.message }), 4000);
        }
      });
    }
    $("exportEdfBtn").addEventListener("click", async () => {
      if (st.blinded || !isActiveCaseLoaded() || !st.recording) return;
      const epoch = st.contextEpoch;
      const button = $("exportEdfBtn");
      const options = exportOptions();
      button.disabled = true;
      clearExportDownloads();
      try {
        const res = await postJSON("/api/export-edf", { case: st.caseSlug, recording: st.recording, run_name: st.runName, ...options });
        if (st.blinded || !exportContextMatches(epoch, options)) return;
        showExportDownloads([
          [res.edf_url, options.deidentify ? "Download de-identified EDF+" : "Download EDF+"],
          [res.csv_url, "Annotation CSV"], [res.json_url, "Provenance JSON"],
        ]);
        toast(() => res.edf_url
          ? t("Exported {count} annotations", { count: res.annotations_exported })
          : t("Annotations ready. This recording format supports CSV export only."));
      } catch (err) {
        if (epoch === st.contextEpoch) toast(() => t("EEG reader export failed: {message}", { message: err.message }), 4000);
      } finally {
        if (epoch === st.contextEpoch) button.disabled = false;
      }
    });
    $("saveProtocolBtn").addEventListener("click", () => {
      if (!isActiveCaseLoaded()) {
        toast(() => t("Active case failed to load — retry switching cases"), 2500);
        return;
      }
      $("protoNameInput").value = `${st.caseSlug}-${st.runName}`;
      $("protoDescInput").value = "";
      const pm = $("protocolModal");
      pm.dataset.boundCase = st.caseSlug || "";
      pm.dataset.boundRecording = st.recording || "";
      pm.dataset.boundRun = st.runName || "baseline";
      pm.dataset.boundEpoch = String(st.contextEpoch || 0);
      pm.classList.remove("hidden");
      setTimeout(() => $("protoNameInput").focus(), 20);
    });
    if ($("closeProtocolModal")) $("closeProtocolModal").addEventListener("click", () => { closeAndUnbindCaseModals(); flushDeferredUiSync(); });
    if ($("cancelProtocolBtn")) $("cancelProtocolBtn").addEventListener("click", () => { closeAndUnbindCaseModals(); flushDeferredUiSync(); });
    if ($("confirmSaveProtocolBtn")) {
      $("confirmSaveProtocolBtn").addEventListener("click", async () => {
        const pm = $("protocolModal");
        const boundCase = pm.dataset.boundCase || "";
        const boundRec = pm.dataset.boundRecording || "";
        const boundRun = pm.dataset.boundRun || "baseline";
        const boundEpoch = pm.dataset.boundEpoch || "";
        if (
          !isActiveCaseLoaded(boundCase) ||
          boundRec !== (st.recording || "") ||
          boundEpoch !== String(st.contextEpoch || 0)
        ) {
          closeAndUnbindCaseModals();
          toast(() => t("Active case/recording changed — reopen Save Protocol for this case"), 3000);
          return;
        }
        const name = $("protoNameInput").value.trim();
        if (!name) { toast(() => t("Please enter a protocol name")); return; }
        const startEpoch = st.contextEpoch || 0;
        try {
          const res = await postJSON("/api/protocol", {
            name,
            description: $("protoDescInput").value.trim(),
            case: boundCase,
            recording: boundRec,
            run_name: boundRun,
          });
          closeAndUnbindCaseModals();
          toast(() => t("Saved protocol '{protocol}'", { protocol: res.protocol_slug }));
          if ((st.contextEpoch || 0) === startEpoch && isActiveCaseLoaded(boundCase)) {
            await bootstrap({ keepWindow: true });
          }
        } catch (err) {
          toast(() => t("Protocol error: {message}", { message: err.message }), 4000);
        }
      });
    }

    // External SOZ Overlay modal
    if ($("openOverlayBtn")) {
      $("openOverlayBtn").addEventListener("click", () => {
        if (!isActiveCaseLoaded()) {
          closeAndUnbindCaseModals();
          toast(() => t("Active case failed to load — retry switching cases before editing overlay"), 3000);
          return;
        }
        const ov = st.active && st.active.clinical_overlay;
        $("ovSourceInput").value = (ov && ov.source) || "";
        $("ovSozInput").value = (ov && (ov.soz_channels || []).join(", ")) || "";
        $("ovResInput").value = (ov && (ov.resected_channels || []).join(", ")) || "";
        $("ovNotesInput").value = (ov && ov.notes) || "";
        const om = $("overlayModal");
        om.dataset.boundCase = st.active.case || "";
        om.dataset.boundRecording = st.recording || "";
        om.dataset.boundRun = st.runName || "baseline";
        om.dataset.boundEpoch = String(st.contextEpoch || 0);
        om.classList.remove("hidden");
      });
      $("closeOverlayModal").addEventListener("click", () => { closeAndUnbindCaseModals(); flushDeferredUiSync(); });
      $("saveOverlayBtn").addEventListener("click", async () => {
        const om = $("overlayModal");
        const boundCase = om.dataset.boundCase || "";
        const boundRec = om.dataset.boundRecording || "";
        const boundRun = om.dataset.boundRun || "baseline";
        const boundEpoch = om.dataset.boundEpoch || "";
        if (
          !isActiveCaseLoaded(boundCase) ||
          boundEpoch !== String(st.contextEpoch || 0)
        ) {
          closeAndUnbindCaseModals();
          toast(() => t("Active case changed — reopen External Clinical Overlay for this case"), 3000);
          return;
        }
        const src = $("ovSourceInput").value.trim();
        if (!src) { toast(() => t("Source citation is required for external overlays.")); return; }
        const startEpoch = st.contextEpoch || 0;
        try {
          await postJSON("/api/overlay", {
            case: boundCase,
            recording: boundRec || st.recording,
            run_name: boundRun || st.runName,
            source: src,
            soz_channels: $("ovSozInput").value,
            resected_channels: $("ovResInput").value,
            notes: $("ovNotesInput").value,
          });
          closeAndUnbindCaseModals();
          if ((st.contextEpoch || 0) !== startEpoch || !isActiveCaseLoaded(boundCase)) {
            toast(() => t("External clinical overlay saved on {caseName}", { caseName: boundCase }));
            return;
          }
          await bootstrap({ caseSlug: boundCase, recording: st.recording, runName: st.runName, keepWindow: true });
          if ((st.contextEpoch || 0) === startEpoch && isActiveCaseLoaded(boundCase)) {
            await buildCaseSheet({ auto: true });
          }
          toast(() => t("External clinical overlay saved & spatial concordance updated"));
        } catch (err) {
          toast(() => t("Overlay error: {message}", { message: err.message }), 4000);
        }
      });
      $("clearOverlayBtn").addEventListener("click", async () => {
        const om = $("overlayModal");
        const boundCase = om.dataset.boundCase || "";
        const boundEpoch = om.dataset.boundEpoch || "";
        if (
          !isActiveCaseLoaded(boundCase) ||
          boundEpoch !== String(st.contextEpoch || 0)
        ) {
          closeAndUnbindCaseModals();
          toast(() => t("Active case changed — reopen External Clinical Overlay for this case"), 3000);
          return;
        }
        const startEpoch = st.contextEpoch || 0;
        try {
          await postJSON("/api/overlay", { case: boundCase, delete: true });
          closeAndUnbindCaseModals();
          if ((st.contextEpoch || 0) !== startEpoch || !isActiveCaseLoaded(boundCase)) {
            toast(() => t("External overlay removed on {caseName}", { caseName: boundCase }));
            return;
          }
          await bootstrap({ caseSlug: boundCase, recording: st.recording, runName: st.runName, keepWindow: true });
          if ((st.contextEpoch || 0) === startEpoch && isActiveCaseLoaded(boundCase)) {
            await buildCaseSheet({ auto: true });
          }
          toast(() => t("External overlay removed"));
        } catch (err) {
          toast(() => t("Overlay error: {message}", { message: err.message }), 4000);
        }
      });
    }

    // Cohort Batch Runner modal
    ["openCohortBtn", "openCohortBtn3"].forEach((id) => {
      if ($(id)) $(id).addEventListener("click", () => { renderCohortModal(); $("cohortModal").classList.remove("hidden"); });
    });
    if ($("closeCohortModal")) $("closeCohortModal").addEventListener("click", () => { $("cohortModal").classList.add("hidden"); flushDeferredUiSync(); });
    if ($("runCohortBtn")) {
      $("runCohortBtn").addEventListener("click", async () => {
        const selectedCases = [...$("cohCasesChecks").querySelectorAll("input:checked")].map((x) => x.value);
        if (!selectedCases.length) { toast(() => t("Select at least one case for the cohort.")); return; }
        $("cohortModal").classList.add("hidden");
        await submitAsyncJob(
          "/api/jobs/cohort",
          {
            targets: selectedCases,
            protocol: $("cohProtocolSelect").value || null,
            cohort_name: $("cohNameInput").value.trim() || null,
          },
          async (coh) => {
            await bootstrap({ keepWindow: true });
            $("cohortModal").classList.remove("hidden");
            toast(() => t("Cohort '{cohort}' done ({completed}/{total})", { cohort: coh.cohort_slug, completed: coh.completed_recordings, total: coh.total_recordings }));
          },
        );
      });
    }

    // Modals (with backdrop click-to-close)
    ["openModal", "branchModal", "overlayModal", "cohortModal", "protocolModal"].forEach((id) => {
      const m = $(id);
      if (m) m.addEventListener("click", (e) => {
        if (e.target === m) {
          m.classList.add("hidden");
          delete m.dataset.boundCase;
          delete m.dataset.boundRecording;
          delete m.dataset.boundRun;
          delete m.dataset.boundEpoch;
          flushDeferredUiSync();
        }
      });
    });
    $("openCaseBtn").addEventListener("click", () => $("openModal").classList.remove("hidden"));
    $("closeOpenModal").addEventListener("click", () => { $("openModal").classList.add("hidden"); flushDeferredUiSync(); });
    $("browsePathBtn").addEventListener("click", e => browseDataset($("customPathInput").value.trim(), e.currentTarget));
    $("importPathBtn").addEventListener("click", () => openDataset($("customPathInput").value.trim()));

    $("newBranchBtn").addEventListener("click", () => {
      if (!isActiveCaseLoaded() || !st.recording) {
        toast(() => t("Active case failed to load — retry switching cases"), 2500);
        return;
      }
      $("branchTargetLabel").textContent = `${st.caseSlug} / ${st.recording}`;
      $("brName").value = "";
      $("brProtocol").value = "";
      $("brExStart").value = "";
      $("brExEnd").value = "";
      $("brExCh").value = "";
      if ($("brState")) $("brState").value = "";
      $("brDetector").value = "STE";
      $("brRefMode").value = "original";
      $("brRms").value = "";
      $("brClassify").checked = false;
      const bm = $("branchModal");
      bm.dataset.boundCase = st.active.case || "";
      bm.dataset.boundRecording = st.recording || "";
      bm.dataset.boundRun = st.runName || "baseline";
      bm.dataset.boundEpoch = String(st.contextEpoch || 0);
      bm.classList.remove("hidden");
    });
    $("closeBranchModal").addEventListener("click", () => { closeAndUnbindCaseModals(); flushDeferredUiSync(); });
    $("presetEx60").addEventListener("click", (e) => { e.preventDefault(); $("brExStart").value = "0"; $("brExEnd").value = "60"; $("brName").value = "excl_0s_60s"; });
    $("presetOnnx").addEventListener("click", (e) => { e.preventDefault(); $("brClassify").checked = true; $("brName").value = "onnx_clf"; });
    $("presetRms6").addEventListener("click", (e) => { e.preventDefault(); $("brRms").value = "6"; $("brName").value = "ste_rms_6"; });

    $("submitBranchBtn").addEventListener("click", async () => {
      const bm = $("branchModal");
      const boundCase = bm.dataset.boundCase || "";
      const boundRec = bm.dataset.boundRecording || "";
      const boundEpoch = bm.dataset.boundEpoch || "";
      if (
        !isActiveCaseLoaded(boundCase) ||
        !boundRec ||
        boundRec !== st.recording ||
        boundEpoch !== String(st.contextEpoch || 0)
      ) {
        closeAndUnbindCaseModals();
        toast(() => t("Active case/recording changed — reopen Sensitivity Branch for this recording"), 3000);
        return;
      }
      closeAndUnbindCaseModals();
      const startEpoch = st.contextEpoch || 0;
      const protoSlug = $("brProtocol").value || null;
      const overrides = {};
      // When a saved protocol is chosen, preserve its detector & reference_mode unless user explicitly overrode them
      if (!protoSlug) {
        overrides.detector = $("brDetector").value;
        overrides.reference_mode = $("brRefMode").value;
      } else {
        if ($("brDetector").value !== "STE") overrides.detector = $("brDetector").value;
        if ($("brRefMode").value !== "original") overrides.reference_mode = $("brRefMode").value;
      }
      const es = parseFloat($("brExStart").value), ee = parseFloat($("brExEnd").value);
      if (Number.isFinite(es) && Number.isFinite(ee) && ee > es) {
        overrides.exclude_intervals = [{ start: es, end: ee, reason: `Exclude [${es}, ${ee})s` }];
      }
      const exCh = $("brExCh").value.split(",").map((x) => x.trim()).filter(Boolean);
      if (exCh.length) overrides.exclude_channels = exCh;
      const stSlice = $("brState") && $("brState").value;
      if (stSlice) overrides.state = stSlice;
      const rms = parseFloat($("brRms").value);
      if (Number.isFinite(rms) && rms > 0) overrides.rms_thres = rms;
      if ($("brClassify").checked) overrides.classifier_enabled = true;

      await submitAsyncJob(
        "/api/jobs/branch",
        {
          target: `${boundCase}/${boundRec}`,
          run_name: $("brName").value.trim() || null,
          protocol: protoSlug,
          overrides,
          parent_run: "baseline",
        },
        async (out) => {
          if ((st.contextEpoch || 0) !== startEpoch || st.caseSlug !== boundCase || st.recording !== boundRec) {
            toast(() => t("Completed branch '{run}' on {caseName}/{recording}", { run: out.run_name, caseName: boundCase, recording: boundRec }));
            return;
          }
          st.runName = out.run_name;
          st.compareRun = out.run_name;
          await bootstrap({ caseSlug: boundCase, recording: boundRec, runName: st.runName });
          switchView("overview");
          const run = st.runName;
          toast(() => t("Completed branch '{run}'", { run }));
        },
      );
    });

    let resizeTimer = null;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (st.view === "overview" && st.active && st.active.run_summary) drawOverviewTimeline(st.active.run_summary);
        else if (st.view === "reader") { drawMultichannel(); drawReaderMinimap(); if (!st.triadCollapsed) drawTriad(); }
        else if (st.view === "sheet") drawSpatialMap();
      }, 60);
    });

    // Bridge messages strictly from our own same-origin embedded MDT Case Sheet iframe
    window.addEventListener("message", async (ev) => {
      if (st.blinded) return;
      const fr = $("sheetFrame");
      if (ev.origin !== window.location.origin || !fr || ev.source !== fr.contentWindow) return;
      const msg = ev.data;
      if (!msg || typeof msg !== "object") return;
      if (msg.type === "brainlens_request_sse_status") {
        const sheetMatches = Boolean(st.loadedSheetKey && st.loadedSheetKey.startsWith(`${st.caseSlug}:${st.runName}:`));
        syncReviewerToSheet();
        syncSheetContext(!sheetMatches);
        forwardSseToSheet("status", null, { online: st.sseConnected, sessionExpired: st.sessionExpired });
        return;
      }
      if (msg.type === "brainlens_request_sheet_context") {
        const sheetMatches = Boolean(st.loadedSheetKey && st.loadedSheetKey.startsWith(`${st.caseSlug}:${st.runName}:`));
        if (sheetMatches) st.sheetStaleRetries = 0;
        syncReviewerToSheet();
        syncSheetContext(!sheetMatches);
        return;
      }
      if (msg.type === "brainlens_composer_state") {
        st.iframeComposerOpen = Boolean(msg.open);
        if (!st.iframeComposerOpen) flushDeferredUiSync();
        return;
      }
      if (msg.type === "brainlens_sheet_stale") {
        if (
          st.view === "sheet" &&
          !st.contextSwitching &&
          msg.case === st.caseSlug &&
          (!msg.primary_run || msg.primary_run === st.runName) &&
          st.sheetStaleRetries < 2
        ) {
          st.sheetStaleRetries += 1;
          await bootstrap({ caseSlug: st.caseSlug, recording: st.recording, runName: st.runName, skipRender: true, lite: true });
          if (st.view === "sheet" && msg.case === st.caseSlug && (!msg.primary_run || msg.primary_run === st.runName)) {
            drawSpatialMap();
            await buildCaseSheet({ auto: true });
          }
        }
        return;
      }
      if (msg.type === "brainlens_report_verdict") {
        // Silently refresh active metadata so View 1 KPIs & View 2 queue stay synced with report verdicts
        if (!st.contextSwitching && (!msg.case || msg.case === st.caseSlug)) {
          await bootstrap({ caseSlug: st.caseSlug, recording: st.recording, runName: st.runName, skipRender: true, lite: true });
        }
        return;
      }
      if (msg.type !== "brainlens_jump_waveform") return;
      const jumpCaseChanged = Boolean(msg.case && msg.case !== st.caseSlug);
      beginContextSwitch(jumpCaseChanged);
      const jumpCase = msg.case || st.caseSlug;
      const jumpRec = msg.recording || st.recording;
      const jumpRun = msg.run_name || st.runName;
      try {
        // Load new case/recording metadata FIRST before resolving shaft, keeping contextSwitching locked!
        await bootstrap({
          caseSlug: jumpCase,
          recording: jumpRec,
          runName: jumpRun,
          targetChannel: msg.channel || null,
          targetView: "reader",
          skipRender: true,
          keepSwitching: true,
          lite: true,
        });
        if (!st.active || st.loadedCaseSlug !== jumpCase) return;
        st.selectedItemId = msg.event_id || null;
        const resolvedSh = st.channel ? resolveChannelShaft(st.channel) : null;
        st.shaft = resolvedSh || (st.active.recording_spec?.modality === "scalp" ? "ALL" : "PAGE:0");
        renderShaftPills();
        const hasStart = msg.start_seconds !== undefined && msg.start_seconds !== null && msg.start_seconds !== "";
        const hasEnd = msg.end_seconds !== undefined && msg.end_seconds !== null && msg.end_seconds !== "";
        const hasCenter = msg.center_seconds !== undefined && msg.center_seconds !== null && msg.center_seconds !== "";
        const t0 = hasStart ? Number(msg.start_seconds) : NaN;
        const t1 = hasEnd ? Number(msg.end_seconds) : NaN;
        const cNum = hasCenter ? Number(msg.center_seconds) : NaN;
        if (msg.event_id) {
          await loadTriadCanvas({ eventId: msg.event_id });
          const tr = st.triadData;
          if (tr && tr.window_seconds) {
            const mid = 0.5 * (tr.window_seconds[0] + tr.window_seconds[1]);
            st.startSec = Math.max(0, mid - st.winSec * 0.5);
            st.viewStartSec = st.startSec;
            $("readerStartInput").value = st.startSec.toFixed(1);
          }
          await loadReaderQueueAndCanvases({ keepWindow: true });
        } else if (Number.isFinite(t0) || Number.isFinite(cNum)) {
          const span = (Number.isFinite(t0) && Number.isFinite(t1) && t1 > t0) ? (t1 - t0) : 0;
          const center = Number.isFinite(cNum)
            ? cNum
            : (Number.isFinite(t0) && Number.isFinite(t1) ? 0.5 * (t0 + t1) : t0);
          const recDur = Number((st.active && st.active.recording_spec && st.active.recording_spec.duration) || 0);
          if (span > 0 && st.winSec < span) {
            $("readerTimebaseSelect").value = "fit";
            $("readerDurSelect").classList.remove("hidden");
            const presets = [0.5, 1, 2, 4, 5, 8, 10, 15, 30];
            st.winSec = presets.find((p) => p >= span) || Math.min(30, Math.ceil(span));
          }
          let nextStart = Math.max(0, center - st.winSec * 0.5);
          if (recDur > 0) {
            nextStart = Math.min(Math.max(0, recDur - st.winSec), nextStart);
          }
          st.startSec = nextStart;
          st.viewStartSec = nextStart;
          if ($("readerStartInput")) $("readerStartInput").value = st.startSec.toFixed(1);
          if ($("readerTimebaseSelect").value === "fit") $("readerDurSelect").value = String(st.winSec);
          await loadReaderQueueAndCanvases({
            keepWindow: true,
            manualTime: center,
            manualSpan: span > 0 ? span : undefined,
          });
        } else {
          await loadReaderQueueAndCanvases();
        }
      } finally {
        st.contextSwitching = false;
        questions.refresh(false);
        if (!st.triadLoading && st.triadData && isRenderedTriadSynced()) {
          updateVerdictPanel(st.triadData);
        }
      }
      syncToAgent();
      toast(() => t("Jumped to {target} from MDT Sheet", { target: msg.channel || msg.recording }));
    });
  }

  function isUserEditingOrModalOpen() {
    const noteEl = $("noteInput");
    const qEl = $("questionInput");
    return (
      hasOpenModal() ||
      st.iframeComposerOpen || questions.editing() ||
      Boolean(noteEl && noteEl.value.trim().length > 0) ||
      Boolean(qEl && qEl.value.trim().length > 0)
    );
  }

  async function applyRemoteUiState(ui) {
    if (st.blinded || st.queueMode === "review_set") return;
    if (!ui || !ui.seq || ui.seq <= st.uiSeq) return;
    st.uiSeq = ui.seq;
    const caseChanged = Boolean(ui.case && ui.case !== st.caseSlug);
    const contextChanged =
      caseChanged ||
      Boolean(ui.recording && ui.recording !== st.recording) ||
      Boolean(ui.run_name && ui.run_name !== st.runName);
    if (contextChanged) {
      beginContextSwitch(caseChanged);
    }
    const nextCase = ui.case || st.caseSlug;
    const nextRec = ui.recording || (caseChanged ? null : st.recording);
    const nextRun = ui.run_name || (caseChanged ? "baseline" : st.runName);
    await bootstrap({
      caseSlug: nextCase,
      recording: nextRec,
      runName: nextRun,
      targetCompareRun: ui.compare_run !== undefined ? ui.compare_run : undefined,
      targetChannel: ui.channel || undefined,
      targetShaft: ui.shaft || undefined,
      targetQueueMode: ui.queue_mode || undefined,
      targetView: ui.view || undefined,
      focusStart: ui.updated_by === "agent_focus_time" && ui.view === "reader" && Number.isFinite(ui.start_seconds)
        ? ui.start_seconds : undefined,
      lite: true,
    });
    toast(() => ui.updated_by === "agent_focus_time" ? t("Reader position updated") : t("Synced with {agent}", { agent: ui.updated_by }));
  }

  function flushDeferredUiSync() {
    if (!st.deferredUiSync || isUserEditingOrModalOpen()) return;
    const pending = st.deferredUiSync;
    st.deferredUiSync = null;
    applyRemoteUiState(pending);
  }

  async function reconcileOnReconnect(refreshView = false) {
    if (st.blinded) return;
    questions.refresh();
    try {
      const ui = await getJSON("/api/ui-state");
      if (ui.pending_action_count !== undefined) {
        st.pendingActionCount = Number(ui.pending_action_count || 0);
        renderSyncBadge();
      }
      if (ui.seq && ui.seq > st.uiSeq && ui.updated_by !== "workbench" && ui.updated_by !== "report_sheet" && ui.updated_by !== "pipeline") {
        if (isUserEditingOrModalOpen()) st.deferredUiSync = ui;
        else await applyRemoteUiState(ui);
        return;
      }
      if (refreshView && st.caseSlug) {
        forwardSseToSheet("resync", { reason: "pageshow" });
        if (st.contextSwitching) return;
        if (hasSavingForCase(st.caseSlug)) {
          st.pendingViewRefresh = true;
        } else {
          st.pendingViewRefresh = false;
          if (st.view === "reader") {
            await bootstrap({ caseSlug: st.caseSlug, recording: st.recording, runName: st.runName, keepWindow: true, skipRender: true, lite: true });
            if (isActiveCaseLoaded()) await loadReaderQueueAndCanvases({ keepWindow: true });
          } else {
            await bootstrap({ caseSlug: st.caseSlug, recording: st.recording, runName: st.runName, keepWindow: true, lite: true });
          }
        }
      }
    } catch (_) {}
  }

  function forwardSseToSheet(event, data = null, extra = {}) {
    const fr = $("sheetFrame");
    if (fr && fr.contentWindow) {
      try {
        fr.contentWindow.postMessage({ type: "brainlens_sse", event, data, ...extra }, window.location.origin);
      } catch (_) {}
    }
  }

  function connectSSE() {
    if (st.blinded || typeof window.EventSource !== "function") return;
    if (st.sseSource) {
      try { st.sseSource.close(); } catch (_) {}
      st.sseSource = null;
    }
    const es = new EventSource("/api/events");
    st.sseSource = es;
    if (!st.ssePageListenersBound) {
      st.ssePageListenersBound = true;
      window.addEventListener("pagehide", () => {
        st.needsRefreshOnReconnect = true;
        if (st.sseSource) {
          try { st.sseSource.close(); } catch (_) {}
          st.sseSource = null;
        }
      });
      window.addEventListener("pageshow", () => {
        st.needsRefreshOnReconnect = true;
        if (!st.sseSource && !st.sessionExpired) {
          connectSSE();
        } else if (st.sseConnected && !st.sessionExpired) {
          st.needsRefreshOnReconnect = false;
          reconcileOnReconnect(true);
        }
      });
    }
    es.onopen = () => {
      const wasReconnect = st.sseEverOpened;
      const forceRefresh = Boolean(st.needsRefreshOnReconnect);
      st.needsRefreshOnReconnect = false;
      st.sseEverOpened = true;
      st.sseConnected = !st.sessionExpired;
      renderSyncBadge();
      forwardSseToSheet("status", null, { online: st.sseConnected, sessionExpired: st.sessionExpired });
      // Refresh active view after pagehide/pageshow so verdicts saved while hidden sync immediately,
      // while keeping benign transient reconnects lightweight with zero flash!
      if ((wasReconnect || forceRefresh) && !st.sessionExpired) reconcileOnReconnect(forceRefresh);
    };
    es.onerror = () => {
      st.sseConnected = false;
      renderSyncBadge();
      forwardSseToSheet("status", null, { online: false, sessionExpired: st.sessionExpired });
    };
    es.addEventListener("ready", () => {
      if (!st.sessionExpired) {
        const forceRefresh = Boolean(st.needsRefreshOnReconnect);
        st.needsRefreshOnReconnect = false;
        reconcileOnReconnect(forceRefresh);
      }
    });
    es.addEventListener("resync", async (ev) => {
      let d = {};
      try {
        d = JSON.parse(ev.data || "{}");
        if (d.reason === "server_restarted") {
          st.sessionExpired = true;
          st.sseConnected = false;
          renderSyncBadge();
          forwardSseToSheet("resync", d);
          toast(() => t("Workbench server restarted — reload page to reconnect"), 5000);
          return;
        }
      } catch (_) {}
      forwardSseToSheet("resync", d);
      await reconcileOnReconnect(true);
    });
    es.addEventListener("ui_state", async (ev) => {
      try {
        const ui = JSON.parse(ev.data || "{}");
        if (ui.pending_action_count !== undefined) {
          st.pendingActionCount = Number(ui.pending_action_count || 0);
          renderSyncBadge();
        }
        if (ui.seq && ui.seq > st.uiSeq && ui.updated_by !== "workbench" && ui.updated_by !== "report_sheet" && ui.updated_by !== "pipeline") {
          if (isUserEditingOrModalOpen()) {
            st.deferredUiSync = ui;
            toast(() => t("Agent updated focus — queued until modal/note closes"), 2200);
          } else {
            await applyRemoteUiState(ui);
          }
        }
      } catch (_) {}
    });
    es.addEventListener("report_actions", (ev) => {
      try {
        const d = JSON.parse(ev.data || "{}");
        if (d.global_pending_count !== undefined) {
          st.pendingActionCount = Number(d.global_pending_count || 0);
          renderSyncBadge();
        } else if (d.pending_count !== undefined && (!d.case || d.case === st.caseSlug)) {
          st.pendingActionCount = Number(d.pending_count || 0);
          renderSyncBadge();
        }
        if (!d.case || d.case === st.caseSlug) questions.refresh();
        forwardSseToSheet("report_actions", d);
      } catch (_) {}
    });
    es.addEventListener("case_updated", async (ev) => {
      try {
        const d = JSON.parse(ev.data || "{}");
        forwardSseToSheet("case_updated", d);
        if (d.case && d.case === st.caseSlug && !hasSavingForCase(st.caseSlug) && !st.contextSwitching && st.view !== "sheet") {
          await bootstrap({ caseSlug: st.caseSlug, recording: st.recording, runName: st.runName, keepWindow: true, lite: true });
        }
      } catch (_) {}
    });
    es.addEventListener("verdict", async (ev) => {
      try {
        const d = JSON.parse(ev.data || "{}");
        forwardSseToSheet("verdict", d);
        if (d.operation_id && st.recentLocalOpIds.has(d.operation_id)) {
          st.recentLocalOpIds.delete(d.operation_id);
          return;
        }
        if (d.case && d.case === st.caseSlug && !hasSavingForCase(st.caseSlug) && !st.contextSwitching) {
          if (st.view === "overview") {
            await bootstrap({ caseSlug: st.caseSlug, recording: st.recording, runName: st.runName, keepWindow: true, lite: true });
          } else if (st.view === "reader") {
            await bootstrap({ caseSlug: st.caseSlug, recording: st.recording, runName: st.runName, skipRender: true, lite: true });
            if (isActiveCaseLoaded()) await loadReaderQueueAndCanvases({ keepWindow: true });
          } else {
            await bootstrap({ caseSlug: st.caseSlug, recording: st.recording, runName: st.runName, skipRender: true, lite: true });
          }
        }
      } catch (_) {}
    });
  }

  wireEvents();
  const initCenter = urlParams.get("center_seconds") !== null ? parseFloat(urlParams.get("center_seconds")) : NaN;
  const initSpan = urlParams.get("span_seconds") !== null ? parseFloat(urlParams.get("span_seconds")) : NaN;
  const initOpts = (!st.selectedItemId && Number.isFinite(initCenter))
    ? { keepWindow: true, manualTime: initCenter, manualSpan: Number.isFinite(initSpan) && initSpan > 0 ? initSpan : undefined }
    : {};
  const focusStart = parseFloat(urlParams.get("start_seconds"));
  if (!st.selectedItemId && Number.isFinite(focusStart)) initOpts.focusStart = focusStart;
  bootstrap(initOpts).finally(() => connectSSE());
})();
