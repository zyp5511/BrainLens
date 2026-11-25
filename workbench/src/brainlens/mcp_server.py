"""Lean, orthogonal 7-tool MCP server for BrainLens Neurophysiology Workbench."""

from __future__ import annotations

import argparse
import contextvars
import functools
import json
import math
import os
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import anyio
import anyio.from_thread
import anyio.to_thread
from mcp.server.fastmcp import FastMCP

from .analytics import compute_overlay_concordance, query_case
from .anatomy import read_anatomy_sidecar, recording_anatomy_coverage, save_anatomy_labels, validate_anatomy_labels
from .bids_export import export_bids_events
from .deid import agent_identity_scope, scrub_agent_payload, scrub_navigation_dates
from .edf_export import export_edf_annotations
from .case_sheet import export_case_sheet
from .cohort import list_cohorts, run_cohort
from .io_cases import navigation_recording_slug, resolve_case_target
from .io_readers import discover_available_inputs, inspect_case as _inspect_case
from .models import ToolkitError, default_workspace_dir, get_env
from .pipeline import analyze_case as _analyze_case, refresh_verdict_tallies
from .protocols import list_protocols, save_protocol
from .pyhfo_runner import check_engine_health
from .storage import WorkbenchStore, sanitize_slug, validate_clinical_overlay
from .storage_actions import NAVIGATION_QUESTION_SOURCES, QUESTION_CASE_REF
from .triad_views import render_evidence_card as _render_evidence_card
from .workbench_http import ensure_workbench_server, is_loopback_http_url

_WORKSPACE_ROOT: Path = default_workspace_dir()


def set_workspace_root(path: str | Path) -> Path:
    global _WORKSPACE_ROOT
    _WORKSPACE_ROOT = Path(path).expanduser().resolve()
    return _WORKSPACE_ROOT


def get_workspace_root() -> Path:
    return _WORKSPACE_ROOT


def _wrap_call(fn, *args, **kwargs) -> dict[str, Any]:
    with agent_identity_scope():
        try:
            result = fn(*args, **kwargs)
        except ToolkitError as exc:
            result = {"error": True, "code": exc.code, "message": exc.message, "details": exc.details}
        except Exception as exc:
            result = {"error": True, "code": "unexpected_error", "message": str(exc)}
        try:
            safe = scrub_agent_payload(result, WorkbenchStore(get_workspace_root(), create_dirs=False))
            for key in ("pending_report_actions", "actions"):
                if not isinstance(result.get(key), list) or not isinstance(safe.get(key), list):
                    continue
                for raw, row in zip(result[key], safe[key]):
                    if not isinstance(raw, dict) or raw.get("source") not in NAVIGATION_QUESTION_SOURCES:
                        continue
                    if not isinstance(row, dict) or not row.get("case"):
                        continue
                    # Navigation labels can contain a calendar date even after native header identity is masked.
                    for field in ("case", "recording", "channel", "marker", "comment", "agent_reply"):
                        if row.get(field) is not None:
                            row[field] = scrub_navigation_dates(row[field])
                    row.pop("case_ref", None)
                    ref = raw.get("case_ref") or ""
                    if raw.get("case") != row["case"] and QUESTION_CASE_REF.fullmatch(ref):
                        row["case_ref"] = ref
            if "resolved_ids" in safe and safe.get("agent_reply"):
                safe["agent_reply"] = scrub_navigation_dates(safe["agent_reply"])
            return safe
        except Exception:
            return {"error": True, "code": "identity_redaction_failed", "message": "Could not safely prepare the agent response."}


mcp = FastMCP(
    "brainlens",
    instructions=(
        "BrainLens — Evidence-grounded SEEG/iEEG/EEG Neurophysiology Workbench. "
        "Use inspect_case -> analyze_case -> query_evidence / render_evidence_card / open_workbench -> "
        "record_judgment -> export_or_protocol. Check open_workbench(action='status') to read live UI focus "
        "and pending_report_actions from the reader, admission timeline and MDT Case Sheet, "
        "and call open_workbench(action='resolve_actions', case=..., action_ids=[...], agent_reply=...) after "
        "addressing them so the clinician sees your reply alongside the question. Separate focal "
        "candidates from widespread multi-shaft synchronous bursts, inspect candidate-free control windows, "
        "preserve provenance per recording, and keep clinical interpretation humble and non-diagnostic. "
        "Exports (EDF+/PPTX) support optional de-identification; also available: external anatomy labels, "
        "evidence map / ictal EI and reader agreement."
    ),
)


# FastMCP calls synchronous tools inline on its event loop, so a multi-minute PyHFO run or a 120 s
# co-review wait used to stall pings, cancellation and progress for the whole stdio session.
# Each tool body below stays a plain synchronous function (importable and directly callable), and
# `_threaded_tool` registers an async shim that runs it in a worker thread instead.
# Bodies still run one at a time (matplotlib pyplot and the stage caches were written for serialized
# calls); `open_workbench(action="wait_action")` only long-polls the local workbench server, so it
# waits outside that lock and does not block status/resolve calls made meanwhile.
_TOOL_LOCK = threading.Lock()
_PROGRESS_CB: contextvars.ContextVar[Callable[[str, float, str], None] | None] = contextvars.ContextVar(
    "brainlens_mcp_progress_cb", default=None
)
_PROGRESS_MIN_INTERVAL_S = 0.5


def _current_progress_cb() -> Callable[[str, float, str], None] | None:
    """Progress callback for the active MCP request (None for direct Python calls or clients without a progress token)."""
    return _PROGRESS_CB.get()


def _progress_forwarder() -> Callable[[str, float, str], None] | None:
    """Build a worker-thread callback that relays `(stage, fraction, detail)` as MCP progress notifications."""
    try:
        ctx = mcp.get_context()
        meta = ctx.request_context.meta
    except Exception:
        return None
    if meta is None or getattr(meta, "progressToken", None) is None:
        return None
    last: dict[str, Any] = {"t": 0.0, "stage": None}

    def _cb(stage: str, pct: float, detail: str) -> None:
        now = time.monotonic()
        frac = max(0.0, min(1.0, float(pct)))
        if stage == last["stage"] and frac < 1.0 and now - last["t"] < _PROGRESS_MIN_INTERVAL_S:
            return
        last["t"], last["stage"] = now, stage
        try:
            # Free-form progress details can contain a source filename before its header is inspected.
            anyio.from_thread.run(ctx.report_progress, frac, 1.0, str(stage))
        except Exception:
            pass

    return _cb


def _run_tool_body(
    body: Callable[..., dict[str, Any]],
    kwargs: dict[str, Any],
    progress_cb: Callable[[str, float, str], None] | None,
    serialize: bool,
) -> dict[str, Any]:
    token = _PROGRESS_CB.set(progress_cb)
    try:
        if serialize:
            with _TOOL_LOCK:
                return body(**kwargs)
        return body(**kwargs)
    finally:
        _PROGRESS_CB.reset(token)


def _threaded_tool(
    *,
    unlocked_when: Callable[[dict[str, Any]], bool] | None = None,
) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Register `body` as an MCP tool executed in a worker thread; return `body` unchanged for direct callers."""

    def _register(body: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        @functools.wraps(body)
        async def _tool(**kwargs: Any) -> dict[str, Any]:
            serialize = not (unlocked_when is not None and unlocked_when(kwargs))
            call = functools.partial(_run_tool_body, body, kwargs, _progress_forwarder(), serialize)
            return await anyio.to_thread.run_sync(call)

        mcp.tool()(_tool)
        return body

    return _register


@_threaded_tool()
def inspect_case(
    target: str = "*",
    modality: str | None = None,
    spatial_sidecar: str | None = None,
    anatomy_sidecar: str | None = None,
    anatomy_source: str | None = None,
    anatomy_patient_native: bool = False,
) -> dict[str, Any]:
    """List workspace cases & local inputs (target='*') or inspect a recording/folder/BIDS case.

    Reads channel geometry, SEEG shafts, sampling rate, physical voltage units, BIDS sidecars,
    published bipolar pairs, 3D coordinates, and per-recording provenance without modifying source files.
    For a registered case, anatomy_sidecar imports external contact/region CSV or BIDS electrodes TSV.
    Supply anatomy_source. Set anatomy_patient_native only for the patient's own MRI/CT co-registration.
    """
    def _impl() -> dict[str, Any]:
        ws = get_workspace_root()
        store = WorkbenchStore(ws)
        anatomy_import = None
        if anatomy_sidecar is not None:
            anatomy_import = read_anatomy_sidecar(anatomy_sidecar, anatomy_source, patient_native=anatomy_patient_native)
            if resolve_case_target(target, store)["kind"] not in {"registered_case", "registered_recording"}:
                raise ToolkitError("case_not_registered", "Anatomy import requires a registered case target.")
        if target.strip() in {"*", "", "list", "workspace"}:
            cases = []
            for cdoc in store.list_cases():
                cslug = cdoc["case"]
                recs = store.list_recordings(cslug)
                cases.append({
                    "case": cslug,
                    "display_name": cdoc.get("display_name", cslug),
                    "recordings": recs,
                    "runs_by_recording": {
                        r: [m.get("run_name") for m in store.list_runs(cslug, r)] for r in recs
                    },
                })
            return {
                "workspace": str(store.root),
                "registered_cases": cases,
                "available_inputs": discover_available_inputs(),
                "saved_protocols": [p["slug"] for p in list_protocols(store.root)],
                "engine": check_engine_health(store.root),
            }
        result = _inspect_case(
            target=target,
            workspace_dir=ws,
            modality=modality,
            spatial_sidecar=spatial_sidecar,
            persist=True,
            progress_cb=_current_progress_cb(),
        )
        if anatomy_import is not None:
            payload, counts = anatomy_import
            result["anatomy_labels"] = save_anatomy_labels(store, result["case"], payload, author="agent")
            result["anatomy_import"] = counts
            result["coverage_by_recording"] = recording_anatomy_coverage(store, result["case"], result["anatomy_labels"])
        return result

    return _wrap_call(_impl)


@_threaded_tool()
def analyze_case(
    target: str,
    run_name: str | None = None,
    overrides: dict[str, Any] | None = None,
    parent_run: str | None = None,
    protocol: str | None = None,
    reason: str | None = None,
    spatial_sidecar: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run full-coverage QC, zero-phase prep, PyHFO detection, cross-shaft synchrony & optional ONNX scoring.

    Automatically reuses unchanged QC/Prepare/Detect stages when branching.
    Common `overrides` keys (unknown keys raise `unknown_override_key`):
    - `detector`: 'STE' | 'MNI' | 'HIL'
    - `band`: [80, 250] or [250, 450]
    - Detector parameters (flat or inside `detector_params`):
      - STE: `rms_thres`, `rms_window`, `min_window` (`min_win`), `min_gap`, `min_osc`, `peak_thres`, `epoch_len` (`epoch_time`)
      - MNI: `min_win` (`min_window`), `min_gap`, `epoch_time` (`epoch_len`), `thrd_perc` (`percentile`), `epo_CHF`, `per_CHF`, `base_seg`, `base_shift`, `base_thrd`, `base_min`
      - HIL: `sd_threshold`, `min_window` (`min_win`), `epoch_time` (`epoch_len`)
    - `reference_mode`: 'original' | 'bipolar' | 'average'
    - `exclude_intervals`: [{'start': 0, 'end': 60, 'channels': ['LB10'], 'reason': '...'}]
    - `exclude_channels`: ['LA1']
    - `classifier_enabled`: True, `classifier_stages`: ['artifact', 'spike', 'ehfo']
    - `start_seconds`, `end_seconds`, `state`
    - `segments`: [{start_seconds, end_seconds, label}, ...] on one recording's true-time clock
    - `marker_segments`: {pattern, before_seconds, after_seconds} selects windows around matching markers
    """
    return _wrap_call(
        _analyze_case,
        target=target,
        workspace_dir=get_workspace_root(),
        run_name=run_name,
        overrides=overrides,
        parent_run=parent_run,
        protocol=protocol,
        reason=reason,
        spatial_sidecar=spatial_sidecar,
        overwrite=overwrite,
        progress_cb=_current_progress_cb(),
    )


@_threaded_tool()
def query_evidence(
    target: str,
    mode: str = "auto",
    run_a: str = "baseline",
    run_b: str | None = None,
    recording: str | None = None,
    channel: str | None = None,
    readers: list[str] | None = None,
    review_set: str | None = None,
    onset_seconds: float | None = None,
    measure: str = "ripple",
    epoch_seconds: float = 10,
    text: str | None = None,
) -> dict[str, Any]:
    """Query focal vs widespread-burst rankings, sensitivity branch diffs, cross-recording concordance, or control windows.

    `mode` options:
    - 'auto': case overview (multi-recording), branch diff (when run_b is set), or recording summary
    - 'recording_summary': single-recording focal/raw tables, synchronous burst windows, QC & spatial status
    - 'compare_runs': symmetric sensitivity comparison between `run_a` and `run_b` with coverage-attributed fate
    - 'compare_recordings': cross-recording concordance without pooling separate recordings
    - 'control_windows': candidate-free background windows for specificity verification
    - 'overlay_concordance': compare focal leaders against externally attached SOZ/Resected clinical overlay
    - 'cohort_summary': inspect saved multi-patient cohort batch tables
    - 'reader_agreement': compare the latest independent reader ratings for one recording and run.
      Supply optional reader names and a frozen review set. Defaults to clinical and research readers.
    - 'evidence_map': per-contact biomarkers, within-recording ranks and descriptive convergence for `run_a`
    - 'epileptogenicity': ictal EI for one recording using `onset_seconds` or a global seizure onset marker
    - 'timeline': admission-relative file boundaries, pieces, gaps and markers. Filter marker labels with `text`.
    - 'trends': quantitative navigation overview for one recording. Choose a band, rms, line_length or hfo_rate
      with `measure`, a 2–60 s `epoch_seconds`, and the HFO source run with `run_a`. Peaks use the original epochs.
    """
    def _impl() -> dict[str, Any]:
        ws = get_workspace_root()
        res = query_case(
            target=target,
            workspace_dir=ws,
            mode=mode,
            run_a=run_a,
            run_b=run_b,
            recording=recording,
            channel=channel,
            readers=readers,
            review_set=review_set,
            onset_seconds=onset_seconds,
            measure=measure,
            epoch_seconds=epoch_seconds,
            text=text,
            progress_cb=_current_progress_cb(),
        )
        if mode in {"timeline", "trends"}:
            # Report actions have absolute submission timestamps. These two projections
            # expose only relative clocks, including dates typed into marker text.
            return res
        c_key = res.get("case") if isinstance(res, dict) else None
        if c_key:
            inbox = WorkbenchStore(ws).load_report_actions(
                case_slug=str(c_key),
                only_pending=True,
                include_claimed=True,
                limit=20,
            )
            if inbox["pending_count"] > 0 or inbox["claimed_count"] > 0:
                res["pending_report_actions"] = inbox["actions"]
                res["pending_action_count"] = inbox["pending_count"]
                res["claimed_action_count"] = inbox["claimed_count"]
                res["latest_action_seq"] = inbox["latest_seq"]
        return res

    result = _wrap_call(_impl)
    if mode in {"timeline", "trends"}:
        return scrub_navigation_dates(result)
    return result


@_threaded_tool()
def render_evidence_card(
    target: str,
    channel: str | None = None,
    recording: str | None = None,
    run_name: str = "baseline",
    event_id: str | None = "top_focal",
    center_seconds: float | None = None,
    window_seconds: float = 0.6,
) -> dict[str, Any]:
    """Render a 3-panel Clinical Triad PNG (Broadband + Adjacent Diff, Bandpass + Envelope, Morlet dB TFR Island).

    `event_id` accepts:
    - 'top_focal': highest-amplitude focal candidate (avoids widespread bursts when focal candidates exist)
    - 'top_burst': representative event inside a widespread multi-shaft synchronous burst
    - 'control': candidate-free background control window on the target channel
    - 'spatial': 2-panel Isometric 3D + Axial Top-Down SEEG shaft projection PNG
    - 'evidence_map': per-contact biomarker heatmap with within-recording percentiles
    - 'epileptogenicity': ictal energy ratios, change detections and normalized EI
    - an explicit event ID (e.g. 'ev_0001'), or pass `center_seconds` directly.
    """
    return _wrap_call(
        _render_evidence_card,
        target=target,
        workspace_dir=get_workspace_root(),
        recording=recording,
        run_name=run_name,
        channel=channel,
        event_id=event_id,
        center_seconds=center_seconds,
        window_seconds=window_seconds,
        return_base64=False,
    )


@_threaded_tool(unlocked_when=lambda kw: kw.get("action") == "wait_action")
def open_workbench(
    action: Literal["focus", "open", "status", "resolve_actions", "wait_action"] = "focus",
    case: str | None = None,
    recording: str | None = None,
    run_name: str | None = None,
    compare_run: str | None = None,
    view: str | None = None,
    channel: str | None = None,
    shaft: str | None = None,
    queue_mode: str | None = None,
    action_ids: list[str] | None = None,
    agent_reply: str | None = None,
    outcome: str = "resolved",
    since_seq: int | None = None,
    wait_timeout: int = 120,
    trigger_only: bool = True,
    claim: bool = True,
    consumer: str = "agent",
    start_seconds: float | None = None,
) -> dict[str, Any]:
    """Start, focus, or sync with the persistent Clinical SEEG Workbench UI and interactive MDT Case Sheet.

    `action`:
    - 'focus': live-navigate open tabs. Supply a recording slug or recording_id and start_seconds to open the Reader.
    - 'open': also launch the URL in the user's desktop browser
    - 'status': return the current UI state and pending_report_actions from the reader, timeline and MDT Case Sheet.
      Questions can precede analysis. Use their saved case, source, role and optional recording, recording_id,
      run_name, channel, start_seconds, end_seconds and marker. All bounds are true seconds within one recording.
      A selected marker carries its label, start_seconds and end_seconds. A case-wide timeline question has no recording.
      When case is masked, pass the action's opaque case_ref as case for status, wait_action and resolve_actions.
    - 'resolve_actions': resolve the specified case and action_ids with an optional agent_reply shown next to the question.
    - 'wait_action': wait for a co-review question or notification for up to wait_timeout seconds (auto-claim with claim=True).
    """
    def _post_live(url: str, token: str | None, payload: dict[str, Any], timeout: float = 4.0) -> dict[str, Any] | None:
        if not token or not is_loopback_http_url(url):
            return None
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-BrainLens-Token": token},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def _impl() -> dict[str, Any]:
        valid_actions = {"focus", "open", "status", "resolve_actions", "wait_action"}
        if action not in valid_actions:
            raise ToolkitError(
                "invalid_action",
                f"Unsupported open_workbench action {action!r}; choose one of {sorted(valid_actions)}.",
            )

        ws = get_workspace_root()
        store = WorkbenchStore(ws, create_dirs=(action != "status"))
        cslug_opt = sanitize_slug(case) if case else None
        if case and action in {"status", "wait_action", "resolve_actions"} and QUESTION_CASE_REF.fullmatch(case):
            cslug_opt = store.case_for_question_ref(case)

        if action == "resolve_actions":
            if not cslug_opt:
                raise ToolkitError("missing_case", "resolve_actions requires explicit `case`.")
            ids = [str(x).strip() for x in (action_ids or []) if str(x).strip()]
            if not ids:
                raise ToolkitError("missing_action_ids", "resolve_actions requires non-empty `action_ids`.")
            # Try notifying the live server so SSE delivers agent_reply immediately; otherwise persist directly.
            meta_file = store.root / "workbench_server.json"
            if meta_file.is_file():
                try:
                    info = json.loads(meta_file.read_text(encoding="utf-8"))
                    live = _post_live(
                        str(info["url"]).rstrip("/") + "/api/report-actions/resolve",
                        info.get("auth_token"),
                        {
                            "case": cslug_opt,
                            "action_ids": ids,
                            "agent_reply": agent_reply or "",
                            "outcome": outcome or "resolved",
                            "resolved_by": consumer or "agent",
                        },
                    )
                    if live is not None:
                        return live
                except Exception:
                    pass
            return store.resolve_report_actions(
                case_slug=cslug_opt,
                action_ids=ids,
                agent_reply=agent_reply or "",
                outcome=outcome or "resolved",
                resolved_by=consumer or "agent",
            )

        if action == "status":
            # Strictly read-only: probe running server metadata without spawning a server or mutating ui_state!
            ui = store.load_ui_state()
            inbox = store.load_report_actions(
                case_slug=cslug_opt,
                only_pending=True,
                include_claimed=True,
                limit=30,
                consumer=consumer or "agent",
            )
            running_url: str | None = None
            running_pid: int | None = None
            meta_file = store.root / "workbench_server.json"
            if meta_file.is_file():
                try:
                    info = json.loads(meta_file.read_text(encoding="utf-8"))
                    base_u = str(info["url"]).rstrip("/")
                    if is_loopback_http_url(base_u):
                        with urllib.request.urlopen(base_u + "/api/health", timeout=0.6) as resp:
                            h = json.loads(resp.read().decode("utf-8"))
                            if h.get("ok") and Path(h.get("workspace", "")).resolve() == store.root:
                                running_url = base_u + "/"
                                running_pid = h.get("pid")
                except Exception:
                    running_url = None
            return {
                "url": running_url,
                "server_running": running_url is not None,
                "pid": running_pid,
                "ui_state": ui,
                "pending_action_count": inbox["pending_count"],
                "claimed_action_count": inbox["claimed_count"],
                "latest_action_seq": inbox["latest_seq"],
                "pending_report_actions": inbox["actions"],
            }

        srv_info = ensure_workbench_server(workspace_dir=ws)
        base_url = str(srv_info["url"]).rstrip("/") + "/"
        tok = srv_info.get("auth_token")

        if action == "wait_action":
            cur_meta = store.load_report_actions(
                case_slug=cslug_opt,
                only_pending=True,
                include_claimed=False,
                limit=1,
                consumer=consumer or "agent",
            )
            eff_since = int(since_seq) if since_seq is not None else int(cur_meta.get("consumer_cursor", 0))
            wparams = {
                "since": str(eff_since),
                "timeout": str(max(1, min(300, int(wait_timeout)))),
                "trigger_only": "1" if trigger_only else "0",
                "claim": "1" if claim else "0",
                "consumer": consumer or "agent",
            }
            if cslug_opt:
                wparams["case"] = cslug_opt
            wait_url = base_url + "api/report-actions/wait?" + urllib.parse.urlencode(wparams)
            headers = {"X-BrainLens-Token": tok} if tok else {}
            req = urllib.request.Request(wait_url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=max(5, min(310, int(wait_timeout) + 5))) as resp:
                return json.loads(resp.read().decode("utf-8"))

        patch: dict[str, Any] = {}
        # action='status' is strictly read-only — NEVER navigate/mutate ui_state when checking status!
        if action in {"focus", "open"}:
            if case:
                cslug = sanitize_slug(case)
                patch["case"] = cslug
                if not recording and store.has_case(cslug):
                    recs = store.list_recordings(cslug)
                    if recs:
                        patch["recording"] = recs[0]
            if recording:
                selected_case = cslug_opt or store.load_ui_state().get("case")
                patch["recording"] = navigation_recording_slug(store, selected_case, recording) if selected_case else sanitize_slug(recording)
            if run_name:
                patch["run_name"] = sanitize_slug(run_name)
            if compare_run is not None:
                patch["compare_run"] = sanitize_slug(compare_run) if compare_run else None
            if view in {"overview", "reader", "sheet"}:
                patch["view"] = view
            if channel is not None:
                patch["channel"] = channel
            if start_seconds is not None:
                if not math.isfinite(start_seconds) or start_seconds < 0:
                    raise ToolkitError("invalid_start", "Reader start_seconds must be finite and nonnegative.")
                patch.update(start_seconds=start_seconds, view=patch.get("view", "reader"), event_id=None, shaft=None)
            if shaft is not None:
                patch["shaft"] = shaft
            if queue_mode in {"focal", "ehfo_fr", "bursts", "controls", "confirmed", "all"}:
                patch["queue_mode"] = queue_mode

        if patch:
            author = "agent_focus_time" if start_seconds is not None else "agent"
            live_ui = _post_live(base_url + "api/ui-state", tok, {**patch, "updated_by": author})
            ui = live_ui if live_ui is not None else store.update_ui_state(patch, updated_by=author)
        else:
            ui = store.load_ui_state()
        inbox = store.load_report_actions(
            case_slug=cslug_opt,
            only_pending=True,
            include_claimed=True,
            limit=30,
            consumer=consumer or "agent",
        )
        qparams = {
            k: v
            for k, v in {
                "case": ui.get("case"),
                "recording": ui.get("recording"),
                "run": ui.get("run_name"),
                "view": ui.get("view"),
                "shaft": shaft,
                "channel": ui.get("channel"),
                "qmode": ui.get("queue_mode"),
                "start_seconds": start_seconds,
            }.items()
            if v or (k == "start_seconds" and v == 0)
        }
        deep_url = base_url + ("?" + urllib.parse.urlencode(qparams) if qparams else "")

        browser_opened = False
        if action == "open" and get_env("BRAINLENS_NO_BROWSER", "BRAINTOOLKIT_NO_BROWSER") != "1":
            try:
                browser_opened = bool(webbrowser.open(deep_url))
            except Exception:
                browser_opened = False

        return {
            "url": base_url,
            "deep_link": deep_url,
            "server_reused": srv_info["reused"],
            "pid": srv_info["pid"],
            "browser_opened": browser_opened,
            "ui_state": ui,
            "pending_action_count": inbox["pending_count"],
            "claimed_action_count": inbox["claimed_count"],
            "latest_action_seq": inbox["latest_seq"],
            "pending_report_actions": inbox["actions"],
        }

    return _wrap_call(_impl)


@_threaded_tool()
def record_judgment(
    case: str,
    recording: str | None = None,
    run_name: str = "baseline",
    channel: str | None = None,
    event_id: str | None = None,
    verdict: str | None = None,
    clipped: bool | None = None,
    note: str | None = None,
    clinical_question: str | None = None,
    clinical_overlay: dict[str, Any] | None = None,
    author: str = "agent",
    role: Literal["agent", "researcher", "engineering_rehearsal", "clinician"] = "agent",
    operation_id: str | None = None,
    anatomy_labels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a clinician or agent verification judgment ('confirm' | 'artifact' | 'uncertain'), clip, case question, or external SOZ/Resected clinical overlay.

    Deduplicates repeated entries and updates run tallies immediately without rerunning signal processing.
    When `clinical_overlay` is supplied (`{source, soz_channels, resected_channels, irritative_channels, notes}`),
    it requires an explicit provenance `source` and returns objective concordance metrics.
    Supply `anatomy_labels` with an external source, a patient_native boolean, and contact/channel
    labels containing regions to replace the case's anatomical annotations.
    """
    def _impl() -> dict[str, Any]:
        store = WorkbenchStore(get_workspace_root())
        cslug = sanitize_slug(case)
        if not store.has_case(cslug):
            raise ToolkitError("case_not_found", f"Case {case!r} is not registered.")
        if verdict is not None and verdict not in {"confirm", "artifact", "uncertain"}:
            raise ToolkitError("invalid_verdict", "verdict must be 'confirm', 'artifact', or 'uncertain'.")
        if role not in {"agent", "researcher", "engineering_rehearsal", "clinician"}:
            raise ToolkitError(
                "invalid_role",
                f"role must be one of 'agent', 'researcher', 'engineering_rehearsal', or 'clinician'; got {role!r}.",
            )

        # Validate ALL composite targets upfront BEFORE mutating any case file (all-or-nothing atomicity!)
        if clinical_overlay is not None:
            validate_clinical_overlay(clinical_overlay)
        if anatomy_labels is not None:
            validate_anatomy_labels(anatomy_labels)
        if clinical_question is not None and not isinstance(clinical_question, str):
            raise ToolkitError("invalid_input", "clinical_question must be a string.")
        if note is not None and not isinstance(note, str):
            raise ToolkitError("invalid_input", "note must be a string.")
        if clipped is not None and not isinstance(clipped, bool):
            raise ToolkitError("invalid_input", "clipped must be a boolean.")

        clin_q = clinical_question.strip() if clinical_question and clinical_question.strip() else None
        has_judgment_write = any(x is not None for x in (event_id, verdict, clipped, note, channel, clin_q))
        rec_slug = sanitize_slug(recording) if recording else None
        rslug = sanitize_slug(run_name)
        op_id = str(operation_id or "").strip() or None

        if has_judgment_write and event_id:
            if not rec_slug:
                raise ToolkitError("missing_recording", f"Event judgment {event_id!r} requires an explicit `recording`.")
            if not store.has_run(cslug, rec_slug, rslug):
                raise ToolkitError("run_not_found", f"Run {cslug}/{rec_slug}/{rslug} does not exist.")
            bundle = store.load_run(cslug, rec_slug, rslug)
            ev_match = next((e for e in bundle.get("events", []) if e.get("id") == event_id), None)
            if ev_match is None:
                raise ToolkitError(
                    "event_not_in_run",
                    f"Event {event_id!r} does not belong to {cslug}/{rec_slug}/{rslug}.",
                )
            if channel and ev_match.get("channel_name") and channel != ev_match.get("channel_name"):
                raise ToolkitError(
                    "event_channel_mismatch",
                    f"Event {event_id!r} belongs to channel {ev_match['channel_name']!r}, not {channel!r}.",
                )

        entry = None
        created_new = True
        warnings_list: list[str] = []
        if has_judgment_write:
            entry, created_new = store.record_judgment_ex(
                cslug,
                {
                    "kind": "event_verdict" if event_id else ("clinical_question" if clin_q else "case_note"),
                    "operation_id": op_id,
                    "recording": rec_slug,
                    "run_name": rslug,
                    "channel": channel,
                    "event_id": event_id,
                    "verdict": verdict,
                    "clipped": clipped,
                    "clinical_question": clin_q,
                    "author": author,
                    "role": role,
                    "note": note or clin_q or "",
                },
                validate_event_ownership=True,
            )
            if created_new and rec_slug and store.has_run(cslug, rec_slug, rslug):
                try:
                    refresh_verdict_tallies(store, cslug, rec_slug, rslug)
                except Exception as exc:
                    warnings_list.append(f"Judgment saved to WAL; deferred run tally refresh ({exc}).")

        if clin_q:
            store.add_open_question(cslug, clin_q)

        overlay_saved = None
        overlay_conc = None
        if clinical_overlay is not None:
            # Always route through save_clinical_overlay(operation_id=op_id): its applied_operation_ids guard
            # heals partial failures on retry while guaranteeing a stale replay never overwrites a newer overlay.
            overlay_saved = store.save_clinical_overlay(
                cslug,
                {**clinical_overlay, "author": author},
                operation_id=op_id,
            )
            recs = store.list_recordings(cslug)
            rec_eff = rec_slug or (recs[0] if recs else None)
            if rec_eff and store.has_run(cslug, rec_eff, rslug):
                overlay_conc = compute_overlay_concordance(store, cslug, rec_slug=rec_eff, run_name=rslug)

        out: dict[str, Any] = {
            "case": cslug,
            "saved_judgment": entry,
            "total_judgments": len(store.load_judgments(cslug)),
        }
        if warnings_list:
            out["warnings"] = warnings_list
        if anatomy_labels is not None:
            out["anatomy_labels"] = save_anatomy_labels(store, cslug, anatomy_labels, author=author, operation_id=op_id)
        if overlay_saved is not None:
            out["clinical_overlay"] = overlay_saved
            out["overlay_concordance"] = overlay_conc
        return out

    return _wrap_call(_impl)


@_threaded_tool()
def export_or_protocol(
    action: Literal["export_case_sheet", "export_mdt_slides", "export_bids_events", "export_edf_annotations", "save_protocol", "list_protocols", "run_cohort", "create_review_set", "export_agreement_report"] = "export_case_sheet",
    case: str | None = None,
    recording: str | None = None,
    primary_run: str = "baseline",
    comparison_run: str | None = None,
    clinical_question: str | None = None,
    protocol_name: str | None = None,
    description: str = "",
    only_confirmed: bool = False,
    exclude_widespread_bursts: bool = True,
    targets: list[str] | str | None = None,
    cohort_name: str | None = None,
    overrides: dict[str, Any] | None = None,
    deidentify: bool = False,
    pseudonym: str | None = None,
    review_set: str | None = None,
    sample_size: int | None = None,
    seed: int | None = None,
    readers: list[str] | None = None,
    max_cards: int = 6,
) -> dict[str, Any]:
    """Export an MDT case sheet or presentation, BIDS events, or EEG reader annotations, manage protocols, or run a cohort batch.

    `action`:
    - 'export_case_sheet': export self-contained MDT HTML sheet with focal vs burst tables, ONNX metrics, 3D spatial figure & Triad cards
    - 'export_mdt_slides': export a PowerPoint deck for the primary run with up to max_cards Clinical Triad figures.
    - 'export_bids_events': export BIDS-compliant *_events.tsv + *_events.json sidecar
    - 'export_edf_annotations': export annotation CSV and provenance JSON for one recording/run, plus an
      annotated EDF+ copy for EDF sources. Set exclude_widespread_bursts=False to include widespread bursts.
    - 'save_protocol': extract reusable analysis parameters (stripping patient-specific exclusions) to protocols/<slug>.json
    - 'list_protocols': list all saved protocols and cohorts in the workspace
    - 'run_cohort': apply `protocol_name` (or baseline) across `targets` (list or folder) and emit cohort_summary.csv + cohort_channels.csv
    - 'create_review_set': freeze all events or a stratified sample for one recording. Supply review_set as the name.
    - 'export_agreement_report': write agreement statistics and provenance to JSON and event ratings and disagreements to CSV.

    Export files retain clinical identity by default. Set deidentify=True to share a case sheet, slide deck, BIDS events,
    or annotated EDF/CSV under an optional pseudonym. MCP responses always mask native header identity.
    """
    def _impl() -> dict[str, Any]:
        ws = get_workspace_root()
        if action == "list_protocols":
            return {"workspace": str(ws), "protocols": list_protocols(ws), "cohorts": list_cohorts(ws)}
        if action == "run_cohort":
            return run_cohort(
                targets=targets,
                protocol=protocol_name,
                cohort_name=cohort_name,
                run_name=primary_run if primary_run != "baseline" or not protocol_name else None,
                overrides=overrides,
                workspace_dir=ws,
                progress_cb=_current_progress_cb(),
            )
        if not case:
            raise ToolkitError("missing_case", f"Provide a case for {action}.")
        if action in {"create_review_set", "export_agreement_report"}:
            from .agreement import export_agreement_report
            from .review_sets import create_review_set

            if not recording:
                raise ToolkitError("missing_recording", "Choose one recording for this action.")
            store = WorkbenchStore(ws)
            if action == "export_agreement_report":
                return export_agreement_report(store, case, recording, primary_run, readers, review_set)
            if not review_set:
                raise ToolkitError("missing_review_set", "Provide a name for the frozen review set.")
            return create_review_set(store, case, recording, primary_run, review_set, sample_size, seed)
        if action == "save_protocol":
            if not protocol_name:
                raise ToolkitError("missing_protocol_name", "Provide `protocol_name` when saving a protocol.")
            return save_protocol(
                name=protocol_name,
                case_slug=sanitize_slug(case),
                recording=recording,
                run_name=primary_run,
                description=description,
                workspace_dir=ws,
            )
        if action in {"export_bids_events", "export_edf_annotations"}:
            exporter = export_bids_events if action == "export_bids_events" else export_edf_annotations
            return exporter(
                case_slug=sanitize_slug(case),
                recording=recording,
                run_name=primary_run,
                only_confirmed=only_confirmed,
                exclude_widespread_bursts=exclude_widespread_bursts,
                deidentify=deidentify,
                pseudonym=pseudonym,
                workspace_dir=ws,
            )
        if action in {"export_case_sheet", "export_mdt_slides"}:
            exporter = export_case_sheet
            options = {}
            if action == "export_mdt_slides":
                from .mdt_slides import export_mdt_slides

                exporter = export_mdt_slides
                options["max_cards"] = max_cards
            return exporter(
                case_slug=sanitize_slug(case),
                workspace_dir=ws,
                primary_run=primary_run,
                comparison_run=comparison_run,
                clinical_question=clinical_question,
                deidentify=deidentify,
                pseudonym=pseudonym,
                **options,
            )
        raise ToolkitError(
            "invalid_action",
            f"Unsupported export or protocol action {action!r}.",
        )

    return _wrap_call(_impl)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BrainLens Neurophysiology Workbench MCP Server (stdio)")
    parser.add_argument(
        "--workspace",
        default=str(default_workspace_dir()),
        help="Path to BrainLens workbench results workspace",
    )
    args = parser.parse_args(argv)
    set_workspace_root(args.workspace)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
