"""Persistent, non-expiring HTTP server for the BrainLens Neurophysiology Workbench."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import secrets
import shutil
import sys
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .agreement import reader_agreement
from .analytics import annotate_summary_anatomy, compare_recordings, compare_runs, compute_overlay_concordance, select_control_windows
from .anatomy import AnatomyMatcher, clear_anatomy_labels, load_anatomy_labels, recording_anatomy_coverage, save_anatomy_labels
from .bids_export import export_bids_events
from .blind_review import blind_event, blind_signal_payload, reader_projection, without_shared_review
from .case_sheet import SHEET_DISPLAY_META, export_case_sheet
from .channel_order import ChannelDisplayOrder
from .cohort import run_cohort
from .deid import SHEET_DEIDENTIFICATION_META
from .edf_export import export_edf_annotations
from .io_readers import discover_available_inputs, inspect_case
from .jobs import JobManager
from .models import ToolkitError, atomic_write_json, canonical_digest, get_verified_sheet_entry, read_json, verify_sheet_trust
from .pipeline import analyze_case, refresh_verdict_tallies
from .protocols import list_protocols, save_protocol
from .pyhfo_runner import check_engine_health
from .review_sets import create_review_set, list_review_sets, load_review_set, reader_event_order
from .storage import WorkbenchStore, run_analysis_signature, sanitize_slug
from .storage_actions import NAVIGATION_QUESTION_SOURCES
from .timeline import admission_timeline, segments_from_markers
from .triad_views import _verify_run_recording_binding, compute_multichannel_batch, compute_triad_view

ASSETS_DIR = Path(__file__).parent / "assets"
MAX_JSON_BODY = 512 * 1024
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


def _export_privacy_options(body: dict[str, Any]) -> dict[str, Any]:
    deidentify = body.get("deidentify", False)
    pseudonym = body.get("pseudonym")
    if not isinstance(deidentify, bool) or (pseudonym is not None and not isinstance(pseudonym, str)):
        raise ToolkitError("invalid_export_options", "Use a boolean for deidentify and text for the optional pseudonym.")
    return {"deidentify": deidentify, "pseudonym": pseudonym}


class _ActionTailIndex:
    """Lock-free-outside-wait incremental tail cache for report_actions.jsonl + state.

    Safe across atomic file replacement (inode/size check) and mid-append partial lines
    (never advances offset past an unterminated trailing line).
    """

    def __init__(self, store: WorkbenchStore) -> None:
        self.store = store
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] = []
        self._offset: int = 0
        self._inode: int | None = None
        self._jsonl_sig: tuple[int, int, int] | None = None
        self._state: dict[str, Any] = {"actions": {}, "consumer_cursors": {}, "legacy_consumed_seq": 0}
        self._state_sig: tuple[int, int, int] | None = None

    def snapshot(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        apath = self.store.report_actions_path()
        spath = self.store.report_actions_state_path()
        with self._lock:
            if apath.is_file():
                st = apath.stat()
                sig = (st.st_ino, st.st_mtime_ns, st.st_size)
                if sig != self._jsonl_sig:
                    if self._inode is not None and st.st_ino != self._inode:
                        self._rows = []
                        self._offset = 0
                    elif st.st_size < self._offset:
                        self._rows = []
                        self._offset = 0
                    self._inode = st.st_ino
                    with apath.open("rb") as fh:
                        fh.seek(self._offset)
                        chunk = fh.read()
                    # Only consume complete newline-terminated lines so mid-append writes are NEVER skipped!
                    last_nl = chunk.rfind(b"\n")
                    if last_nl >= 0:
                        complete = chunk[: last_nl + 1]
                        self._offset += len(complete)
                        for raw_line in complete.decode("utf-8", errors="replace").splitlines():
                            line = raw_line.strip()
                            if not line:
                                continue
                            try:
                                self._rows.append(json.loads(line))
                            except Exception:
                                continue
                    if last_nl == len(chunk) - 1 or not chunk:
                        self._jsonl_sig = sig
                    else:
                        self._jsonl_sig = None
            else:
                self._rows = []
                self._offset = 0
                self._inode = None
                self._jsonl_sig = None

            if spath.is_file():
                st2 = spath.stat()
                sig2 = (st2.st_ino, st2.st_mtime_ns, st2.st_size)
                if sig2 != self._state_sig:
                    self._state = self.store._load_action_state()
                    self._state_sig = sig2
            elif self._state_sig is not None or self.store.report_actions_cursor_path().is_file():
                self._state = self.store._load_action_state()
                self._state_sig = None

            return list(self._rows), dict(self._state)

    def query(
        self,
        *,
        case_slug: str | None = None,
        since_seq: int = 0,
        only_pending: bool = False,
        include_claimed: bool = True,
        trigger_only: bool = False,
        limit: int = 100,
        consumer: str = "default",
    ) -> dict[str, Any]:
        rows, state = self.snapshot()
        return self.store.project_report_actions(
            rows,
            state,
            case_slug=case_slug,
            since_seq=since_seq,
            only_pending=only_pending,
            include_claimed=include_claimed,
            trigger_only=trigger_only,
            limit=limit,
            consumer=consumer,
        )


class _SSEHub:
    """In-memory ring buffer + epoch-tagged condition broadcaster for /api/events SSE clients."""

    def __init__(self, maxlen: int = 250) -> None:
        self._cond = threading.Condition()
        self.epoch = secrets.token_hex(4)
        self._seq = 0
        self._ring: deque[tuple[int, str, dict[str, Any]]] = deque(maxlen=maxlen)

    def publish(self, ev_type: str, payload: dict[str, Any]) -> int:
        with self._cond:
            self._seq += 1
            self._ring.append((self._seq, ev_type, payload))
            self._cond.notify_all()
            return self._seq

    def wake_all(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def check_resume(self, last_raw: str) -> tuple[int, str | None]:
        """Parse Last-Event-ID ('epoch:seq' or legacy int) and detect server restart or ring overflow."""
        with self._cond:
            raw = (last_raw or "").strip()
            if not raw:
                return self._seq, None
            if ":" in raw:
                ep, seq_s = raw.split(":", 1)
                if ep != self.epoch:
                    return self._seq, "server_restarted"
                try:
                    last_id = int(seq_s)
                except ValueError:
                    return self._seq, "invalid_cursor"
            else:
                try:
                    last_id = int(raw)
                except ValueError:
                    return self._seq, "invalid_cursor"
                if last_id > self._seq:
                    return self._seq, "server_restarted"
            if last_id > self._seq:
                return self._seq, "server_restarted"
            if self._ring and last_id < self._ring[0][0] - 1:
                return self._seq, "ring_overflow"
            return last_id, None

    def next_events(self, last_id: int, timeout: float = 15.0) -> tuple[list[tuple[int, str, dict[str, Any]]], bool]:
        with self._cond:
            if self._ring and last_id < self._ring[0][0] - 1:
                return [], True
            hits = [item for item in self._ring if item[0] > last_id]
            if hits:
                return hits, False
            self._cond.wait(timeout=timeout)
            if self._ring and last_id < self._ring[0][0] - 1:
                return [], True
            return [item for item in self._ring if item[0] > last_id], False

    def current_seq(self) -> int:
        with self._cond:
            return self._seq


class WorkbenchHTTPServer:
    def __init__(self, workspace_dir: str | Path | None = None, host: str = "127.0.0.1", port: int = 0):
        self.store = WorkbenchStore(workspace_dir)
        self.jobs = JobManager()
        self._evidence_jobs: dict[tuple, str] = {}
        self._evidence_job_lock = threading.Lock()
        self.host = host
        self.requested_port = port
        self.auth_token = secrets.token_urlsafe(24)
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self._watch_thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._lock = threading.RLock()
        self._action_cond = threading.Condition()
        self._action_index = _ActionTailIndex(self.store)
        self._sse = _SSEHub()
        self._last_sse_ui_seq: int = -1
        self._last_sse_judg_sigs: dict[str, tuple[int, int]] = {}
        self._last_sse_act_sigs: tuple[tuple[int, int] | None, tuple[int, int] | None] = (None, None)

    @property
    def port(self) -> int:
        assert self.httpd is not None
        return int(self.httpd.server_port)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def _allowed_origins(self) -> set[str]:
        p = self.port
        return {f"http://127.0.0.1:{p}", f"http://localhost:{p}", f"http://[::1]:{p}"}

    def _is_verified_sheet(self, case_slug: str, fpath: Path, raw_bytes: bytes | None = None) -> bool:
        try:
            data = raw_bytes if raw_bytes is not None else fpath.read_bytes()
            return verify_sheet_trust(self.store.workspace_id, sanitize_slug(case_slug), fpath.name, data)
        except Exception:
            return False

    def _check_verified_sheet_state(
        self,
        case_slug: str,
        fpath: Path,
        raw_bytes: bytes | None = None,
        *,
        active_run: str | None = None,
        check_judgments: bool = True,
    ) -> str:
        """Classify a sheet for display, including older exports that require regeneration."""
        try:
            data = raw_bytes if raw_bytes is not None else fpath.read_bytes()
            head = data[:4096]
            entry = get_verified_sheet_entry(self.store.workspace_id, sanitize_slug(case_slug), fpath.name, data)
            if not isinstance(entry, dict):
                return "unverified"
            # Direct downloads serve the signed, static sharing snapshot. A request
            # to export a specific active run still checks freshness after rendering.
            if active_run is None and SHEET_DEIDENTIFICATION_META.encode("ascii") in head:
                return "shared"
            # Rebuild the embedded report from the saved run when the old layout
            # would still show ambiguous model scores and unused reviewer roles.
            if active_run is not None and SHEET_DISPLAY_META.encode("ascii") not in head:
                return "stale"
            if b'<meta name="brainlens-deidentification" content="header-subject-v1"' in head:
                # Rebuild cached sheets from the former always-masked local renderer.
                return "stale"
            saved_sig = entry.get("run_signature")
            if not isinstance(saved_sig, dict) or not saved_sig.get("primary"):
                return "stale"
            eff_run = active_run or entry.get("primary_run") or saved_sig.get("primary_run")
            if not eff_run:
                return "stale"
            ref_runs = list((saved_sig.get("referenced_runs") or {}).keys())
            cur_sig = self.store.case_sheet_run_signature(
                case_slug,
                str(eff_run),
                comparison_run=saved_sig.get("comparison_run"),
                referenced_runs=ref_runs,
            )
            # Reject sheet as stale if ANY recording in primary, comparison, or referenced runs was replaced/re-inspected!
            all_cur_fps = self.store.case_recording_fingerprints(case_slug)
            sig_run_maps: list[dict[str, Any]] = [
                saved_sig.get("primary") or {},
                saved_sig.get("comparison") or {},
            ]
            for ref_m in (saved_sig.get("referenced_runs") or {}).values():
                if isinstance(ref_m, dict):
                    sig_run_maps.append(ref_m)
            for r_map in sig_run_maps:
                for r_key, r_entry in r_map.items():
                    run_fp = (r_entry or {}).get("recording_fingerprint") if isinstance(r_entry, dict) else None
                    cur_fp = all_cur_fps.get(r_key)
                    if run_fp and cur_fp and run_fp != cur_fp:
                        return "stale"
            # A sheet predating external anatomy labels rendered the absent overlay.
            saved_sig = {"anatomy_labels_digest": canonical_digest(None), **saved_sig}
            if check_judgments:
                return "fresh" if saved_sig == cur_sig else "stale"
            analysis_keys = (
                "primary_run",
                "primary",
                "comparison_run",
                "comparison",
                "referenced_runs",
                "clinical_overlay_digest",
                "anatomy_labels_digest",
                "clip_signature",
                "open_questions_digest",
                "clinical_notes_digest",
                "recording_spatial_digest",
                "recording_fingerprints",
                "evidence_map_digests",
            )
            if any(
                k not in saved_sig
                for k in (
                    "referenced_runs",
                    "clinical_overlay_digest",
                    "clip_signature",
                    "open_questions_digest",
                    "clinical_notes_digest",
                    "recording_spatial_digest",
                    "recording_fingerprints",
                )
            ):
                return "stale"
            return "fresh" if all(saved_sig.get(k) == cur_sig.get(k) for k in analysis_keys) else "stale"
        except Exception:
            return "unverified"

    def _is_verified_fresh_sheet(
        self,
        case_slug: str,
        fpath: Path,
        active_run: str,
        *,
        check_judgments: bool = True,
    ) -> bool:
        """Verify machine-local HMAC trust AND exact match to active, comparison, & referenced run generations."""
        return (
            self._check_verified_sheet_state(
                case_slug,
                fpath,
                active_run=active_run,
                check_judgments=check_judgments,
            )
            == "fresh"
        )

    def _inject_token_html(self, raw: bytes) -> bytes:
        meta_tag = (
            f'<meta name="brainlens-token" content="{self.auth_token}" />'
            f'<meta name="brainlens-workspace-id" content="{self.store.workspace_id}" />'
        )
        txt = raw.decode("utf-8", errors="replace")
        if "</head>" in txt:
            txt = txt.replace("</head>", f"  {meta_tag}\n</head>", 1)
        else:
            txt = meta_tag + "\n" + txt
        return txt.encode("utf-8")

    @staticmethod
    def _inject_stale_meta_html(raw: bytes) -> bytes:
        meta_tag = '<meta name="brainlens-sheet-stale" content="1" />'
        txt = raw.decode("utf-8", errors="replace")
        if "</head>" in txt:
            txt = txt.replace("</head>", f"  {meta_tag}\n</head>", 1)
        else:
            txt = meta_tag + "\n" + txt
        return txt.encode("utf-8")

    @staticmethod
    def _file_sig(p: Path) -> tuple[int, int] | None:
        try:
            st = p.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def _scan_judgment_sigs(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        croot = self.store.root / "cases"
        if not croot.is_dir():
            return out
        for d in croot.iterdir():
            if not d.is_dir():
                continue
            sig = self._file_sig(d / "judgments.jsonl")
            if sig is not None:
                out[d.name] = sig
        return out

    def _scan_case_evidence_sigs(self) -> dict[str, tuple[Any, ...]]:
        """Track committed run generations & clinical overlays per case for immediate SSE stale-sheet invalidation."""
        out: dict[str, tuple[Any, ...]] = {}
        croot = self.store.root / "cases"
        if not croot.is_dir():
            return out
        for cdir in croot.iterdir():
            if not cdir.is_dir():
                continue
            items: list[tuple[str, Any]] = []
            c_sig = self._file_sig(cdir / "case.json")
            if c_sig is not None:
                items.append(("case", c_sig))
            ov_sig = self._file_sig(cdir / "clinical_overlay.json")
            if ov_sig is not None:
                items.append(("overlay", ov_sig))
            anatomy_sig = self._file_sig(cdir / "anatomy_labels.json")
            if anatomy_sig is not None:
                items.append(("anatomy", anatomy_sig))
            rroot = cdir / "recordings"
            if rroot.is_dir():
                for rec_d in sorted(rroot.iterdir()):
                    rec_sig = self._file_sig(rec_d / "recording.json")
                    if rec_sig is not None:
                        items.append((f"rec:{rec_d.name}", rec_sig))
                    runs_d = rec_d / "runs"
                    if not runs_d.is_dir():
                        continue
                    for rd in sorted(runs_d.iterdir()):
                        if not rd.is_dir() or rd.name.startswith("."):
                            continue
                        cur_l = rd / "current"
                        if cur_l.is_symlink():
                            try:
                                items.append((f"{rec_d.name}/{rd.name}", os.readlink(cur_l)))
                                continue
                            except OSError:
                                pass
                        csig = self._file_sig(rd / "CURRENT.json") or self._file_sig(rd / "run_meta.json")
                        if csig is not None:
                            items.append((f"{rec_d.name}/{rd.name}", csig))
            out[cdir.name] = tuple(items)
        return out

    def _mark_act_sigs_published(self) -> None:
        self._last_sse_act_sigs = (
            self._file_sig(self.store.report_actions_path()),
            self._file_sig(self.store.report_actions_state_path()),
        )

    def _mark_judgment_published(self, case_slug: str) -> None:
        sig = self._file_sig(self.store.judgments_path(case_slug))
        if sig is not None:
            self._last_sse_judg_sigs[sanitize_slug(case_slug)] = sig

    def _watch_external_files(self) -> None:
        """Detect out-of-process writes (ui_state, report_actions, judgments, runs, overlays) and push SSE wakeups."""
        ui_sig = self._file_sig(self.store.ui_state_path())
        act_sig = self._file_sig(self.store.report_actions_path())
        st_sig = self._file_sig(self.store.report_actions_state_path())
        judg_sigs = self._scan_judgment_sigs()
        ev_sigs = self._scan_case_evidence_sigs()
        self._last_sse_judg_sigs = dict(judg_sigs)
        self._last_sse_act_sigs = (act_sig, st_sig)
        while not self._stop_evt.wait(0.6):
            try:
                n_ui = self._file_sig(self.store.ui_state_path())
                if n_ui != ui_sig:
                    ui_sig = n_ui
                    ui = self.store.load_ui_state()
                    seq_val = int(ui.get("seq", 0))
                    if seq_val != self._last_sse_ui_seq:
                        self._last_sse_ui_seq = seq_val
                        acts = self._action_index.query(only_pending=True, limit=1)
                        self._sse.publish("ui_state", {
                            **ui,
                            "pending_action_count": acts["pending_count"],
                            "claimed_action_count": acts["claimed_count"],
                            "latest_action_seq": acts["latest_seq"],
                        })
                n_act = self._file_sig(self.store.report_actions_path())
                n_st = self._file_sig(self.store.report_actions_state_path())
                if (n_act, n_st) != (act_sig, st_sig):
                    act_sig, st_sig = n_act, n_st
                    if (n_act, n_st) != self._last_sse_act_sigs:
                        self._last_sse_act_sigs = (n_act, n_st)
                        with self._action_cond:
                            self._action_cond.notify_all()
                        acts = self._action_index.query(only_pending=True, limit=1)
                        self._sse.publish("report_actions", {
                            "latest_seq": acts["latest_seq"],
                            "pending_count": acts["pending_count"],
                            "claimed_count": acts["claimed_count"],
                            "global_pending_count": acts["pending_count"],
                        })
                n_judg = self._scan_judgment_sigs()
                if n_judg != judg_sigs:
                    for cslug, sig in n_judg.items():
                        if judg_sigs.get(cslug) != sig and self._last_sse_judg_sigs.get(cslug) != sig:
                            self._last_sse_judg_sigs[cslug] = sig
                            self._sse.publish("verdict", {"case": cslug, "source": "external"})
                    judg_sigs = n_judg
                n_ev = self._scan_case_evidence_sigs()
                if n_ev != ev_sigs:
                    changed_cases = {c for c in set(n_ev) | set(ev_sigs) if n_ev.get(c) != ev_sigs.get(c)}
                    ev_sigs = n_ev
                    for cslug in sorted(changed_cases):
                        self._sse.publish("case_updated", {"case": cslug, "source": "external"})
            except Exception:
                continue

    def start(self) -> str:
        with self._lock:
            if self.httpd is not None:
                return self.url
            srv = self

            class Handler(BaseHTTPRequestHandler):
                protocol_version = "HTTP/1.1"

                def log_message(self, *_args: Any) -> None:
                    return

                def _check_host(self) -> bool:
                    raw_host = (self.headers.get("Host") or "").strip().lower()
                    if not raw_host:
                        return False
                    if raw_host.startswith("["):
                        if "]" not in raw_host:
                            return False
                        host_only, rest = raw_host.split("]", 1)
                        host_only += "]"
                        port_part = rest[1:] if rest.startswith(":") else ""
                    elif ":" in raw_host:
                        host_only, port_part = raw_host.rsplit(":", 1)
                    else:
                        host_only, port_part = raw_host, ""
                    if host_only not in ALLOWED_HOSTS:
                        return False
                    if port_part and port_part != str(srv.port):
                        return False
                    return True

                def _write_cors_headers(self) -> None:
                    origin = (self.headers.get("Origin") or "").strip()
                    if origin and origin in srv._allowed_origins():
                        self.send_header("Access-Control-Allow-Origin", origin)
                        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
                        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-BrainLens-Token, X-BTK-Token")
                        self.send_header("Vary", "Origin")

                def do_OPTIONS(self) -> None:
                    if not self._check_host():
                        self._json(403, {"error": {"code": "forbidden_host", "message": "Disallowed Host header."}})
                        return
                    self.send_response(204)
                    self._write_cors_headers()
                    self.send_header("Content-Length", "0")
                    self.send_header("Connection", "close")
                    self.end_headers()

                def _verify_write_security(self) -> None:
                    if not self._check_host():
                        raise ToolkitError("forbidden_host", "Disallowed Host header.")
                    origin = (self.headers.get("Origin") or "").strip()
                    if origin and origin not in srv._allowed_origins():
                        raise ToolkitError("forbidden_origin", f"Cross-origin write rejected ({origin}).")
                    ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
                    if ctype != "application/json":
                        raise ToolkitError("invalid_content_type", "Write requests require Content-Type: application/json.")
                    tok = (self.headers.get("X-BrainLens-Token") or self.headers.get("X-BTK-Token") or "").strip()
                    if not tok or not secrets.compare_digest(tok, srv.auth_token):
                        raise ToolkitError("unauthorized", "Missing or invalid X-BrainLens-Token session header.")

                def _send_headers(
                    self,
                    code: int,
                    length: int,
                    ctype: str,
                    extra_headers: dict[str, str] | None = None,
                ) -> None:
                    self.send_response(code)
                    self._write_cors_headers()
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(length))
                    self.send_header("Cache-Control", "no-store")
                    if extra_headers:
                        for hk, hv in extra_headers.items():
                            self.send_header(hk, hv)
                    self.send_header("Connection", "close")
                    self.end_headers()

                def _send(
                    self,
                    code: int,
                    body: bytes,
                    ctype: str,
                    extra_headers: dict[str, str] | None = None,
                ) -> None:
                    self._send_headers(code, len(body), ctype, extra_headers)
                    try:
                        self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

                def _json(self, code: int, data: Any) -> None:
                    self._send(
                        code,
                        json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                        "application/json; charset=utf-8",
                    )

                def _body(self) -> dict[str, Any]:
                    n = int(self.headers.get("Content-Length") or "0")
                    if n <= 0:
                        return {}
                    if n > MAX_JSON_BODY:
                        raise ToolkitError("payload_too_large", "JSON body exceeds limit.")
                    obj = json.loads(self.rfile.read(n).decode("utf-8"))
                    if not isinstance(obj, dict):
                        raise ToolkitError("invalid_json", "Expected JSON object.")
                    return obj

                def _handle_sse(self) -> None:
                    last_raw = self.headers.get("Last-Event-ID") or ""
                    last_id, resync_reason = srv._sse.check_resume(last_raw)
                    ep = srv._sse.epoch
                    self.send_response(200)
                    self._write_cors_headers()
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    try:
                        self.wfile.write(b"retry: 2000\n\n")
                        if resync_reason:
                            frame = (
                                f"id: {ep}:{last_id}\n"
                                f"event: resync\n"
                                f"data: {json.dumps({'reason': resync_reason, 'epoch': ep, 'latest_seq': last_id})}\n\n"
                            )
                            self.wfile.write(frame.encode("utf-8"))
                        else:
                            frame = (
                                f"id: {ep}:{last_id}\n"
                                f"event: ready\n"
                                f"data: {json.dumps({'epoch': ep, 'latest_seq': last_id})}\n\n"
                            )
                            self.wfile.write(frame.encode("utf-8"))
                        self.wfile.flush()
                        while not srv._stop_evt.is_set():
                            evts, overflowed = srv._sse.next_events(last_id, timeout=15.0)
                            if srv._stop_evt.is_set():
                                break
                            if overflowed:
                                last_id = srv._sse.current_seq()
                                frame = (
                                    f"id: {ep}:{last_id}\n"
                                    f"event: resync\n"
                                    f"data: {json.dumps({'reason': 'ring_overflow', 'epoch': ep, 'latest_seq': last_id})}\n\n"
                                )
                                self.wfile.write(frame.encode("utf-8"))
                                self.wfile.flush()
                                continue
                            if not evts:
                                self.wfile.write(b": ping\n\n")
                                self.wfile.flush()
                                continue
                            for eid, etype, payload in evts:
                                last_id = eid
                                frame = (
                                    f"id: {ep}:{eid}\n"
                                    f"event: {etype}\n"
                                    f"data: {json.dumps(payload, ensure_ascii=False, allow_nan=False)}\n\n"
                                )
                                self.wfile.write(frame.encode("utf-8"))
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass

                def do_GET(self) -> None:
                    if not self._check_host():
                        self._json(403, {"error": {"code": "forbidden_host", "message": "Disallowed Host header."}})
                        return
                    u = urlsplit(self.path)
                    path = u.path
                    query = parse_qs(u.query, keep_blank_values=True)
                    qs = {k: v[-1] for k, v in query.items() if v[-1] or (path == "/api/multichannel" and k in {"display_lff", "display_hff"})}
                    try:
                        if "blind_for" in query:
                            reader = (qs.get("blind_for") or "").strip()
                            if not reader:
                                raise ToolkitError("missing_reader", "Provide a reviewer name for blinded requests.")
                            qs["blind_for"] = reader
                            # Shared activity streams, reports and export files contain cross-reader data.
                            if path.startswith(("/api/", "/exports/", "/cohorts/")) and path not in {
                                "/api/health", "/api/bootstrap", "/api/queue", "/api/triad", "/api/multichannel",
                                "/api/review-sets", "/api/ui-state", "/api/comparisons",
                            }:
                                raise ToolkitError("blinded_review", "This shared view is unavailable during blinded review.")
                            if path == "/api/ui-state":
                                self._json(200, {})
                                return
                        if path == "/api/events":
                            self._handle_sse()
                            return

                        if path == "/api/health":
                            self._json(200, {
                                "ok": True,
                                "service": "brainlens-workbench",
                                "pid": os.getpid(),
                                "workspace": str(srv.store.root),
                                "workspace_id": srv.store.workspace_id,
                                "url": srv.url,
                            })
                            return

                        if path == "/api/ui-state":
                            ui = srv.store.load_ui_state()
                            acts = srv._action_index.query(only_pending=True, limit=1)
                            self._json(200, {
                                **ui,
                                "pending_action_count": acts["pending_count"],
                                "claimed_action_count": acts["claimed_count"],
                                "latest_action_seq": acts["latest_seq"],
                            })
                            return

                        if path == "/api/case-verdicts":
                            cslug = qs.get("case") or ""
                            if not cslug or not srv.store.has_case(cslug):
                                self._json(404, {"error": {"code": "case_not_found", "message": f"Case not found: {cslug}"}})
                                return
                            self._json(200, srv.store.case_verdicts_projection(cslug))
                            return

                        if path == "/api/sheet-freshness":
                            cslug = sanitize_slug(qs.get("case") or "")
                            leaf = Path(qs.get("file") or "").name
                            exp_hash = (qs.get("export_hash") or "").strip()
                            if not cslug or not leaf or leaf.startswith(".") or not srv.store.has_case(cslug):
                                self._json(200, {"fresh": False, "state": "unverified"})
                                return
                            exports_root = (srv.store.case_dir(cslug) / "exports").resolve()
                            fpath = (exports_root / leaf).resolve()
                            if not fpath.is_file() or not fpath.is_relative_to(exports_root):
                                self._json(200, {"fresh": False, "state": "missing"})
                                return
                            raw_bytes = fpath.read_bytes()
                            state = srv._check_verified_sheet_state(cslug, fpath, raw_bytes, check_judgments=False)
                            if state == "fresh" and exp_hash:
                                if f'"export_hash": "{exp_hash}"'.encode("utf-8") not in raw_bytes:
                                    state = "superseded"
                            self._json(200, {"fresh": state == "fresh", "state": state})
                            return

                        if path == "/api/report-actions":
                            since = int(qs.get("since", "0") or "0")
                            only_p = qs.get("pending") in {"1", "true", "yes"}
                            inc_c = qs.get("include_claimed", "1") not in {"0", "false", "no"}
                            limit = int(qs.get("limit", "100") or "100")
                            self._json(
                                200,
                                srv._action_index.query(
                                    case_slug=qs.get("case") or None,
                                    since_seq=since,
                                    only_pending=only_p,
                                    include_claimed=inc_c,
                                    limit=limit,
                                    consumer=qs.get("consumer") or "default",
                                ),
                            )
                            return

                        if path == "/api/report-actions/wait":
                            since = int(qs.get("since", "0") or "0")
                            timeout = max(0.5, min(300.0, float(qs.get("timeout", "120") or "120")))
                            trigger_only = qs.get("trigger_only") in {"1", "true", "yes"}
                            claim = qs.get("claim") in {"1", "true", "yes"}
                            if claim:
                                origin = (self.headers.get("Origin") or "").strip()
                                if origin and origin not in srv._allowed_origins():
                                    raise ToolkitError("forbidden_origin", f"Cross-origin claim rejected ({origin}).")
                                tok = (self.headers.get("X-BrainLens-Token") or self.headers.get("X-BTK-Token") or "").strip()
                                if not tok or not secrets.compare_digest(tok, srv.auth_token):
                                    raise ToolkitError("unauthorized", "wait?claim=1 requires X-BrainLens-Token.")
                            consumer = qs.get("consumer") or "agent"
                            lease_sec = float(qs.get("lease_seconds", "180") or "180")
                            c_filter = qs.get("case") or None
                            res = srv._wait_for_report_actions(
                                since_seq=since,
                                case_slug=c_filter,
                                timeout_seconds=timeout,
                                trigger_only=trigger_only,
                                claim=claim,
                                consumer=consumer,
                                lease_seconds=lease_sec,
                            )
                            self._json(200, res)
                            return

                        if path == "/api/bootstrap":
                            self._json(200, srv._bootstrap(qs))
                            return

                        if path == "/api/anatomy":
                            cslug = qs.get("case", "")
                            if not cslug or not srv.store.has_case(cslug):
                                raise ToolkitError("case_not_found", "Specify a registered case for anatomy labels.")
                            anatomy = load_anatomy_labels(srv.store, cslug)
                            self._json(200, {
                                "case": cslug, "anatomy_labels": anatomy,
                                "coverage_by_recording": recording_anatomy_coverage(srv.store, cslug, anatomy),
                            })
                            return

                        if path in {"/api/review-sets", "/api/agreement"}:
                            case, rec, run = srv._review_target(qs)
                            if path == "/api/review-sets":
                                result = {"review_sets": list_review_sets(srv.store, case, rec, run)}
                            else:
                                readers = [r.strip() for r in qs["readers"].split(",")] if qs.get("readers") else None
                                result = reader_agreement(srv.store, case, rec, run, readers, qs.get("review_set"))
                            self._json(200, result)
                            return

                        if path in {"/api/evidence-map", "/api/epileptogenicity", "/api/timeline", "/api/trends"}:
                            # A cold read starts work and an EI selection can change the case map.
                            origin = self.headers.get("Origin")
                            site = self.headers.get("Sec-Fetch-Site")
                            if (origin and origin not in srv._allowed_origins()) or site in {"cross-site", "same-site"}:
                                raise ToolkitError("forbidden_origin", "Cross-origin evidence requests are not allowed.")
                            if path == "/api/timeline":
                                self._json(200, admission_timeline(srv.store, qs.get("case", ""), text=qs.get("text"),
                                                                   include_absolute=True))
                                return
                            if path == "/api/trends":
                                from .trends import recording_trends

                                case, rec, _ = srv._review_target(qs)
                                self._json(200, recording_trends(srv.store, case, rec, measure=qs.get("measure", "ripple"),
                                                                 epoch_seconds=qs.get("epoch_seconds", 10), run=qs.get("run_name")))
                                return
                            status, payload = srv._evidence_response(path, qs)
                            self._json(status, payload)
                            return

                        if path == "/api/multichannel":
                            result = compute_multichannel_batch(
                                srv.store,
                                case_slug=qs["case"],
                                rec_slug=qs["recording"],
                                run_name=qs.get("run_name", "baseline"),
                                shaft=qs.get("shaft") or None,
                                focus_channel=qs.get("focus_channel") or None,
                                page=int(qs.get("page", "0")),
                                page_size=int(qs.get("page_size", "16")),
                                start_seconds=float(qs.get("start_seconds", "0")),
                                duration_seconds=float(qs.get("duration_seconds", "8")),
                                montage=qs.get("montage", "acquired"),
                                display_filter=qs.get("display_filter", "broadband"),
                                display_notch=qs.get("display_notch") in {"1", "true", "yes"},
                                display_lff=qs.get("display_lff"),
                                display_hff=qs.get("display_hff"),
                                source_only=qs.get("source_only") == "1",
                            )
                            if qs.get("blind_for"):
                                result = blind_signal_payload(srv.store, result, qs["blind_for"], qs.get("blind_role"))
                            self._json(200, result)
                            return

                        if path == "/api/jobs":
                            self._json(200, {"jobs": srv.jobs.list_jobs()})
                            return

                        if path.startswith("/api/jobs/"):
                            jid = unquote(path.split("/api/jobs/", 1)[1]).strip("/")
                            job = srv.jobs.get(jid)
                            if job is None:
                                self._json(404, {"error": f"Job not found: {jid}"})
                            else:
                                self._json(200, job)
                            return

                        if path == "/api/cohorts":
                            self._json(200, {"cohorts": srv.store.list_cohorts()})
                            return

                        if path.startswith("/cohorts/"):
                            parts = [unquote(p) for p in path.split("/") if p]
                            if len(parts) == 3:
                                _, cohort_slug, fname = parts
                                fpath = (srv.store.cohort_dir(cohort_slug) / Path(fname).name).resolve()
                                if fpath.is_file() and fpath.is_relative_to(srv.store.cohort_dir(cohort_slug).resolve()):
                                    ctype = "text/csv; charset=utf-8" if fpath.suffix == ".csv" else "application/json; charset=utf-8"
                                    self._send(200, fpath.read_bytes(), ctype)
                                    return
                            self._json(404, {"error": "Cohort file not found"})
                            return

                        if path == "/api/triad":
                            c_raw = (qs.get("case") or "").strip()
                            r_raw = (qs.get("recording") or "").strip()
                            c_clean = sanitize_slug(c_raw)
                            r_clean = sanitize_slug(r_raw)
                            if not c_raw or not srv.store.has_case(c_clean):
                                raise ToolkitError("case_not_found", f"Case {c_raw!r} is not registered.")
                            if not r_raw or r_clean not in srv.store.list_recordings(c_clean):
                                raise ToolkitError("recording_not_found", f"Recording {r_raw!r} not found in case {c_clean!r}.")
                            c_sec = float(qs["center_seconds"]) if qs.get("center_seconds") not in {None, ""} else None
                            result = compute_triad_view(
                                target=f"{c_clean}/{r_clean}",
                                workspace_dir=srv.store.root,
                                recording=r_clean,
                                run_name=qs.get("run_name", "baseline"),
                                channel=qs.get("channel") or None,
                                event_id=qs.get("event_id") or None,
                                center_seconds=c_sec,
                                window_seconds=float(qs.get("window_seconds", "0.6")),
                                allow_filesystem_persist=False,
                                source_only=qs.get("source_only") == "1",
                            )
                            if qs.get("blind_for"):
                                result = blind_signal_payload(srv.store, result, qs["blind_for"], qs.get("blind_role"))
                            self._json(200, result)
                            return

                        if path == "/api/queue":
                            self._json(200, srv._queue(qs))
                            return

                        if path == "/api/comparisons":
                            self._json(200, srv._comparisons(qs))
                            return

                        if path.startswith("/exports/"):
                            parts = [unquote(p) for p in path.split("/") if p]
                            if len(parts) == 3:
                                _, cslug, fname = parts
                                leaf = Path(fname).name
                                exports_root = (srv.store.case_dir(cslug) / "exports").resolve()
                                fpath = (exports_root / leaf).resolve()
                                if (
                                    not leaf.startswith(".")
                                    and fpath.is_file()
                                    and fpath.is_relative_to(exports_root)
                                ):
                                    suf = fpath.suffix.lower()
                                    mime_map = {
                                        ".html": "text/html; charset=utf-8",
                                        ".tsv": "text/tab-separated-values; charset=utf-8",
                                        ".csv": "text/csv; charset=utf-8",
                                        ".edf": "application/octet-stream",
                                        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                        ".json": "application/json; charset=utf-8",
                                        ".png": "image/png",
                                        ".svg": "image/svg+xml",
                                    }
                                    ctype = mime_map.get(suf, "application/octet-stream")
                                    if suf in {".edf", ".csv", ".pptx"}:
                                        safe_dl = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf)
                                        with fpath.open("rb") as fh:
                                            self._send_headers(200, os.fstat(fh.fileno()).st_size, ctype, {
                                                "X-Content-Type-Options": "nosniff",
                                                "Content-Disposition": f'attachment; filename="{safe_dl}"',
                                            })
                                            try:
                                                shutil.copyfileobj(fh, self.wfile, length=1024 * 1024)
                                            except (BrokenPipeError, ConnectionResetError):
                                                pass
                                        return
                                    raw_bytes = fpath.read_bytes()
                                    # Verify against out-of-workspace machine-local HMAC-SHA256 trust store AND
                                    # live analysis generation freshness (stale bookmarks open strictly offline read-only!)
                                    sheet_state = (
                                        srv._check_verified_sheet_state(cslug, fpath, raw_bytes, check_judgments=False)
                                        if suf == ".html"
                                        else "unverified"
                                    )
                                    extra_hdrs: dict[str, str] = {"X-Content-Type-Options": "nosniff"}
                                    if sheet_state == "shared":
                                        safe_dl = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf)
                                        extra_hdrs["Content-Disposition"] = f'attachment; filename="{safe_dl}"'
                                    elif sheet_state == "fresh":
                                        raw_bytes = srv._inject_token_html(raw_bytes)
                                    elif sheet_state == "stale":
                                        raw_bytes = srv._inject_stale_meta_html(raw_bytes)
                                    elif suf in {".html", ".htm", ".svg", ".xhtml", ".xml"}:
                                        # Isolate unverified active content from the workbench origin
                                        safe_dl = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf)
                                        extra_hdrs["Content-Security-Policy"] = "sandbox"
                                        extra_hdrs["Content-Disposition"] = f'attachment; filename="{safe_dl}"'
                                    self._send(200, raw_bytes, ctype, extra_headers=extra_hdrs)
                                    return
                            self._json(404, {"error": "Export file not found"})
                            return

                        if path == "/favicon.ico":
                            svg = (
                                b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
                                b'<rect width="24" height="24" rx="6" fill="#1d4ed8"/>'
                                b'<polyline points="2 12 6 12 9 4 13 20 16 10 19 12 22 12" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
                                b'</svg>'
                            )
                            self._send(200, svg, "image/svg+xml")
                            return

                        rel = "index.html" if path in {"", "/"} else path.lstrip("/")
                        asset = (ASSETS_DIR / rel).resolve()
                        if not asset.is_relative_to(ASSETS_DIR.resolve()) or not asset.is_file():
                            self._json(404, {"error": f"Not found: {path}"})
                            return
                        mime = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
                        if mime.startswith("text/") or mime in {"application/javascript", "application/json"}:
                            mime += "; charset=utf-8"
                        raw_asset = asset.read_bytes()
                        if rel == "index.html":
                            raw_asset = srv._inject_token_html(raw_asset)
                        self._send(200, raw_asset, mime)
                    except ToolkitError as exc:
                        code_map = {
                            "blinded_review": 403,
                            "stale_review_set": 409,
                            "stale_run_generation": 409,
                            "run_recording_mismatch": 409,
                            "unauthorized": 401,
                            "forbidden_origin": 403,
                            "forbidden_host": 403,
                        }
                        self._json(code_map.get(exc.code, 400), {"error": exc.to_dict()})
                    except Exception as exc:
                        self._json(500, {"error": {"code": "internal_error", "message": str(exc)}})

                def do_POST(self) -> None:
                    self._handle_write()

                def do_DELETE(self) -> None:
                    self._handle_write()

                def _handle_write(self) -> None:
                    u = urlsplit(self.path)
                    path = u.path
                    try:
                        self._verify_write_security()
                        body = self._body()
                        req_ws = str(body.get("workspace_id") or "").strip()
                        if req_ws and req_ws != srv.store.workspace_id:
                            raise ToolkitError(
                                "workspace_mismatch",
                                f"Request belongs to workspace {req_ws!r}, but active server is {srv.store.workspace_id!r}.",
                            )
                        if self.command == "DELETE" and path != "/api/anatomy":
                            self._json(404, {"error": f"Unknown DELETE route: {path}"})
                            return
                        if path == "/api/marker-segments":
                            case, rec, _ = srv._review_target(body)
                            segments = segments_from_markers(srv.store.load_recording(case, rec), body.get("pattern"),
                                                             body.get("before_seconds", 0), body.get("after_seconds", 0))
                            self._json(200, {"case": case, "recording": rec, "segments": segments})
                            return
                        if path == "/api/anatomy":
                            query_case_name = parse_qs(u.query).get("case", [""])[-1]
                            cslug = body.get("case") or query_case_name
                            if not isinstance(cslug, str) or not cslug or not srv.store.has_case(cslug):
                                raise ToolkitError("case_not_found", "Specify a registered case for anatomy labels.")
                            if query_case_name and query_case_name != cslug:
                                raise ToolkitError("invalid_input", "Conflicting anatomy case targets in URL and body.")
                            cslug = sanitize_slug(cslug)
                            if self.command == "DELETE":
                                clear_anatomy_labels(srv.store, cslug)
                                anatomy = None
                            else:
                                anatomy = save_anatomy_labels(srv.store, cslug, body, operation_id=body.get("operation_id"))
                            srv._sse.publish("case_updated", {"case": cslug, "source": "anatomy"})
                            self._json(200, {"ok": True, "case": cslug, "anatomy_labels": anatomy})
                            return
                        if path == "/api/review-sets":
                            case, rec, run = srv._review_target(body)
                            result = create_review_set(
                                srv.store, case, rec, run, body.get("name"),
                                sample_size=body.get("sample_size"), seed=body.get("seed"),
                            )
                            self._json(200, result)
                            return
                        if path == "/api/ui-state":
                            st = srv.store.update_ui_state(body, updated_by=str(body.get("updated_by") or "workbench"))
                            srv._last_sse_ui_seq = int(st.get("seq", 0))
                            acts = srv._action_index.query(only_pending=True, limit=1)
                            srv._sse.publish("ui_state", {
                                **st,
                                "pending_action_count": acts["pending_count"],
                                "claimed_action_count": acts["claimed_count"],
                                "latest_action_seq": acts["latest_seq"],
                            })
                            self._json(200, st)
                            return

                        if path == "/api/open-case":
                            target = str(body["target"]).strip()
                            if body.get("analyze", True):
                                out = analyze_case(target, workspace_dir=srv.store.root, spatial_sidecar=body.get("spatial_sidecar"))
                                cslug = out["case"]
                                rec0 = out["runs"][0]["recording"]
                                lead = out["runs"][0]["leading_focal_channels"]
                                top_ch = lead[0]["channel"] if lead else None
                                ui = srv.store.update_ui_state(
                                    {"case": cslug, "recording": rec0, "run_name": out["run_name"], "channel": top_ch, "view": "overview"},
                                    updated_by="workbench_open",
                                )
                                srv._sse.publish("case_updated", {"case": cslug, "source": "open_case"})
                                self._json(200, {"ok": True, "result": out, "ui_state": ui})
                            else:
                                insp = inspect_case(target, workspace_dir=srv.store.root, spatial_sidecar=body.get("spatial_sidecar"))
                                ui = srv.store.update_ui_state(
                                    {"case": insp["case"], "recording": insp["recordings"][0]["recording"], "run_name": "baseline"},
                                    updated_by="workbench_open",
                                )
                                srv._sse.publish("case_updated", {"case": insp["case"], "source": "open_case"})
                                self._json(200, {"ok": True, "result": insp, "ui_state": ui})
                            return

                        if path == "/api/branch":
                            out = analyze_case(
                                target=str(body["target"]),
                                workspace_dir=srv.store.root,
                                run_name=body.get("run_name"),
                                overrides=body.get("overrides"),
                                parent_run=body.get("parent_run") or "baseline",
                                protocol=body.get("protocol"),
                                reason=body.get("reason"),
                                overwrite=bool(body.get("overwrite", False)),
                            )
                            ui = srv.store.update_ui_state(
                                {
                                    "case": out["case"],
                                    "run_name": out["run_name"],
                                    "compare_run": out["run_name"],
                                },
                                updated_by="workbench_branch",
                            )
                            srv._sse.publish("case_updated", {"case": out["case"], "source": "branch"})
                            self._json(200, {"ok": True, "result": out, "ui_state": ui})
                            return

                        if path == "/api/judgment":
                            self._json(200, srv._save_judgment(body))
                            return

                        if path == "/api/report-action":
                            self._json(200, srv._record_report_action(body))
                            return

                        if path == "/api/report-actions/claim":
                            res = srv.store.claim_report_actions(
                                case_slug=body.get("case"),
                                action_ids=list(body.get("action_ids") or []),
                                consumer=str(body.get("consumer") or "agent"),
                                lease_seconds=float(body.get("lease_seconds") or 180.0),
                            )
                            srv._mark_act_sigs_published()
                            with srv._action_cond:
                                srv._action_cond.notify_all()
                            gmeta = srv._action_index.query(only_pending=True, limit=1)
                            srv._sse.publish("report_actions", {
                                "case": res.get("case"),
                                "latest_seq": res.get("latest_seq"),
                                "pending_count": res.get("pending_count"),
                                "claimed_count": res.get("claimed_count"),
                                "global_pending_count": gmeta["pending_count"],
                            })
                            self._json(200, res)
                            return

                        if path == "/api/report-actions/release":
                            res = srv.store.release_report_actions(
                                case_slug=str(body.get("case") or ""),
                                action_ids=list(body.get("action_ids") or []),
                                consumer=str(body.get("consumer") or "agent"),
                            )
                            srv._mark_act_sigs_published()
                            with srv._action_cond:
                                srv._action_cond.notify_all()
                            gmeta = srv._action_index.query(only_pending=True, limit=1)
                            srv._sse.publish("report_actions", {
                                "case": res.get("case"),
                                "latest_seq": res.get("latest_seq"),
                                "pending_count": res.get("pending_count"),
                                "claimed_count": res.get("claimed_count"),
                                "global_pending_count": gmeta["pending_count"],
                            })
                            self._json(200, res)
                            return

                        if path == "/api/report-actions/resolve":
                            res = srv.store.resolve_report_actions(
                                case_slug=str(body.get("case") or ""),
                                action_ids=list(body.get("action_ids") or []),
                                agent_reply=str(body.get("agent_reply") or ""),
                                outcome=str(body.get("outcome") or "resolved"),
                                resolved_by=str(body.get("resolved_by") or "agent"),
                            )
                            srv._mark_act_sigs_published()
                            with srv._action_cond:
                                srv._action_cond.notify_all()
                            gmeta = srv._action_index.query(only_pending=True, limit=1)
                            srv._sse.publish("report_actions", {
                                "case": res.get("case"),
                                "latest_seq": res.get("latest_seq"),
                                "pending_count": res.get("pending_count"),
                                "claimed_count": res.get("claimed_count"),
                                "global_pending_count": gmeta["pending_count"],
                                "resolved_ids": res.get("resolved_ids"),
                            })
                            self._json(200, res)
                            return

                        if path == "/api/export-slides":
                            from .mdt_slides import export_mdt_slides

                            cslug = sanitize_slug(str(body["case"]))
                            privacy = _export_privacy_options(body)

                            def _run_slides(cb):
                                cb("render", 0.1, "Building MDT slides")
                                exp = export_mdt_slides(
                                    case_slug=cslug, workspace_dir=srv.store.root,
                                    primary_run=str(body.get("primary_run") or "baseline"),
                                    comparison_run=body.get("comparison_run"),
                                    clinical_question=body.get("clinical_question"),
                                    max_cards=body.get("max_cards", 6),
                                    **privacy,
                                )
                                filename = Path(exp["slides_path"]).name
                                return {**exp, "download_url": f"/exports/{cslug}/{filename}"}

                            self._json(202, srv.jobs.submit("export_mdt_slides", "MDT slides", _run_slides))
                            return

                        if path == "/api/export-sheet":
                            cslug = sanitize_slug(str(body["case"]))
                            primary_run = str(body.get("primary_run") or "baseline")
                            privacy = _export_privacy_options(body)
                            exp: dict[str, Any] = {}
                            rp = Path()
                            for _attempt in range(3):
                                exp = export_case_sheet(
                                    case_slug=cslug,
                                    workspace_dir=srv.store.root,
                                    primary_run=primary_run,
                                    comparison_run=body.get("comparison_run"),
                                    clinical_question=body.get("clinical_question"),
                                    **privacy,
                                )
                                rp = Path(exp["report_path"])
                                # Re-render immediately if an underlying run was overwritten mid-export!
                                if srv._is_verified_fresh_sheet(cslug, rp, primary_run, check_judgments=False):
                                    break
                            self._json(200, {
                                **exp,
                                "sheet_url": f"/exports/{cslug}/{rp.name}",
                                "sheet_mtime_ns": rp.stat().st_mtime_ns if rp.is_file() else 0,
                            })
                            return

                        if path == "/api/protocol":
                            res = save_protocol(
                                name=str(body["name"]),
                                case_slug=str(body["case"]),
                                recording=body.get("recording"),
                                run_name=str(body.get("run_name") or "baseline"),
                                description=str(body.get("description") or ""),
                                workspace_dir=srv.store.root,
                            )
                            self._json(200, {"ok": True, **res})
                            return

                        if path == "/api/export-bids":
                            cslug = sanitize_slug(str(body["case"]))
                            exp = export_bids_events(
                                case_slug=cslug,
                                recording=body.get("recording"),
                                run_name=str(body.get("run_name") or "baseline"),
                                only_confirmed=bool(body.get("only_confirmed", False)),
                                exclude_widespread_bursts=bool(body.get("exclude_widespread_bursts", True)),
                                workspace_dir=srv.store.root,
                                **_export_privacy_options(body),
                            )
                            tsv_name = Path(exp["tsv_path"]).name
                            json_name = Path(exp["json_path"]).name
                            self._json(200, {
                                **exp,
                                "tsv_url": f"/exports/{cslug}/{tsv_name}",
                                "json_url": f"/exports/{cslug}/{json_name}",
                            })
                            return

                        if path == "/api/export-edf":
                            cslug = sanitize_slug(str(body["case"]))
                            exp = export_edf_annotations(
                                case_slug=cslug,
                                recording=body.get("recording"),
                                run_name=str(body.get("run_name") or "baseline"),
                                only_confirmed=bool(body.get("only_confirmed", False)),
                                exclude_widespread_bursts=bool(body.get("exclude_widespread_bursts", False)),
                                workspace_dir=srv.store.root,
                                **_export_privacy_options(body),
                            )
                            for kind in ("edf", "csv", "json"):
                                path_value = exp[f"{kind}_path"]
                                exp[f"{kind}_url"] = f"/exports/{cslug}/{Path(path_value).name}" if path_value else None
                            self._json(200, exp)
                            return

                        if path == "/api/overlay":
                            cslug = sanitize_slug(str(body["case"]))
                            if body.get("delete"):
                                srv.store.delete_clinical_overlay(cslug)
                                srv._sse.publish("case_updated", {"case": cslug, "source": "overlay"})
                                self._json(200, {"ok": True, "overlay": None, "concordance": {"available": False}})
                                return
                            ov = srv.store.save_clinical_overlay(cslug, body)
                            srv._sse.publish("case_updated", {"case": cslug, "source": "overlay"})
                            rec_s = body.get("recording") or (srv.store.list_recordings(cslug)[0] if srv.store.list_recordings(cslug) else None)
                            rname = str(body.get("run_name") or "baseline")
                            conc = (
                                compute_overlay_concordance(srv.store, cslug, rec_slug=rec_s, run_name=rname)
                                if (rec_s and srv.store.has_run(cslug, sanitize_slug(rec_s), sanitize_slug(rname)))
                                else {"available": False}
                            )
                            self._json(200, {"ok": True, "overlay": ov, "concordance": conc})
                            return

                        if path == "/api/jobs/branch":
                            tgt = str(body["target"])
                            lbl = f"Branch {body.get('run_name') or 'auto'} on {tgt}"

                            def _run_br(cb):
                                return analyze_case(
                                    target=tgt,
                                    workspace_dir=srv.store.root,
                                    run_name=body.get("run_name"),
                                    overrides=body.get("overrides"),
                                    parent_run=body.get("parent_run") or "baseline",
                                    protocol=body.get("protocol"),
                                    reason=body.get("reason"),
                                    overwrite=bool(body.get("overwrite", False)),
                                    progress_cb=cb,
                                )

                            def _done_br(out):
                                srv.store.update_ui_state(
                                    {"case": out["case"], "run_name": out["run_name"], "compare_run": out["run_name"]},
                                    updated_by="workbench_branch",
                                )
                                srv._sse.publish("case_updated", {"case": out["case"], "source": "branch_job"})

                            job = srv.jobs.submit("branch", lbl, _run_br, on_success=_done_br)
                            self._json(202, job)
                            return

                        if path == "/api/jobs/open-case":
                            tgt = str(body["target"]).strip()

                            def _run_open(cb):
                                return analyze_case(
                                    tgt,
                                    workspace_dir=srv.store.root,
                                    spatial_sidecar=body.get("spatial_sidecar"),
                                    progress_cb=cb,
                                )

                            def _done_open(out):
                                rec0 = out["runs"][0]["recording"]
                                lead = out["runs"][0]["leading_focal_channels"]
                                top_ch = lead[0]["channel"] if lead else None
                                srv.store.update_ui_state(
                                    {"case": out["case"], "recording": rec0, "run_name": out["run_name"], "channel": top_ch, "view": "overview"},
                                    updated_by="workbench_open",
                                )
                                srv._sse.publish("case_updated", {"case": out["case"], "source": "open_job"})

                            job = srv.jobs.submit("open_case", f"Analyze {tgt}", _run_open, on_success=_done_open)
                            self._json(202, job)
                            return

                        if path == "/api/jobs/cohort":
                            proto = body.get("protocol") or None
                            cname = body.get("cohort_name") or None
                            tgts = body.get("targets") or "*"

                            def _run_coh(cb):
                                return run_cohort(
                                    targets=tgts,
                                    protocol=proto,
                                    cohort_name=cname,
                                    run_name=body.get("run_name"),
                                    overrides=body.get("overrides"),
                                    workspace_dir=srv.store.root,
                                    overwrite=bool(body.get("overwrite", False)),
                                    progress_cb=cb,
                                )

                            job = srv.jobs.submit("cohort", f"Cohort ({proto or 'baseline'})", _run_coh)
                            self._json(202, job)
                            return

                        self._json(404, {"error": f"Unknown route: {path}"})
                    except ToolkitError as exc:
                        code_map = {
                            "unauthorized": 401,
                            "forbidden_origin": 403,
                            "forbidden_host": 403,
                            "workspace_mismatch": 409,
                            "review_set_exists": 409,
                            "stale_review_set": 409,
                            "stale_sheet": 409,
                            "run_recording_mismatch": 409,
                            "stale_run_generation": 409,
                            "invalid_content_type": 415,
                        }
                        self._json(code_map.get(exc.code, 400), {"error": exc.to_dict()})
                    except Exception as exc:
                        self._json(500, {"error": {"code": "internal_error", "message": str(exc)}})

            class _QuietHTTPServer(ThreadingHTTPServer):
                def handle_error(self, request: Any, client_address: Any) -> None:
                    _, exc_val, _ = sys.exc_info()
                    if isinstance(exc_val, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
                        return
                    super().handle_error(request, client_address)

            try:
                self.store.reconcile_pending_judgments()
            except Exception:
                pass
            try:
                self.httpd = _QuietHTTPServer((self.host, self.requested_port), Handler)
            except OSError:
                self.httpd = _QuietHTTPServer((self.host, 0), Handler)
            self.httpd.daemon_threads = True
            self._stop_evt.clear()
            self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True, name="brainlens-workbench-http")
            self.thread.start()
            self._watch_thread = threading.Thread(target=self._watch_external_files, daemon=True, name="brainlens-workbench-watch")
            self._watch_thread.start()
            meta_p = self.store.root / "workbench_server.json"
            atomic_write_json(
                meta_p,
                {
                    "url": self.url,
                    "host": self.host,
                    "port": self.port,
                    "pid": os.getpid(),
                    "workspace_id": self.store.workspace_id,
                    "auth_token": self.auth_token,
                },
                mode=0o600,
            )
            return self.url

    def stop(self) -> None:
        self._stop_evt.set()
        self._sse.wake_all()
        with self._action_cond:
            self._action_cond.notify_all()
        with self._lock:
            h, self.httpd = self.httpd, None
        if h is not None:
            h.shutdown()
            h.server_close()

    def _review_target(self, params: dict[str, Any]) -> tuple[str, str, str]:
        case = params.get("case") or ""
        rec = params.get("recording") or ""
        if not isinstance(case, str) or not case.strip() or not self.store.has_case(case):
            raise ToolkitError("case_not_found", "Select a registered case for the review.")
        if not isinstance(rec, str) or not rec.strip() or sanitize_slug(rec) not in self.store.list_recordings(case):
            raise ToolkitError("recording_not_found", "Select a recording in this case for the review.")
        run_name = params.get("run_name") or "baseline"
        if not isinstance(run_name, str):
            raise ToolkitError("invalid_run_name", "Provide a run name as a string.")
        return sanitize_slug(case), sanitize_slug(rec), sanitize_slug(run_name)

    def _evidence_response(self, path: str, qs: dict[str, str]) -> tuple[int, dict[str, Any]]:
        from .evidence_map import get_evidence_map, get_epileptogenicity

        case = qs.get("case", "")
        recording = qs.get("recording", "")
        if not case or not self.store.has_case(case):
            raise ToolkitError("case_not_found", f"Case {case!r} is not registered.")
        if recording not in self.store.list_recordings(case):
            raise ToolkitError("recording_not_found", f"Recording {recording!r} not found in case {case!r}.")

        options: dict[str, Any] = {"recording": recording}
        if path == "/api/evidence-map":
            compute = get_evidence_map
            options["run_name"] = qs.get("run_name", "baseline")
            label = "Evidence map"
        else:
            compute = get_epileptogenicity
            if qs.get("onset_seconds"):
                try:
                    options["onset_seconds"] = float(qs["onset_seconds"])
                except ValueError as exc:
                    raise ToolkitError("invalid_onset", "onset_seconds must be a number in recording seconds.") from exc
            label = "Epileptogenicity Index"

        cached = compute(self.store, case, **options, cached_only=True)
        if cached is not None:
            if path == "/api/epileptogenicity":
                # Selecting a cached onset also selects the EI shown in the map and card.
                return 200, compute(self.store, case, **options)
            return 200, cached

        # An overwrite or re-registration may occur while a previous job is still running.
        fingerprint = self.store.load_recording(case, recording).fingerprint
        generation = None
        if path == "/api/evidence-map":
            resolved = self.store.resolve_run_gen_dir(case, recording, options["run_name"])
            if resolved is None:
                raise ToolkitError("run_not_found", f"No saved run {options['run_name']!r} for {recording!r}.")
            generation = resolved[1]
        key = (path, case, tuple(options.items()), fingerprint, generation)

        def run(progress):
            try:
                progress("compute", 0.1, f"Computing {label} for {recording}")
                return compute(self.store, case, **options)
            finally:
                with self._evidence_job_lock:
                    self._evidence_jobs.pop(key, None)

        with self._evidence_job_lock:
            existing = self.jobs.get(self._evidence_jobs.get(key, ""))
            if existing is not None:
                return 202, existing
            job = self.jobs.submit(path.rsplit("/", 1)[-1], f"{label} · {case}/{recording}", run)
            self._evidence_jobs[key] = job["job_id"]
            return 202, job

    def _bootstrap(self, qs: dict[str, str]) -> dict[str, Any]:
        reader = qs.get("blind_for")
        ui = self.store.load_ui_state()
        cases = self.store.list_cases()
        case_names = [c["case"] for c in cases]
        active_case = qs.get("case") or ui.get("case") or (case_names[0] if case_names else None)
        if active_case and not self.store.has_case(active_case):
            active_case = case_names[0] if case_names else None

        lite = qs.get("lite") in {"1", "true", "yes"}
        avail_inputs = [] if lite else discover_available_inputs()
        protocols = [] if reader else list_protocols(self.store.root)

        if not active_case:
            return {
                "workspace_id": self.store.workspace_id,
                "cases": [],
                "available_inputs": avail_inputs,
                "protocols": protocols,
                "engine": {} if lite else check_engine_health(self.store.root),
                "ui_state": {} if reader else ui,
                "active": None,
            }

        cdoc = self.store.load_case(active_case)
        recs = self.store.list_recordings(active_case)
        active_rec = (
            qs.get("recording")
            or (ui.get("recording") if ui.get("case") == active_case else None)
            or (recs[0] if recs else None)
        )
        if active_rec not in recs:
            active_rec = recs[0] if recs else None

        runs_by_rec = {r: self.store.list_runs(active_case, r) for r in recs}
        rec_run_names = [m["run_name"] for m in runs_by_rec.get(active_rec or "", []) if m.get("run_name")]
        req_run = (qs.get("run_name") or "").strip()
        active_run = req_run or ui.get("run_name") or (rec_run_names[0] if rec_run_names else "baseline")
        if rec_run_names and active_run not in rec_run_names:
            # If caller explicitly requested a run that exists on disk, NEVER silently fall back to rec_run_names[0]!
            if not (active_rec and self.store.has_run(active_case, active_rec, active_run)):
                active_run = rec_run_names[0]

        rec_spec = self.store.load_recording(active_case, active_rec) if active_rec else None
        active_bundle = (
            self.store.load_run(active_case, active_rec, active_run)
            if (active_rec and self.store.has_run(active_case, active_rec, active_run))
            else None
        )
        run_summary = active_bundle["summary"] if active_bundle is not None else None
        active_rmeta = (active_bundle or {}).get("meta") or {}
        run_asig = (
            str(active_rmeta.get("analysis_signature") or run_analysis_signature(active_bundle))
            if active_bundle is not None
            else None
        )
        run_agen = (
            (str(active_rmeta.get("analysis_generation_id") or active_rmeta.get("generation_id") or "") or None)
            if active_bundle is not None
            else None
        )

        # Existing exported sheet matched specifically to active_run, verified by this machine's trust store,
        # AND matching active_run's current analysis generation signature (stale/overwritten/copied/legacy sheets
        # return sheet_url=None so View 3 immediately rebuilds a fresh sheet!)
        exp_dir = self.store.case_dir(active_case) / "exports"
        run_sheet = exp_dir / f"{active_case}_{sanitize_slug(active_run)}_case_sheet.html"
        legacy_sheet = exp_dir / f"{active_case}_case_sheet.html"
        sheet_path: Path | None = None
        if not reader and run_sheet.is_file() and self._is_verified_fresh_sheet(active_case, run_sheet, active_run):
            sheet_path = run_sheet
        elif not reader and legacy_sheet.is_file() and self._is_verified_fresh_sheet(active_case, legacy_sheet, active_run):
            try:
                txt = legacy_sheet.read_text(encoding="utf-8", errors="ignore")
                m_meta = re.search(r'name="(?:brainlens|btk)-primary-run"\s+content="([^"]+)"', txt[:4096])
                m_json = re.search(r'"primary_run"\s*:\s*"([^"]+)"', txt)
                legacy_primary = (
                    m_meta.group(1).strip() if m_meta else (m_json.group(1).strip() if m_json else None)
                )
                if legacy_primary is not None and sanitize_slug(legacy_primary) == sanitize_slug(active_run):
                    sheet_path = legacy_sheet
            except Exception:
                sheet_path = None
        sheet_url = f"/exports/{active_case}/{sheet_path.name}" if sheet_path is not None else None
        if reader:
            all_judgments = []
            if active_bundle and active_rec:
                all_judgments, _ = reader_projection(
                    self.store, active_case, active_rec, active_run, active_bundle, reader, qs.get("blind_role"),
                    blinded=True,
                )
            run_summary = without_shared_review(run_summary)
            cdoc = {"case": active_case}
            ui_state = {}
        else:
            all_judgments = self.store.load_judgments(active_case)
            acts_meta = self._action_index.query(only_pending=True, limit=1)
            ui_state = {
                **ui,
                "pending_action_count": acts_meta["pending_count"],
                "claimed_action_count": acts_meta["claimed_count"],
                "latest_action_seq": acts_meta["latest_seq"],
            }

        avail_states = sorted({
            m.label
            for m in (rec_spec.markers if rec_spec else [])
            if m.label and not m.label.upper().startswith("BAD")
        })
        overlay = None if reader else self.store.load_clinical_overlay(active_case)
        anatomy = load_anatomy_labels(self.store, active_case)
        if run_summary is not None:
            run_summary = annotate_summary_anatomy(run_summary, anatomy, recording=rec_spec)
        recording_doc = rec_spec.model_dump(mode="json") if rec_spec else None
        if recording_doc:
            matcher = AnatomyMatcher(anatomy)
            order = ChannelDisplayOrder(rec_spec)
            for row in recording_doc["channels"]:
                row["anatomy"] = matcher.match(row["name"], row.get("contacts", []))
                row["display_shaft"] = order.depth_shaft(row["id"], row["name"])
            recording_doc["channels"].sort(key=lambda row: order.position(row["id"], row["name"]))
        overlay_conc = (
            compute_overlay_concordance(self.store, active_case, rec_slug=active_rec, run_name=active_run)
            if (overlay and active_rec and self.store.has_run(active_case, active_rec, active_run))
            else {"available": False}
        )

        return {
            "workspace_id": self.store.workspace_id,
            "cases": [{"case": c["case"], "recording_count": c.get("recording_count", 0), "origin_kinds": c.get("origin_kinds", [])} for c in cases],
            "available_inputs": avail_inputs,
            "protocols": protocols,
            "cohorts": [] if reader else self.store.list_cohorts(),
            "jobs": [] if reader else self.jobs.list_jobs(limit=10),
            "engine": {} if lite else check_engine_health(self.store.root),
            "ui_state": ui_state,
            "active": {
                "case": active_case,
                "case_doc": cdoc,
                "recordings": recs,
                "active_recording": active_rec,
                "recording_spec": recording_doc,
                "available_states": avail_states,
                "runs_by_recording": {
                    r: [
                        {
                            "run_name": m.get("run_name"),
                            "parent_run": m.get("parent_run"),
                            "reason": None if reader else m.get("reason"),
                            "counts": without_shared_review(m.get("counts")) if reader else m.get("counts"),
                            "classifier_status": m.get("classifier_status"),
                        }
                        for m in rlist
                    ]
                    for r, rlist in runs_by_rec.items()
                },
                "active_run": active_run,
                "run_analysis_signature": run_asig,
                "run_analysis_generation_id": run_agen,
                "run_summary": run_summary,
                "clinical_overlay": overlay,
                "anatomy_labels": anatomy,
                "overlay_concordance": overlay_conc,
                "judgments": all_judgments[-40:],
                "judgment_revision": len(all_judgments),
                "sheet_url": sheet_url,
                "sheet_primary_run": active_run if sheet_path is not None else None,
                "sheet_mtime_ns": sheet_path.stat().st_mtime_ns if sheet_path is not None else 0,
            },
        }

    def _queue(self, qs: dict[str, str]) -> dict[str, Any]:
        cslug = qs["case"]
        rslug = qs["recording"]
        rname = qs.get("run_name", "baseline")
        qmode = qs.get("mode", "focal")
        qsort = qs.get("sort", "time")
        ch = qs.get("channel") or None

        offset = max(0, int(qs.get("offset", "0") or "0"))
        limit = max(1, min(500, int(qs.get("limit", "150") or "150")))
        if qmode != "review_set" and not self.store.has_run(cslug, rslug, rname):
            return {"mode": qmode, "sort": qsort, "total": 0, "offset": offset, "limit": limit, "has_more": False, "items": []}

        if qmode == "controls":
            ctrls = select_control_windows(self.store, cslug, rslug, run_name=rname, channel=ch, count=6)
            page_ctrl = ctrls[offset : offset + limit]
            return {
                "mode": "controls",
                "sort": qsort,
                "total": len(ctrls),
                "offset": offset,
                "limit": limit,
                "has_more": (offset + len(page_ctrl)) < len(ctrls),
                "items": page_ctrl,
            }

        bundle = self.store.load_run(cslug, rslug, rname)
        events = bundle["events"]
        blind_for = qs.get("blind_for")
        own: dict[str, dict[str, Any]] = {}
        if qmode == "review_set" or blind_for:
            reader = (blind_for or qs.get("reader") or "").strip()
            if not reader:
                raise ToolkitError("missing_reader", "Provide the reviewer's name for the queue.")
            role = qs.get("blind_role") if blind_for else qs.get("reader_role")
            _, own = reader_projection(self.store, cslug, rslug, rname, bundle, reader, role, blinded=bool(blind_for))

        if qmode == "review_set":
            name = qs.get("review_set")
            if not name:
                raise ToolkitError("missing_review_set", "Select a review set for the queue.")
            doc = load_review_set(self.store, cslug, rslug, rname, name, bundle=bundle)
            order = reader_event_order(doc, reader)
            by_id = {e["id"]: e for e in events}
            page = [by_id[eid] for eid in order[offset:offset + limit]]
            judged = sum(
                any(state.get(k) for k in ("verdict", "researcher_verdict", "rehearsal_verdict", "agent_verdict"))
                for eid in order if (state := own.get(eid))
            )
            return {
                "mode": qmode, "sort": "reader_order", "total": len(order), "offset": offset,
                "limit": limit, "has_more": offset + len(page) < len(order),
                "items": [blind_event(e, own) for e in page] if blind_for else page,
                "progress": {"judged": judged, "total": len(order)},
            }

        if blind_for:
            events = [blind_event(e, own) for e in events]
        filtered: list[dict[str, Any]] = []
        for ev in events:
            if ch and ev["channel_name"] != ch and ev["channel_id"] != ch:
                continue
            if qmode == "focal" and ev.get("synchrony_class") == "widespread_burst":
                continue
            if qmode == "ehfo_fr":
                lbls = ev.get("labels") or {}
                is_hi = (
                    ev.get("synchrony_class") != "widespread_burst"
                    and (
                        lbls.get("ehfo") == 1
                        or lbls.get("spike") == 1
                        or ev.get("dominant_band") == "fast_ripple"
                    )
                )
                if not is_hi:
                    continue
            if qmode == "bursts" and ev.get("synchrony_class") != "widespread_burst":
                continue
            if qmode == "confirmed":
                confirm = ev.get("clinician_verdict") == "confirm" or ev.get("clipped")
                if blind_for:
                    confirm = confirm or any(
                        ev.get(f"{role}_verdict") == "confirm" or ev.get(f"{role}_clipped")
                        for role in ("researcher", "rehearsal")
                    )
                if not confirm:
                    continue
            filtered.append(ev)

        if qsort == "ehfo":
            filtered.sort(
                key=lambda e: (
                    -((e.get("scores") or {}).get("ehfo") if (e.get("scores") or {}).get("ehfo") is not None else -1.0),
                    -((e.get("scores") or {}).get("artifact") if (e.get("scores") or {}).get("artifact") is not None else -1.0),
                    e["start_seconds"],
                )
            )
        elif qsort == "freq":
            filtered.sort(key=lambda e: (-(e.get("peak_freq_hz") or 0.0), e["start_seconds"]))
        elif qsort == "amplitude":
            filtered.sort(key=lambda e: (-(e.get("peak_to_peak_uv") or 0.0), e["start_seconds"]))
        else:
            filtered.sort(key=lambda e: (e["start_seconds"], e["channel_name"]))

        page_items = filtered[offset : offset + limit]
        matcher = AnatomyMatcher(load_anatomy_labels(self.store, cslug))
        contacts = {row["channel_id"]: row.get("contacts", []) for row in bundle["summary"].get("channels", [])}
        page_items = [
            {**event, "anatomy": matcher.match(event["channel_name"], contacts.get(event["channel_id"], []))}
            for event in page_items
        ]
        return {
            "mode": qmode,
            "sort": qsort,
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "has_more": (offset + len(page_items)) < len(filtered),
            "items": page_items,
        }

    def _comparisons(self, qs: dict[str, str]) -> dict[str, Any]:
        cslug = qs["case"]
        rslug = qs.get("recording") or self.store.list_recordings(cslug)[0]
        run_a = qs.get("run_a", "baseline")
        run_b = qs.get("run_b") or None
        if qs.get("blind_for"):
            return {"case": cslug, "recording": rslug, "run_a": run_a, "run_b": run_b, "cross_recording": None, "branch": None}

        recs_a = [r for r in self.store.list_recordings(cslug) if self.store.has_run(cslug, r, run_a)]
        cross = compare_recordings(self.store, cslug, run_name=run_a) if len(recs_a) > 1 else None

        if not run_b:
            others = [m["run_name"] for m in self.store.list_runs(cslug, rslug) if m.get("run_name") and m["run_name"] != run_a]
            if others:
                run_b = others[0]
        branch = (
            compare_runs(self.store, cslug, rslug, run_a, run_b)
            if (run_b and self.store.has_run(cslug, rslug, run_a) and self.store.has_run(cslug, rslug, run_b))
            else None
        )
        return {"case": cslug, "recording": rslug, "run_a": run_a, "run_b": run_b, "cross_recording": cross, "branch": branch}

    def _validate_event_ownership(
        self,
        cslug: str,
        rec_slug: str | None,
        run_name: str,
        ev_id: str | None,
        channel: str | None,
    ) -> None:
        """Strictly require recording + existing run + matching event_id & channel whenever event_id is supplied."""
        if not ev_id:
            return
        if not rec_slug or not str(rec_slug).strip():
            raise ToolkitError("missing_recording", f"Event judgment {ev_id!r} requires an explicit `recording`.")
        rec_clean = sanitize_slug(str(rec_slug))
        run_clean = sanitize_slug(str(run_name or "baseline"))
        if not self.store.has_run(cslug, rec_clean, run_clean):
            raise ToolkitError("run_not_found", f"Run {cslug}/{rec_clean}/{run_clean} does not exist.")
        run_bundle = self.store.load_run(cslug, rec_clean, run_clean)
        ev_match = next((e for e in run_bundle.get("events", []) if e.get("id") == ev_id), None)
        if ev_match is None:
            raise ToolkitError(
                "event_not_in_run",
                f"Event {ev_id!r} does not belong to {cslug}/{rec_clean}/{run_clean}.",
            )
        if channel and ev_match.get("channel_name") and channel != ev_match.get("channel_name"):
            raise ToolkitError(
                "event_channel_mismatch",
                f"Event {ev_id!r} belongs to channel {ev_match['channel_name']!r}, not {channel!r}.",
            )

    def _save_judgment(self, body: dict[str, Any]) -> dict[str, Any]:
        cslug = sanitize_slug(str(body["case"]))
        if not self.store.has_case(cslug):
            raise ToolkitError("case_not_found", f"Case {cslug!r} is not registered.")
        clin_q = str(body.get("clinical_question") or "").strip() or None

        verdict = body.get("verdict")
        if verdict is not None and verdict not in {"confirm", "artifact", "uncertain"}:
            raise ToolkitError("invalid_verdict", "verdict must be 'confirm', 'artifact', or 'uncertain'.")

        rec_slug = body.get("recording")
        run_name = str(body.get("run_name") or "baseline")
        ev_id = body.get("event_id")
        if body.get("review_set"):
            case, rec, run = self._review_target(body)
            review_set = load_review_set(self.store, case, rec, run, body["review_set"])
            if (verdict is not None or ev_id is not None) and ev_id not in review_set["event_ids"]:
                raise ToolkitError("event_not_in_review_set", "Select an event in the active review set.")
            # record_judgment checks these bindings again inside its WAL write lock.
            for key in ("recording_fingerprint", "analysis_signature"):
                if body.get(key) and body[key] != review_set[key]:
                    raise ToolkitError("stale_review_set", "The displayed event and review set belong to different analyses.")
            body = {
                **body,
                "recording_fingerprint": review_set["recording_fingerprint"],
                "analysis_signature": review_set["analysis_signature"],
            }
        self._validate_event_ownership(cslug, rec_slug, run_name, ev_id, body.get("channel"))

        entry = self.store.record_judgment(
            cslug,
            {
                "kind": str(body.get("kind") or ("event_verdict" if ev_id else ("clinical_question" if clin_q else "interval_note"))),
                "operation_id": body.get("operation_id"),
                "recording": sanitize_slug(str(rec_slug)) if rec_slug else None,
                "recording_fingerprint": body.get("recording_fingerprint"),
                "analysis_signature": body.get("analysis_signature"),
                "analysis_generation_id": body.get("analysis_generation_id"),
                "run_name": sanitize_slug(run_name),
                "channel": body.get("channel"),
                "event_id": ev_id,
                "center_seconds": body.get("center_seconds"),
                "interval_seconds": body.get("interval_seconds"),
                "verdict": verdict,
                "clipped": body.get("clipped"),
                "clinical_question": clin_q,
                "author": str(body.get("author") or "clinician"),
                "role": str(body.get("role") or "clinician"),
                "note": str(body.get("note") or (clin_q or "")),
            },
            validate_event_ownership=True,
        )
        if clin_q:
            self.store.add_open_question(cslug, clin_q)
        if rec_slug and self.store.has_run(cslug, sanitize_slug(str(rec_slug)), sanitize_slug(run_name)):
            refresh_verdict_tallies(self.store, cslug, sanitize_slug(str(rec_slug)), sanitize_slug(run_name))
        self._mark_judgment_published(cslug)
        j_rev = len(self.store.load_judgments(cslug))
        if clin_q:
            self._sse.publish("case_updated", {"case": cslug, "source": "clinical_question"})
        self._sse.publish("verdict", {
            "case": cslug,
            "recording": sanitize_slug(str(rec_slug)) if rec_slug else None,
            "run_name": sanitize_slug(run_name),
            "channel": body.get("channel"),
            "event_id": ev_id,
            "verdict": verdict,
            "role": entry.get("role"),
            "clipped": body.get("clipped"),
            "clinical_question": clin_q,
            "operation_id": entry.get("operation_id"),
            "judgment_revision": j_rev,
        })
        if body.get("blind_for"):
            return {"ok": True}
        return {"ok": True, "judgment": entry, "judgment_revision": j_rev}

    def _verify_sheet_action_freshness(self, cslug: str | None, body: dict[str, Any]) -> None:
        """Reject mutating actions from stale/superseded MDT sheets with HTTP 409 stale_sheet."""
        req_hash = str(body.get("export_hash") or "").strip()
        is_sheet_src = str(body.get("source") or "") == "case_sheet"
        if not is_sheet_src and not req_hash:
            return
        if not cslug or not self.store.has_case(cslug):
            raise ToolkitError("stale_sheet", "MDT Case Sheet refers to an unknown case; reload the latest sheet.", state="missing")
        if not req_hash:
            raise ToolkitError("stale_sheet", "MDT Case Sheet request is missing its export_hash version token.", state="missing_hash")
        leaf = Path(str(body.get("export_filename") or "")).name
        if not leaf:
            p_run = sanitize_slug(str(body.get("primary_run") or body.get("run_name") or "baseline"))
            leaf = f"{cslug}_{p_run}_case_sheet.html"
        if leaf.startswith("."):
            raise ToolkitError("stale_sheet", "Invalid MDT Case Sheet filename.", state="unverified")
        exports_root = (self.store.case_dir(cslug) / "exports").resolve()
        fpath = (exports_root / leaf).resolve()
        if not fpath.is_file() or not fpath.is_relative_to(exports_root):
            raise ToolkitError("stale_sheet", "MDT Case Sheet no longer exists on disk; reload the latest sheet.", state="missing")
        raw_bytes = fpath.read_bytes()
        state = self._check_verified_sheet_state(cslug, fpath, raw_bytes, check_judgments=False)
        if state == "fresh" and f'"export_hash": "{req_hash}"'.encode("utf-8") not in raw_bytes:
            state = "superseded"
        if state != "fresh":
            raise ToolkitError(
                "stale_sheet",
                f"MDT Case Sheet is outdated ({state}); reload the latest sheet before submitting.",
                state=state,
            )

    def _navigation_question(self, body: dict[str, Any], case: str | None) -> dict[str, Any]:
        """Bind a reader/timeline question to registered metadata, including before the first analysis."""
        if body.get("action") != "ask_agent" or body.get("verdict") is not None or body.get("event_id") is not None:
            raise ToolkitError("invalid_action", "Reader and timeline questions must use ask_agent without a verdict or event_id.")
        if not case or not self.store.has_case(case):
            raise ToolkitError("case_not_found", "Select a registered case.")
        if body.get("blind_for"):
            raise ToolkitError("blinded_review", "Agent questions are unavailable during blinded review.")
        role = body.get("role")
        if role not in ("clinician", "researcher", "engineering_rehearsal"):
            raise ToolkitError("invalid_role", "Select a clinician, researcher, or engineering rehearsal reviewer.")
        question = body.get("comment")
        if not isinstance(question, str) or not 0 < len(question.strip()) <= 1200:
            raise ToolkitError("invalid_question", "Enter a question of up to 1200 characters.")

        source = body["source"]
        item = {"case": case, "source": source, "action": "ask_agent", "anchor_type": source,
                "comment": question.strip(), "role": role, "author": body.get("author") or role,
                "operation_id": body.get("operation_id")}
        slug = body.get("recording")
        if not slug:
            if source == "waveform_reader" or any(body.get(k) is not None for k in (
                "run_name", "channel", "start_seconds", "end_seconds", "marker",
            )):
                raise ToolkitError("missing_recording", "Select a recording for a recording-specific question.")
            return item

        if not isinstance(slug, str):
            raise ToolkitError("invalid_recording", "Select a registered recording.")
        rec = self.store.load_recording(case, slug)
        item.update(recording=rec.slug, recording_id=rec.id)
        channel = body.get("channel")
        channels = {label for ch in rec.channels for label in (ch.id, ch.name)}
        if run_name := body.get("run_name"):
            if not isinstance(run_name, str):
                raise ToolkitError("invalid_run", "Select a saved analysis run.")
            bundle = self.store.load_run(case, rec.slug, run_name, reconcile=False)
            _verify_run_recording_binding(rec, bundle, run_name)
            item["run_name"] = run_name
            channels.update(row["channel_name"] for row in bundle["summary"].get("channels", []))
        if channel is not None:
            if not isinstance(channel, str) or channel not in channels:
                raise ToolkitError("invalid_channel", "The selected channel is unavailable in this recording and run.")
            item["channel"] = channel

        start, end = body.get("start_seconds"), body.get("end_seconds")
        if source == "waveform_reader" or start is not None or end is not None:
            # The waveform response reports bounds to five decimal places; clamp the last rounded sample edge.
            if (type(start) not in (int, float) or type(end) not in (int, float)
                    or not (0 <= start <= end <= rec.duration + 0.000005)
                    or (source == "waveform_reader" and start >= min(end, rec.duration))):
                raise ToolkitError("invalid_window", "Select a true-time window within the recording.")
            item.update(start_seconds=min(start, rec.duration), end_seconds=min(end, rec.duration))
        if (marker := body.get("marker")) is not None:
            if not isinstance(marker, dict) or not any(
                m.label == marker.get("label") and m.start == marker.get("start_seconds") and m.end == marker.get("end_seconds")
                for m in rec.markers
            ):
                raise ToolkitError("invalid_marker", "The selected marker does not belong to this recording.")
            item["marker"] = {k: marker[k] for k in ("label", "start_seconds", "end_seconds")}
        return item

    def _record_report_action(self, body: dict[str, Any]) -> dict[str, Any]:
        req_ws = str(body.get("workspace_id") or "").strip()
        if str(body.get("source") or "") == "case_sheet" and req_ws != self.store.workspace_id:
            raise ToolkitError(
                "workspace_mismatch",
                f"Report belongs to workspace {req_ws!r}, but active server is {self.store.workspace_id!r}.",
            )
        cslug = sanitize_slug(str(body["case"])) if body.get("case") else None
        navigation = body.get("source") in NAVIGATION_QUESTION_SOURCES
        # Split navigation telemetry from clinician inbox intents!
        if not navigation and str(body.get("action") or "") == "inspect_waveform":
            if cslug and (body.get("channel") or body.get("recording")):
                patch: dict[str, Any] = {"case": cslug, "view": "reader"}
                if body.get("recording"):
                    patch["recording"] = sanitize_slug(str(body["recording"]))
                if body.get("run_name"):
                    patch["run_name"] = sanitize_slug(str(body["run_name"]))
                if body.get("channel"):
                    patch["channel"] = str(body["channel"])
                if body.get("event_id"):
                    patch["event_id"] = str(body["event_id"])
                elif body.get("start_seconds") is not None or body.get("center_seconds") is not None:
                    patch["event_id"] = None
                    try:
                        s0 = float(body["start_seconds"]) if body.get("start_seconds") is not None else None
                        s1 = float(body["end_seconds"]) if body.get("end_seconds") is not None else None
                        ctr = (
                            float(body["center_seconds"])
                            if body.get("center_seconds") is not None
                            else (0.5 * (s0 + s1) if (s0 is not None and s1 is not None) else s0)
                        )
                        if ctr is not None:
                            span = (s1 - s0) if (s0 is not None and s1 is not None and s1 > s0) else 0.0
                            win = max(8.0, min(30.0, span)) if span > 0 else 8.0
                            patch["start_seconds"] = max(0.0, round(ctr - 0.5 * win, 4))
                            patch["window_seconds"] = round(win, 4)
                    except (TypeError, ValueError):
                        pass
                ui = self.store.update_ui_state(patch, updated_by="report_sheet")
                self._last_sse_ui_seq = int(ui.get("seq", 0))
                self._sse.publish("ui_state", ui)
            meta = self._action_index.query(case_slug=cslug, only_pending=True, limit=1)
            return {"ok": True, "telemetry_only": True, "latest_seq": meta["latest_seq"], "pending_count": meta["pending_count"]}

        # Reject mutating actions from stale/superseded MDT sheets BEFORE appending to report_actions or judgments!
        self._verify_sheet_action_freshness(cslug, body)
        if navigation:
            body = self._navigation_question(body, cslug)

        # Pre-validate verdict & event ownership BEFORE appending to report_actions.jsonl (zero orphaned inbox items!)
        verdict_val = body.get("verdict")
        if verdict_val is not None:
            if verdict_val not in {"confirm", "artifact", "uncertain"}:
                raise ToolkitError("invalid_verdict", "verdict must be 'confirm', 'artifact', or 'uncertain'.")
            if not cslug or not self.store.has_case(cslug):
                raise ToolkitError("case_not_found", f"Case {cslug!r} is not registered.")
            self._validate_event_ownership(
                cslug,
                body.get("recording"),
                str(body.get("run_name") or "baseline"),
                body.get("event_id"),
                body.get("channel"),
            )

        entry, created = self.store.append_report_action_ex(body)
        # Sync explicit report verdicts into judgments.jsonl idempotently, preserving caller role (never forcing clinician!)
        if cslug and entry.get("verdict") in {"confirm", "artifact", "uncertain"} and self.store.has_case(cslug):
            req_role = str(body.get("role") or entry.get("role") or "").strip()
            if not req_role:
                eff_role = "clinician"
                rec_tok = entry.get("recording")
                if rec_tok:
                    try:
                        rec_obj = self.store.load_recording(cslug, sanitize_slug(str(rec_tok)))
                        if rec_obj.provenance.is_synthetic:
                            eff_role = "engineering_rehearsal"
                    except Exception:
                        pass
            else:
                eff_role = req_role
            self._save_judgment({
                "case": cslug,
                "operation_id": f"rep_{entry['id']}",
                "recording": entry.get("recording"),
                "run_name": entry.get("run_name") or "baseline",
                "channel": entry.get("channel"),
                "event_id": entry.get("event_id"),
                "verdict": entry.get("verdict"),
                "note": entry.get("comment") or f"Report action ({entry.get('anchor_type')})",
                "author": entry.get("author") or ("rehearsal" if eff_role == "engineering_rehearsal" else "clinician"),
                "role": eff_role,
                "kind": "report_verdict",
            })

        self._mark_act_sigs_published()
        with self._action_cond:
            self._action_cond.notify_all()

        meta = self._action_index.query(case_slug=cslug, only_pending=True, limit=1)
        gmeta = self._action_index.query(only_pending=True, limit=1)
        self._sse.publish("report_actions", {
            "case": cslug,
            "action_id": entry["id"],
            "latest_seq": meta["latest_seq"],
            "pending_count": meta["pending_count"],
            "claimed_count": meta["claimed_count"],
            "global_pending_count": gmeta["pending_count"],
        })
        return {
            "ok": True,
            "action": entry,
            "deduplicated": not created,
            "latest_seq": meta["latest_seq"],
            "pending_count": meta["pending_count"],
            "claimed_count": meta["claimed_count"],
            "global_pending_count": gmeta["pending_count"],
        }

    def _wait_for_report_actions(
        self,
        *,
        since_seq: int = 0,
        case_slug: str | None = None,
        timeout_seconds: float = 120.0,
        trigger_only: bool = False,
        claim: bool = False,
        consumer: str = "agent",
        lease_seconds: float = 180.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds

        def _check_outside_lock() -> dict[str, Any] | None:
            # Scan ALL unaddressed actions for triggers BEFORE pagination (never miss a trigger buried behind 100+ notes!)
            probe = self._action_index.query(
                case_slug=case_slug,
                since_seq=since_seq,
                only_pending=True,
                include_claimed=False,
                trigger_only=trigger_only,
                limit=0,
                consumer=consumer,
            )
            if not probe["actions"]:
                return None
            # When woken by trigger_only, bundle unaddressed pending actions AND guarantee all waking triggers are included
            if trigger_only:
                bundle = self._action_index.query(
                    case_slug=case_slug,
                    since_seq=0,
                    only_pending=True,
                    include_claimed=False,
                    trigger_only=False,
                    limit=100,
                    consumer=consumer,
                )
                seen_ids = {str(a["id"]) for a in bundle["actions"]}
                merged = list(bundle["actions"])
                for trig in probe["actions"]:
                    if str(trig["id"]) not in seen_ids:
                        merged.append(trig)
                        seen_ids.add(str(trig["id"]))
                bundle = {**bundle, "actions": merged}
            else:
                bundle = probe
            if claim and bundle["actions"]:
                ids = [str(a["id"]) for a in bundle["actions"]]
                claim_res = self.store.claim_report_actions(
                    case_slug=case_slug,
                    action_ids=ids,
                    consumer=consumer,
                    lease_seconds=lease_seconds,
                )
                actually_claimed = set(claim_res.get("claimed_ids") or [])
                if not actually_claimed:
                    # Another concurrent consumer won the race; do NOT double-deliver!
                    return None
                self._mark_act_sigs_published()
                after = self._action_index.query(case_slug=case_slug, only_pending=True, limit=1, consumer=consumer)
                gmeta = self._action_index.query(only_pending=True, limit=1)
                owned_actions = [
                    {**a, "status": "claimed", "claimed_by": consumer}
                    for a in bundle["actions"]
                    if str(a["id"]) in actually_claimed
                ]
                bundle = {
                    **bundle,
                    "actions": owned_actions,
                    "claimed_ids": list(claim_res.get("claimed_ids") or []),
                    "pending_count": after["pending_count"],
                    "claimed_count": after["claimed_count"],
                }
                self._sse.publish("report_actions", {
                    "case": bundle["case"],
                    "latest_seq": bundle["latest_seq"],
                    "pending_count": after["pending_count"],
                    "claimed_count": after["claimed_count"],
                    "global_pending_count": gmeta["pending_count"],
                    "claimed_ids": bundle["claimed_ids"],
                })
            return {
                "woke": True,
                "timed_out": False,
                "case": bundle["case"],
                "actions": bundle["actions"],
                "claimed_ids": bundle.get("claimed_ids", []),
                "latest_seq": bundle["latest_seq"],
                "case_latest_seq": bundle["case_latest_seq"],
                "pending_count": bundle["pending_count"],
                "claimed_count": bundle["claimed_count"],
            }

        # IMPORTANT: Never perform file I/O while holding self._action_cond!
        hit = _check_outside_lock()
        if hit is not None:
            return hit
        while not self._stop_evt.is_set():
            rem = deadline - time.monotonic()
            if rem <= 0:
                break
            with self._action_cond:
                self._action_cond.wait(timeout=min(1.0, rem))
            hit = _check_outside_lock()
            if hit is not None:
                return hit

        cur = self._action_index.query(case_slug=case_slug, only_pending=True, include_claimed=False, limit=1, consumer=consumer)
        return {
            "woke": False,
            "timed_out": True,
            "case": cur["case"],
            "actions": [],
            "claimed_ids": [],
            "latest_seq": cur["latest_seq"],
            "case_latest_seq": cur["case_latest_seq"],
            "pending_count": cur["pending_count"],
            "claimed_count": cur["claimed_count"],
        }


_ACTIVE_SERVER: WorkbenchHTTPServer | None = None
_ACTIVE_LOCK = threading.RLock()


def is_loopback_http_url(url: str) -> bool:
    """Return True strictly for http://127.0.0.1, http://localhost, or http://[::1] URLs."""
    try:
        parsed = urlsplit(str(url).strip())
    except Exception:
        return False
    return parsed.scheme == "http" and (parsed.hostname or "").lower() in {"127.0.0.1", "localhost", "::1"}


def ensure_workbench_server(workspace_dir: str | Path | None = None, preferred_port: int = 8765) -> dict[str, Any]:
    global _ACTIVE_SERVER
    store = WorkbenchStore(workspace_dir)
    meta_file = store.root / "workbench_server.json"
    if meta_file.is_file():
        try:
            info = read_json(meta_file)
            base_u = str(info["url"]).rstrip("/")
            if is_loopback_http_url(base_u):
                with urllib.request.urlopen(base_u + "/api/health", timeout=0.8) as resp:
                    h = json.loads(resp.read().decode("utf-8"))
                    if h.get("ok") and Path(h.get("workspace", "")).resolve() == store.root:
                        return {
                            "url": str(info["url"]),
                            "reused": True,
                            "pid": h.get("pid"),
                            "auth_token": info.get("auth_token"),
                            "workspace_id": h.get("workspace_id") or store.workspace_id,
                        }
        except Exception:
            pass

    with _ACTIVE_LOCK:
        if _ACTIVE_SERVER is not None and _ACTIVE_SERVER.httpd is not None and _ACTIVE_SERVER.store.root == store.root:
            return {
                "url": _ACTIVE_SERVER.url,
                "reused": True,
                "pid": os.getpid(),
                "auth_token": _ACTIVE_SERVER.auth_token,
                "workspace_id": store.workspace_id,
            }
        srv = WorkbenchHTTPServer(workspace_dir=store.root, port=preferred_port)
        url = srv.start()
        _ACTIVE_SERVER = srv
        return {
            "url": url,
            "reused": False,
            "pid": os.getpid(),
            "auth_token": srv.auth_token,
            "workspace_id": store.workspace_id,
        }
