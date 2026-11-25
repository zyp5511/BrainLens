"""Real headless Chrome CDP browser regression tests (P0 #3 cross-case SSE isolation, Reader controls, Triad & Sheet)."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest

WORKBENCH_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKBENCH_DIR.parent
_REPO_SYNTH = PROJECT_ROOT / "inputs" / "synthetic_seeg.edf"
_FIXTURE_SYNTH = Path(__file__).resolve().parent / "fixtures" / "synthetic_seeg.edf"
SYNTHETIC_EDF = _REPO_SYNTH if _REPO_SYNTH.is_file() else _FIXTURE_SYNTH
SRC_DIR = WORKBENCH_DIR / "src"
if importlib.util.find_spec("brainlens") is None and SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlens.io_readers import inspect_case
from brainlens.models import atomic_write_json, read_json
from brainlens.pipeline import analyze_case
from brainlens.storage import WorkbenchStore
from brainlens.workbench_http import WorkbenchHTTPServer


def _find_chrome() -> str | None:
    pw_shells = sorted(
        (Path.home() / "Library" / "Caches" / "ms-playwright").glob("chromium_headless_shell-*/**/chrome-headless-shell"),
        reverse=True,
    ) if (Path.home() / "Library" / "Caches" / "ms-playwright").is_dir() else []
    cands = [
        os.environ.get("BRAINLENS_CHROME_PATH", ""),
        shutil.which("chrome-headless-shell") or "",
        *(str(p) for p in pw_shells),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("google-chrome") or "",
        shutil.which("chromium") or "",
    ]
    for c in cands:
        if c and Path(c).is_file():
            return c
    return None


class _MiniCDP:
    def __init__(self, ws_url: str) -> None:
        assert ws_url.startswith("ws://")
        rest = ws_url[5:]
        host_port, path = rest.split("/", 1)
        host, port_s = host_port.split(":")
        self.sock = socket.create_connection((host, int(port_s)), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET /{path} HTTP/1.1\r\n"
            f"Host: {host_port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.sock.recv(4096)
        assert b"101 " in buf.split(b"\r\n", 1)[0], buf
        self._extra = buf.split(b"\r\n\r\n", 1)[1]
        self._id = 0
        self.exceptions: list[str] = []
        self.console_errors: list[str] = []

    def _recv_exact(self, n: int) -> bytes:
        while len(self._extra) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EOFError("CDP websocket closed")
            self._extra += chunk
        out = self._extra[:n]
        self._extra = self._extra[n:]
        return out

    def _read_frame(self) -> tuple[int, bytes]:
        b1, b2 = self._recv_exact(2)
        opcode = b1 & 0x0F
        masked = (b2 & 0x80) != 0
        length = b2 & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        mask = self._recv_exact(4) if masked else b""
        payload = self._recv_exact(length)
        if masked:
            payload = bytes(payload[i] ^ mask[i % 4] for i in range(length))
        return opcode, payload

    def _send_text(self, text: str) -> None:
        data = text.encode("utf-8")
        mask = os.urandom(4)
        header = bytearray([0x81])
        n = len(data)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", n))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", n))
        header.extend(mask)
        masked = bytes(data[i] ^ mask[i % 4] for i in range(n))
        self.sock.sendall(bytes(header) + masked)

    def _record_event(self, msg: dict[str, Any]) -> None:
        m = msg.get("method")
        p = msg.get("params") or {}
        if m == "Runtime.exceptionThrown":
            det = p.get("exceptionDetails") or {}
            txt = det.get("text") or ""
            ex = (det.get("exception") or {}).get("description") or ""
            self.exceptions.append(f"{txt}: {ex}")
        elif m == "Log.entryAdded":
            ent = p.get("entry") or {}
            if ent.get("level") == "error":
                self.console_errors.append(f"[Log] {ent.get('text', '')}")
        elif m == "Runtime.consoleAPICalled" and p.get("type") == "error":
            args = [str(a.get("value", a.get("description", ""))) for a in p.get("args", [])]
            self.console_errors.append("[Console] " + " ".join(args))

    def send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._id += 1
        cid = self._id
        self._send_text(json.dumps({"id": cid, "method": method, "params": params or {}}))
        while True:
            op, payload = self._read_frame()
            if op == 8:
                raise EOFError("CDP closed")
            if op == 9:
                self.sock.sendall(b"\x8a\x80\x00\x00\x00\x00")
                continue
            if op != 1:
                continue
            msg = json.loads(payload.decode("utf-8"))
            if msg.get("id") == cid:
                if "error" in msg:
                    raise RuntimeError(f"CDP error {method}: {msg['error']}")
                return msg.get("result", {})
            self._record_event(msg)

    def eval(self, expr: str) -> Any:
        res = self.send(
            "Runtime.evaluate",
            {"expression": expr, "returnByValue": True, "awaitPromise": True},
        )
        if res.get("exceptionDetails"):
            raise RuntimeError(f"JS exception: {res['exceptionDetails']}")
        return (res.get("result") or {}).get("value")

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


def _build_two_case_fixture(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    rng = np.random.default_rng(7)
    sfreq = 1024.0
    n_samps = int(sfreq * 20)
    # CaseA: 24 channels across 3 shafts (tests All-channel scroll + row height + control + selection)
    ch_a = [f"LA{i}" for i in range(1, 9)] + [f"LB{i}" for i in range(1, 9)] + [f"LC{i}" for i in range(1, 9)]
    info_a = mne.create_info(ch_a, sfreq=sfreq, ch_types="seeg")
    raw_a = mne.io.RawArray(rng.normal(0, 25e-6, size=(len(ch_a), n_samps)), info_a, verbose="ERROR")
    f_a = tmp_path / "CaseA.fif"
    raw_a.save(f_a, overwrite=True, verbose="ERROR")
    (tmp_path / "CaseA.json").write_text(json.dumps({"reference": "common", "line_freq": 60.0}), encoding="utf-8")

    # CaseB: distinct second case for P0 #3 cross-case SSE switch test
    ch_b = ["RA1", "RA2", "RB1", "RB2"]
    info_b = mne.create_info(ch_b, sfreq=sfreq, ch_types="seeg")
    raw_b = mne.io.RawArray(rng.normal(0, 20e-6, size=(len(ch_b), n_samps)), info_b, verbose="ERROR")
    f_b = tmp_path / "CaseB.fif"
    raw_b.save(f_b, overwrite=True, verbose="ERROR")
    (tmp_path / "CaseB.json").write_text(json.dumps({"reference": "common", "line_freq": 60.0}), encoding="utf-8")

    analyze_case(str(f_a), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    analyze_case(str(f_b), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})

    # Populate 2 realistic events on CaseA (at 3.0s and 10.0s on LB4) and valid intervals [0, 20s]
    store = WorkbenchStore(ws)
    rdir_a = store.run_generation_dir("CaseA", "CaseA", "baseline")
    meta_a = read_json(rdir_a / "run_meta.json")
    det_p = Path(meta_a["detection_json_path"])
    det = read_json(det_p)
    sum_a = read_json(rdir_a / "summary.json")
    lb4 = next(c for c in sum_a["channels"] if c["channel_name"] == "LB4")
    det["status"] = "complete"
    det["valid_intervals"] = {c["channel_id"]: [[0, n_samps]] for c in sum_a["channels"]}
    det["events"] = [
        {
            "id": "ev_casea_001",
            "channel_id": lb4["channel_id"],
            "channel_name": "LB4",
            "shaft": "LB",
            "start_sample": int(3.0 * sfreq),
            "end_sample": int(3.08 * sfreq),
            "start_seconds": 3.0,
            "end_seconds": 3.08,
            "scores": {},
            "labels": {},
        },
        {
            "id": "ev_casea_002",
            "channel_id": lb4["channel_id"],
            "channel_name": "LB4",
            "shaft": "LB",
            "start_sample": int(10.0 * sfreq),
            "end_sample": int(10.08 * sfreq),
            "start_seconds": 10.0,
            "end_seconds": 10.08,
            "scores": {},
            "labels": {},
        },
    ]
    atomic_write_json(det_p, det)
    clf_p = Path(meta_a["classification_json_path"])
    atomic_write_json(clf_p, {**det, "classifier_status": "skipped", "classification": {"requested": [], "stages": {}}})
    from brainlens.pipeline import refresh_verdict_tallies
    refresh_verdict_tallies(store, "CaseA", "CaseA", "baseline")

    # Populate 1 realistic event on CaseB (at 4.0s on RA1) for pagehide/pageshow verdict sync verification
    rdir_b = store.run_generation_dir("CaseB", "CaseB", "baseline")
    meta_b = read_json(rdir_b / "run_meta.json")
    det_pb = Path(meta_b["detection_json_path"])
    det_b = read_json(det_pb)
    sum_b = read_json(rdir_b / "summary.json")
    ra1 = next(c for c in sum_b["channels"] if c["channel_name"] == "RA1")
    det_b["status"] = "complete"
    det_b["valid_intervals"] = {c["channel_id"]: [[0, n_samps]] for c in sum_b["channels"]}
    det_b["events"] = [
        {
            "id": "ev_CaseB_001",
            "channel_id": ra1["channel_id"],
            "channel_name": "RA1",
            "shaft": "RA",
            "start_sample": int(4.0 * sfreq),
            "end_sample": int(4.08 * sfreq),
            "start_seconds": 4.0,
            "end_seconds": 4.08,
            "scores": {},
            "labels": {},
        },
    ]
    atomic_write_json(det_pb, det_b)
    clf_pb = Path(meta_b["classification_json_path"])
    atomic_write_json(clf_pb, {**det_b, "classifier_status": "skipped", "classification": {"requested": [], "stages": {}}})
    refresh_verdict_tallies(store, "CaseB", "CaseB", "baseline")
    return ws


@pytest.mark.skipif(_find_chrome() is None, reason="Google Chrome / Chromium not installed")
def test_browser_cdp_p0_cross_case_and_pro_reader_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    chrome_bin = _find_chrome()
    assert chrome_bin is not None
    ws = _build_two_case_fixture(tmp_path)
    srv = WorkbenchHTTPServer(workspace_dir=ws, port=0)
    base_url = srv.start().rstrip("/")
    tok = srv.auth_token

    # Find free debug port for Chrome
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dbg_port = s.getsockname()[1]

    user_data = tmp_path / "chrome_profile"
    chrome = subprocess.Popen(
        [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            f"--remote-debugging-port={dbg_port}",
            f"--user-data-dir={user_data}",
            "--window-size=1280,820",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cdp: _MiniCDP | None = None
    try:
        ws_url = None
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{dbg_port}/json/list", timeout=1.0) as r:
                    tabs = json.loads(r.read().decode("utf-8"))
                    page_tab = next((t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")), None)
                    if page_tab:
                        ws_url = page_tab["webSocketDebuggerUrl"]
                        break
            except Exception:
                time.sleep(0.15)
        assert ws_url, "Chrome CDP endpoint did not become ready"

        cdp = _MiniCDP(ws_url)
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Log.enable")
        cdp.send("Emulation.setDeviceMetricsOverride", {
            "width": 1280,
            "height": 820,
            "deviceScaleFactor": 2,
            "mobile": False,
        })

        # 1) Navigate directly to Pro Waveform Reader on CaseA
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&recording=CaseA&run=baseline&view=reader"})
        for _ in range(40):
            ready = cdp.eval("Boolean(document.getElementById('caseSelect')?.value === 'CaseA' && document.querySelectorAll('#queueList .q-item').length >= 2)")
            if ready:
                break
            time.sleep(0.15)
        assert cdp.eval("document.getElementById('caseSelect').value") == "CaseA"

        # 2) Verify zero horizontal overflow + pixel-exact Triad canvas at DPR=2
        layout_ok = cdp.eval("""(() => {
            const docOverflow = document.documentElement.scrollWidth - window.innerWidth;
            const trWrap = document.querySelector('.triad-canvas-wrap');
            const trCanvas = document.getElementById('triadCanvas');
            return {
                docOverflow,
                trWrapH: trWrap ? Math.round(trWrap.getBoundingClientRect().height) : 0,
                trCanvasH: trCanvas ? Math.round(trCanvas.getBoundingClientRect().height) : 0,
            };
        })()""")
        assert layout_ok["docOverflow"] <= 1, layout_ok
        assert abs(layout_ok["trWrapH"] - layout_ok["trCanvasH"]) <= 2, layout_ok

        # 3) Control window span change MUST stay QUIET CONTROL (never reset to top_focal!)
        cdp.eval("document.getElementById('showControlBtn').click()")
        for _ in range(30):
            if cdp.eval("document.getElementById('syncClassBadge').textContent.trim()") == "QUIET CONTROL":
                break
            time.sleep(0.1)
        assert cdp.eval("document.getElementById('syncClassBadge').textContent.trim()") == "QUIET CONTROL"
        cdp.eval("""(() => {
            const s = document.getElementById('triadWinSelect');
            s.value = '1.0';
            s.dispatchEvent(new Event('change', { bubbles: true }));
        })()""")
        time.sleep(0.6)
        assert cdp.eval("document.getElementById('syncClassBadge').textContent.trim()") == "QUIET CONTROL"
        assert "Quiet control" in cdp.eval("document.getElementById('triadHeaderText').textContent")

        # 4) Vertical All-channel scrolling + Row Height mode (poll until 'All (' traces finish loading!)
        cdp.eval("[...document.querySelectorAll('#shaftPills button')].find(b => b.textContent.trim().startsWith('All'))?.click()")
        for _ in range(35):
            if (cdp.eval("document.getElementById('pageLabel')?.textContent || ''") or "").startswith("All ("):
                break
            time.sleep(0.08)
        assert (cdp.eval("document.getElementById('pageLabel')?.textContent || ''") or "").startswith("All (")
        cdp.eval("""(() => {
            const rh = document.getElementById('rowHeightSelect');
            rh.value = '40';
            rh.dispatchEvent(new Event('change', { bubbles: true }));
        })()""")
        time.sleep(0.15)
        rect = cdp.eval("(() => { const r = document.getElementById('mcCanvas').getBoundingClientRect(); return {x: r.left + r.width/2, y: r.top + r.height/2}; })()")
        cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseWheel",
            "x": rect["x"],
            "y": rect["y"],
            "deltaX": 0,
            "deltaY": 320,
        })
        time.sleep(0.25)
        # Switch to Measure tool and drag a caliper -> must display honest 'disp P2P' HUD (poll observable state!)
        cdp.eval("document.querySelector('#mcToolModes button[data-tool=\"measure\"]').click()")
        cdp.send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": rect["x"] - 40, "y": rect["y"], "button": "left", "clickCount": 1})
        cdp.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": rect["x"] + 60, "y": rect["y"], "buttons": 1})
        cdp.send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": rect["x"] + 60, "y": rect["y"], "button": "left", "clickCount": 1})
        hud_txt = ""
        for _ in range(30):
            hud_txt = cdp.eval("document.getElementById('mcReadoutHud').textContent") or ""
            if "disp P2P" in hud_txt:
                break
            time.sleep(0.08)
        if "disp P2P" not in hud_txt:
            diag = cdp.eval("({caseSel: document.getElementById('caseSelect')?.value, activeTool: document.querySelector('#mcToolModes button.active')?.dataset?.tool, hudHidden: document.getElementById('mcReadoutHud')?.classList?.contains('hidden'), pageLbl: document.getElementById('pageLabel')?.textContent, toast: document.getElementById('toast')?.textContent})")
            raise AssertionError(f"Expected honest 'disp P2P' label in caliper HUD, got: {hud_txt!r}, diag={diag}, exc={cdp.exceptions}, console={cdp.console_errors}")

        # 5) Round 3 P0 #2 Regression: DELAY CaseB bootstrap while SSE switches CaseA -> CaseB;
        #    dragging & clicking Clip on still-rendered CaseA waveform MUST NEVER write to CaseB!
        cdp.eval("document.querySelector('#mcToolModes button[data-tool=\"exclude\"]').click()")
        cdp.send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": rect["x"] - 50, "y": rect["y"], "button": "left", "clickCount": 1})
        cdp.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": rect["x"] + 50, "y": rect["y"], "buttons": 1})
        cdp.send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": rect["x"] + 50, "y": rect["y"], "button": "left", "clickCount": 1})
        for _ in range(25):
            if cdp.eval("!document.getElementById('selectionBar').classList.contains('hidden')"):
                break
            time.sleep(0.06)
        assert cdp.eval("!document.getElementById('selectionBar').classList.contains('hidden')") is True
        assert "CaseA" in cdp.eval("document.getElementById('selectionText').textContent")

        # Wrap window.fetch to delay CaseB's /api/bootstrap response by 900ms so we exercise the exact in-flight window!
        cdp.eval("""(() => {
            const origFetch = window.fetch.bind(window);
            window.__brainlensDelayedCaseB = false;
            window.fetch = async (...args) => {
                const url = String(args[0] || '');
                if (!window.__brainlensDelayedCaseB && url.includes('/api/bootstrap') && url.includes('case=CaseB')) {
                    window.__brainlensDelayedCaseB = true;
                    await new Promise((r) => setTimeout(r, 900));
                }
                return origFetch(...args);
            };
        })()""")

        # Remote agent switches ui_state to CaseB via HTTP POST (triggers live SSE ui_state event)
        req = urllib.request.Request(
            f"{base_url}/api/ui-state",
            data=json.dumps({
                "case": "CaseB",
                "recording": "CaseB",
                "run_name": "baseline",
                "view": "reader",
                "updated_by": "agent",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-BrainLens-Token": tok},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200

        # Wait until delayed CaseB bootstrap starts (still in-flight while CaseA canvas is shown!)
        for _ in range(30):
            if cdp.eval("Boolean(window.__brainlensDelayedCaseB)"):
                break
            time.sleep(0.05)
        assert cdp.eval("Boolean(window.__brainlensDelayedCaseB)") is True

        # During in-flight switch: previous CaseA selectionBar MUST already be hidden,
        # AND attempting a new exclude-drag + clip click on the old canvas MUST NOT write to CaseB,
        # AND an interleaved SSE verdict on CaseA MUST NOT hijack focus back to CaseA!
        assert cdp.eval("document.getElementById('selectionBar').classList.contains('hidden')") is True
        cdp.send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": rect["x"] - 45, "y": rect["y"], "button": "left", "clickCount": 1})
        cdp.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": rect["x"] + 45, "y": rect["y"], "buttons": 1})
        cdp.send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": rect["x"] + 45, "y": rect["y"], "button": "left", "clickCount": 1})
        time.sleep(0.1)
        assert cdp.eval("document.getElementById('selectionBar').classList.contains('hidden')") is True
        cdp.eval("document.getElementById('clipSelectionBtn').click()")

        # Fire an interleaved CaseA verdict while CaseA -> CaseB switch is still in flight!
        req_va = urllib.request.Request(
            f"{base_url}/api/judgment",
            data=json.dumps({
                "case": "CaseA",
                "recording": "CaseA",
                "run_name": "baseline",
                "channel": "LB4",
                "event_id": "ev_casea_001",
                "verdict": "uncertain",
                "author": "Dr. Interleave",
                "role": "clinician",
                "note": "Interleaved verdict during remote switch",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-BrainLens-Token": tok},
            method="POST",
        )
        with urllib.request.urlopen(req_va, timeout=5) as rva:
            assert rva.status == 200

        # Wait for CaseB bootstrap + waveforms to finish loading (MUST land on CaseB, NEVER stuck on CaseA!)
        for _ in range(40):
            if cdp.eval("document.getElementById('caseSelect').value") == "CaseB":
                break
            time.sleep(0.12)
        assert cdp.eval("document.getElementById('caseSelect').value") == "CaseB"
        assert cdp.eval("document.getElementById('selectionBar').classList.contains('hidden')") is True
        # ZERO cross-case journal writes in CaseB!
        assert WorkbenchStore(ws).load_judgments("CaseB") == []

        # 6) Round 3 P1 Regression: pagehide -> external verdict persisted while hidden -> pageshow MUST refresh UI!
        for _ in range(30):
            if "CaseB" in (cdp.eval("document.getElementById('triadHeaderText').textContent") or "") or "RA1" in (cdp.eval("document.getElementById('verdictEventMeta').textContent") or ""):
                break
            time.sleep(0.1)
        assert "unreviewed" in (cdp.eval("document.getElementById('verdictEventMeta').textContent") or "")
        cdp.eval("window.dispatchEvent(new Event('pagehide'))")
        time.sleep(0.15)
        # Persist a clinician confirm verdict on CaseB's event while page is hidden (SSE closed)
        req_v = urllib.request.Request(
            f"{base_url}/api/judgment",
            data=json.dumps({
                "case": "CaseB",
                "recording": "CaseB",
                "run_name": "baseline",
                "channel": "RA1",
                "event_id": "ev_CaseB_001",
                "verdict": "confirm",
                "author": "Dr. Remote",
                "role": "clinician",
                "note": "Saved while page hidden",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-BrainLens-Token": tok},
            method="POST",
        )
        with urllib.request.urlopen(req_v, timeout=5) as rv:
            assert rv.status == 200
        # Wake page with pageshow -> SSE reconnect MUST refresh active verdict without manual reload!
        cdp.eval("window.dispatchEvent(new Event('pageshow'))")
        meta_after = ""
        for _ in range(40):
            meta_after = cdp.eval("document.getElementById('verdictEventMeta').textContent") or ""
            if "confirm" in meta_after:
                break
            time.sleep(0.12)
        assert "confirm" in meta_after, f"Expected verdict to sync to 'confirm' after pageshow, got: {meta_after!r}"

        # 7) Round 4 P1 Regression: set Reviewer role = researcher -> switch to View 3 MDT Sheet ->
        #    clicking Confirm on MDT card MUST record role='researcher' (NOT 'clinician')!
        cdp.eval("""(() => {
            const sel = document.getElementById('authorRoleSelect');
            sel.value = 'researcher';
            sel.dispatchEvent(new Event('change', { bubbles: true }));
            const inp = document.getElementById('authorInput');
            inp.value = 'Res. Alice';
            inp.dispatchEvent(new Event('input', { bubbles: true }));
        })()""")
        cdp.eval("document.querySelector('#viewNav button[data-view=\"sheet\"]').click()")
        for _ in range(40):
            src = cdp.eval("document.getElementById('sheetFrame').getAttribute('src') || ''")
            if "CaseB_baseline_case_sheet.html" in src:
                break
            time.sleep(0.2)
        assert "CaseB_baseline_case_sheet.html" in cdp.eval("document.getElementById('sheetFrame').getAttribute('src') || ''")
        # Wait until embedded MDT Sheet connects live and renders the RA1 card
        for _ in range(40):
            ready_m = cdp.eval("Boolean(document.getElementById('sheetFrame')?.contentDocument?.body?.classList?.contains('live-connected') && document.getElementById('sheetFrame')?.contentDocument?.querySelector('.fig-card[data-event-id=\"ev_CaseB_001\"] button[data-verdict=\"confirm\"]'))")
            if ready_m:
                break
            time.sleep(0.12)
        assert cdp.eval("Boolean(document.getElementById('sheetFrame')?.contentDocument?.querySelector('.fig-card[data-event-id=\"ev_CaseB_001\"] button[data-verdict=\"confirm\"]'))") is True

        # Click Confirm as researcher on MDT Sheet -> journal MUST store role='researcher' & author='Res. Alice'!
        cdp.eval("document.getElementById('sheetFrame').contentDocument.querySelector('.fig-card[data-event-id=\"ev_CaseB_001\"] button[data-verdict=\"confirm\"]').click()")
        for _ in range(30):
            js_b = WorkbenchStore(ws).load_judgments("CaseB")
            if any(j.get("role") == "researcher" and j.get("verdict") == "confirm" for j in js_b):
                break
            time.sleep(0.1)
        js_b = WorkbenchStore(ws).load_judgments("CaseB")
        res_entry = next((j for j in js_b if j.get("kind") == "report_verdict"), None)
        assert res_entry is not None and res_entry.get("role") == "researcher" and res_entry.get("author") == "Res. Alice", js_b

        # 8) Round 4 P1 Regression: MDT Sheet hidden during clinician verdict flip (confirm -> artifact)
        #    MUST update card .vtag to 'artifact' on pageshow!
        cdp.eval("window.dispatchEvent(new Event('pagehide'))")
        time.sleep(0.15)
        req_flip = urllib.request.Request(
            f"{base_url}/api/judgment",
            data=json.dumps({
                "case": "CaseB",
                "recording": "CaseB",
                "run_name": "baseline",
                "channel": "RA1",
                "event_id": "ev_CaseB_001",
                "verdict": "artifact",
                "author": "Dr. Attending",
                "role": "clinician",
                "note": "Flipped to artifact while MDT hidden",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-BrainLens-Token": tok},
            method="POST",
        )
        with urllib.request.urlopen(req_flip, timeout=5) as rfl:
            assert rfl.status == 200
        cdp.eval("window.dispatchEvent(new Event('pageshow'))")
        vtag_after = ""
        for _ in range(40):
            vtag_after = cdp.eval("document.getElementById('sheetFrame')?.contentDocument?.querySelector('.fig-card[data-event-id=\"ev_CaseB_001\"] .vtag')?.textContent?.trim() || ''") or ""
            if vtag_after == "artifact":
                break
            time.sleep(0.12)
        assert vtag_after == "artifact", f"Expected MDT card .vtag to refresh to 'artifact' after pageshow, got: {vtag_after!r}"

        assert cdp.exceptions == [], f"JS exceptions: {cdp.exceptions}"
        assert cdp.console_errors == [], f"Console errors: {cdp.console_errors}"
    finally:
        if cdp is not None:
            cdp.close()
        chrome.terminate()
        try:
            chrome.wait(timeout=4)
        except Exception:
            chrome.kill()
        srv.stop()



def test_cdp_round5_sheet_switch_click_lock_and_researcher_role_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real CDP tests for Round 5 P0 #4 (MDT Sheet click lock during Case B->A switch) & P1 #1 (researcher role after synthetic visit)."""
    chrome_bin = _find_chrome()
    if not chrome_bin:
        pytest.skip("Google Chrome / Chromium not installed")

    import brainlens.workbench_http as http_mod
    from brainlens.case_sheet import export_case_sheet
    from brainlens.pipeline import refresh_verdict_tallies

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r5_cdp"))
    ws = _build_two_case_fixture(tmp_path)
    store = WorkbenchStore(ws)

    # Also register a synthetic demo case (provenance.is_synthetic == True) to test P1 #1 role preservation!
    syn_edf = SYNTHETIC_EDF
    analyze_case(str(syn_edf), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})

    # Populate 1 event on CaseB so CaseB's pre-built MDT Sheet has a real clickable Uncertain button
    rdir_b = store.run_generation_dir("CaseB", "CaseB", "baseline")
    meta_b = read_json(rdir_b / "run_meta.json")
    sum_b = read_json(rdir_b / "summary.json")
    ra1 = next(c for c in sum_b["channels"] if c["channel_name"] == "RA1")
    det_b = read_json(Path(meta_b["detection_json_path"]))
    det_b["status"] = "complete"
    det_b["valid_intervals"] = {c["channel_id"]: [[0, 1024 * 20]] for c in sum_b["channels"]}
    det_b["events"] = [
        {
            "id": "ev_caseb_r5",
            "channel_id": ra1["channel_id"],
            "channel_name": "RA1",
            "shaft": "RA",
            "start_sample": int(4.0 * 1024),
            "end_sample": int(4.08 * 1024),
            "start_seconds": 4.0,
            "end_seconds": 4.08,
            "scores": {},
            "labels": {},
        }
    ]
    atomic_write_json(Path(meta_b["detection_json_path"]), det_b)
    clf_b = read_json(Path(meta_b["classification_json_path"]))
    clf_b["events"] = det_b["events"]
    atomic_write_json(Path(meta_b["classification_json_path"]), clf_b)
    refresh_verdict_tallies(store, "CaseB", "CaseB", "baseline")

    # Pre-build ONLY CaseB's sheet; CaseA has NO sheet yet so switching CaseB -> CaseA triggers buildCaseSheet()!
    export_case_sheet("CaseB", workspace_dir=ws, primary_run="baseline")

    # Slow down export_case_sheet for CaseA by 1.2s so the mid-switch window is 100% deterministic!
    orig_export = http_mod.export_case_sheet
    export_started = [False]

    def _slow_export(*args: Any, **kwargs: Any):
        c_arg = kwargs.get("case_slug") or (args[0] if args else "")
        if str(c_arg) == "CaseA":
            export_started[0] = True
            time.sleep(1.2)
        return orig_export(*args, **kwargs)

    monkeypatch.setattr(http_mod, "export_case_sheet", _slow_export)

    srv = WorkbenchHTTPServer(ws, host="127.0.0.1", port=0)
    base_url = srv.start()
    user_data = tmp_path / "chrome_r5_prof"
    user_data.mkdir()

    s_tmp = socket.socket()
    s_tmp.bind(("127.0.0.1", 0))
    dbg_port = s_tmp.getsockname()[1]
    s_tmp.close()

    chrome = subprocess.Popen(
        [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--window-size=1280,860",
            f"--remote-debugging-port={dbg_port}",
            f"--user-data-dir={user_data}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cdp: _MiniCDP | None = None
    try:
        ws_url = None
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{dbg_port}/json/list", timeout=1) as r:
                    tabs = json.loads(r.read().decode("utf-8"))
                page = next((t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")), None)
                if page:
                    ws_url = page["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.15)
        assert ws_url, "Headless Chrome did not expose CDP page target"

        cdp = _MiniCDP(ws_url)
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Log.enable")
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseB&view=sheet"})

        # Wait for CaseB's embedded MDT Sheet to connect live and render its Uncertain button
        for _ in range(80):
            ready_b = cdp.eval(
                "Boolean(document.getElementById('sheetFrame')?.contentDocument?.body?.classList?.contains('live-connected')"
                " && document.getElementById('sheetFrame')?.contentDocument?.querySelector('.fig-card[data-event-id=\"ev_caseb_r5\"] button[data-verdict=\"uncertain\"]'))"
            )
            if ready_b:
                break
            time.sleep(0.12)
        assert ready_b is True

        # Measure REAL viewport coordinates of CaseB's Uncertain button inside the iframe
        btn_xy = cdp.eval(
            "(() => {"
            "  const fr = document.getElementById('sheetFrame');"
            "  const frR = fr.getBoundingClientRect();"
            "  const btn = fr.contentDocument.querySelector('.fig-card[data-event-id=\"ev_caseb_r5\"] button[data-verdict=\"uncertain\"]');"
            "  btn.scrollIntoView({block: 'center'});"
            "  const r = btn.getBoundingClientRect();"
            "  return { x: frR.left + r.left + r.width / 2, y: frR.top + r.top + r.height / 2 };"
            "})()"
        )
        assert btn_xy and btn_xy["x"] > 0 and btn_xy["y"] > 0, btn_xy

        # (A) P0 #4: Switch CaseB -> CaseA and immediately fire a REAL physical mouse click at CaseB's Uncertain button!
        cdp.eval(
            "(() => {"
            "  const sel = document.getElementById('caseSelect');"
            "  sel.value = 'CaseA';"
            "  sel.dispatchEvent(new Event('change', { bubbles: true }));"
            "})()"
        )
        # Wait briefly until CaseA export is in flight (parent already switching / building CaseA sheet)
        for _ in range(40):
            if export_started[0]:
                break
            time.sleep(0.05)
        # Verify iframe is locked/cleared immediately during switch!
        frame_locked = cdp.eval(
            "(() => {"
            "  const fr = document.getElementById('sheetFrame');"
            "  return { pe: fr.style.pointerEvents, src: fr.getAttribute('src') || '' };"
            "})()"
        )
        assert frame_locked["pe"] == "none" and frame_locked["src"] == "about:blank", frame_locked

        # Dispatch real CDP mouse click at CaseB's former Uncertain coordinates while CaseA builds
        cdp.send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": btn_xy["x"], "y": btn_xy["y"], "button": "left", "clickCount": 1})
        cdp.send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": btn_xy["x"], "y": btn_xy["y"], "button": "left", "clickCount": 1})

        # Wait until CaseA's fresh MDT Sheet finishes building, loads, and unlocks!
        for _ in range(80):
            ready_a = cdp.eval(
                "(() => {"
                "  const fr = document.getElementById('sheetFrame');"
                "  return Boolean("
                "    (fr.getAttribute('src') || '').includes('CaseA_baseline_case_sheet.html')"
                "    && fr.style.pointerEvents !== 'none'"
                "    && fr.contentDocument?.body?.classList?.contains('live-connected')"
                "    && fr.contentDocument?.querySelector('.fig-card[data-event-id=\"ev_casea_001\"] button[data-verdict=\"uncertain\"]')"
                "  );"
                "})()"
            )
            if ready_a:
                break
            time.sleep(0.12)
        assert ready_a is True, "CaseA MDT Sheet did not load and unlock cleanly"

        # CaseB MUST have ZERO cross-case verdicts written from that mid-switch physical mouse click!
        assert WorkbenchStore(ws).load_judgments("CaseB") == [], WorkbenchStore(ws).load_judgments("CaseB")

        # (B) P1 #1: Explicitly choose 'researcher', confirm on CaseA, visit synthetic_seeg, return to CaseA -> MUST stay 'researcher'!
        cdp.eval(
            "(() => {"
            "  document.querySelector('.view-tab[data-view=\"reader\"]').click();"
            "})()"
        )
        for _ in range(50):
            if cdp.eval("Boolean(document.getElementById('vConfirmBtn') && !document.getElementById('vConfirmBtn').disabled)"):
                break
            time.sleep(0.1)
        cdp.eval(
            "(() => {"
            "  const rSel = document.getElementById('authorRoleSelect');"
            "  const aInp = document.getElementById('authorInput');"
            "  aInp.value = 'Res. Dana';"
            "  aInp.dispatchEvent(new Event('input', { bubbles: true }));"
            "  rSel.value = 'researcher';"
            "  rSel.dispatchEvent(new Event('change', { bubbles: true }));"
            "})()"
        )
        cdp.eval("document.getElementById('vConfirmBtn').click()")
        for _ in range(40):
            if len(WorkbenchStore(ws).load_judgments("CaseA")) >= 1:
                break
            time.sleep(0.1)
        sum1 = WorkbenchStore(ws).load_run("CaseA", "CaseA", "baseline")["summary"]["counts"]
        assert sum1["researcher_confirmed"] == 1 and sum1["clinician_confirmed"] == 0, sum1

        # Visit synthetic_seeg (auto-switches role to engineering_rehearsal)
        cdp.eval(
            "(() => {"
            "  const sel = document.getElementById('caseSelect');"
            "  sel.value = 'synthetic_seeg';"
            "  sel.dispatchEvent(new Event('change', { bubbles: true }));"
            "})()"
        )
        for _ in range(50):
            if cdp.eval("document.getElementById('caseSelect').value === 'synthetic_seeg' && document.getElementById('authorRoleSelect').value === 'engineering_rehearsal'"):
                break
            time.sleep(0.1)
        assert cdp.eval("document.getElementById('authorRoleSelect').value") == "engineering_rehearsal"

        # Return to CaseA WITHOUT manually editing role -> MUST restore 'researcher', NEVER promote to 'clinician'!
        cdp.eval(
            "(() => {"
            "  const sel = document.getElementById('caseSelect');"
            "  sel.value = 'CaseA';"
            "  sel.dispatchEvent(new Event('change', { bubbles: true }));"
            "})()"
        )
        for _ in range(60):
            if cdp.eval(
                "document.getElementById('caseSelect').value === 'CaseA'"
                " && !document.getElementById('vConfirmBtn').disabled"
                " && (document.getElementById('verdictEventMeta')?.textContent || '').includes('LB4')"
            ):
                break
            time.sleep(0.12)
        role_after = cdp.eval("document.getElementById('authorRoleSelect').value")
        assert role_after == "researcher", f"Researcher was wrongly promoted after visiting synthetic demo: {role_after!r}"

        # Confirm second event (or confirm again) and assert clinician_confirmed stays 0!
        cdp.eval("document.getElementById('vConfirmBtn').click()")
        for _ in range(40):
            if len(WorkbenchStore(ws).load_judgments("CaseA")) >= 2:
                break
            time.sleep(0.1)
        js_a = WorkbenchStore(ws).load_judgments("CaseA")
        assert len(js_a) >= 2 and all(j.get("role") == "researcher" for j in js_a), js_a
        sum2 = WorkbenchStore(ws).load_run("CaseA", "CaseA", "baseline")["summary"]["counts"]
        assert sum2["clinician_confirmed"] == 0 and sum2["researcher_confirmed"] >= 1, sum2

        assert cdp.exceptions == [], f"JS exceptions: {cdp.exceptions}"
        assert cdp.console_errors == [], f"Console errors: {cdp.console_errors}"
    finally:
        if cdp is not None:
            cdp.close()
        chrome.terminate()
        try:
            chrome.wait(timeout=4)
        except Exception:
            chrome.kill()
        srv.stop()



@pytest.mark.skipif(_find_chrome() is None, reason="Google Chrome / Chromium not installed")
def test_cdp_round7_mdt_confirm_updates_kpi_live_and_stale_bookmark_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real Headless Chrome CDP regression for Round 7 MDT review counter live update & stale bookmark guard."""
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r7_cdp"))
    chrome_bin = _find_chrome()
    assert chrome_bin is not None

    ws = _build_two_case_fixture(tmp_path)
    srv = WorkbenchHTTPServer(workspace_dir=ws, port=0)
    base_url = srv.start().rstrip("/")

    user_data = tmp_path / "chrome_r7_profile"
    user_data.mkdir()
    s_tmp = socket.socket()
    s_tmp.bind(("127.0.0.1", 0))
    dbg_port = s_tmp.getsockname()[1]
    s_tmp.close()

    chrome = subprocess.Popen(
        [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--window-size=1280,860",
            f"--remote-debugging-port={dbg_port}",
            f"--user-data-dir={user_data}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cdp: _MiniCDP | None = None
    try:
        ws_url = None
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{dbg_port}/json/list", timeout=1) as r:
                    tabs = json.loads(r.read().decode("utf-8"))
                page = next((t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")), None)
                if page:
                    ws_url = page["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.15)
        assert ws_url, "Headless Chrome did not expose CDP page target"

        cdp = _MiniCDP(ws_url)
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Log.enable")
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&view=sheet"})

        # 1) Wait for CaseA embedded MDT Sheet to connect live
        for _ in range(80):
            ready = cdp.eval(
                "Boolean(document.getElementById('sheetFrame')?.contentDocument?.body?.classList?.contains('live-connected')"
                " && document.getElementById('sheetFrame')?.contentDocument?.querySelector('.fig-card[data-event-id=\"ev_casea_001\"] button[data-verdict=\"confirm\"]'))"
            )
            if ready:
                break
            time.sleep(0.12)
        assert ready is True
        before_kpi = cdp.eval("document.getElementById('sheetFrame').contentDocument.querySelector('.kpi-review')?.textContent || ''")
        assert "0 of 2 focal candidates" in before_kpi and "confirmed" not in before_kpi, before_kpi

        # 2) Click Confirm as clinician on the MDT Sheet -> SAME PAGE KPI MUST update live to 1/2 (1 confirmed)!
        cdp.eval("document.getElementById('sheetFrame').contentDocument.querySelector('.fig-card[data-event-id=\"ev_casea_001\"] button[data-verdict=\"confirm\"]').click()")
        live_kpi = ""
        for _ in range(60):
            live_kpi = cdp.eval("document.getElementById('sheetFrame').contentDocument.querySelector('.kpi-review')?.textContent || ''") or ""
            if "1 of 2 focal candidates" in live_kpi and "1 confirmed" in live_kpi:
                break
            time.sleep(0.1)
        assert "1 of 2 focal candidates" in live_kpi and "1 confirmed" in live_kpi, f"Same-page MDT KPI stale: {live_kpi!r}"

        # 3) Fresh re-navigation to View 3 MUST ALSO show 1 of 2 focal candidates (1 confirmed)!
        cdp.send("Page.navigate", {"url": "about:blank"})
        time.sleep(0.2)
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&view=sheet"})
        re_kpi = ""
        for _ in range(80):
            re_kpi = cdp.eval(
                "document.getElementById('sheetFrame')?.contentDocument?.body?.classList?.contains('live-connected')"
                " ? (document.getElementById('sheetFrame').contentDocument.querySelector('.kpi-review')?.textContent || '') : ''"
            ) or ""
            if "1 of 2 focal candidates" in re_kpi and "1 confirmed" in re_kpi:
                break
            time.sleep(0.12)
        assert "1 of 2 focal candidates" in re_kpi and "1 confirmed" in re_kpi, f"Re-navigated MDT KPI stale: {re_kpi!r}"

        # 4) Direct bookmark to /exports/... AFTER baseline overwrite (2 -> 0) MUST open strictly offline-static (NOT live)!
        analyze_case("CaseA/CaseA", workspace_dir=ws, run_name="baseline", overwrite=True, overrides={"detector_enabled": False})
        cdp.send("Page.navigate", {"url": f"{base_url}/exports/CaseA/CaseA_baseline_case_sheet.html"})
        bm_state: dict[str, Any] = {}
        for _ in range(50):
            bm_state = cdp.eval(
                "({offline: document.body?.classList?.contains('offline-static'), live: document.body?.classList?.contains('live-connected'), pill: document.getElementById('offlineOpenLink')?.textContent?.trim() || ''})"
            ) or {}
            if bm_state.get("offline"):
                break
            time.sleep(0.1)
        assert bm_state.get("offline") is True and bm_state.get("live") is False, bm_state
        assert "Outdated Snapshot" in bm_state.get("pill", ""), bm_state

        assert cdp.exceptions == [], f"JS exceptions: {cdp.exceptions}"
        assert cdp.console_errors == [], f"Console errors: {cdp.console_errors}"
    finally:
        if cdp is not None:
            cdp.close()
        chrome.terminate()
        try:
            chrome.wait(timeout=4)
        except Exception:
            chrome.kill()
        srv.stop()



@pytest.mark.skipif(_find_chrome() is None, reason="Google Chrome / Chromium not installed")
def test_cdp_round8_live_sheet_auto_rebuilds_on_overwrite_overlay_and_unclip_bookmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real Headless Chrome CDP regression for Round 8 overlay invalidation, already-open sheet overwrite sync, & unclip bookmark."""
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r8_cdp"))
    chrome_bin = _find_chrome()
    assert chrome_bin is not None

    ws = _build_two_case_fixture(tmp_path)
    srv = WorkbenchHTTPServer(workspace_dir=ws, port=0)
    base_url = srv.start().rstrip("/")

    user_data = tmp_path / "chrome_r8_profile"
    user_data.mkdir()
    s_tmp = socket.socket()
    s_tmp.bind(("127.0.0.1", 0))
    dbg_port = s_tmp.getsockname()[1]
    s_tmp.close()

    chrome = subprocess.Popen(
        [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--window-size=1280,860",
            f"--remote-debugging-port={dbg_port}",
            f"--user-data-dir={user_data}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cdp: _MiniCDP | None = None
    try:
        ws_url = None
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{dbg_port}/json/list", timeout=1) as r:
                    tabs = json.loads(r.read().decode("utf-8"))
                page = next((t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")), None)
                if page:
                    ws_url = page["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.15)
        assert ws_url, "Headless Chrome did not expose CDP page target"

        cdp = _MiniCDP(ws_url)
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Log.enable")

        def _set_overlay(soz: list[str], src: str) -> None:
            req = urllib.request.Request(
                f"{base_url}/api/overlay",
                data=json.dumps({"case": "CaseA", "source": src, "soz_channels": soz}).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-BrainLens-Token": srv.auth_token},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                assert r.status == 200

        # 1) Set SOZ=LB4, open View 3 -> shows LB4 & 2 focal events
        _set_overlay(["LB4"], "SOZ rev 1")
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&view=sheet"})
        for _ in range(80):
            ok1 = cdp.eval(
                "Boolean(document.getElementById('sheetFrame')?.contentDocument?.body?.classList?.contains('live-connected')"
                " && (document.getElementById('sheetFrame').contentDocument.body.textContent || '').includes('SOZ rev 1'))"
            )
            if ok1:
                break
            time.sleep(0.12)
        assert ok1 is True

        # 2) Update SOZ -> LA1 while View 3 is open -> MUST auto-rebuild to show SOZ rev 2 / LA1!
        _set_overlay(["LA1"], "SOZ rev 2")
        ok2 = False
        for _ in range(80):
            ok2 = cdp.eval(
                "Boolean(document.getElementById('sheetFrame')?.contentDocument?.body?.classList?.contains('live-connected')"
                " && (document.getElementById('sheetFrame').contentDocument.body.textContent || '').includes('SOZ rev 2'))"
            )
            if ok2:
                break
            time.sleep(0.12)
        assert ok2 is True, "Already-open View 3 did not auto-refresh after /api/overlay update"

        # 3) Overwrite CaseA baseline (2 -> 0 events) in Python while View 3 stays open ->
        #    MUST detect overwrite via SSE watcher & auto-rebuild to 0 focal events!
        analyze_case("CaseA/CaseA", workspace_dir=ws, run_name="baseline", overwrite=True, overrides={"detector_enabled": False})
        ok3 = False
        for _ in range(80):
            ok3 = cdp.eval(
                "Boolean(document.getElementById('sheetFrame')?.contentDocument?.body?.classList?.contains('live-connected')"
                " && document.getElementById('sheetFrame').contentDocument.querySelector('.big.accent')?.textContent?.trim() === '0')"
            )
            if ok3:
                break
            time.sleep(0.12)
        assert ok3 is True, "Already-open View 3 stayed stale with 2 events after 2->0 overwrite!"

        # 4) Unclip bookmark on CaseB: clip -> export -> unclip -> direct bookmark MUST open offline-static (NOT live)!
        from brainlens.case_sheet import export_case_sheet
        srv._save_judgment({
            "case": "CaseB",
            "recording": "CaseB",
            "run_name": "baseline",
            "event_id": "ev_CaseB_001",
            "channel": "RA1",
            "role": "clinician",
            "author": "Dr. B",
            "clipped": True,
        })
        export_case_sheet("CaseB", workspace_dir=ws, primary_run="baseline")
        srv._save_judgment({
            "case": "CaseB",
            "recording": "CaseB",
            "run_name": "baseline",
            "event_id": "ev_CaseB_001",
            "channel": "RA1",
            "role": "clinician",
            "author": "Dr. B",
            "clipped": False,
        })
        cdp.send("Page.navigate", {"url": f"{base_url}/exports/CaseB/CaseB_baseline_case_sheet.html"})
        bm_u: dict[str, Any] = {}
        for _ in range(50):
            bm_u = cdp.eval(
                "({offline: document.body?.classList?.contains('offline-static'), live: document.body?.classList?.contains('live-connected'), pill: document.getElementById('offlineOpenLink')?.textContent?.trim() || ''})"
            ) or {}
            if bm_u.get("offline"):
                break
            time.sleep(0.1)
        assert bm_u.get("offline") is True and bm_u.get("live") is False, bm_u
        assert "Outdated Snapshot" in bm_u.get("pill", ""), bm_u

        assert cdp.exceptions == [], f"JS exceptions: {cdp.exceptions}"
        assert cdp.console_errors == [], f"Console errors: {cdp.console_errors}"
    finally:
        if cdp is not None:
            cdp.close()
        chrome.terminate()
        try:
            chrome.wait(timeout=4)
        except Exception:
            chrome.kill()
        srv.stop()



def test_cdp_round9_stale_sheet_click_rejected_409_and_interval_clip_jump_centers_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chrome_bin = _find_chrome()
    if not chrome_bin:
        pytest.skip("Headless Chrome/Chromium binary not available")

    from brainlens.case_sheet import export_case_sheet
    from brainlens.pipeline import refresh_verdict_tallies

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r9_cdp"))
    ws = _build_two_case_fixture(tmp_path)
    store = WorkbenchStore(ws)

    # Give CaseB a real event at 4.0s on RA1 so jumping from a [10, 14]s interval clip
    # MUST NOT fall back to selecting the 4.0s event!
    rdir_b = store.run_generation_dir("CaseB", "CaseB", "baseline")
    meta_b = read_json(rdir_b / "run_meta.json")
    sum_b = read_json(rdir_b / "summary.json")
    ra1 = next(c for c in sum_b["channels"] if c["channel_name"] == "RA1")
    det_b = read_json(Path(meta_b["detection_json_path"]))
    det_b["status"] = "complete"
    det_b["valid_intervals"] = {c["channel_id"]: [[0, 1024 * 20]] for c in sum_b["channels"]}
    det_b["events"] = [
        {
            "id": "ev_caseb_4s",
            "channel_id": ra1["channel_id"],
            "channel_name": "RA1",
            "shaft": "RA",
            "start_sample": int(4.0 * 1024),
            "end_sample": int(4.08 * 1024),
            "start_seconds": 4.0,
            "end_seconds": 4.08,
            "scores": {},
            "labels": {},
        }
    ]
    atomic_write_json(Path(meta_b["detection_json_path"]), det_b)
    clf_b = read_json(Path(meta_b["classification_json_path"]))
    clf_b["events"] = det_b["events"]
    atomic_write_json(Path(meta_b["classification_json_path"]), clf_b)
    refresh_verdict_tallies(store, "CaseB", "CaseB", "baseline")

    store.save_clinical_overlay("CaseA", {"source": "SOZ_v1", "soz_channels": ["LB4"]})
    export_case_sheet("CaseA", workspace_dir=ws, primary_run="baseline")

    srv = WorkbenchHTTPServer(workspace_dir=ws, host="127.0.0.1", port=0)
    base_url = srv.start().rstrip("/")
    user_data = tmp_path / "chrome_r9_profile"
    user_data.mkdir()
    s_tmp = socket.socket()
    s_tmp.bind(("127.0.0.1", 0))
    dbg_port = s_tmp.getsockname()[1]
    s_tmp.close()

    chrome = subprocess.Popen(
        [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--window-size=1440,960",
            f"--remote-debugging-port={dbg_port}",
            f"--user-data-dir={user_data}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cdp: _MiniCDP | None = None
    try:
        ws_url = None
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{dbg_port}/json/list", timeout=1) as r:
                    tabs = json.loads(r.read().decode("utf-8"))
                page = next((t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")), None)
                if page:
                    ws_url = page["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.15)
        assert ws_url, "Headless Chrome did not expose CDP page target"

        cdp = _MiniCDP(ws_url)
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Log.enable")

        # (A) Stale MDT sheet click while /api/sheet-freshness is delayed:
        #     Mutate SOZ LB4 -> LA1, click Confirm on old open sheet -> server rejects 409,
        #     0 clinician verdicts persisted, and page transitions to offline-static!
        cdp.send("Page.navigate", {"url": f"{base_url}/exports/CaseA/CaseA_baseline_case_sheet.html"})
        for _ in range(60):
            if cdp.eval("Boolean(document.body?.classList?.contains('live-connected') && document.querySelector('[data-act=\"verdict\"][data-verdict=\"confirm\"]'))"):
                break
            time.sleep(0.1)
        assert cdp.eval("document.body.classList.contains('live-connected')") is True

        # Stall /api/sheet-freshness in Chrome so the page STILL thinks it's live when SOZ changes
        cdp.send("Fetch.enable", {"patterns": [{"urlPattern": "*/api/sheet-freshness*"}]})
        req_ov = urllib.request.Request(
            f"{base_url}/api/overlay",
            data=json.dumps({"case": "CaseA", "source": "SOZ_v2", "soz_channels": ["LA1"]}).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-BrainLens-Token": srv.auth_token},
            method="POST",
        )
        with urllib.request.urlopen(req_ov, timeout=5) as r_ov:
            assert r_ov.status == 200
        time.sleep(0.2)
        # Click Confirm on the now-stale page!
        cdp.eval("document.querySelector('[data-act=\"verdict\"][data-verdict=\"confirm\"]').click()")
        became_offline = False
        for _ in range(50):
            if cdp.eval("document.body.classList.contains('offline-static') && !document.body.classList.contains('live-connected')"):
                became_offline = True
                break
            time.sleep(0.1)
        assert became_offline is True, "Stale sheet did not lock out after HTTP 409 rejection!"
        assert store.load_judgments("CaseA") == [], f"Stale sheet wrote verdict: {store.load_judgments('CaseA')}"
        cdp.send("Fetch.disable")

        # (B) Clipped intervals [10, 14] and [11, 13] on CaseB:
        #     View 3 MUST render 2 distinct interval cards, and clicking Waveforms on [10, 14]
        #     MUST center Reader around 12s covering [10, 14] (NOT jump to the 4s event)!
        for idx_iv, iv in enumerate(([10.0, 14.0], [11.0, 13.0])):
            srv._save_judgment({
                "case": "CaseB",
                "operation_id": f"cdp_iv_{idx_iv}",
                "recording": "CaseB",
                "run_name": "baseline",
                "channel": "RA1",
                "center_seconds": 12.0,
                "interval_seconds": iv,
                "clipped": True,
                "role": "clinician",
                "author": "Dr. Interval",
                "note": f"Selected interval {iv}",
            })
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseB&recording=CaseB&run=baseline&view=sheet"})
        cards_info: list[dict[str, Any]] = []
        for _ in range(80):
            cards_info = cdp.eval(
                "(() => {"
                "  const fr = document.getElementById('sheetFrame');"
                "  if (!fr || !fr.contentDocument) return [];"
                "  return [...fr.contentDocument.querySelectorAll('.fig-card[data-start-seconds]')]"
                "    .filter((c) => c.getAttribute('data-start-seconds'))"
                "    .map((c) => ({s0: Number(c.getAttribute('data-start-seconds')), s1: Number(c.getAttribute('data-end-seconds')), cap: c.querySelector('.fig-cap')?.textContent?.trim() || ''}));"
                "})()"
            ) or []
            if len(cards_info) >= 2 and any(abs(c["s0"] - 10.0) < 0.05 and abs(c["s1"] - 14.0) < 0.05 for c in cards_info):
                break
            time.sleep(0.15)
        assert any(abs(c["s0"] - 10.0) < 0.05 and abs(c["s1"] - 14.0) < 0.05 for c in cards_info), cards_info
        assert any(abs(c["s0"] - 11.0) < 0.05 and abs(c["s1"] - 13.0) < 0.05 for c in cards_info), cards_info

        # Click Waveforms on the [10, 14] card inside the View 3 iframe!
        clicked = cdp.eval(
            "(() => {"
            "  const fr = document.getElementById('sheetFrame');"
            "  const card = [...fr.contentDocument.querySelectorAll('.fig-card[data-start-seconds]')]"
            "    .find((c) => Math.abs(Number(c.getAttribute('data-start-seconds')) - 10) < 0.05 && Math.abs(Number(c.getAttribute('data-end-seconds')) - 14) < 0.05);"
            "  if (!card) return false;"
            "  card.querySelector('[data-act=\"jump_waveform\"]').click();"
            "  return true;"
            "})()"
        )
        assert clicked is True
        reader_state: dict[str, Any] = {}
        for _ in range(60):
            reader_state = cdp.eval(
                "({"
                "  isReader: document.getElementById('view-reader')?.classList?.contains('active') || false,"
                "  start: Number(document.getElementById('readerStartInput')?.value || 0),"
                "  dur: Number(document.getElementById('readerDurSelect')?.value || 0),"
                "  meta: document.getElementById('verdictEventMeta')?.textContent?.trim() || '',"
                "  badge: document.getElementById('syncClassBadge')?.textContent?.trim() || ''"
                "})"
            ) or {}
            if reader_state.get("isReader") and "[10.00," in reader_state.get("meta", ""):
                break
            time.sleep(0.12)
        assert reader_state.get("isReader") is True, reader_state
        v_start = float(reader_state["start"])
        v_end = v_start + float(reader_state["dur"])
        assert v_start <= 10.01 and v_end >= 13.99, f"Reader viewport [{v_start}, {v_end}] missed [10, 14]: {reader_state}"
        assert "[10.00, 14.00]s" in reader_state["meta"] or "[10.00, 13.99" in reader_state["meta"], reader_state
        assert reader_state.get("badge") == "WINDOW", reader_state

        assert cdp.exceptions == [], f"JS exceptions: {cdp.exceptions}"
        # Allow the intentional HTTP 409 network response console message during stale-sheet click test
        non_409_errs = [m for m in cdp.console_errors if "409" not in m]
        assert non_409_errs == [], f"Console errors: {non_409_errs}"
    finally:
        if cdp is not None:
            cdp.close()
        chrome.terminate()
        try:
            chrome.wait(timeout=4)
        except Exception:
            chrome.kill()
        srv.stop()


def test_cdp_round10_late_verdict_save_preserves_case_switch_and_retracted_note_locks_bookmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chrome_bin = _find_chrome()
    if not chrome_bin:
        pytest.skip("Headless Chrome/Chromium binary not available")

    import brainlens.workbench_http as http_mod
    from brainlens.case_sheet import export_case_sheet

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r10_cdp"))
    ws = _build_two_case_fixture(tmp_path)
    store = WorkbenchStore(ws)

    srv = WorkbenchHTTPServer(workspace_dir=ws, host="127.0.0.1", port=0)

    # Deterministically stall CaseA's /api/judgment AND CaseB's /api/bootstrap so CaseA's save response
    # arrives AFTER the user switched to CaseB while CaseB's bootstrap is still in-flight!
    orig_save = srv._save_judgment
    orig_boot = srv._bootstrap
    save_in_flight = [False]
    boot_b_started = [False]

    def _delayed_save(body: dict[str, Any]) -> dict[str, Any]:
        if body.get("case") == "CaseA" and body.get("verdict") == "confirm":
            save_in_flight[0] = True
            # Wait up to 3s for CaseB bootstrap to begin so save resolves mid-switch!
            t_end = time.monotonic() + 3.0
            while not boot_b_started[0] and time.monotonic() < t_end:
                time.sleep(0.03)
        return orig_save(body)

    def _delayed_boot(qs: dict[str, str]) -> dict[str, Any]:
        if qs.get("case") == "CaseB" and save_in_flight[0]:
            boot_b_started[0] = True
            time.sleep(0.6)
        return orig_boot(qs)

    monkeypatch.setattr(srv, "_save_judgment", _delayed_save)
    monkeypatch.setattr(srv, "_bootstrap", _delayed_boot)

    base_url = srv.start().rstrip("/")
    user_data = tmp_path / "chrome_r10_profile"
    user_data.mkdir()
    s_tmp = socket.socket()
    s_tmp.bind(("127.0.0.1", 0))
    dbg_port = s_tmp.getsockname()[1]
    s_tmp.close()

    chrome = subprocess.Popen(
        [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--window-size=1440,960",
            f"--remote-debugging-port={dbg_port}",
            f"--user-data-dir={user_data}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cdp: _MiniCDP | None = None
    try:
        ws_url = None
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{dbg_port}/json/list", timeout=1) as r:
                    tabs = json.loads(r.read().decode("utf-8"))
                page = next((t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")), None)
                if page:
                    ws_url = page["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.15)
        assert ws_url, "Headless Chrome did not expose CDP page target"

        cdp = _MiniCDP(ws_url)
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Log.enable")

        # (A) Late save response MUST NOT undo user's CaseA -> CaseB switch!
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&recording=CaseA&run=baseline&view=reader"})
        for _ in range(60):
            if cdp.eval(
                "document.getElementById('caseSelect')?.value === 'CaseA'"
                " && !document.getElementById('vConfirmBtn')?.disabled"
                " && (document.getElementById('verdictEventMeta')?.textContent || '').includes('LB4')"
                " && (document.getElementById('minimapReadout')?.textContent || '').includes('/')"
            ):
                break
            time.sleep(0.1)
        time.sleep(0.15)

        # Click Confirm on CaseA (fires delayed POST /api/judgment), then immediately switch to CaseB!
        cdp.eval("document.getElementById('vConfirmBtn').click()")
        for _ in range(40):
            if save_in_flight[0]:
                break
            time.sleep(0.05)
        assert save_in_flight[0] is True

        cdp.eval(
            "(() => {"
            "  const sel = document.getElementById('caseSelect');"
            "  sel.value = 'CaseB';"
            "  sel.dispatchEvent(new Event('change', { bubbles: true }));"
            "})()"
        )
        # Wait for BOTH CaseA save and CaseB bootstrap to finish, then verify CaseB WON!
        time.sleep(1.2)
        final_state = cdp.eval(
            "({"
            "  selCase: document.getElementById('caseSelect')?.value || '',"
            "  meta: document.getElementById('verdictEventMeta')?.textContent?.trim() || ''"
            "})"
        )
        assert boot_b_started[0] is True
        assert final_state["selCase"] == "CaseB", f"Late CaseA save undid CaseB switch: {final_state}"
        assert "LB4" not in final_state["meta"], f"CaseA event leaked onto CaseB screen: {final_state}"
        assert store.load_ui_state().get("case") == "CaseB", store.load_ui_state()
        assert len(store.load_judgments("CaseA")) == 1 and len(store.load_judgments("CaseB")) == 0

        # (B) Clinical note retraction MUST lock direct MDT sheet bookmark strictly offline-static!
        orig_save({
            "case": "CaseA",
            "operation_id": "cdp_r10_n1",
            "recording": "CaseA",
            "run_name": "baseline",
            "channel": "LB4",
            "note": "Initial right onset impression",
            "role": "clinician",
            "author": "Dr. A",
        })
        export_case_sheet("CaseA", workspace_dir=ws, primary_run="baseline")
        cdp.send("Page.navigate", {"url": f"{base_url}/exports/CaseA/CaseA_baseline_case_sheet.html"})
        for _ in range(50):
            if cdp.eval("document.body?.classList?.contains('live-connected')"):
                break
            time.sleep(0.1)
        assert cdp.eval("document.body.classList.contains('live-connected')") is True

        orig_save({
            "case": "CaseA",
            "operation_id": "cdp_r10_n2",
            "recording": "CaseA",
            "run_name": "baseline",
            "channel": "LB4",
            "note": "WITHDRAW earlier lateralization",
            "role": "clinician",
            "author": "Dr. A",
        })
        cdp.send("Page.navigate", {"url": f"{base_url}/exports/CaseA/CaseA_baseline_case_sheet.html"})
        bm_st: dict[str, Any] = {}
        for _ in range(50):
            bm_st = cdp.eval(
                "({offline: document.body?.classList?.contains('offline-static'), live: document.body?.classList?.contains('live-connected')})"
            ) or {}
            if bm_st.get("offline"):
                break
            time.sleep(0.1)
        assert bm_st.get("offline") is True and bm_st.get("live") is False, bm_st

        assert cdp.exceptions == [], f"JS exceptions: {cdp.exceptions}"
        assert cdp.console_errors == [], f"Console errors: {cdp.console_errors}"
    finally:
        if cdp is not None:
            cdp.close()
        chrome.terminate()
        try:
            chrome.wait(timeout=4)
        except Exception:
            chrome.kill()
        srv.stop()


def test_cdp_round11_case_switch_blocks_stale_overlay_and_question_and_allows_case_b_vote_while_case_a_saves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chrome_bin = _find_chrome()
    if not chrome_bin:
        pytest.skip("Headless Chrome/Chromium binary not available")

    import threading

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r11_cdp"))
    ws = _build_two_case_fixture(tmp_path)
    store = WorkbenchStore(ws)

    # Seed CaseA with an External Clinical Overlay; leave CaseB with NO overlay
    store.save_clinical_overlay(
        "CaseA",
        {
            "source": "CaseA Video-EEG Conference",
            "soz_channels": ["LB4"],
            "resected_channels": [],
            "notes": "CaseA-only SOZ note",
        },
    )

    srv = WorkbenchHTTPServer(workspace_dir=ws, host="127.0.0.1", port=0)
    orig_save = srv._save_judgment
    case_a_entered = threading.Event()
    release_case_a = threading.Event()

    def _gated_save(body: dict[str, Any]) -> dict[str, Any]:
        if body.get("case") == "CaseA" and body.get("verdict") == "confirm":
            case_a_entered.set()
            release_case_a.wait(timeout=8.0)
        return orig_save(body)

    monkeypatch.setattr(srv, "_save_judgment", _gated_save)
    base_url = srv.start().rstrip("/")

    user_data = tmp_path / "chrome_r11_profile"
    user_data.mkdir()
    s_tmp = socket.socket()
    s_tmp.bind(("127.0.0.1", 0))
    dbg_port = s_tmp.getsockname()[1]
    s_tmp.close()

    chrome = subprocess.Popen(
        [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--window-size=1440,960",
            f"--remote-debugging-port={dbg_port}",
            f"--user-data-dir={user_data}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cdp: _MiniCDP | None = None
    try:
        ws_url = None
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{dbg_port}/json/list", timeout=1) as r:
                    tabs = json.loads(r.read().decode("utf-8"))
                page = next((t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")), None)
                if page:
                    ws_url = page["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.15)
        assert ws_url, "Headless Chrome did not expose CDP page target"

        cdp = _MiniCDP(ws_url)
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Log.enable")

        # (A) SOZ Overlay modal MUST close & unbind on CaseA -> CaseB switch and block forced stale save!
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&recording=CaseA&run=baseline&view=sheet"})
        for _ in range(60):
            if cdp.eval("document.getElementById('caseSelect')?.value === 'CaseA' && !!document.getElementById('openOverlayBtn')"):
                break
            time.sleep(0.1)

        cdp.eval("document.getElementById('openOverlayBtn').click()")
        ov_open = cdp.eval(
            "({"
            "  hidden: document.getElementById('overlayModal').classList.contains('hidden'),"
            "  src: document.getElementById('ovSourceInput').value,"
            "  soz: document.getElementById('ovSozInput').value,"
            "  boundCase: document.getElementById('overlayModal').dataset.boundCase || ''"
            "})"
        )
        assert ov_open == {"hidden": False, "src": "CaseA Video-EEG Conference", "soz": "LB4", "boundCase": "CaseA"}, ov_open

        # Switch to CaseB while overlay modal is open!
        cdp.eval(
            "(() => {"
            "  const sel = document.getElementById('caseSelect');"
            "  sel.value = 'CaseB';"
            "  sel.dispatchEvent(new Event('change', { bubbles: true }));"
            "})()"
        )
        for _ in range(60):
            if cdp.eval("document.getElementById('caseSelect')?.value === 'CaseB'"):
                break
            time.sleep(0.05)
        time.sleep(0.4)
        ov_after = cdp.eval(
            "({"
            "  hidden: document.getElementById('overlayModal').classList.contains('hidden'),"
            "  boundCase: document.getElementById('overlayModal').dataset.boundCase || ''"
            "})"
        )
        assert ov_after == {"hidden": True, "boundCase": ""}, f"Overlay modal stayed open/bound across switch: {ov_after}"
        # Even force-clicking #saveOverlayBtn directly MUST NOT write CaseA's overlay onto CaseB!
        cdp.eval("document.getElementById('saveOverlayBtn').click()")
        time.sleep(0.35)
        assert store.load_clinical_overlay("CaseB") is None, f"CaseA overlay leaked to CaseB: {store.load_clinical_overlay('CaseB')}"

        # (B) #questionInput unsaved draft on CaseA MUST NOT leak into CaseB via Rebuild!
        cdp.eval(
            "(() => {"
            "  const sel = document.getElementById('caseSelect');"
            "  sel.value = 'CaseA';"
            "  sel.dispatchEvent(new Event('change', { bubbles: true }));"
            "})()"
        )
        time.sleep(0.5)
        cdp.eval("document.getElementById('questionInput').value = 'CaseA unsaved MDT question?'")
        cdp.eval(
            "(() => {"
            "  const sel = document.getElementById('caseSelect');"
            "  sel.value = 'CaseB';"
            "  sel.dispatchEvent(new Event('change', { bubbles: true }));"
            "  document.getElementById('rebuildSheetBtn').click();"
            "})()"
        )
        time.sleep(0.8)
        # Also click Rebuild again once CaseB finishes loading!
        cdp.eval("document.getElementById('rebuildSheetBtn').click()")
        time.sleep(0.8)
        assert store.load_case("CaseB").get("open_questions", []) == [], (
            f"CaseA question leaked into CaseB: {store.load_case('CaseB')}"
        )
        assert store.load_case("CaseA").get("open_questions", []) == []

        # (C) Clicking CaseB's enabled Confirm while CaseA's Confirm POST is STILL in flight MUST persist immediately!
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&recording=CaseA&run=baseline&view=reader"})
        for _ in range(60):
            if cdp.eval(
                "document.getElementById('caseSelect')?.value === 'CaseA'"
                " && !document.getElementById('vConfirmBtn')?.disabled"
                " && (document.getElementById('verdictEventMeta')?.textContent || '').includes('LB4')"
                " && (document.getElementById('minimapReadout')?.textContent || '').includes('/')"
            ):
                break
            time.sleep(0.1)
        time.sleep(0.15)

        cdp.eval("document.getElementById('vConfirmBtn').click()")
        assert case_a_entered.wait(timeout=5.0), "CaseA confirm POST did not start"

        # Switch to CaseB while CaseA's POST is STILL blocked on the server!
        cdp.eval(
            "(() => {"
            "  const sel = document.getElementById('caseSelect');"
            "  sel.value = 'CaseB';"
            "  sel.dispatchEvent(new Event('change', { bubbles: true }));"
            "})()"
        )
        for _ in range(60):
            if cdp.eval(
                "document.getElementById('caseSelect')?.value === 'CaseB'"
                " && !document.getElementById('vConfirmBtn')?.disabled"
                " && (document.getElementById('verdictEventMeta')?.textContent || '').includes('RA1')"
            ):
                break
            time.sleep(0.1)

        assert not release_case_a.is_set()
        cdp.eval("document.getElementById('vConfirmBtn').click()")
        for _ in range(50):
            if len(store.load_judgments("CaseB")) >= 1:
                break
            time.sleep(0.1)
        assert len(store.load_judgments("CaseB")) == 1, (
            f"CaseB Confirm click was silently dropped while CaseA save was in flight: {store.load_judgments('CaseB')}"
        )
        assert store.load_judgments("CaseB")[0]["verdict"] == "confirm"
        assert store.load_judgments("CaseB")[0]["channel"] == "RA1"

        # Now release CaseA's slow POST and verify BOTH judgments exist & UI stays cleanly on CaseB!
        release_case_a.set()
        for _ in range(50):
            if len(store.load_judgments("CaseA")) >= 1:
                break
            time.sleep(0.1)
        time.sleep(0.4)
        assert len(store.load_judgments("CaseA")) == 1
        assert len(store.load_judgments("CaseB")) == 1
        assert cdp.eval("document.getElementById('caseSelect')?.value") == "CaseB"
        assert cdp.exceptions == [], f"JS exceptions: {cdp.exceptions}"
        assert cdp.console_errors == [], f"Console errors: {cdp.console_errors}"
    finally:
        release_case_a.set()
        if cdp is not None:
            cdp.close()
        chrome.terminate()
        try:
            chrome.wait(timeout=4)
        except Exception:
            chrome.kill()
        srv.stop()


def test_cdp_round12_failed_case_switch_blocks_overlay_and_next_event_isolates_note_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chrome_bin = _find_chrome()
    if not chrome_bin:
        pytest.skip("Headless Chrome/Chromium binary not available")

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r12_cdp"))
    ws = _build_two_case_fixture(tmp_path)
    store = WorkbenchStore(ws)
    store.save_clinical_overlay(
        "CaseA",
        {
            "source": "CaseA Video-EEG Conference",
            "soz_channels": ["LB4"],
            "resected_channels": [],
            "notes": "CaseA-only SOZ note",
        },
    )

    srv = WorkbenchHTTPServer(workspace_dir=ws, host="127.0.0.1", port=0)
    orig_boot = srv._bootstrap
    failed_once = [False]

    def _flaky_boot(qs: dict[str, str]) -> dict[str, Any]:
        if qs.get("case") == "CaseB" and not failed_once[0]:
            failed_once[0] = True
            raise RuntimeError("Simulated transient bootstrap I/O error on CaseB")
        return orig_boot(qs)

    monkeypatch.setattr(srv, "_bootstrap", _flaky_boot)
    base_url = srv.start().rstrip("/")

    user_data = tmp_path / "chrome_r12_profile"
    user_data.mkdir()
    s_tmp = socket.socket()
    s_tmp.bind(("127.0.0.1", 0))
    dbg_port = s_tmp.getsockname()[1]
    s_tmp.close()

    chrome = subprocess.Popen(
        [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--window-size=1440,960",
            f"--remote-debugging-port={dbg_port}",
            f"--user-data-dir={user_data}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cdp: _MiniCDP | None = None
    try:
        ws_url = None
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{dbg_port}/json/list", timeout=1) as r:
                    tabs = json.loads(r.read().decode("utf-8"))
                page = next((t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")), None)
                if page:
                    ws_url = page["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.15)
        assert ws_url, "Headless Chrome did not expose CDP page target"

        cdp = _MiniCDP(ws_url)
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Log.enable")

        # (A) P0 #2: Transient bootstrap failure on CaseA -> CaseB MUST NOT allow CaseA's overlay to open or save onto CaseB!
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&recording=CaseA&run=baseline&view=sheet"})
        for _ in range(60):
            if cdp.eval("document.getElementById('caseSelect')?.value === 'CaseA' && !!document.getElementById('openOverlayBtn')"):
                break
            time.sleep(0.1)

        cdp.eval(
            "(() => {"
            "  const sel = document.getElementById('caseSelect');"
            "  sel.value = 'CaseB';"
            "  sel.dispatchEvent(new Event('change', { bubbles: true }));"
            "})()"
        )
        for _ in range(40):
            if failed_once[0]:
                break
            time.sleep(0.05)
        time.sleep(0.4)
        assert failed_once[0] is True

        # Try clicking Open Overlay & Save Overlay right after the failed bootstrap!
        cdp.eval("document.getElementById('openOverlayBtn').click()")
        ov_state = cdp.eval(
            "({"
            "  hidden: document.getElementById('overlayModal').classList.contains('hidden'),"
            "  boundCase: document.getElementById('overlayModal').dataset.boundCase || ''"
            "})"
        )
        assert ov_state == {"hidden": True, "boundCase": ""}, f"Overlay opened after failed bootstrap: {ov_state}"
        cdp.eval("document.getElementById('saveOverlayBtn').click()")
        time.sleep(0.35)
        assert store.load_clinical_overlay("CaseB") is None, f"CaseA overlay leaked onto CaseB after failed switch: {store.load_clinical_overlay('CaseB')}"

        # (B) P1 #6: Typing a note on Event 1 (3.000s), clicking Next -> Event 2 (10.000s), then Confirm
        #     MUST NOT attach Event 1's note to Event 2, and Prev MUST restore Event 1's draft!
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&recording=CaseA&run=baseline&view=reader"})
        for _ in range(60):
            if cdp.eval(
                "document.getElementById('caseSelect')?.value === 'CaseA'"
                " && !document.getElementById('vConfirmBtn')?.disabled"
                " && (document.getElementById('verdictEventMeta')?.textContent || '').includes('3.000')"
            ):
                break
            time.sleep(0.1)

        cdp.eval("document.getElementById('noteInput').value = 'Exclusive observation for 3.000s event'")
        cdp.eval("document.getElementById('nextEvBtn').click()")
        for _ in range(60):
            if cdp.eval(
                "!document.getElementById('vConfirmBtn')?.disabled"
                " && (document.getElementById('verdictEventMeta')?.textContent || '').includes('10.000')"
            ):
                break
            time.sleep(0.1)
        note_on_ev2 = cdp.eval("document.getElementById('noteInput').value")
        assert note_on_ev2 == "", f"Event 1 note leaked into Event 2 input box: {note_on_ev2!r}"

        cdp.eval("document.getElementById('vConfirmBtn').click()")
        for _ in range(50):
            if len(store.load_judgments("CaseA")) >= 1:
                break
            time.sleep(0.1)
        j_list = store.load_judgments("CaseA")
        assert len(j_list) == 1 and j_list[0]["event_id"] == "ev_casea_002", j_list
        assert (j_list[0].get("note") or "") == "", f"Event 1 note leaked into Event 2 WAL judgment: {j_list[0]}"

        # Clicking Prev back to Event 1 (3.000s) MUST restore Event 1's unsaved draft!
        cdp.eval("document.getElementById('prevEvBtn').click()")
        for _ in range(60):
            if cdp.eval("(document.getElementById('verdictEventMeta')?.textContent || '').includes('3.000')"):
                break
            time.sleep(0.1)
        restored = cdp.eval("document.getElementById('noteInput').value")
        assert restored == "Exclusive observation for 3.000s event", f"Event 1 draft was lost: {restored!r}"
        assert cdp.exceptions == [], f"JS exceptions: {cdp.exceptions}"
    finally:
        if cdp is not None:
            cdp.close()
        chrome.terminate()
        try:
            chrome.wait(timeout=4)
        except Exception:
            chrome.kill()
        srv.stop()


def test_cdp_round13_failed_case_run_clears_display_blocks_protocol_and_isolates_cursor_and_clip_drafts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chrome_bin = _find_chrome()
    if not chrome_bin:
        pytest.skip("Headless Chrome/Chromium binary not available")

    import threading

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r13_cdp"))
    ws = _build_two_case_fixture(tmp_path)
    store = WorkbenchStore(ws)
    # Seed a second branch 'sensitivity' on CaseA so #runSelect can switch baseline -> sensitivity
    analyze_case(
        "CaseA",
        workspace_dir=ws,
        run_name="sensitivity",
        parent_run="baseline",
        overrides={"detector_enabled": False, "reason": "Sensitivity branch"},
    )

    srv = WorkbenchHTTPServer(workspace_dir=ws, host="127.0.0.1", port=0)
    orig_boot = srv._bootstrap
    orig_save = srv._save_judgment
    fail_case_b_once = [False]
    fail_sens_once = [False]
    stall_e1_entered = threading.Event()
    release_e1 = threading.Event()
    stall_e1_active = [False]

    def _flaky_boot(qs: dict[str, str]) -> dict[str, Any]:
        if qs.get("case") == "CaseB" and not fail_case_b_once[0]:
            fail_case_b_once[0] = True
            raise RuntimeError("Transient CaseB bootstrap error")
        if qs.get("case") == "CaseA" and qs.get("run_name") == "sensitivity" and not fail_sens_once[0]:
            fail_sens_once[0] = True
            raise RuntimeError("Transient sensitivity run bootstrap error")
        return orig_boot(qs)

    def _gated_save(body: dict[str, Any]) -> dict[str, Any]:
        if stall_e1_active[0] and body.get("event_id") == "ev_casea_001":
            stall_e1_entered.set()
            release_e1.wait(timeout=8.0)
        return orig_save(body)

    monkeypatch.setattr(srv, "_bootstrap", _flaky_boot)
    monkeypatch.setattr(srv, "_save_judgment", _gated_save)
    base_url = srv.start().rstrip("/")

    user_data = tmp_path / "chrome_r13_profile"
    user_data.mkdir()
    s_tmp = socket.socket()
    s_tmp.bind(("127.0.0.1", 0))
    dbg_port = s_tmp.getsockname()[1]
    s_tmp.close()

    chrome = subprocess.Popen(
        [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--window-size=1440,960",
            f"--remote-debugging-port={dbg_port}",
            f"--user-data-dir={user_data}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cdp: _MiniCDP | None = None
    try:
        ws_url = None
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{dbg_port}/json/list", timeout=1) as r:
                    tabs = json.loads(r.read().decode("utf-8"))
                page = next((t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")), None)
                if page:
                    ws_url = page["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.15)
        assert ws_url, "Headless Chrome did not expose CDP page target"

        cdp = _MiniCDP(ws_url)
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Log.enable")

        # (A) P0 #3: Failed CaseA -> CaseB switch MUST clear CaseA's canvases & metadata (ZERO stale Patient A display!)
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&recording=CaseA&run=baseline&view=reader"})
        for _ in range(60):
            if cdp.eval(
                "document.getElementById('caseSelect')?.value === 'CaseA'"
                " && !document.getElementById('vConfirmBtn')?.disabled"
                " && (document.getElementById('verdictEventMeta')?.textContent || '').includes('LB4')"
                " && (document.getElementById('minimapReadout')?.textContent || '').includes('/')"
            ):
                break
            time.sleep(0.1)
        time.sleep(0.15)
        before_snap = cdp.eval(
            "({"
            "  mc: document.getElementById('mcCanvas').toDataURL(),"
            "  tr: document.getElementById('triadCanvas').toDataURL(),"
            "  meta: document.getElementById('verdictEventMeta').textContent"
            "})"
        )
        assert "LB4" in before_snap["meta"]

        cdp.eval(
            "(() => {"
            "  const sel = document.getElementById('caseSelect');"
            "  sel.value = 'CaseB';"
            "  sel.dispatchEvent(new Event('change', { bubbles: true }));"
            "})()"
        )
        for _ in range(40):
            if fail_case_b_once[0]:
                break
            time.sleep(0.05)
        time.sleep(0.45)
        after_fail = cdp.eval(
            "({"
            "  mc: document.getElementById('mcCanvas').toDataURL(),"
            "  tr: document.getElementById('triadCanvas').toDataURL(),"
            "  meta: document.getElementById('verdictEventMeta').textContent,"
            "  hdr: document.getElementById('triadHeaderText').textContent,"
            "  disabled: document.getElementById('vConfirmBtn').disabled"
            "})"
        )
        assert after_fail["mc"] != before_snap["mc"], "CaseA mcCanvas pixels stayed visible after failed switch to CaseB"
        assert after_fail["tr"] != before_snap["tr"], "CaseA triadCanvas pixels stayed visible after failed switch to CaseB"
        assert "LB4" not in after_fail["meta"] and "LB4" not in after_fail["hdr"], after_fail
        assert after_fail["disabled"] is True

        # (B) P1 #4: Failed runSelect switch (baseline -> sensitivity) MUST block Save Protocol from exporting baseline!
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&recording=CaseA&run=baseline&view=overview"})
        for _ in range(60):
            if cdp.eval("document.getElementById('runSelect')?.value === 'baseline'"):
                break
            time.sleep(0.1)
        cdp.eval(
            "(() => {"
            "  const sel = document.getElementById('runSelect');"
            "  sel.value = 'sensitivity';"
            "  sel.dispatchEvent(new Event('change', { bubbles: true }));"
            "})()"
        )
        for _ in range(40):
            if fail_sens_once[0]:
                break
            time.sleep(0.05)
        time.sleep(0.4)
        cdp.eval("document.getElementById('saveProtocolBtn').click()")
        assert cdp.eval("document.getElementById('protocolModal').classList.contains('hidden')") is True

        # (C) P1 #5 & #6: Delayed E1 save MUST NOT clear identical text typed on E2!
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&recording=CaseA&run=baseline&view=reader"})
        for _ in range(60):
            if cdp.eval(
                "document.getElementById('caseSelect')?.value === 'CaseA'"
                " && !document.getElementById('vConfirmBtn')?.disabled"
                " && (document.getElementById('verdictEventMeta')?.textContent || '').includes('3.000')"
            ):
                break
            time.sleep(0.1)

        stall_e1_active[0] = True
        cdp.eval("document.getElementById('noteInput').value = 'Shared observation text'")
        cdp.eval("document.getElementById('vConfirmBtn').click()")
        assert stall_e1_entered.wait(timeout=5.0), "E1 confirm did not enter stalled server handler"

        # While E1 save is stalled, move to E2 (10.000s) and type the EXACT SAME text!
        cdp.eval("document.getElementById('nextEvBtn').click()")
        for _ in range(60):
            if cdp.eval("(document.getElementById('verdictEventMeta')?.textContent || '').includes('10.000')"):
                break
            time.sleep(0.1)
        cdp.eval("document.getElementById('noteInput').value = 'Shared observation text'")

        # Now release E1's delayed save and verify E2's note box is NOT wiped!
        release_e1.set()
        for _ in range(50):
            if len(store.load_judgments("CaseA")) >= 1:
                break
            time.sleep(0.1)
        time.sleep(0.5)
        e2_note = cdp.eval("document.getElementById('noteInput').value")
        assert e2_note == "Shared observation text", f"Delayed E1 save wiped E2's draft: {e2_note!r}"
        assert cdp.eval("(document.getElementById('verdictEventMeta')?.textContent || '').includes('10.000')") is True

        # (D) P1 #6a: Saving a dragged Interval Clip MUST NOT wipe E2's unsaved #noteInput draft!
        cdp.eval(
            "(() => {"
            "  const mc = document.getElementById('mcCanvas');"
            "  const r = mc.getBoundingClientRect();"
            "  const y = r.top + 80;"
            "  const x0 = r.left + 160;"
            "  const x1 = r.left + 260;"
            "  mc.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, clientX: x0, clientY: y, shiftKey: true, button: 0 }));"
            "  window.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: x1, clientY: y, shiftKey: true, button: 0 }));"
            "  window.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, clientX: x1, clientY: y, shiftKey: true, button: 0 }));"
            "})()"
        )
        assert cdp.eval("document.getElementById('selectionBar').classList.contains('hidden')") is False
        cdp.eval("document.getElementById('clipSelectionBtn').click()")
        # The JSONL append precedes the response and the client-side bootstrap/triad refresh.
        # The selection bar closes only after saveVerdict has completed that refresh.
        for _ in range(60):
            if cdp.eval("document.getElementById('selectionBar').classList.contains('hidden')"):
                break
            time.sleep(0.1)
        assert cdp.eval("document.getElementById('selectionBar').classList.contains('hidden')"), "Clip refresh did not finish"
        assert any(j.get("interval_seconds") for j in store.load_judgments("CaseA")), "Interval clip was not saved"
        assert cdp.eval("document.getElementById('noteInput').value") == "Shared observation text", (
            "Interval clip wiped unsaved #noteInput event draft!"
        )

        # (E) Use the waveform hover readout to locate two named, event-free lanes under the actual layout.
        def select_cursor(channel: str, fraction: float) -> None:
            point = cdp.eval("""(([channel, fraction]) => {
                const mc = document.getElementById('mcCanvas');
                const rect = mc.getBoundingClientRect();
                const x = Math.round(rect.left + rect.width * fraction);
                for (let y = Math.ceil(rect.top); y < rect.bottom; y += 4) {
                    mc.dispatchEvent(new MouseEvent('mousemove', {clientX: x, clientY: y, bubbles: true}));
                    const [time, lane] = document.getElementById('mcReadoutHud').textContent.split(' · ');
                    if (lane === channel) return {x, y, time: parseFloat(time)};
                }
                return null;
            })(""" + json.dumps([channel, fraction]) + ")")
            assert point is not None, f"Cannot find waveform lane {channel}"
            click = {"x": point["x"], "y": point["y"], "button": "left", "clickCount": 1}
            cdp.send("Input.dispatchMouseEvent", {"type": "mousePressed", **click})
            cdp.send("Input.dispatchMouseEvent", {"type": "mouseReleased", **click})
            for _ in range(60):
                meta = cdp.eval("""(() => {
                    const meta = document.getElementById('verdictEventMeta');
                    return {channel: meta.querySelector('strong')?.textContent,
                            window: meta.querySelector('code')?.textContent, text: meta.textContent,
                            ready: !document.getElementById('vClipBtn').disabled};
                })()""")
                if meta["ready"] and meta.get("channel") == channel and "manual cursor window" in meta["text"]:
                    start, end = map(float, meta["window"].removeprefix("[").removesuffix("]s").split(","))
                    if abs((start + end) / 2 - point["time"]) <= .01:
                        return
                time.sleep(0.1)
            pytest.fail(f"Cursor did not reach {channel} at {point['time']} s: {meta}")

        select_cursor("LB1", 0.33)
        cdp.eval("document.getElementById('noteInput').value = 'Cursor window 1 draft'")
        select_cursor("LB2", 0.55)
        win2_note = cdp.eval("document.getElementById('noteInput').value")
        assert win2_note == "", f"Cursor window 1 draft bled into cursor window 2: {win2_note!r}"
        select_cursor("LB1", 0.33)
        assert cdp.eval("document.getElementById('noteInput').value") == "Cursor window 1 draft"

        assert cdp.exceptions == [], f"JS exceptions: {cdp.exceptions}"
    finally:
        release_e1.set()
        if cdp is not None:
            cdp.close()
        chrome.terminate()
        try:
            chrome.wait(timeout=4)
        except Exception:
            chrome.kill()
        srv.stop()


def test_cdp_round14_failed_switch_clears_overview_sheet_stale_reader_409_and_cursor_note_clip_span(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chrome_bin = _find_chrome()
    if not chrome_bin:
        pytest.skip("Headless Chrome/Chromium binary not available")

    import edfio

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r14_cdp"))
    ws = _build_two_case_fixture(tmp_path)
    store = WorkbenchStore(ws)
    store.save_clinical_overlay(
        "CaseA",
        {
            "source": "Reviewed preconference",
            "soz_channels": ["LB4"],
            "resected_channels": ["LC4"],
            "notes": "CaseA SOZ",
        },
    )

    srv = WorkbenchHTTPServer(workspace_dir=ws, host="127.0.0.1", port=0)
    orig_boot = srv._bootstrap
    fail_case_b_once = [False]

    def _flaky_boot(qs: dict[str, str]) -> dict[str, Any]:
        if qs.get("case") == "CaseB" and not fail_case_b_once[0]:
            fail_case_b_once[0] = True
            raise RuntimeError("Transient CaseB bootstrap error")
        return orig_boot(qs)

    monkeypatch.setattr(srv, "_bootstrap", _flaky_boot)
    base_url = srv.start().rstrip("/")

    user_data = tmp_path / "chrome_r14_profile"
    user_data.mkdir()
    s_tmp = socket.socket()
    s_tmp.bind(("127.0.0.1", 0))
    dbg_port = s_tmp.getsockname()[1]
    s_tmp.close()

    chrome = subprocess.Popen(
        [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--window-size=1440,960",
            f"--remote-debugging-port={dbg_port}",
            f"--user-data-dir={user_data}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cdp: _MiniCDP | None = None
    try:
        ws_url = None
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{dbg_port}/json/list", timeout=1) as r:
                    tabs = json.loads(r.read().decode("utf-8"))
                page = next((t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")), None)
                if page:
                    ws_url = page["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.15)
        assert ws_url, "Headless Chrome did not expose CDP page target"

        cdp = _MiniCDP(ws_url)
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Log.enable")

        # (A) P0 #5: Failed CaseA -> CaseB switch MUST clear Overview timeline/QC AND Sheet SOZ/Resected overlay bar!
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&recording=CaseA&run=baseline&view=overview"})
        for _ in range(60):
            if cdp.eval(
                "document.getElementById('caseSelect')?.value === 'CaseA'"
                " && (document.getElementById('overviewDiagnosticsBox')?.textContent || '').length > 10"
                " && (document.getElementById('overlaySummaryBar')?.textContent || '').includes('LB4')"
            ):
                break
            time.sleep(0.1)
        time.sleep(0.2)
        ov_before = cdp.eval(
            "({"
            "  tl: document.getElementById('overviewTimelineCanvas').toDataURL(),"
            "  diag: document.getElementById('overviewDiagnosticsBox').textContent,"
            "  ovText: document.getElementById('overlaySummaryBar').textContent,"
            "  ovHidden: document.getElementById('overlaySummaryBar').classList.contains('hidden')"
            "})"
        )
        assert "LB4" in ov_before["ovText"] and ov_before["ovHidden"] is False

        cdp.eval(
            "(() => {"
            "  const sel = document.getElementById('caseSelect');"
            "  sel.value = 'CaseB';"
            "  sel.dispatchEvent(new Event('change', { bubbles: true }));"
            "})()"
        )
        for _ in range(40):
            if fail_case_b_once[0]:
                break
            time.sleep(0.05)
        time.sleep(0.45)
        ov_failed = cdp.eval(
            "({"
            "  tl: document.getElementById('overviewTimelineCanvas').toDataURL(),"
            "  diag: document.getElementById('overviewDiagnosticsBox').textContent,"
            "  ovText: document.getElementById('overlaySummaryBar').textContent,"
            "  ovHidden: document.getElementById('overlaySummaryBar').classList.contains('hidden')"
            "})"
        )
        assert ov_failed["ovHidden"] is True and ov_failed["ovText"] == "", f"CaseA SOZ overlay leaked after failed switch: {ov_failed}"
        assert ov_failed["diag"] == "", f"CaseA diagnostics leaked after failed switch: {ov_failed}"
        assert ov_failed["tl"] != ov_before["tl"], "CaseA overviewTimelineCanvas pixels stayed visible after failed switch!"

        # (B) P1 #4a & #4b: Manual cursor Save Observation MUST NOT jump to Event 0, and Clip on widened 1.0s cursor
        #     MUST record the exact 1.0s interval and export a 1.0s evidence card!
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&recording=CaseA&run=baseline&view=reader"})
        for _ in range(60):
            if cdp.eval(
                "document.getElementById('caseSelect')?.value === 'CaseA'"
                " && !document.getElementById('vConfirmBtn')?.disabled"
                " && (document.getElementById('minimapReadout')?.textContent || '').includes('/')"
            ):
                break
            time.sleep(0.1)
        time.sleep(0.2)

        # Click a manual cursor window away from events
        cdp.eval(
            "(() => {"
            "  const mc = document.getElementById('mcCanvas');"
            "  const r = mc.getBoundingClientRect();"
            "  mc.dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: r.left + 210, clientY: r.top + 52, button: 0 }));"
            "})()"
        )
        for _ in range(60):
            if cdp.eval(
                "(document.getElementById('verdictEventMeta')?.textContent || '').includes('manual cursor window')"
                " && !document.getElementById('vClipBtn')?.disabled"
            ):
                break
            time.sleep(0.1)
        cur_meta_before = cdp.eval("document.getElementById('verdictEventMeta')?.textContent || ''")
        assert "manual cursor window" in cur_meta_before, cur_meta_before

        cdp.eval("document.getElementById('noteInput').value = 'Bookmark cursor finding'")
        cdp.eval("document.getElementById('saveNoteOnlyBtn').click()")
        for _ in range(50):
            if len(store.load_judgments("CaseA")) >= 1:
                break
            time.sleep(0.1)
        time.sleep(0.45)
        cur_meta_after = cdp.eval("document.getElementById('verdictEventMeta')?.textContent || ''")
        assert "manual cursor window" in cur_meta_after, f"Save Observation jumped away from manual cursor to: {cur_meta_after!r}"

        # Widen cursor window to 1.0s and click Clip ★ -> MUST archive ~1.0s interval (NOT 0.60s or Event 0!)
        cdp.eval(
            "(() => {"
            "  const sel = document.getElementById('triadWinSelect');"
            "  sel.value = '1.0';"
            "  sel.dispatchEvent(new Event('change', { bubbles: true }));"
            "})()"
        )
        for _ in range(60):
            m_txt = cdp.eval("document.getElementById('verdictEventMeta')?.textContent || ''")
            if "manual cursor window" in m_txt and m_txt != cur_meta_after and not cdp.eval("document.getElementById('vClipBtn')?.disabled"):
                break
            time.sleep(0.1)
        cdp.eval("document.getElementById('vClipBtn').click()")
        for _ in range(50):
            if len(store.load_judgments("CaseA")) >= 2:
                break
            time.sleep(0.1)
        judg_list = store.load_judgments("CaseA")
        clip_row = next((j for j in judg_list if j.get("clipped") is True), None)
        assert clip_row is not None and clip_row.get("event_id") is None, f"Clip archived wrong event instead of manual cursor: {judg_list}"
        iv = clip_row.get("interval_seconds")
        assert isinstance(iv, list) and len(iv) == 2 and abs((iv[1] - iv[0]) - 1.0) < 0.05, f"Widened 1.0s cursor span lost: {clip_row}"

        # (C) P0 #6: Open Reader tab on CaseA Event 0 -> replace recording & re-inspect -> clicking Confirm MUST fail (HTTP 409, zero WAL write)!
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&recording=CaseA&run=baseline&view=reader"})
        for _ in range(60):
            if cdp.eval(
                "document.getElementById('caseSelect')?.value === 'CaseA'"
                " && !document.getElementById('vConfirmBtn')?.disabled"
                " && (document.getElementById('verdictEventMeta')?.textContent || '').includes('3.000')"
            ):
                break
            time.sleep(0.1)
        wal_before_replace = len(store.load_judgments("CaseA"))
        rec_a_spec = store.load_recording("CaseA", "CaseA")
        # Replace CaseA source recording with modified signal and re-inspect under same case
        p_a = Path(rec_a_spec.path)
        t_rep = np.arange(rec_a_spec.n_samples, dtype=np.float64) / rec_a_spec.sfreq
        if p_a.suffix.lower() == ".edf":
            edfio.Edf([
                edfio.EdfSignal(5.0 * np.sin(2.0 * np.pi * 7.0 * t_rep), sampling_frequency=int(rec_a_spec.sfreq), label=c.name, physical_dimension="uV")
                for c in rec_a_spec.channels
            ]).write(p_a)
        else:
            import mne
            info_rep = mne.create_info([c.name for c in rec_a_spec.channels], sfreq=rec_a_spec.sfreq, ch_types="seeg")
            mne.io.RawArray(np.full((len(rec_a_spec.channels), rec_a_spec.n_samples), 5e-6), info_rep, verbose="ERROR").save(p_a, overwrite=True, verbose="ERROR")
        inspect_case(str(p_a), workspace_dir=ws, case_name="CaseA", persist=True)
        # Now click Confirm in the STILL-OPEN stale Reader tab!
        cdp.eval("document.getElementById('vConfirmBtn').click()")
        time.sleep(0.8)
        assert len(store.load_judgments("CaseA")) == wal_before_replace, (
            f"Stale Reader tab persisted verdict after recording replacement: {store.load_judgments('CaseA')[wal_before_replace:]}"
        )
        assert cdp.exceptions == [], f"JS exceptions: {cdp.exceptions}"
    finally:
        if cdp is not None:
            cdp.close()
        chrome.terminate()
        try:
            chrome.wait(timeout=4)
        except Exception:
            chrome.kill()
        srv.stop()


def test_cdp_round15_failed_rec_switch_retry_co_review_cursor_draft_stability_and_control_clip_overwrite_409(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chrome_bin = _find_chrome()
    if not chrome_bin:
        pytest.skip("Headless Chrome/Chromium binary not available")

    import threading
    import mne
    from brainlens.case_sheet import export_case_sheet
    from brainlens.io_readers import inspect_case
    from brainlens.models import atomic_write_json, read_json
    from brainlens.pipeline import analyze_case, refresh_verdict_tallies

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r15_cdp"))
    ws = _build_two_case_fixture(tmp_path)
    store = WorkbenchStore(ws)

    # Also build a two-recording single case `patient` (`first` with LB4, `second` with RA1) for P1 #4a!
    p_dir = tmp_path / "two_rec_patient"
    p_dir.mkdir()
    rng = np.random.default_rng(1515)
    t_arr = np.arange(20 * 1024, dtype=np.float64) / 1024.0
    sig1 = rng.normal(0.0, 8e-6, size=(2, len(t_arr)))
    sig2 = rng.normal(0.0, 8e-6, size=(2, len(t_arr)))
    info1 = mne.create_info(["LB4", "LB5"], sfreq=1024.0, ch_types="seeg")
    info2 = mne.create_info(["RA1", "RA2"], sfreq=1024.0, ch_types="seeg")
    mne.io.RawArray(sig1, info1, verbose="ERROR").save(p_dir / "first.fif", overwrite=True, verbose="ERROR")
    mne.io.RawArray(sig2, info2, verbose="ERROR").save(p_dir / "second.fif", overwrite=True, verbose="ERROR")
    (p_dir / "first.json").write_text(json.dumps({"reference": "common", "line_freq": 60.0}), encoding="utf-8")
    (p_dir / "second.json").write_text(json.dumps({"reference": "common", "line_freq": 60.0}), encoding="utf-8")
    inspect_case(str(p_dir), workspace_dir=ws, case_name="patient", persist=True)
    for rec_n, ch_n, ev_id, t0_ev in (("first", "LB4", "ev_first_1", 3.0), ("second", "RA1", "ev_second_1", 4.0)):
        analyze_case(f"patient/{rec_n}", workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
        b_p = store.load_run("patient", rec_n, "baseline", reconcile=False)
        det_p = Path(b_p["meta"]["detection_json_path"])
        det_d = read_json(det_p)
        det_d["status"] = "complete"
        det_d["events"] = [{
            "id": ev_id,
            "channel_id": "ch0000",
            "channel_name": ch_n,
            "shaft": ch_n[:2],
            "start_sample": int(round(t0_ev * 1024)),
            "end_sample": int(round((t0_ev + 0.08) * 1024)),
            "start_seconds": t0_ev,
            "end_seconds": t0_ev + 0.08,
            "duration_ms": 80.0,
            "peak_to_peak_uv": 90.0,
            "peak_freq_hz": 300.0,
            "dominant_band": "fast_ripple",
            "synchrony_class": "focal",
            "coactive_shaft_count": 1,
            "coactive_channel_count": 1,
        }]
        atomic_write_json(det_p, det_d)
        atomic_write_json(
            Path(b_p["meta"]["classification_json_path"]),
            {**det_d, "classifier_status": "skipped", "classification": {"requested": [], "stages": {}}},
        )
        refresh_verdict_tallies(store, "patient", rec_n, "baseline")

    srv = WorkbenchHTTPServer(workspace_dir=ws, host="127.0.0.1", port=0)
    orig_boot = srv._bootstrap
    orig_save = srv._save_judgment
    fail_second_once = [False]
    gate_ctrl_clip = [False]
    ctrl_entered = threading.Event()
    ctrl_release = threading.Event()

    def _hooked_boot(qs: dict[str, str]) -> dict[str, Any]:
        if qs.get("case") == "patient" and qs.get("recording") == "second" and not fail_second_once[0]:
            fail_second_once[0] = True
            raise RuntimeError("Transient second recording bootstrap error")
        return orig_boot(qs)

    def _hooked_save(body: dict[str, Any]) -> dict[str, Any]:
        if gate_ctrl_clip[0] and body.get("case") == "CaseA" and body.get("clipped") is True:
            ctrl_entered.set()
            ctrl_release.wait(timeout=8.0)
        return orig_save(body)

    monkeypatch.setattr(srv, "_bootstrap", _hooked_boot)
    monkeypatch.setattr(srv, "_save_judgment", _hooked_save)
    base_url = srv.start().rstrip("/")

    user_data = tmp_path / "chrome_r15_profile"
    user_data.mkdir()
    s_tmp = socket.socket()
    s_tmp.bind(("127.0.0.1", 0))
    dbg_port = s_tmp.getsockname()[1]
    s_tmp.close()

    chrome = subprocess.Popen(
        [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--window-size=1440,960",
            f"--remote-debugging-port={dbg_port}",
            f"--user-data-dir={user_data}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cdp: _MiniCDP | None = None
    try:
        ws_url = None
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{dbg_port}/json/list", timeout=1) as r:
                    tabs = json.loads(r.read().decode("utf-8"))
                page = next((t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")), None)
                if page:
                    ws_url = page["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.15)
        assert ws_url, "Headless Chrome did not expose CDP page target"

        cdp = _MiniCDP(ws_url)
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Log.enable")

        # (A) P1 #4a: Failed same-case recording switch first -> second MUST clear active pill, NOT render first Overview
        #     or unverified second Reader, and allow direct click on `second` pill to retry & save Confirm!
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=patient&recording=first&run=baseline&view=reader"})
        for _ in range(60):
            if cdp.eval(
                "document.querySelector('#recPills button.active')?.textContent === 'first'"
                " && !document.getElementById('vConfirmBtn')?.disabled"
                " && (document.getElementById('verdictEventMeta')?.textContent || '').includes('LB4')"
            ):
                break
            time.sleep(0.1)
        assert cdp.eval("document.querySelector('#recPills button.active')?.textContent") == "first"

        # Click `second` pill (fails transiently!)
        cdp.eval("Array.from(document.querySelectorAll('#recPills button')).find(b => b.textContent === 'second').click()")
        for _ in range(40):
            if fail_second_once[0]:
                break
            time.sleep(0.05)
        time.sleep(0.45)

        failed_st = cdp.eval(
            "({"
            "  activePill: document.querySelector('#recPills button.active')?.textContent || '',"
            "  disabled: document.getElementById('vConfirmBtn')?.disabled,"
            "  list: document.getElementById('queueList')?.textContent || ''"
            "})"
        )
        assert failed_st["activePill"] == "", f"Stale 'first' pill stayed highlighted after failed switch: {failed_st}"
        assert failed_st["disabled"] is True and failed_st["list"] == "", failed_st

        # Switch to Overview -> MUST NOT render stale `first` Overview KPIs!
        cdp.eval("document.querySelector('.view-tab[data-view=\"overview\"]').click()")
        time.sleep(0.35)
        ov_txt = cdp.eval("document.getElementById('kpiGrid')?.textContent || ''")
        assert ov_txt == "", f"Stale 'first' Overview rendered after failed recording switch: {ov_txt!r}"

        # Switch back to Reader -> MUST NOT render unverified `second` RA1 or re-enable Confirm!
        cdp.eval("document.querySelector('.view-tab[data-view=\"reader\"]').click()")
        time.sleep(0.45)
        rd_fail = cdp.eval(
            "({"
            "  activePill: document.querySelector('#recPills button.active')?.textContent || '',"
            "  disabled: document.getElementById('vConfirmBtn')?.disabled,"
            "  meta: document.getElementById('verdictEventMeta')?.textContent || ''"
            "})"
        )
        assert rd_fail["activePill"] == "" and rd_fail["disabled"] is True and "RA1" not in rd_fail["meta"], rd_fail

        # Now click `second` pill directly to retry -> MUST load `second` cleanly, highlight `second`, & persist Confirm!
        cdp.eval("Array.from(document.querySelectorAll('#recPills button')).find(b => b.textContent === 'second').click()")
        for _ in range(60):
            if cdp.eval(
                "document.querySelector('#recPills button.active')?.textContent === 'second'"
                " && !document.getElementById('vConfirmBtn')?.disabled"
                " && (document.getElementById('verdictEventMeta')?.textContent || '').includes('RA1')"
            ):
                break
            time.sleep(0.1)
        assert cdp.eval("document.querySelector('#recPills button.active')?.textContent") == "second"
        cdp.eval("document.getElementById('vConfirmBtn').click()")
        for _ in range(50):
            if len(store.load_judgments("patient")) >= 1:
                break
            time.sleep(0.1)
        j_pat = store.load_judgments("patient")
        assert len(j_pat) == 1 and j_pat[0]["recording"] == "second" and j_pat[0]["verdict"] == "confirm", j_pat

        # (B) P1 #4b: 6 consecutive co-review SSE verdict refreshes MUST preserve cursor [5.88, 6.48]s & unsaved #noteInput draft!
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&recording=CaseA&run=baseline&view=reader&channel=LB4&center_seconds=6.18&span_seconds=0.6"})
        for _ in range(60):
            if cdp.eval(
                "document.getElementById('caseSelect')?.value === 'CaseA'"
                " && (document.getElementById('verdictEventMeta')?.textContent || '').includes('[5.88, 6.48]s')"
                " && !document.getElementById('vClipBtn')?.disabled"
            ):
                break
            time.sleep(0.1)
        start_meta = cdp.eval("document.getElementById('verdictEventMeta')?.textContent || ''")
        assert "[5.88, 6.48]s" in start_meta, start_meta
        draft_txt = "Solo finding for LB4 cursor [5.88, 6.48]s"
        cdp.eval(f"document.getElementById('noteInput').value = {json.dumps(draft_txt)}")

        # Trigger 6 real external verdicts on ev_casea_001 via HTTP POST -> 6 SSE refreshes!
        for k in range(6):
            req_v = urllib.request.Request(
                f"{base_url}/api/judgment",
                data=json.dumps({
                    "case": "CaseA",
                    "recording": "CaseA",
                    "run_name": "baseline",
                    "channel": "LB4",
                    "event_id": "ev_casea_001",
                    "verdict": "confirm" if k % 2 == 0 else "uncertain",
                    "author": "Co-Reviewer",
                    "role": "clinician",
                    "note": f"Co-review pass {k}",
                    "operation_id": f"r15_corev_{k}",
                }).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-BrainLens-Token": srv.auth_token},
                method="POST",
            )
            with urllib.request.urlopen(req_v, timeout=5) as resp_v:
                assert resp_v.status == 200
            time.sleep(0.35)

        after6 = cdp.eval(
            "({"
            "  meta: document.getElementById('verdictEventMeta')?.textContent || '',"
            "  draft: document.getElementById('noteInput')?.value || ''"
            "})"
        )
        assert "[5.88, 6.48]s" in after6["meta"], f"Co-review SSE shrank cursor window: {after6}"
        assert after6["draft"] == draft_txt, f"Co-review SSE wiped clinician draft: {after6}"

        # (C) P1 #1: Paused Quiet Control Clip rejected HTTP 409 after baseline overwritten to detector=skipped & lp=20Hz!
        cdp.eval("document.getElementById('showControlBtn').click()")
        for _ in range(60):
            if cdp.eval(
                "(document.getElementById('triadHeaderText')?.textContent || '').includes('Quiet control: ripple RMS')"
                " && !document.getElementById('vClipBtn')?.disabled"
            ):
                break
            time.sleep(0.1)
        cdp.eval("document.getElementById('noteInput').value = 'Detector-verified clean background reference'")
        wal_before_ctrl = len(store.load_judgments("CaseA"))
        gate_ctrl_clip[0] = True
        cdp.eval("document.getElementById('vClipBtn').click()")
        assert ctrl_entered.wait(timeout=5.0), "Quiet Control Clip POST did not reach server!"

        # While browser's Quiet Control Clip POST is paused, overwrite CaseA/baseline to detector=disabled, lowpass=20Hz!
        analyze_case(
            "CaseA/CaseA",
            workspace_dir=ws,
            run_name="baseline",
            overwrite=True,
            overrides={"detector_enabled": False, "lowpass": 20.0},
        )
        ctrl_release.set()
        time.sleep(0.8)

        # Verify paused Quiet Control Clip was rejected (zero new WAL rows) & absent from fresh MDT sheet!
        assert len(store.load_judgments("CaseA")) == wal_before_ctrl, (
            f"Stale Quiet Control Clip persisted after overwrite: {store.load_judgments('CaseA')[wal_before_ctrl:]}"
        )
        sh_ovw = export_case_sheet("CaseA", workspace_dir=ws, primary_run="baseline")
        html_ovw = Path(sh_ovw["report_path"]).read_text(encoding="utf-8")
        assert "Detector-verified clean background reference" not in html_ovw

        assert cdp.exceptions == [], f"JS exceptions: {cdp.exceptions}"
    finally:
        ctrl_release.set()
        if cdp is not None:
            cdp.close()
        chrome.terminate()
        try:
            chrome.wait(timeout=4)
        except Exception:
            chrome.kill()
        srv.stop()


def _reader_wait(cdp: _MiniCDP, expr: str) -> Any:
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        try:
            result = cdp.eval(expr)
            if result:
                return result
        except RuntimeError as exc:
            # Chrome may discard the old execution context between Page.reload and evaluation.
            if not any(msg in str(exc) for msg in ("Execution context was destroyed", "Cannot find context")):
                raise
        time.sleep(0.05)
    pytest.fail(f"Reader did not reach {expr!r}; JS exceptions: {cdp.exceptions}")


def _reader_change(cdp: _MiniCDP, element_id: str, value: str) -> None:
    cdp.eval(f"""(() => {{
        const el = document.getElementById({json.dumps(element_id)});
        el.value = {json.dumps(value)};
        el.dispatchEvent(new Event('change', {{bubbles: true}}));
        el.blur();
    }})()""")


@pytest.fixture
def reader_cdp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    chrome_bin = _find_chrome()
    if not chrome_bin:
        pytest.skip("Google Chrome / Chromium not installed")
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    ws = _build_two_case_fixture(tmp_path)
    server = WorkbenchHTTPServer(workspace_dir=ws, port=0)
    base_url = server.start().rstrip("/")
    profile = tmp_path / "chrome_profile"
    chrome = None
    cdp = None
    try:
        chrome = subprocess.Popen([
            chrome_bin, "--headless=new", "--disable-gpu", "--no-first-run",
            "--remote-debugging-port=0", f"--user-data-dir={profile}", "about:blank",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        port_file = profile / "DevToolsActivePort"
        deadline = time.monotonic() + 12
        while not port_file.exists() and time.monotonic() < deadline:
            assert chrome.poll() is None, "Chrome exited before enabling CDP"
            time.sleep(0.05)
        assert port_file.exists(), "Chrome CDP did not start"
        port = int(port_file.read_text().splitlines()[0])
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as response:
            tab = next(t for t in json.load(response) if t["type"] == "page")
        cdp = _MiniCDP(tab["webSocketDebuggerUrl"])
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Emulation.setDeviceMetricsOverride", {
            "width": 1280, "height": 820, "deviceScaleFactor": 2, "mobile": False,
        })
        # Inspect the rendered pulse and waveform width in CSS pixels at DPR 2.
        cdp.send("Page.addScriptToEvaluateOnNewDocument", {"source": """
            window.readerFrame = {labels: [], plotWidth: 0, pulse: null};
            const proto = CanvasRenderingContext2D.prototype;
            const traces = new WeakMap();
            let path = [];
            for (const method of ['clearRect', 'fillText', 'drawImage', 'beginPath', 'moveTo', 'lineTo', 'stroke']) {
                const original = proto[method];
                proto[method] = function(...args) {
                    // Save the first non-horizontal trace segment on each offscreen waveform rebuild.
                    if (!this.canvas.isConnected) {
                        if (method === 'clearRect') traces.set(this.canvas, {});
                        const trace = traces.get(this.canvas);
                        if (trace && trace.dy === undefined) {
                            if (method === 'moveTo') trace.y = args[1];
                            if (method === 'lineTo' && Number.isFinite(trace.y) && trace.y !== args[1]) {
                                trace.dy = args[1] - trace.y;
                            }
                        }
                    }
                    if (this.canvas.id === 'mcCanvas') {
                        if (method === 'clearRect') { readerFrame.labels = []; readerFrame.pulse = null; }
                        if (method === 'fillText') readerFrame.labels.push(args[0]);
                        if (method === 'drawImage') {
                            readerFrame.plotWidth = args.at(-2);
                            readerFrame.traceDy = traces.get(args[0])?.dy;
                        }
                        if (method === 'beginPath') path = [];
                        if (method === 'moveTo' || method === 'lineTo') path.push(args);
                        if (method === 'stroke' && path.length === 4) {
                            const [a, b, c, d] = path;
                            if (a[0] === b[0] && b[1] === c[1] && c[0] === d[0] && a[1] === d[1]) {
                                readerFrame.pulse = {width: c[0] - b[0], height: a[1] - b[1]};
                            }
                        }
                    }
                    return original.apply(this, args);
                };
            }
            window.readerRequest = () => {
                const urls = performance.getEntriesByType('resource').map(r => new URL(r.name));
                return urls.filter(u => u.pathname === '/api/multichannel').at(-1)?.searchParams;
            };
        """})
        cdp.send("Page.navigate", {"url": f"{base_url}/?case=CaseA&recording=CaseA&run=baseline&view=reader"})
        _reader_wait(cdp, "window.readerFrame?.plotWidth && document.querySelectorAll('#queueList .q-item').length >= 2")
        yield cdp
        assert cdp.exceptions == []
    finally:
        if cdp:
            cdp.close()
        if chrome:
            chrome.terminate()
            try:
                chrome.wait(timeout=4)
            except subprocess.TimeoutExpired:
                chrome.kill()
                chrome.wait()
        server.stop()


def test_reader_conventions_calibration_and_timebase(reader_cdp: _MiniCDP) -> None:
    cdp = reader_cdp
    assert cdp.eval("document.getElementById('readerSensitivitySelect').value") == "auto"
    assert cdp.eval("document.getElementById('readerTimebaseSelect').value") == "fit"
    _reader_wait(cdp, "readerFrame.labels.some(s => s.includes('µV/ch'))")
    _reader_change(cdp, "readerSensitivitySelect", "100")
    _reader_wait(cdp, "readerFrame.labels.some(s => s.includes('100 µV/mm') && s.includes('screen not calibrated'))")
    cdp.eval("document.getElementById('calibrateScreenBtn').click()")
    _reader_wait(cdp, "document.getElementById('screenCalibrationDialog').open")
    ruler_width = cdp.eval("document.getElementById('calibrationRuler').getBoundingClientRect().width")
    assert ruler_width > 0
    # Enter a known physical scale independent of DPR and the default 96 dpi.
    cdp.eval(f"document.getElementById('calibrationLengthMm').value = {ruler_width / 4}")
    cdp.eval("document.getElementById('saveCalibrationBtn').click()")
    calibrated = "window.readerFrame?.labels.some(s => s.includes('100 µV/mm (calibrated)'))"
    _reader_wait(cdp, calibrated)
    frame = cdp.eval("readerFrame")
    assert "1 s" in frame["labels"]
    pulse_uv = float(next(s.split()[0] for s in frame["labels"] if s.endswith(" µV")))
    assert frame["pulse"]["height"] == pytest.approx(pulse_uv * 4 / 100)
    assert frame["pulse"]["width"] == pytest.approx(frame["plotWidth"] / 8)
    _reader_change(cdp, "readerSensitivitySelect", "50")
    _reader_wait(cdp, "readerFrame.labels.some(s => s.includes('50 µV/mm (calibrated)'))")
    assert cdp.eval("readerFrame.traceDy") == pytest.approx(frame["traceDy"] * 2)
    _reader_change(cdp, "readerTimebaseSelect", "30")
    _reader_wait(cdp, """Math.abs(Number(readerRequest()?.get('duration_seconds'))
                                      - readerFrame.plotWidth / (4 * 30)) < 0.01""")
    _reader_wait(cdp, r"""Math.abs(Number(document.getElementById('readerTimebaseReadout').textContent
        .match(/([\d.]+)\s*mm\/s/)?.[1]) - 30) < 0.1""")
    cdp.send("Emulation.setDeviceMetricsOverride", {
        "width": 1000, "height": 820, "deviceScaleFactor": 2, "mobile": False,
    })
    _reader_wait(cdp, f"readerFrame.plotWidth < {frame['plotWidth']} - 50")
    _reader_wait(cdp, """Math.abs(Number(readerRequest()?.get('duration_seconds'))
                                      - readerFrame.plotWidth / (4 * 30)) < 0.01""")
    _reader_change(cdp, "readerTimebaseSelect", "fit")
    _reader_wait(cdp, """!document.getElementById('readerDurSelect').classList.contains('hidden')
                         && Number(readerRequest()?.get('duration_seconds')) === 8""")

    old_origin = cdp.eval("performance.timeOrigin")
    cdp.send("Page.reload", {"ignoreCache": True})
    _reader_wait(cdp, f"performance.timeOrigin > {old_origin} && window.readerFrame?.plotWidth > 0")
    _reader_change(cdp, "readerSensitivitySelect", "100")
    _reader_wait(cdp, calibrated)
    _reader_change(cdp, "readerTimebaseSelect", "30")
    _reader_wait(cdp, """Math.abs(Number(readerRequest()?.get('duration_seconds'))
                                      - readerFrame.plotWidth / (4 * 30)) < 0.01""")


def test_reader_conventions_filter_request_and_notice(reader_cdp: _MiniCDP) -> None:
    cdp = reader_cdp
    _reader_change(cdp, "dispFilterSelect", "eeg_1_70")
    _reader_wait(cdp, """Number(document.getElementById('displayLff').value) === 1
                          && Number(document.getElementById('displayHff').value) === 70""")
    cdp.eval("document.getElementById('displayNotchChk').click()")
    _reader_wait(cdp, """document.getElementById('readerFilterReadout').textContent
                         .includes('LFF 1 Hz · HFF 70 Hz · Notch 60 Hz')""")
    _reader_change(cdp, "displayLff", "3")
    _reader_change(cdp, "displayHff", "1000")
    _reader_wait(cdp, """readerRequest()?.get('display_lff') === '3'
                          && readerRequest()?.get('display_hff') === '1000'""")
    # At 1024 Hz, 1000 Hz is beyond the trace's 512 Hz Nyquist ceiling.
    _reader_wait(cdp, """(() => {
        const notice = document.getElementById('montageNotice');
        return notice.offsetWidth > 0 && /clamp|limit/i.test(notice.textContent);
    })()""")
    _reader_change(cdp, "displayLff", "600")
    _reader_wait(cdp, """readerRequest()?.get('display_lff') === '600'
                          && /broadband/i.test(document.getElementById('montageNotice').textContent)""")


def test_reader_conventions_page_keys_and_help(reader_cdp: _MiniCDP) -> None:
    cdp = reader_cdp

    def press(key: str, code: str, keycode: int, modifiers: int = 0) -> None:
        for event in ("keyDown", "keyUp"):
            cdp.send("Input.dispatchKeyEvent", {
                "type": event, "key": key, "code": code,
                "windowsVirtualKeyCode": keycode, "modifiers": modifiers,
            })

    assert cdp.eval("document.getElementById('readerPageStep').value") == "0.85"
    cdp.eval("document.activeElement.blur()")
    press("Home", "Home", 36)
    _reader_wait(cdp, "Number(document.getElementById('readerStartInput').value) === 0")
    press("PageDown", "PageDown", 34)
    _reader_wait(cdp, "Math.abs(Number(readerRequest()?.get('start_seconds')) - 6.8) < 0.01")
    press("PageUp", "PageUp", 33)
    _reader_wait(cdp, "Number(readerRequest()?.get('start_seconds')) === 0")
    _reader_change(cdp, "readerPageStep", "1")
    press("PageDown", "PageDown", 34)
    _reader_wait(cdp, "Number(readerRequest()?.get('start_seconds')) === 8")
    press("End", "End", 35)
    _reader_wait(cdp, "Number(readerRequest()?.get('start_seconds')) === 12")
    press("?", "Slash", 191, modifiers=8)
    _reader_wait(cdp, "document.getElementById('readerShortcutDialog').open")
    help_text = cdp.eval("document.getElementById('readerShortcutDialog').innerText").lower().replace(" ", "")
    for shortcut in ("pageup", "pagedown", "home", "end"):
        assert shortcut in help_text
    press("Home", "Home", 36)
    assert cdp.eval("Number(document.getElementById('readerStartInput').value)") == 12
    press("Escape", "Escape", 27)
    _reader_wait(cdp, "!document.getElementById('readerShortcutDialog').open")
    press("Home", "Home", 36)
    _reader_wait(cdp, "Number(readerRequest()?.get('start_seconds')) === 0")










def test_i18n_reader_state_sheet_and_reload(reader_cdp: _MiniCDP) -> None:
    cdp = reader_cdp
    assert cdp.eval("document.documentElement?.lang") == "en"
    _reader_change(cdp, "readerSensitivitySelect", "100")
    _reader_wait(cdp, "readerFrame.labels.some(s => s.includes('100 µV/mm') && s.includes('screen not calibrated'))")
    _reader_wait(cdp, "document.querySelector('#verdictEventMeta strong')?.textContent === 'LB4'")
    cdp.eval("""(() => {
        const note = document.getElementById('noteInput');
        note.value = 'LB4 draft / 未保存';
        note.dispatchEvent(new Event('input', {bubbles: true}));
        document.getElementById('languageBtn').click();
    })()""")
    _reader_wait(cdp, """document.documentElement?.lang === 'zh-CN'
                         && readerFrame.labels.some(s => s.includes('100 µV/mm') && s.includes('校准'))""")
    assert cdp.eval("document.getElementById('noteInput').value") == "LB4 draft / 未保存"
    assert "临床观察" in cdp.eval("document.getElementById('noteInput').placeholder")
    assert cdp.eval("document.querySelector('#blindedToggle').parentElement.textContent.trim()") == "盲评"
    assert cdp.eval("document.querySelector('#queueList .q-item strong').textContent.trim()") == "LB4"

    cdp.eval("""document.querySelector('.view-tab[data-view="sheet"]').click()""")
    _reader_wait(cdp, """document.querySelector('#view-sheet.active')
                         && document.getElementById('sheetFrame').contentDocument?.querySelector('footer')
                            ?.textContent.includes('not a standalone surgical resection boundary')""")
    assert cdp.eval("document.getElementById('sheetFrame').contentDocument.documentElement.lang") == "en"
    assert "去标识化" in cdp.eval("document.getElementById('deidentifyExport').closest('label').innerText")
    assert cdp.eval("document.getElementById('rebuildSheetBtn').textContent.trim()") == "重新生成"

    old_origin = cdp.eval("performance.timeOrigin")
    cdp.send("Page.reload", {"ignoreCache": True})
    _reader_wait(cdp, f"""performance.timeOrigin > {old_origin} && document.documentElement?.lang === 'zh-CN'
                          && document.getElementById('caseSelect')?.value === 'CaseA'
                          && document.querySelector('.view-tab[data-view="overview"]')?.textContent.trim() === '概览'""")
    cdp.eval("document.getElementById('languageBtn').click()")
    _reader_wait(cdp, """document.documentElement?.lang === 'en'
                         && document.querySelector('.view-tab[data-view="reader"]').textContent.trim() === 'Waveform Reader'""")
    assert cdp.eval("document.getElementById('noteInput').placeholder") == "Clinical observation note..."
    assert cdp.eval("document.getElementById('deidentifyExport').closest('label').textContent.trim()") == "De-identify for sharing"


def test_i18n_overview_agreement_and_evidence_results(reader_cdp: _MiniCDP) -> None:
    cdp = reader_cdp
    cdp.eval("""document.querySelector('.view-tab[data-view="overview"]').click()""")
    _reader_wait(cdp, "!document.getElementById('evidencePanel').hidden")
    cdp.eval("""document.querySelector('#agreementPanel summary').click()""")
    _reader_wait(cdp, "document.getElementById('agreementResults').textContent.includes('Disagreements (0)')")
    cdp.eval("document.getElementById('loadEvidenceMap').click()")
    _reader_wait(cdp, "document.querySelector('#evidenceResult thead')?.textContent.includes('Convergence / available')")
    contacts = cdp.eval("[...document.querySelectorAll('[data-evidence-contact]')].map(b => b.textContent)")
    assert "LB4" in contacts
    cdp.eval("document.getElementById('languageBtn').click()")
    _reader_wait(cdp, """document.querySelector('#evidenceResult thead')?.textContent.includes('触点')
                         && document.querySelector('#agreementResults h3')?.textContent.includes('判读分歧（0）')""")
    assert cdp.eval("document.querySelector('#agreementPanel summary').innerText") == "评分者间一致性"
    assert cdp.eval("document.getElementById('evidenceTitle').innerText") == "证据图谱"
    scope = cdp.eval("document.querySelector('#evidenceResult > p:last-child').textContent")
    assert "医疗器械" in scope and "切除" in scope
    assert cdp.eval("[...document.querySelectorAll('[data-evidence-contact]')].map(b => b.textContent)") == contacts

    cdp.eval("document.getElementById('eiOnsetSeconds').value = '10'")
    cdp.eval("document.getElementById('loadEpileptogenicity').click()")
    _reader_wait(cdp, """document.querySelector('#evidenceResult [data-er-row]')
                         && document.getElementById('evidenceStatus').textContent.includes('发作')""")
    assert "能量比" in cdp.eval("document.querySelector('#evidenceResult thead').innerText")
    cdp.eval("document.getElementById('languageBtn').click()")
    _reader_wait(cdp, """document.querySelector('#evidenceResult thead')?.textContent.includes('Normalized EI')
                         && document.getElementById('agreementResults').textContent.includes('Disagreements (0)')""")
    assert cdp.eval("document.getElementById('eiOnsetSeconds').value") == "10"


def test_i18n_browser_language_default_and_explicit_choice(reader_cdp: _MiniCDP) -> None:
    cdp = reader_cdp
    cdp.send("Page.addScriptToEvaluateOnNewDocument", {"source": """
        Object.defineProperty(navigator, 'language', {get: () => 'zh-Hant-TW'});
    """})
    cdp.send("Page.reload", {"ignoreCache": True})
    _reader_wait(cdp, """document.documentElement?.lang === 'zh-CN' && window.readerFrame?.plotWidth > 0""")
    assert cdp.eval("brainlensI18n.t('Missing English label for {channel}', {channel: 'LA1'})") == "Missing English label for LA1"
    cdp.eval("document.getElementById('languageBtn').click()")
    _reader_wait(cdp, "document.documentElement?.lang === 'en'")
    old_origin = cdp.eval("performance.timeOrigin")
    cdp.send("Page.reload", {"ignoreCache": True})
    _reader_wait(cdp, f"""performance.timeOrigin > {old_origin} && window.readerFrame?.plotWidth > 0
                          && document.documentElement?.lang === 'en' && navigator.language === 'zh-Hant-TW'""")
