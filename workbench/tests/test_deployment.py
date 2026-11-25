"""Deployment invariants and offline/launcher tests for BrainLens (Phase 5)."""

from __future__ import annotations

import hashlib
import http.client
import importlib.util
import io
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

WORKBENCH_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKBENCH_DIR.parent
SRC_DIR = WORKBENCH_DIR / "src"
if importlib.util.find_spec("brainlens") is None and SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlens.workbench_http import WorkbenchHTTPServer

DOCKERFILE = PROJECT_ROOT / "docker" / "Dockerfile"
COMPOSE_FILE = PROJECT_ROOT / "docker" / "compose.yaml"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
DEPLOYMENT_DOC = PROJECT_ROOT / "docs" / "deployment.md"
BUILD_BUNDLE_SCRIPT = PROJECT_ROOT / "scripts" / "build_offline_bundle.sh"
INSTALL_OFFLINE_SCRIPT = PROJECT_ROOT / "scripts" / "install_offline.sh"
LAUNCHERS = [
    WORKBENCH_DIR / "bin" / "brainlens",
    WORKBENCH_DIR / "bin" / "brainlens-mcp",
    WORKBENCH_DIR / "bin" / "brainlens-workbench",
]


def _dockerfile_instructions(text: str) -> list[tuple[str, str]]:
    """Collapse backslash continuations and return (KEYWORD, body) pairs."""
    logical_lines: list[str] = []
    buf = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not buf and (not stripped or stripped.startswith("#")):
            continue
        if stripped.endswith("\\"):
            buf += stripped[:-1].rstrip() + " "
        else:
            buf += stripped
            logical_lines.append(buf.strip())
            buf = ""
    if buf.strip():
        logical_lines.append(buf.strip())

    out: list[tuple[str, str]] = []
    for line in logical_lines:
        parts = line.split(None, 1)
        if not parts:
            continue
        out.append((parts[0].upper(), parts[1] if len(parts) > 1 else ""))
    return out


class _KeepBufferBytesIO(io.BytesIO):
    """BytesIO that retains its written bytes after BaseHTTPRequestHandler.finish() closes wfile."""

    def __init__(self, initial: bytes = b"") -> None:
        super().__init__(initial)
        self.captured: bytes = b""

    def close(self) -> None:
        if not self.closed:
            self.captured = self.getvalue()
        super().close()


class _InMemorySocket:
    """Minimal socket adapter for driving ThreadingHTTPServer.RequestHandlerClass without TCP connect."""

    def __init__(self, raw_request: bytes) -> None:
        self._r = io.BytesIO(raw_request)
        self._w = _KeepBufferBytesIO()

    def makefile(self, mode: str, _buffering: int = -1):
        return self._r if "r" in mode else self._w

    def sendall(self, data: bytes) -> None:
        self._w.write(data)

    def close(self) -> None:
        pass

    @property
    def response_bytes(self) -> bytes:
        return self._w.captured if self._w.closed else self._w.getvalue()


def _http_get_with_host(srv: WorkbenchHTTPServer, path: str, host_header: str) -> tuple[int, dict]:
    """Send a GET request with an explicit Host header; fall back to in-process handler if sandbox blocks TCP connect."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5.0)
        try:
            conn.putrequest("GET", path, skip_host=True)
            conn.putheader("Host", host_header)
            conn.endheaders()
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(body)
        finally:
            conn.close()
    except PermissionError:
        assert srv.httpd is not None
        raw_req = f"GET {path} HTTP/1.1\r\nHost: {host_header}\r\nConnection: close\r\n\r\n".encode("ascii")
        sock = _InMemorySocket(raw_req)
        srv.httpd.RequestHandlerClass(sock, ("127.0.0.1", 0), srv.httpd)
        head, _, body_bytes = sock.response_bytes.partition(b"\r\n\r\n")
        status_line = head.split(b"\r\n", 1)[0].decode("ascii")
        status_code = int(status_line.split()[1])
        return status_code, json.loads(body_bytes.decode("utf-8"))


def test_dockerfile_multistage_dependencies_and_pyhfo() -> None:
    assert DOCKERFILE.is_file(), f"Missing {DOCKERFILE}"
    text = DOCKERFILE.read_text(encoding="utf-8")
    insts = _dockerfile_instructions(text)

    from_lines = [body for kw, body in insts if kw == "FROM"]
    assert len(from_lines) >= 2, "Dockerfile must be multi-stage"
    assert any(body.startswith("python:3.12-slim") for body in from_lines), "Dockerfile must use python:3.12-slim"

    # Must build BrainLens wheel and install python-pptx>=1.0,<2
    assert "pip wheel" in text and "brainlens-*.whl" in text
    assert "python-pptx>=1.0,<2" in text

    # Optional WITH_PYHFO build arg uses pre-fetched context (never clones/fetches at runtime or build)
    assert any(kw == "ARG" and body.startswith("WITH_PYHFO") for kw, body in insts)
    assert "docker/vendor" in text
    assert "/opt/pyhfo-venv" in text
    for env_key in ("BRAINLENS_PYHFO_SOURCE", "BRAINLENS_PYHFO_PYTHON", "BRAINLENS_PYHFO_MODEL_CACHE"):
        assert env_key in text, f"Missing {env_key} in Dockerfile"
    assert "git clone" not in text, "Dockerfile must never clone PyHFO"
    assert "huggingface.co" not in text, "Dockerfile must never download models from HuggingFace"


def test_dockerfile_security_and_runtime_invariants() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    insts = _dockerfile_instructions(text)

    # Non-root USER must appear in the final stage and never revert to root
    last_from_idx = max(i for i, (kw, _) in enumerate(insts) if kw == "FROM")
    final_insts = insts[last_from_idx + 1 :]
    user_insts = [body.strip() for kw, body in final_insts if kw == "USER"]
    assert user_insts, "Final stage must declare a non-root USER"
    assert user_insts[-1].lower() not in {"root", "0", "0:0"}, f"Final USER must be non-root, got {user_insts[-1]!r}"

    # Required env vars, volumes, /recordings mount point, HEALTHCHECK, ENTRYPOINT
    assert "BRAINLENS_WORKSPACE=/data/workbench_results" in text
    assert "BRAINLENS_NO_BROWSER=1" in text
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "/recordings" in text
    assert any(kw == "VOLUME" and "/data" in body for kw, body in final_insts)

    healthchecks = [body for kw, body in final_insts if kw == "HEALTHCHECK"]
    assert healthchecks, "Final stage must declare a HEALTHCHECK"
    assert "/api/health" in healthchecks[-1] and "127.0.0.1:8765" in healthchecks[-1]

    entrypoints = [body for kw, body in final_insts if kw == "ENTRYPOINT"]
    assert entrypoints, "Final stage must declare an ENTRYPOINT"
    ep = entrypoints[-1]
    assert "brainlens-workbench" in ep
    assert "--host" in ep and "0.0.0.0" in ep
    assert "--port" in ep and "8765" in ep


def test_compose_loopback_only_and_volumes() -> None:
    assert COMPOSE_FILE.is_file(), f"Missing {COMPOSE_FILE}"
    text = COMPOSE_FILE.read_text(encoding="utf-8")

    # Never invoke --host 0.0.0.0 in compose.yaml
    assert "--host 0.0.0.0" not in text
    assert '"--host", "0.0.0.0"' not in text

    # Extract all published port specs (via PyYAML if available, else regex fallback)
    port_specs: list[str] = []
    if importlib.util.find_spec("yaml") is not None:
        import yaml  # type: ignore[import-untyped]

        doc = yaml.safe_load(text)
        services = doc.get("services") or {}
        assert "brainlens" in services
        svc = services["brainlens"]
        for p in svc.get("ports") or []:
            assert isinstance(p, str), f"Expected short loopback port syntax, got {p!r}"
            port_specs.append(p)
        volumes = [str(v) for v in (svc.get("volumes") or [])]
        assert any(v.endswith(":/data") for v in volumes), "compose.yaml must mount /data"
        assert any(v.endswith(":/recordings:ro") for v in volumes), "compose.yaml must mount /recordings read-only"
        env = svc.get("environment") or {}
        assert env.get("BRAINLENS_WORKSPACE") == "/data/workbench_results"
        assert str(env.get("BRAINLENS_NO_BROWSER")) == "1"
    else:
        in_ports = False
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "ports:":
                in_ports = True
                continue
            if in_ports:
                if stripped.startswith("- "):
                    port_specs.append(stripped[2:].strip().strip("\"'"))
                else:
                    in_ports = False
        assert ":/data" in text
        assert ":/recordings:ro" in text
        assert "BRAINLENS_WORKSPACE: /data/workbench_results" in text
        assert "BRAINLENS_NO_BROWSER:" in text

    assert port_specs, "compose.yaml must declare loopback port publishing"
    for spec in port_specs:
        # Must strictly bind 127.0.0.1:<port>:<port> or [::1]:<port>:<port>
        m = re.fullmatch(r"(127\.0\.0\.1|\[::1\]):(\d+):(\d+)", spec)
        assert m is not None, f"Port spec {spec!r} is not strictly loopback-bound!"
        assert m.group(2) == m.group(3), f"Host/container port mismatch in {spec!r} (rejected by _check_host)"
    assert "127.0.0.1:8765:8765" in port_specs


def test_no_host_0000_outside_container_entrypoint() -> None:
    pattern = re.compile(r"--host[\s=\"',]+0\.0\.0\.0")

    # 1. Dockerfile: only allowed on ENTRYPOINT (or CMD) instructions
    for kw, body in _dockerfile_instructions(DOCKERFILE.read_text(encoding="utf-8")):
        if pattern.search(body):
            assert kw == "ENTRYPOINT", f"--host 0.0.0.0 found outside ENTRYPOINT in Dockerfile ({kw}: {body})"

    # 2. Compose, launchers, scripts, and package Python code must never pass --host 0.0.0.0
    checked_paths: list[Path] = [
        COMPOSE_FILE,
        *LAUNCHERS,
        BUILD_BUNDLE_SCRIPT,
        INSTALL_OFFLINE_SCRIPT,
        *sorted((WORKBENCH_DIR / "src" / "brainlens").glob("*.py")),
    ]
    for path in checked_paths:
        content = path.read_text(encoding="utf-8")
        assert not pattern.search(content), f"Found forbidden '--host 0.0.0.0' in {path.relative_to(PROJECT_ROOT)}"


def test_dockerignore_excludes_venvs_results_build_git_and_raw_inputs() -> None:
    assert DOCKERIGNORE.is_file(), f"Missing {DOCKERIGNORE}"
    lines = {
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for required in (".git", ".venv/", "results/", "workbench_results/", "build/"):
        assert required in lines, f".dockerignore missing {required!r}"
    for ext in (".edf", ".bdf", ".eeg", ".fif"):
        assert any(ln.startswith("inputs/") and ln.endswith(ext) for ln in lines), (
            f".dockerignore must exclude raw {ext} files under inputs/"
        )
    assert "workbench/" not in lines
    assert "docker/vendor/" not in lines


def test_ui_assets_have_no_external_cdn_or_webfonts() -> None:
    assets_dir = WORKBENCH_DIR / "src" / "brainlens" / "assets"
    url_re = re.compile(r"https?://[^\s'\"<>)]+", re.IGNORECASE)
    for asset in ("index.html", "style.css", "app.js"):
        text = (assets_dir / asset).read_text(encoding="utf-8")
        urls = [u for u in url_re.findall(text) if not u.startswith("http://www.w3.org/")]
        assert not urls, f"Unexpected external URLs in UI asset {asset}: {urls}"


def test_workbench_http_host_check_accepts_matching_loopback_and_rejects_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    # Bind 0.0.0.0 when permitted (mirrors container binding); fall back to 127.0.0.1 if sandbox forbids 0.0.0.0.
    try:
        srv = WorkbenchHTTPServer(workspace_dir=tmp_path / "ws", host="0.0.0.0", port=0)
        srv.start()
    except OSError:
        srv = WorkbenchHTTPServer(workspace_dir=tmp_path / "ws", host="127.0.0.1", port=0)
        srv.start()

    try:
        port = srv.port
        # Allowed write origins stay strictly loopback on the exact matching port (even when bound to 0.0.0.0)
        origins = srv._allowed_origins()
        assert origins == {f"http://127.0.0.1:{port}", f"http://localhost:{port}", f"http://[::1]:{port}"}
        assert f"http://0.0.0.0:{port}" not in origins

        # Matching loopback Host headers succeed
        for host_hdr in (f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"):
            status, payload = _http_get_with_host(srv, "/api/health", host_hdr)
            assert status == 200, f"Expected 200 for Host {host_hdr}, got {status}: {payload}"
            assert payload.get("ok") is True

        # Mismatched port (e.g. -p 127.0.0.1:9000:8765) and non-loopback Host headers are rejected
        for bad_hdr in (f"127.0.0.1:{port + 1}", f"localhost:{port + 1}", f"192.168.1.50:{port}", f"evil.example.com:{port}"):
            status, payload = _http_get_with_host(srv, "/api/health", bad_hdr)
            assert status == 403, f"Expected 403 for Host {bad_hdr}, got {status}: {payload}"
            assert payload["error"]["code"] == "forbidden_host"
    finally:
        srv.stop()


def test_launchers_remove_releases_fallback_and_pass_bash_n() -> None:
    for launcher in LAUNCHERS:
        assert launcher.is_file(), f"Missing launcher {launcher}"
        assert os.access(launcher, os.X_OK), f"Launcher {launcher} must be executable"
        text = launcher.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash\n")
        assert "set -euo pipefail" in text
        assert ".releases/" not in text, f"Stale .releases/ reference still in {launcher}"
        assert "BRAINLENS_PYTHON" in text
        assert ".venv/bin/python" in text
        assert "command -v python3" in text

        res = subprocess.run(["bash", "-n", str(launcher)], capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"bash -n failed on {launcher}: {res.stderr}"


def test_launcher_error_handling_and_valid_execution(tmp_path: Path) -> None:
    launcher = WORKBENCH_DIR / "bin" / "brainlens"

    # 1. Non-executable BRAINLENS_PYTHON fails fast with clear error
    env_bad = {**os.environ, "BRAINLENS_PYTHON": str(tmp_path / "does-not-exist")}
    res_bad = subprocess.run([str(launcher), "--help"], env=env_bad, capture_output=True, text=True, check=False)
    assert res_bad.returncode != 0
    assert res_bad.stdout == ""
    assert "BRAINLENS_PYTHON" in res_bad.stderr

    # 2. Interpreter missing BrainLens deps prints clear error (no traceback) and nothing on stdout
    stub_py = tmp_path / "bare-python"
    stub_py.write_text(f"#!/usr/bin/env bash\nexec {sys.executable!r} -S \"$@\"\n", encoding="utf-8")
    stub_py.chmod(stub_py.stat().st_mode | stat.S_IXUSR)
    env_bare = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "BTK_PYTHON"}}
    env_bare["BRAINLENS_PYTHON"] = str(stub_py)
    res_bare = subprocess.run([str(launcher), "--help"], env=env_bare, capture_output=True, text=True, check=False)
    assert res_bare.returncode != 0
    assert res_bare.stdout == ""
    assert "cannot import required BrainLens dependencies" in res_bare.stderr
    assert "Traceback" not in res_bare.stderr

    # 3. Valid interpreter (current pytest Python with BrainLens deps) runs --help cleanly
    env_ok = {**os.environ, "BRAINLENS_PYTHON": sys.executable}
    res_ok = subprocess.run([str(launcher), "--help"], env=env_ok, capture_output=True, text=True, check=False)
    assert res_ok.returncode == 0, f"Expected valid launcher to succeed: {res_ok.stderr}"
    assert "BrainLens" in res_ok.stdout


def test_offline_bundle_scripts_syntax_and_invariants() -> None:
    for script in (BUILD_BUNDLE_SCRIPT, INSTALL_OFFLINE_SCRIPT):
        assert script.is_file(), f"Missing {script}"
        assert os.access(script, os.X_OK), f"{script} must be executable"
        text = script.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash\n")
        assert "set -euo pipefail" in text

        syntax = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
        assert syntax.returncode == 0, f"bash -n failed on {script}: {syntax.stderr}"

        help_res = subprocess.run([str(script), "--help"], capture_output=True, text=True, check=False)
        assert help_res.returncode == 0, f"--help failed on {script}: {help_res.stderr}"
        assert "Usage:" in help_res.stdout

    build_txt = BUILD_BUNDLE_SCRIPT.read_text(encoding="utf-8")
    assert "pip wheel" in build_txt
    assert "pip download" in build_txt
    assert "--platform" in build_txt
    assert "--python-version" in build_txt
    assert "--with-pyhfo" in build_txt
    assert "python-pptx>=1.0,<2" in build_txt
    assert "SHA256SUMS" in build_txt
    assert "install_offline.sh" in build_txt

    install_txt = INSTALL_OFFLINE_SCRIPT.read_text(encoding="utf-8")
    assert "--no-index" in install_txt
    assert "--find-links" in install_txt
    assert "SHA256SUMS" in install_txt


def test_install_offline_verifies_sha256sums_and_rejects_tampered_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "offline-bundle"
    wheels = bundle / "wheels"
    wheels.mkdir(parents=True)
    dummy_whl = wheels / "brainlens-1.0.0-py3-none-any.whl"
    payload = b"fake-wheel-bytes-for-checksum-verification\n"
    dummy_whl.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (bundle / "SHA256SUMS").write_text(f"{digest}  wheels/{dummy_whl.name}\n", encoding="utf-8")

    ok = subprocess.run(
        [str(INSTALL_OFFLINE_SCRIPT), "--bundle", str(bundle), "--verify-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, f"Expected valid SHA256SUMS verification to pass: {ok.stderr}"

    # Tamper with wheel contents and confirm checksum verification fails
    dummy_whl.write_bytes(payload + b"tampered")
    bad = subprocess.run(
        [str(INSTALL_OFFLINE_SCRIPT), "--bundle", str(bundle), "--verify-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode != 0, "Expected tampered bundle to fail SHA256SUMS check"


def test_deployment_docs_cover_required_topics() -> None:
    assert DEPLOYMENT_DOC.is_file(), f"Missing {DEPLOYMENT_DOC}"
    doc = DEPLOYMENT_DOC.read_text(encoding="utf-8")
    required_snippets = (
        "127.0.0.1:8765:8765",
        "forbidden_host",
        "/data/workbench_results",
        "/recordings",
        "WITH_PYHFO=1",
        "c1c7d3fec990e8661604080664766dc1297cbfdc",
        "build_offline_bundle.sh",
        "install_offline.sh",
        "SHA256SUMS",
        "--no-index",
        "export_edf_annotations",
        "brainlens export-edf",
        "export_mdt_slides",
        "brainlens export-slides",
        "python-pptx",
        "brainlens anatomy",
        "not a medical device",
        "Multi-user",
        "Backup",
        "Upgrade",
    )
    for snippet in required_snippets:
        assert snippet in doc, f"docs/deployment.md is missing required topic/snippet: {snippet!r}"
