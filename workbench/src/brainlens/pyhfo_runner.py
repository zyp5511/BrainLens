"""Standalone process-isolated PyHFO runner & ONNX model resolver."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from importlib.util import find_spec
from operator import itemgetter
from pathlib import Path
from typing import Any

import numpy as np

from .analysis_windows import prepared_windows
from .models import (
    AnalysisConfig,
    ChannelSpec,
    ClassifierConfig,
    EngineSettings,
    IS_SOURCE_CHECKOUT,
    LEGACY_ENGINE_JSON,
    LEGACY_USER_ENGINE_JSON,
    USER_ENGINE_JSON,
    ToolkitError,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    get_env,
    merge_intervals,
    read_json,
    true_runs_mask,
)

_STAGES = ("artifact", "spike", "ehfo")
# Worker input bound. Keep this out of serialized AnalysisConfig and existing stage fingerprints.
MAX_CHUNK_SECONDS = 600.0
_STAGE_TAGS = {"artifact": "a", "spike": "s", "ehfo": "e"}
REQUIRED_PYHFO_MODULES = (
    "src/hfo_app.py",
    "src/param/param_detector.py",
    "src/param/param_filter.py",
    "src/utils/utils_feature.py",
)
COMPATIBLE_PYHFO_HINT = (
    "Public upstream PyHFO: https://github.com/roychowdhuryresearch/pyHFO (verified commit c1c7d3fec990e8661604080664766dc1297cbfdc)."
)
_ENGINE_PROBE_CACHE: dict[tuple[str, str, int, int, tuple[int, ...]], tuple[bool, list[str], str | None]] = {}


def _missing_pyhfo_modules(source: Path) -> list[str]:
    return [rel for rel in REQUIRED_PYHFO_MODULES if not (source / rel).is_file()]


def _site_packages_mtimes(python_exe: Path) -> tuple[int, ...]:
    """Best-effort mtime fingerprint of the interpreter's site-packages directory(ies) for cache invalidation."""
    mtimes: list[int] = []
    env_root = python_exe.parent.parent
    candidates = [env_root / "Lib" / "site-packages"]
    lib_dir = env_root / "lib"
    if lib_dir.is_dir():
        try:
            for child in sorted(lib_dir.iterdir()):
                sp = child / "site-packages"
                if sp.is_dir():
                    candidates.append(sp)
        except OSError:
            pass
    for sp in candidates:
        try:
            if sp.is_dir():
                mtimes.append(int(sp.stat().st_mtime_ns))
        except OSError:
            continue
    return tuple(mtimes)


def _probe_worker_imports(source: Path | None, python_exe: Path, model_cache: str | None) -> tuple[bool, list[str], str | None]:
    """Verify the configured PyHFO interpreter can actually import HFO_App and worker dependencies."""
    if not python_exe.is_file():
        return False, [], f"PyHFO Python executable not found at {python_exe}"
    try:
        py_mt = int(python_exe.stat().st_mtime_ns)
        app_py = source / "src" / "hfo_app.py" if source else None
        app_mt = app_py.stat().st_mtime_ns if app_py and app_py.is_file() else 0
    except OSError:
        py_mt, app_mt = 0, 0
    cache_key = (str(source), str(python_exe), py_mt, app_mt, _site_packages_mtimes(python_exe))
    if cache_key in _ENGINE_PROBE_CACHE:
        return _ENGINE_PROBE_CACHE[cache_key]

    env = _sanitized_worker_env(source, model_cache)
    try:
        res = subprocess.run(
            [str(python_exe), str(Path(__file__).with_name("_pyhfo_worker.py")), "--probe"],
            cwd=source or Path(__file__).parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=45.0,
            check=False,
        )
    except Exception as exc:
        # Never cache failed probes so fixing the environment in place works immediately!
        return False, [], f"Failed to execute PyHFO Python probe: {exc}"

    marker = next(
        (ln for ln in (res.stdout or "").splitlines() if ln.startswith(("BRAINLENS_PROBE:", "BTK_PROBE:"))),
        None,
    )
    if res.returncode == 0 and marker:
        try:
            payload = json.loads(marker.split(":", 1)[1])
            ok = bool(payload.get("ok"))
            missing = list(payload.get("missing") or [])
            reason = None if ok else f"PyHFO Python environment failed worker import check: {'; '.join(missing[:4])}"
            out = (ok, missing, reason)
            if ok:
                _ENGINE_PROBE_CACHE[cache_key] = out
            return out
        except Exception:
            pass
    err_tail = ((res.stderr or res.stdout or "").strip().splitlines() or ["probe exited non-zero"])[-1]
    return False, [err_tail], f"PyHFO Python probe failed (rc={res.returncode}): {err_tail}"
_SAFE_ENV_EXACT = frozenset({
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TMPDIR",
    "TMP",
    "TEMP",
    "NUMBA_CACHE_DIR",
    "MNE_DONTWRITE_HOME",
    "MPLCONFIGDIR",
    "LANG",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "SYSTEMROOT",
    "COMSPEC",
})
_SAFE_ENV_PREFIXES = ("LC_", "DYLD_", "LD_LIBRARY_PATH")


def _is_owned_by_current_user(path: Path) -> bool:
    if not hasattr(os, "getuid"):
        return True
    try:
        return path.stat().st_uid == os.getuid()
    except OSError:
        return False


def _sanitized_worker_env(source: Path | None, model_cache: str | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for k, v in os.environ.items():
        if k in _SAFE_ENV_EXACT or any(k.startswith(p) for p in _SAFE_ENV_PREFIXES):
            env[k] = v
    env.update({
        "BRAINLENS_PYHFO_MODE": "source" if source else "package",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        "BRAINLENS_LAUNCHER_PID": str(os.getpid()),
    })
    if source:
        env.update(PYTHONPATH=str(source), PYTHONNOUSERSITE="1")
    else:
        for key in ("PYHFO_HOME", "PYHFO_ONNX_DIR", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            if os.environ.get(key):
                env[key] = str(Path(os.environ[key]).expanduser().resolve())
    if model_cache:
        env["PYHFO_ONNX_DIR"] = str(Path(model_cache).expanduser().resolve())
    return env


def resolve_engine_settings(workspace_dir: Path | None = None) -> EngineSettings:
    # 1. Explicit environment variables (clean-machine & container friendly; legacy BTK_* supported)
    env_src = (get_env("BRAINLENS_PYHFO_SOURCE", "BTK_PYHFO_SOURCE") or "").strip()
    env_py = (get_env("BRAINLENS_PYHFO_PYTHON", "BTK_PYHFO_PYTHON") or "").strip()
    if env_py:
        raw_env: dict[str, Any] = {
            "source_root": env_src or None,
            "python_executable": env_py,
            "model_cache": (get_env("BRAINLENS_PYHFO_MODEL_CACHE", "BTK_PYHFO_MODEL_CACHE") or "").strip() or None,
        }
        max_w = get_env("BRAINLENS_PYHFO_MAX_WORKERS", "BTK_PYHFO_MAX_WORKERS")
        if max_w:
            raw_env["max_workers"] = int(max_w)
        timeout_v = get_env("BRAINLENS_PYHFO_TIMEOUT", "BTK_PYHFO_TIMEOUT")
        if timeout_v:
            raw_env["timeout_seconds"] = float(timeout_v)
        return EngineSettings.model_validate(raw_env)

    # 2. User config (~/.brainlens/engine.json, falling back to legacy ~/.braintoolkit/engine.json);
    #    workspace engine.json requires explicit opt-in (BRAINLENS_ALLOW_WORKSPACE_ENGINE=1) so untrusted
    #    shared workspaces cannot select an interpreter.
    candidates: list[tuple[Path, bool]] = [(USER_ENGINE_JSON, False), (LEGACY_USER_ENGINE_JSON, False)]
    allow_ws_eng = get_env("BRAINLENS_ALLOW_WORKSPACE_ENGINE", "BTK_ALLOW_WORKSPACE_ENGINE", "0")
    if workspace_dir is not None and allow_ws_eng == "1":
        candidates.append((Path(workspace_dir) / "engine.json", True))
    if IS_SOURCE_CHECKOUT:
        candidates.append((LEGACY_ENGINE_JSON, True))

    for c, require_same_uid in candidates:
        if c.is_file():
            if require_same_uid and not _is_owned_by_current_user(c):
                continue
            raw = read_json(c)
            if isinstance(raw, dict) and isinstance(raw.get("engine"), dict):
                raw = raw["engine"]
            return EngineSettings.model_validate(raw)
    if find_spec("pyhfo") is not None:
        return EngineSettings(python_executable=sys.executable)
    raise ToolkitError(
        "engine_not_configured",
        "Install `brainlens[pyhfo]` or run `brainlens configure-engine --python <pyhfo_venv_python>`. "
        "For a legacy checkout, also supply --source <pyhfo_dir>.",
    )


def save_engine_settings(
    *,
    source_root: str | Path | None = None,
    python_executable: str | Path,
    model_cache: str | Path | None = None,
    max_workers: int = 2,
    timeout_seconds: float = 1200.0,
    scope: str = "user",
    workspace_dir: Path | None = None,
    fetch_models: bool = False,
) -> dict[str, Any]:
    py = Path(python_executable).expanduser().absolute()
    mc = str(Path(model_cache).expanduser().resolve()) if model_cache else None
    settings = EngineSettings(
        source_root=str(Path(source_root).expanduser().resolve()) if source_root else None,
        python_executable=str(py),
        model_cache=mc,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
    )
    if scope == "workspace":
        if workspace_dir is None:
            raise ToolkitError("missing_workspace", "workspace_dir is required for scope='workspace'.")
        out_path = Path(workspace_dir).expanduser().resolve() / "engine.json"
    else:
        out_path = USER_ENGINE_JSON
    runner = PyHFORunner(settings)  # Validate the selected engine before replacing a working configuration.
    fetched = runner.fetch_models() if fetch_models else None
    atomic_write_json(out_path, settings.model_dump(mode="json"), mode=0o600)
    return {
        "saved_to": str(out_path),
        "scope": scope,
        "engine": settings.model_dump(mode="json"),
        "health": check_engine_health(workspace_dir),
        **({"fetched_models": fetched} if fetched else {}),
    }


def check_engine_health(workspace_dir: Path | None = None) -> dict[str, Any]:
    try:
        cfg = resolve_engine_settings(workspace_dir)
    except Exception as exc:
        return {"configured": False, "available": False, "reason": str(exc)}
    return _engine_health(cfg)


def _source_version(source: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "log", "-1", "--format=%h", "--abbrev=12", "--", "src"],
            capture_output=True, text=True, timeout=3, check=True,
        )
        revision = result.stdout.strip()
        return f"git:{revision}" if revision else None
    except (OSError, subprocess.SubprocessError):
        return None


def _engine_health(cfg: EngineSettings) -> dict[str, Any]:
    src = Path(cfg.source_root).expanduser() if cfg.source_root else None
    py = Path(cfg.python_executable).expanduser().absolute()
    missing_mods = _missing_pyhfo_modules(src) if src else []
    py_ok = py.is_file()
    models: list[str] = []
    model_errors: dict[str, str] = {}
    version = _source_version(src) if src and not missing_mods else None
    import_ok = False
    missing_imports: list[str] = []
    probe_reason: str | None = None
    if not missing_mods and py_ok:
        try:
            runner = PyHFORunner(cfg)
            if runner.package:
                version = runner.package["version"]
            import_ok, missing_imports, probe_reason = _probe_worker_imports(src, py, cfg.model_cache)
            if src:
                models = list(runner.resolve_models(ClassifierConfig()))
            else:
                model_paths, model_errors = runner.package_models(_STAGES)
                models = list(model_paths)
        except ToolkitError as exc:
            if import_ok:
                model_errors = {stage: str(exc) for stage in _STAGES}
            else:
                probe_reason = str(exc)
    avail = (not missing_mods) and py_ok and import_ok
    reason = None
    if missing_mods:
        reason = f"Incompatible PyHFO checkout (missing {', '.join(missing_mods)}). {COMPATIBLE_PYHFO_HINT}"
    elif not py_ok:
        reason = f"PyHFO Python executable not found at {py}"
    elif not import_ok:
        reason = probe_reason or "PyHFO Python interpreter failed worker import verification."
    return {
        "configured": True,
        "available": avail,
        "import_ok": import_ok,
        "reason": reason,
        "missing_modules": missing_mods,
        "missing_imports": missing_imports,
        "mode": "source" if src else "package",
        "pyhfo_version": version,
        "source_root": str(src) if src else None,
        "python_executable": str(py),
        "local_onnx_models": models,
        "model_errors": model_errors,
        **({"model_setup": "brainlens configure-engine --fetch-models"} if model_errors and not src else {}),
        "max_workers": cfg.max_workers,
    }


def _kill_group(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        try:
            proc.wait(timeout=2.0)
            return
        except subprocess.TimeoutExpired:
            continue


def _detection_chunks(start: int, stop: int, limit: int, margin: int, minimum: int) -> Iterator[tuple[int, int]]:
    """Tile a valid run with overlapping inputs whose edge-trimmed interiors cover the run."""
    if stop - start > limit:
        if limit < minimum or limit <= 2 * margin:
            raise ToolkitError(
                "detector_chunk_too_short",
                "The detector chunk limit must accommodate a minimum segment and both detector edge margins.",
            )
        # Two margins cover the seam itself. An extra margin (or a minimum segment)
        # leaves room for a complete event straddling it to appear in a usable interior.
        # A very large configured minimum should extend the final input backwards,
        # without forcing every other chunk to advance by a single sample.
        overlap = min(limit - 1, max(2 * margin + 1, min(limit // 2, max(3 * margin, minimum))))
        while stop - start > limit:
            yield start, start + limit
            start = min(start + limit - overlap, stop - minimum)
    yield start, stop


class PyHFORunner:
    """Process-isolated bridge to an installed PyHFO package or a legacy checkout."""

    def __init__(self, settings: EngineSettings):
        self.settings = settings
        self.source = Path(settings.source_root).expanduser().resolve() if settings.source_root else None
        self.python = Path(settings.python_executable).expanduser().absolute()
        missing_mods = _missing_pyhfo_modules(self.source) if self.source else []
        if missing_mods:
            raise ToolkitError(
                "engine_incompatible",
                f"PyHFO checkout at {self.source} is missing required worker modules: {', '.join(missing_mods)}. {COMPATIBLE_PYHFO_HINT}",
                missing_modules=missing_mods,
            )
        if not self.python.is_file():
            raise ToolkitError("engine_python_missing", f"PyHFO Python missing at {self.python}")
        self.package = None if self.source else self._package_command("describe")

    def _package_command(self, action: str, *tags: str) -> dict[str, Any]:
        helper = Path(__file__).with_name("_pyhfo_package.py")
        env = _sanitized_worker_env(None, self.settings.model_cache)
        if action == "fetch":
            env.pop("HF_HUB_OFFLINE")
            env.pop("TRANSFORMERS_OFFLINE")
        code = "engine_incompatible" if action == "describe" else "engine_models_unavailable"
        try:
            res = subprocess.run(
                [str(self.python), str(helper), action, *tags],
                cwd=helper.parent, env=env, capture_output=True, text=True,
                timeout=self.settings.timeout_seconds if action == "fetch" else 45,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ToolkitError(code, f"PyHFO package {action} failed in {self.python}: {exc}") from exc
        try:
            reply = json.loads(res.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            detail = res.stderr.strip().splitlines() or ["No JSON response from the engine interpreter."]
            raise ToolkitError(code, f"PyHFO package {action} failed in {self.python}: {detail[-1]}") from exc
        if res.returncode or "error" in reply:
            error = reply.get("error") or res.stderr.strip()
            raise ToolkitError(code, f"PyHFO package {action} failed in {self.python}: {error}")
        return reply

    def package_models(
        self, stages: Sequence[str], *, fetch: bool = False,
    ) -> tuple[dict[str, str], dict[str, str]]:
        reply = self._package_command("fetch" if fetch else "models", *(_STAGE_TAGS[s] for s in stages))
        paths, errors = {}, {}
        for stage in stages:
            tag = _STAGE_TAGS[stage]
            if tag in reply["paths"]:
                paths[stage] = reply["paths"][tag]
            else:
                errors[stage] = reply["errors"][tag]
        return paths, errors

    def fetch_models(self) -> dict[str, str]:
        if self.source:
            raise ToolkitError(
                "engine_incompatible", "--fetch-models requires PyHFO 4.x. Use --python <pyhfo_4_venv_python> to select it.",
            )
        paths, errors = self.package_models(_STAGES, fetch=True)
        if errors:
            raise ToolkitError(
                "engine_models_unavailable",
                f"Could not fetch pinned ONNX models: {', '.join(errors)}. {next(iter(errors.values()))}",
                stages=errors,
            )
        return paths

    def source_identity(self) -> dict[str, Any]:
        common = {
            "name": "PyHFO",
            "mode": "source" if self.source else "package",
            "worker_sha256": file_sha256(Path(__file__).with_name("_pyhfo_worker.py")),
            "python_executable": str(self.python),
        }
        if self.package is not None:
            return {**common, **self.package}
        files = {
            str(p.relative_to(self.source)): file_sha256(p)
            for p in sorted((self.source / "src").rglob("*.py"))
        }
        return {
            **common,
            "source_root": str(self.source),
            "source_sha256": canonical_digest(files),
        }

    def resolve_models(self, cfg: ClassifierConfig, stages: Sequence[str] | None = None) -> dict[str, str]:
        out = dict(cfg.model_sources)
        if self.package is not None:
            needed = [s for s in (_STAGES if stages is None else stages) if s not in out]
            if needed:
                paths, errors = self.package_models(needed)
                if errors:
                    raise ToolkitError(
                        "engine_models_unavailable",
                        f"ONNX models unavailable offline: {', '.join(errors)}. "
                        "Run `brainlens configure-engine --fetch-models` with network access.",
                        stages=errors,
                    )
                out.update(paths)
            return out
        if self.settings.model_cache:
            root = Path(self.settings.model_cache).expanduser().resolve()
            for stage, tag in _STAGE_TAGS.items():
                if stage in out:
                    continue
                for rel in (f"model_{tag}/model.onnx", f"model_{tag}.onnx", f"{stage}/model.onnx"):
                    cand = root / rel
                    if cand.is_file() and cand.with_suffix(".spec.json").is_file():
                        out[stage] = str(cand)
                        break
        return out

    def _invoke(self, action: str, payload: dict[str, Any], workdir: Path) -> dict[str, Any]:
        workdir.mkdir(parents=True, exist_ok=True)
        req_file = workdir / "request.json"
        res_file = workdir / "result.json"
        res_file.unlink(missing_ok=True)
        atomic_write_json(req_file, {"action": action, **payload})

        env = _sanitized_worker_env(self.source, self.settings.model_cache)

        cmd = [str(self.python), str(Path(__file__).with_name("_pyhfo_worker.py")), str(req_file), str(res_file)]
        with (workdir / "worker.log").open("wb") as log_fh:
            proc = subprocess.Popen(
                cmd,
                cwd=self.source,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            t0 = time.monotonic()
            try:
                while proc.poll() is None:
                    if time.monotonic() - t0 > self.settings.timeout_seconds:
                        raise ToolkitError("engine_timeout", "PyHFO worker timed out.")
                    time.sleep(0.05)
                if not res_file.is_file():
                    raise ToolkitError("engine_failed", f"PyHFO exited ({proc.returncode}) without result.json.")
                resp = read_json(res_file)
                if resp.get("status") == "worker_failed" or proc.returncode != 0:
                    raise ToolkitError("engine_failed", resp.get("message", "PyHFO worker failed."))
                return resp
            finally:
                _kill_group(proc)

    def detect(self, prepared: dict[str, Any], config: AnalysisConfig, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        channels = [ChannelSpec.model_validate(c) for c in prepared["channels"]]
        sfreq = float(prepared["sfreq"])
        first_sample = int(prepared["source_first_sample"])

        if not config.detector.enabled:
            res = {
                "status": "skipped",
                "reason": "detector_disabled",
                "events": [],
                "valid_intervals": {c.id: [] for c in channels},
                "segments": [],
                "backend": {},
                "skipped_channels": {},
                "event_backend": {},
                "warnings": [],
            }
            atomic_write_json(output_dir / "detection.json", res)
            return res

        if any(not float(b).is_integer() for b in config.detector.band):
            raise ToolkitError("band_not_integer", "PyHFO requires integer Hz detector band edges.")

        data = np.load(prepared["data_path"], mmap_mode="r", allow_pickle=False)
        valid = np.load(prepared["valid_path"], mmap_mode="r", allow_pickle=False)
        min_samples = int(math.ceil(config.detector.minimum_segment_seconds * sfreq))
        margin = int(math.ceil(config.detector.edge_seconds * sfreq))
        chunk_limit = int(math.floor(MAX_CHUNK_SECONDS * sfreq))
        hi_hz = config.detector.band[1]

        # Masks and stored sample intervals use packed source samples. A boundary at a
        # gap can have valid samples on both sides, even when the detector margin is zero.
        windows = prepared_windows(prepared)
        grouped: dict[tuple[int, int, float, int], list[int]] = {}
        valid_interiors: dict[str, list[list[int]]] = {c.id: [] for c in channels}
        chunked_channels: set[str] = set()
        skipped: dict[str, Any] = {}
        for idx, ch in enumerate(channels):
            if ch.unit != "V":
                skipped[ch.id] = {"reason": "unverified_units", "unit": ch.unit}
                continue
            nyq = min(float(ch.source_sfreq or sfreq), sfreq) / 2.0
            lp_candidates = [
                v for v in (ch.source_lowpass, prepared.get("input_lowpass"), (prepared.get("effective_filters") or {}).get("lowpass"))
                if v is not None and math.isfinite(float(v))
            ]
            eff_lp = min(lp_candidates) if lp_candidates else None
            if hi_hz >= nyq or (eff_lp is not None and hi_hz > eff_lp):
                skipped[ch.id] = {"reason": "band_exceeds_bandwidth", "nyquist_hz": nyq, "lowpass_hz": eff_lp}
                continue
            bound_lp = min(nyq, eff_lp) if eff_lp is not None else nyq
            for piece_idx, window in enumerate(windows):
                lo, hi = window.array_start, window.array_stop
                for left, right in true_runs_mask(valid[idx, lo:hi]):
                    a, b = lo + left, lo + right
                    if b - a < min_samples:
                        continue
                    if b - a > 2 * margin:
                        valid_interiors[ch.id].append([window.offset + a + margin, window.offset + b - margin])
                    if b - a > chunk_limit:
                        chunked_channels.add(ch.id)
                    for x, y in _detection_chunks(a, b, chunk_limit, margin, min_samples):
                        grouped.setdefault((x, y, bound_lp, piece_idx), []).append(idx)

        backend_id = self.source_identity()
        all_events: list[dict[str, Any]] = []
        event_backend: dict[str, Any] = {}
        segments: list[dict[str, Any]] = []
        warnings: list[str] = []
        earlier_events: dict[str, list[tuple[int, int]]] = {}

        for seg_num, ((a, b, acq_lp, piece_idx), idxs) in enumerate(sorted(grouped.items()), start=1):
            seg_id = f"seg_{seg_num:03d}"
            seg_dir = output_dir / seg_id
            seg_chs = [channels[i] for i in idxs]
            total_ch_samps = len(idxs) * (b - a)
            n_jobs = self.settings.max_workers if total_ch_samps >= self.settings.parallel_min_channel_samples or len(idxs) >= 8 else 1

            obs = self._invoke(
                "detect",
                {
                    "data_path": prepared["data_path"],
                    "indexes": idxs,
                    "start": a,
                    "stop": b,
                    "sfreq": sfreq,
                    "channel_names": [c.name for c in seg_chs],
                    "lowpass": acq_lp,
                    "detector": config.detector.model_dump(mode="json"),
                    "n_jobs": n_jobs,
                },
                seg_dir,
            )
            backend_id["versions"] = obs.get("versions", {})
            ch_by_name = {c.name: c for c in seg_chs}
            sig_hashes = {
                channels[i].id: hashlib.sha256(np.ascontiguousarray(data[i, a:b]).view(np.uint8)).hexdigest()
                for i in idxs
            }
            window = windows[piece_idx]
            source_start = window.offset + a
            source_stop = window.offset + b
            in_lo = source_start + margin
            in_hi = source_stop - margin
            piece = window.piece
            provenance = ({"segment_index": window.segment_index, "segment_label": config.segments[window.segment_index].label}
                          if window.segment_index is not None else {})
            chunk_events: dict[str, list[tuple[int, int]]] = {}

            for raw_ev in obs.get("events", []):
                ch = ch_by_name[raw_ev["channel_name"]]
                ev_s0 = source_start + round(raw_ev["start_sample"] * sfreq / obs["backend_sfreq"])
                ev_s1 = source_start + round(raw_ev["end_sample"] * sfreq / obs["backend_sfreq"])
                if ev_s0 < in_lo or ev_s1 > in_hi or ev_s1 <= ev_s0:
                    continue
                if ch.id in chunked_channels:
                    # Keep the first complete candidate in chunk order, including when
                    # the backend shifts its onset/end on the overlapping input.
                    earlier = earlier_events.get(ch.id, [])
                    j = bisect_right(earlier, ev_s0, key=itemgetter(1))
                    if j < len(earlier) and earlier[j][0] < ev_s1:
                        continue
                    chunk_events.setdefault(ch.id, []).append((ev_s0, ev_s1))
                eid = "ev_" + canonical_digest({
                    "rec": prepared["recording_fingerprint"],
                    "ch": ch.id,
                    "s0": ev_s0,
                    "s1": ev_s1,
                    "det": config.detector.model_dump(mode="json"),
                    "sig": sig_hashes[ch.id],
                })[:20]
                all_events.append({
                    "id": eid,
                    "channel_id": ch.id,
                    "channel_name": ch.name,
                    "shaft": ch.shaft,
                    "start_sample": ev_s0,
                    "end_sample": ev_s1,
                    "source_start_sample": first_sample + ev_s0,
                    "source_end_sample": first_sample + ev_s1,
                    "start_seconds": piece.time_at(ev_s0, sfreq) if prepared.get("pieces") else round(ev_s0 / sfreq, 6),
                    "end_seconds": piece.time_at(ev_s1, sfreq) if prepared.get("pieces") else round(ev_s1 / sfreq, 6),
                    "detector": config.detector.name,
                    "segment_id": seg_id,
                    **provenance,
                    "scores": {},
                    "labels": {},
                })
                event_backend[eid] = {
                    "segment_id": seg_id,
                    "index": raw_ev["index"],
                    "start_sample": raw_ev["start_sample"],
                    "end_sample": raw_ev["end_sample"],
                    "channel_name": ch.name,
                }

            for cid, new in chunk_events.items():
                active = [span for span in earlier_events.get(cid, []) if span[1] > in_lo]
                earlier_events[cid] = merge_intervals(active + new)
            segments.append({
                "id": seg_id,
                **provenance,
                "status": "complete",
                "start_sample": source_start,
                "end_sample": source_stop,
                "channel_ids": [c.id for c in seg_chs],
                "session_path": obs["session_path"],
                "backend_sfreq": obs["backend_sfreq"],
                "backend_samples": obs["backend_samples"],
            })
            warnings.extend(obs.get("warnings", []))

        all_events.sort(key=lambda e: (e["start_sample"], e["channel_id"], e["end_sample"]))

        det_doc = {
            "status": "complete" if segments else "skipped",
            "events": all_events,
            "valid_intervals": valid_interiors,
            "segments": segments,
            "backend": backend_id,
            "skipped_channels": skipped,
            "event_backend": event_backend,
            "warnings": list(dict.fromkeys(warnings)),
        }
        atomic_write_json(output_dir / "detection.json", det_doc)
        return det_doc

    def classify(
        self,
        detection: dict[str, Any],
        config: ClassifierConfig,
        output_dir: Path,
        prior_classification: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        res = copy.deepcopy(detection)
        requested = [st for st in _STAGES if getattr(config, st)] if config.enabled else []
        stages = list(requested)
        if any(st in stages for st in ("spike", "ehfo")) and "artifact" not in stages:
            stages.insert(0, "artifact")

        for ev in res["events"]:
            ev["scores"], ev["labels"] = {}, {}

        if not stages:
            res["classifier_status"] = "skipped"
            res["classification"] = {"requested": [], "stages": {}, "score_cache_by_segment": {}}
            atomic_write_json(output_dir / "classification.json", res)
            return res

        if not res["events"]:
            res["classifier_status"] = "complete"
            res["classification"] = {"requested": requested, "stages": {}, "score_cache_by_segment": {}}
            atomic_write_json(output_dir / "classification.json", res)
            return res

        ev_by_id = {e["id"]: e for e in res["events"]}
        sources = self.resolve_models(config, stages)
        prior_meta = (prior_classification or {}).get("classification", {})
        prior_seg_caches = prior_meta.get("score_cache_by_segment", {})
        prior_seg_hashes = prior_meta.get("model_hashes_by_segment", {})
        prior_seg_sessions = prior_meta.get("session_path_by_segment", {})

        seg_caches_out: dict[str, Any] = {}
        seg_hashes_out: dict[str, Any] = {}
        seg_sessions_out: dict[str, str] = {}
        stage_summaries: dict[str, Any] = {}
        warnings = list(res.get("warnings", []))

        def _remap_relocated_session_path(sp_raw: Any) -> str | None:
            if not sp_raw:
                return None
            p_sp = Path(str(sp_raw))
            if p_sp.is_file():
                return str(p_sp)
            # If workspace was moved, locate workspace root from output_dir (.../cases/<case>/...) and remap all /cases/ splits
            norm_sp = str(sp_raw).replace("\\", "/")
            for anc in output_dir.resolve().parents:
                if (anc / "cases").is_dir():
                    search_end = len(norm_sp)
                    while search_end > 0:
                        idx = norm_sp.rfind("/cases/", 0, search_end)
                        if idx < 0:
                            break
                        cand = (anc / norm_sp[idx + 1 :]).resolve()
                        if cand.is_file() and cand.is_relative_to(anc):
                            return str(cand)
                        search_end = idx + len("/cases/") - 1
                    break
            return str(sp_raw)

        for seg in detection.get("segments", []):
            sid = seg["id"]
            ev_map = {
                eid: mp
                for eid, mp in detection.get("event_backend", {}).items()
                if mp["segment_id"] == sid and eid in ev_by_id
            }
            if not ev_map:
                continue
            eff_sess_path = _remap_relocated_session_path(seg.get("session_path"))
            prior_sess = _remap_relocated_session_path(prior_seg_sessions.get(sid))
            same_sess = bool(prior_sess) and prior_sess == eff_sess_path
            obs = self._invoke(
                "classify",
                {
                    "session_path": eff_sess_path,
                    "prior_session_path": prior_sess if same_sess else None,
                    "event_map": ev_map,
                    "model_sources": sources,
                    "stages": stages,
                    "thresholds": config.thresholds,
                    "n_jobs": self.settings.max_workers,
                    "score_cache": prior_seg_caches.get(sid, {}) if same_sess else {},
                    "model_hashes": prior_seg_hashes.get(sid, {}) if same_sess else {},
                },
                output_dir / f"clf_{sid}",
            )
            seg_caches_out[sid] = obs.get("score_cache", {})
            seg_sessions_out[sid] = str(eff_sess_path or "")
            seg_hashes_out[sid] = {
                st: {"model": st_d.get("model_sha256"), "spec": st_d.get("spec_sha256")}
                for st, st_d in obs.get("stages", {}).items()
            }
            stage_summaries[sid] = obs.get("stages", {})
            for eid, st_vals in obs.get("assessments", {}).items():
                for st, v in st_vals.items():
                    ev_by_id[eid]["scores"][st] = v["score"]
                    ev_by_id[eid]["labels"][st] = v["label"]
            warnings.extend(obs.get("warnings", []))

        stages_rollup: dict[str, dict[str, Any]] = {}
        for st in stages:
            st_entries = [seg_st[st] for seg_st in stage_summaries.values() if st in seg_st]
            if not st_entries:
                stages_rollup[st] = {"status": "skipped"}
            elif all(e.get("status") == "complete" for e in st_entries):
                stages_rollup[st] = {"status": "complete", "model_sha256": st_entries[0].get("model_sha256")}
            else:
                bad = next((e for e in st_entries if e.get("status") != "complete"), st_entries[0])
                stages_rollup[st] = {
                    "status": bad.get("status", "unavailable"),
                    "reason": bad.get("reason") or bad.get("error"),
                }

        all_statuses = [
            st_d.get("status")
            for seg_st in stage_summaries.values()
            for st_d in seg_st.values()
        ]
        n_ok = sum(s == "complete" for s in all_statuses)
        n_bad = sum(s != "complete" for s in all_statuses)
        res["classifier_status"] = "complete" if (n_ok and not n_bad) else ("partial" if n_ok else "failed")
        res["warnings"] = list(dict.fromkeys(warnings))
        res["classification"] = {
            "requested": requested,
            "required": stages,
            "stages": stages_rollup,
            "segments": stage_summaries,
            "score_cache_by_segment": seg_caches_out,
            "model_hashes_by_segment": seg_hashes_out,
            "session_path_by_segment": seg_sessions_out,
        }
        atomic_write_json(output_dir / "classification.json", res)
        return res
