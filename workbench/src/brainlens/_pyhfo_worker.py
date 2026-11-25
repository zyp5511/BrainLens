"""Standalone isolated PyHFO subprocess worker (runs inside engine-venv, zero brainlens imports)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import signal
import sys
import threading
import traceback
import types
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
from pathlib import Path


_ENGINE_NAMESPACE = "pyhfo" if os.environ.get("BRAINLENS_PYHFO_MODE") == "package" else "src"


def _engine_module(name: str) -> types.ModuleType:
    return import_module(f"{_ENGINE_NAMESPACE}.{name}")


def _ensure_headless_pyhfo_compat() -> None:
    """Allow importing stock public roychowdhuryresearch/pyHFO without PyQt5/pyqtgraph GUI libraries."""
    if _ENGINE_NAMESPACE == "pyhfo" or "src.utils.utils_plotting" in sys.modules:
        return
    try:
        import PyQt5  # noqa: F401
        import pyqtgraph  # noqa: F401
        return
    except ImportError:
        pass
    stub = types.ModuleType("src.utils.utils_plotting")
    stub.plot_feature = None  # type: ignore[attr-defined]
    sys.modules["src.utils.utils_plotting"] = stub


def _start_watchdog(expected_ppid: int, pgid: int) -> None:
    def _watch() -> None:
        while True:
            if os.getppid() != expected_ppid:
                os.killpg(pgid, signal.SIGKILL)
                return
            threading.Event().wait(0.15)

    if os.getppid() != expected_ppid:
        os.killpg(pgid, signal.SIGKILL)
    threading.Thread(target=_watch, daemon=True, name="brainlens-parent-watchdog").start()


def _init_pool_worker(expected_ppid: int, pgid: int) -> None:
    _start_watchdog(expected_ppid, pgid)


class _BoundedPool:
    """Reuse one bounded ProcessPoolExecutor across upstream PyHFO channel/event tasks."""

    def __init__(self, n_jobs: int, supervised: bool):
        self.n_jobs = n_jobs
        self.supervised = supervised
        self.pool: ProcessPoolExecutor | None = None
        self.saved: list[tuple[object, object]] = []

    def __enter__(self):
        if self.n_jobs > 1:
            from HFODetector import utils as det_utils
            utils_feature = _engine_module("utils.utils_feature")

            for mod in (det_utils, utils_feature):
                self.saved.append((mod, mod.ProcessPoolExecutor))
                mod.ProcessPoolExecutor = self._borrow
        return self

    def _borrow(self, max_workers: int):
        owner = self

        class _Lease:
            def __enter__(self):
                if owner.pool is None:
                    kwargs = {}
                    if owner.supervised:
                        kwargs = {"initializer": _init_pool_worker, "initargs": (os.getpid(), os.getpgrp())}
                    owner.pool = ProcessPoolExecutor(
                        max_workers=owner.n_jobs,
                        mp_context=multiprocessing.get_context("spawn"),
                        **kwargs,
                    )
                return owner.pool

            def __exit__(self, exc_type, _exc, _tb):
                if exc_type is not None:
                    owner.close()
                return False

        return _Lease()

    def close(self) -> None:
        p, self.pool = self.pool, None
        if p is not None:
            p.shutdown(wait=True, cancel_futures=True)

    def __exit__(self, _et, _ev, _tb):
        try:
            self.close()
        finally:
            for mod, orig in reversed(self.saved):
                mod.ProcessPoolExecutor = orig
            self.saved.clear()
        return False


def _atomic_json(path: Path, data: dict) -> None:
    tmp = Path(f"{path}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, allow_nan=False, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while blk := fh.read(1024 * 1024):
            h.update(blk)
    return h.hexdigest()


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "executable": sys.executable}
    for pkg in ("numpy", "scipy", "mne", "HFODetector", "onnxruntime", "torch"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _do_detect(req: dict, workdir: Path, supervised: bool) -> dict:
    import mne
    import numpy as np
    _ensure_headless_pyhfo_compat()
    HFO_App = _engine_module("hfo_app").HFO_App

    n_jobs = max(1, int(req.get("n_jobs", 1)))
    data = np.load(req["data_path"], mmap_mode="r", allow_pickle=False)
    values = np.asarray(data[req["indexes"], req["start"] : req["stop"]], dtype=np.float64)
    rate = float(req["sfreq"])
    backend_rate = math.ceil(rate)

    raw = mne.io.RawArray(values, mne.create_info(req["channel_names"], rate, ch_types="seeg"), verbose="ERROR")
    if req.get("lowpass") is not None:
        with raw.info._unlock():
            raw.info["lowpass"] = min(float(req["lowpass"]), rate / 2.0)
    if rate != backend_rate:
        raw.resample(backend_rate, npad="auto", n_jobs=1, verbose="ERROR")

    in_fif = workdir / "input_raw.fif"
    raw.save(in_fif, fmt="double", overwrite=True, verbose="ERROR")
    det = req["detector"]

    try:
        run_detection = getattr(_engine_module("agent_api.pipeline"), "run_detection", None)
    except ImportError:
        run_detection = None

    warnings: list[str] = []
    if run_detection is not None:
        payload = {
            "input_path": str(in_fif),
            "output_dir": str(workdir / "upstream"),
            "biomarker": "HFO",
            "detector": det["name"],
            "filter": {"fp": det["band"][0], "fs": det["band"][1]},
            "detector_params": {**det["parameters"], "pass_band": det["band"][0], "stop_band": det["band"][1]},
            "classifier": {"enabled": False},
            "exports": ["csv", "session", "summary_json"],
            "session_format": "pybrain",
            "n_jobs": n_jobs,
        }
        with _BoundedPool(n_jobs, supervised):
            obs = run_detection(payload)
        if obs.get("status") != "complete":
            return obs
        session_path = obs["artifacts"]["session"]
        app = HFO_App.import_app(session_path)
        warnings.extend(list(obs.get("warnings", [])))
    else:
        # Direct execution against public upstream roychowdhuryresearch/pyHFO (HFO_App + ParamFilter + ParamDetector)
        detector_params = _engine_module("param.param_detector")
        ParamFilter = _engine_module("param.param_filter").ParamFilter

        det_name = str(det["name"]).upper()
        if det_name == "STE":
            det_param = detector_params.ParamSTE(sample_freq=backend_rate, n_jobs=n_jobs)
        elif det_name == "MNI":
            det_param = detector_params.ParamMNI(sample_freq=backend_rate, n_jobs=n_jobs)
        elif det_name == "HIL":
            det_param = detector_params.ParamHIL(sample_freq=backend_rate, n_jobs=n_jobs)
        else:
            raise ValueError(f"Unsupported detector {det['name']!r}")
        for k, v in (det.get("parameters") or {}).items():
            if not hasattr(det_param, k):
                raise ValueError(f"Unsupported parameter for {det_param.__class__.__name__}: {k}")
            setattr(det_param, k, v)
        det_param.pass_band = int(det["band"][0])
        det_param.stop_band = int(det["band"][1])
        det_param.sample_freq = backend_rate
        det_param.n_jobs = n_jobs

        app = HFO_App()
        app.load_raw(raw, file_path=str(in_fif))
        app.set_n_jobs(n_jobs)
        pf = ParamFilter(fp=int(det["band"][0]), fs=int(det["band"][1]), sample_freq=int(app.sample_freq))
        app.set_filter_parameter(pf)
        app.filter_eeg_data(pf)
        app.set_detector(detector_params.ParamDetector(det_param, detector_type=det_name))
        with _BoundedPool(n_jobs, supervised):
            app.detect_biomarker()
        out_dir = workdir / "upstream"
        out_dir.mkdir(parents=True, exist_ok=True)
        session_path = str(out_dir / "session.pybrain")
        app.export_app(session_path)
        dur_s = raw.n_times / float(backend_rate)
        if len(app.event_features.starts) == 0 and dur_s < 60.0:
            warnings.append(
                f"Recording is only {dur_s:.1f}s long; HFO detectors need a longer segment "
                "(at least 60s of interictal data is recommended) to estimate detection thresholds, "
                "so 0 events is expected. Load a longer recording."
            )

    feat = app.event_features
    events = [
        {"index": i, "channel_name": str(ch), "start_sample": int(s), "end_sample": int(e)}
        for i, (ch, s, e) in enumerate(zip(feat.channel_names, feat.starts, feat.ends))
    ]
    if det["name"] == "MNI":
        warnings.append("Upstream MNI uses an unseeded random baseline; exact event equality across runs is not guaranteed.")
    # input_raw.fif is only the hand-off format for upstream detection (a float64 copy of the prepared segment,
    # ~200 MB per HUP134 run). The exported session checkpoint carries its own signal arrays and is what later
    # classification reloads, and the prepared data.npy stays the canonical cached signal, so drop the copy.
    try:
        in_fif.unlink(missing_ok=True)
    except OSError:
        pass
    return {
        "status": "complete",
        "events": events,
        "session_path": session_path,
        "backend_sfreq": float(app.sample_freq),
        "backend_samples": int(raw.n_times),
        "effective_parameters": {"filter": app.param_filter.to_dict(), "detector": app.param_detector.to_dict()},
        "versions": _versions(),
        "warnings": warnings,
    }


class _StandalonePreProcessing:
    """Pure-NumPy spectrogram crop matching PyHFO PreProcessing for stock upstream checkouts."""

    def __init__(self, d: dict):
        self.image_size = int(d["image_size"])
        self.freq_range = list(d["freq_range_hz"])
        self.fs = d["fs"]
        self.event_length = d["time_range_ms"][1]
        self.crop_time = d["selected_window_size_ms"]
        self.crop_freq = list(d["selected_freq_range_hz"])

    def process_biomarker_feature(self, feature):
        import numpy as np

        data = feature.get_features()
        freq_low, freq_high = float(feature.freq_range[0]), float(feature.freq_range[1])
        ev_len = float(max(feature.time_range))
        img_sz = self.image_size
        f_idx_low = img_sz - img_sz / (freq_high - freq_low) * (float(self.crop_freq[0]) - freq_low)
        f_idx_high = img_sz - img_sz / (freq_high - freq_low) * (float(self.crop_freq[1]) - freq_low)
        crop_f = np.array([f_idx_high, f_idx_low]).astype(int)
        crop_range_idx = float(self.crop_time) / ev_len * img_sz
        crop_t = (img_sz // 2) + np.array([-crop_range_idx, crop_range_idx]).astype(int)
        for v in (f_idx_low, f_idx_high, crop_t[0], crop_t[1]):
            if v < 0 or v > img_sz:
                raise ValueError(f"Model crop index {v:.1f} outside [0, {img_sz}]")
        return data[:, :, crop_f[0] : crop_f[1], crop_t[0] : crop_t[1]]


class _StandaloneOnnxModel:
    """Torch-free ONNX classifier runner matching PyHFO OnnxBiomarkerModel for stock upstream checkouts."""

    def __init__(self, onnx_path: str, spec_path: str | None = None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.in_channels = inp.shape[1] if len(inp.shape) >= 2 and isinstance(inp.shape[1], int) else None

    def predict_proba(self, x, batch_size: int = 256):
        import numpy as np

        x = np.asarray(x)
        if x.shape[0] == 0:
            return np.empty((0, 1), dtype=np.float32)
        if self.in_channels is not None and x.shape[1] > self.in_channels:
            x = x[:, : self.in_channels]
        x = x.astype(np.float64).copy()
        for c in range(min(x.shape[1], 3)):
            ch = x[:, c]
            flat = ch.reshape(ch.shape[0], -1)
            lo = flat.min(axis=1, keepdims=True)
            hi = flat.max(axis=1, keepdims=True)
            span = np.where(hi - lo == 0, 1.0, hi - lo)
            x[:, c] = (255.0 * (flat - lo) / span).reshape(ch.shape)
        x = x.astype(np.float32)
        outs = [self.session.run(None, {self.input_name: x[s : s + batch_size]})[0] for s in range(0, x.shape[0], batch_size)]
        return np.concatenate(outs, axis=0)


def _do_classify(req: dict, workdir: Path, supervised: bool) -> dict:
    import numpy as np
    _ensure_headless_pyhfo_compat()
    HFO_App = _engine_module("hfo_app").HFO_App

    try:
        PreProcessing = getattr(_engine_module("preprocessing"), "PreProcessing", None)
    except ImportError:
        PreProcessing = None
    try:
        OnnxBiomarkerModel = getattr(_engine_module("utils.onnx_inference"), "OnnxBiomarkerModel", None)
    except ImportError:
        OnnxBiomarkerModel = None

    app = HFO_App.import_app(req["session_path"])
    n_jobs = max(1, int(req.get("n_jobs", 1)))
    app.set_n_jobs(n_jobs)
    stages = req["stages"]
    event_map = req["event_map"]

    with _BoundedPool(n_jobs, supervised):
        app.generate_HFO_features()
    feat = app.event_features

    assessments = {
        eid: {st: {"evaluated": False, "reason": "not_executed", "score": None, "label": None} for st in stages}
        for eid in event_map
    }
    details: dict = {}
    warnings: list[str] = []
    score_cache: dict = {}
    artifact_labels = np.full(len(feat.starts), -1, dtype=int)

    for stage in stages:
        threshold = float(req.get("thresholds", {}).get(stage, 0.5))
        details[stage] = {"status": "running", "threshold": threshold}
        try:
            if stage != "artifact" and details.get("artifact", {}).get("status") != "complete":
                raise ValueError("Required artifact stage did not complete")
            src = req["model_sources"].get(stage)
            if not src:
                raise ValueError(f"No local ONNX model configured for stage '{stage}'")
            # Keep the snapshot filenames: HF symlinks the ONNX and its spec to separate blobs.
            mpath = Path(src).expanduser().absolute()
            if mpath.is_dir():
                mpath = mpath / "model.onnx"
            spath = mpath.with_suffix(".spec.json")
            if not spath.is_file() and mpath.is_symlink():
                spath = mpath.resolve().with_suffix(".spec.json")
            if not mpath.is_file() or not spath.is_file():
                raise ValueError(f"Missing model.onnx or .spec.json for '{stage}'")

            mhash, shash = _sha256(mpath), _sha256(spath)
            spec = json.loads(spath.read_text(encoding="utf-8"))
            preproc = spec.get("preprocessing")
            if not preproc:
                raise ValueError("Model .spec.json lacks preprocessing contract")

            crop = preproc["selected_freq_range_hz"]
            if float(crop[0]) < feat.freq_range[0] or float(crop[1]) > feat.freq_range[1]:
                raise ValueError(f"Model crop {crop} Hz exceeds recording feature bandwidth {feat.freq_range} Hz")

            if stage == "artifact":
                eligible = (feat.starts > app.sample_freq) & (feat.ends < len(app.eeg_data[0]) - app.sample_freq)
                inelig = "outside_1s_edge_support"
            else:
                eligible = artifact_labels > 0
                inelig = "rejected_by_artifact_stage"

            target_idx = sorted({m["index"] for m in event_map.values() if eligible[m["index"]]})
            for eid, m in event_map.items():
                if not eligible[m["index"]]:
                    assessments[eid][stage]["reason"] = inelig

            processor = PreProcessing.from_dict(preproc) if PreProcessing is not None else _StandalonePreProcessing(preproc)
            images = processor.process_biomarker_feature(feat)
            if stage == "ehfo":
                images = np.repeat(images[:, :1, :, :], repeats=3, axis=1)

            prior_cache = req.get("score_cache", {}).get(stage, {})
            prior_hashes = req.get("model_hashes", {}).get(stage, {})
            same_session = bool(req.get("prior_session_path")) and req.get("prior_session_path") == req.get("session_path")
            reusable = same_session and prior_hashes.get("model") == mhash and prior_hashes.get("spec") == shash

            # Map eligible event IDs to backend feature indices and reuse ONLY by exact content-hashed event_id
            eid_by_idx: dict[int, list[str]] = {}
            for eid, m in event_map.items():
                if eligible[m["index"]]:
                    eid_by_idx.setdefault(int(m["index"]), []).append(eid)

            retained_by_eid: dict[str, float] = {}
            pending_idx: list[int] = []
            for idx in target_idx:
                eids = eid_by_idx.get(idx, [])
                if reusable and eids and all(eid in prior_cache for eid in eids):
                    for eid in eids:
                        retained_by_eid[eid] = float(prior_cache[eid])
                else:
                    pending_idx.append(idx)

            if pending_idx:
                model = (
                    OnnxBiomarkerModel(str(mpath), spec_path=str(spath))
                    if OnnxBiomarkerModel is not None
                    else _StandaloneOnnxModel(str(mpath), spec_path=str(spath))
                )
                try:
                    preds = np.asarray(model.predict_proba(images[pending_idx], batch_size=256)).reshape(-1)
                finally:
                    del model
                for idx, val in zip(pending_idx, preds):
                    for eid in eid_by_idx.get(idx, []):
                        retained_by_eid[eid] = float(val)

            score_cache[stage] = dict(retained_by_eid)
            for eid, m in event_map.items():
                if eid in retained_by_eid:
                    idx = m["index"]
                    sc = retained_by_eid[eid]
                    lbl = int(sc > threshold)
                    assessments[eid][stage] = {"evaluated": True, "reason": None, "score": sc, "label": lbl}
                    if stage == "artifact":
                        artifact_labels[idx] = lbl

            details[stage].update(
                status="complete",
                evaluated=len(target_idx),
                fresh_inferences=len(pending_idx),
                reused_scores=len(target_idx) - len(pending_idx),
                model_sha256=mhash,
                spec_sha256=shash,
            )
        except Exception as exc:
            details[stage].update(status="unavailable", error=str(exc))
            warnings.append(f"{stage} unavailable: {exc}")
            for eid in event_map:
                assessments[eid][stage]["reason"] = f"unavailable: {exc}"

    n_fail = sum(v["status"] != "complete" for v in details.values())
    status = "complete" if n_fail == 0 else ("failed" if n_fail == len(details) else "partial")
    return {
        "status": status,
        "stages": details,
        "assessments": assessments,
        "score_cache": score_cache,
        "versions": _versions(),
        "warnings": warnings,
    }


def main() -> int:
    if sys.argv[1:] == ["--probe"]:
        _ensure_headless_pyhfo_compat()
        modules = ["numpy", "scipy", "mne", "pandas", "p_tqdm", "HFODetector", "onnxruntime"]
        modules.extend(f"{_ENGINE_NAMESPACE}.{name}" for name in
                       ("hfo_app", "param.param_detector", "param.param_filter", "utils.utils_feature"))
        missing = []
        for name in modules:
            try:
                import_module(name)
            except Exception as exc:
                missing.append(f"{name}: {exc}")
        print("BRAINLENS_PROBE:" + json.dumps({"ok": not missing, "missing": missing}))
        return 0

    supervised = False
    launcher_pid = os.environ.get("BRAINLENS_LAUNCHER_PID") or os.environ.get("BTK_LAUNCHER_PID")
    if launcher_pid and os.getpgrp() == os.getpid():
        _start_watchdog(int(launcher_pid), os.getpid())
        supervised = True
        os.environ["BRAINLENS_POOL_PPID"] = str(os.getpid())
        os.environ["BRAINLENS_POOL_PGID"] = str(os.getpgrp())
    multiprocessing.set_start_method("spawn", force=True)

    req_path, res_path = Path(sys.argv[1]), Path(sys.argv[2])
    try:
        req = json.loads(req_path.read_text(encoding="utf-8"))
        act = req["action"]
        if act == "detect":
            out = _do_detect(req, req_path.parent, supervised)
        elif act == "classify":
            out = _do_classify(req, req_path.parent, supervised)
        else:
            raise ValueError(f"Unsupported action: {act}")
        _atomic_json(res_path, out)
        return 0
    except Exception as exc:
        traceback.print_exc()
        _atomic_json(res_path, {"status": "worker_failed", "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__mp_main__":
    pool_ppid = os.environ.get("BRAINLENS_POOL_PPID") or os.environ.get("BTK_POOL_PPID")
    pool_pgid = os.environ.get("BRAINLENS_POOL_PGID") or os.environ.get("BTK_POOL_PGID")
    if pool_ppid and pool_pgid:
        _start_watchdog(int(pool_ppid), int(pool_pgid))
    _ensure_headless_pyhfo_compat()
    _engine_module("utils.utils_detector")

if __name__ == "__main__":
    raise SystemExit(main())
