"""Multi-channel SEEG reader batches & Clinical Triad (Broadband + Bandpass + Morlet TFR) renderer."""

from __future__ import annotations

import atexit
import base64
import io
import math
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, fftconvolve, hilbert, iirnotch, sosfiltfilt, tf2sos

from .analytics import select_control_windows
from .anatomy import AnatomyMatcher, load_anatomy_labels
from .channel_order import ChannelDisplayOrder
from .clinical_display import model_score_labels
from .io_readers import (
    create_verified_snapshot_dir,
    open_mne_raw,
    read_scaled_volts,
    resolve_case_target,
    verify_source_recording_integrity,
)
from .io_integrity import (
    pin_window_files, quick_source_signature, recording_is_large, stat_signature, stat_signature_is_settled, verify_raw_filenames,
)
from .models import ChannelSpec, RecordingPiece, RecordingSpec, ToolkitError, merge_intervals, read_json
from .signal_prep import resolve_channel
from .storage import WorkbenchStore, run_analysis_signature, sanitize_slug


def _load_optional_run_bundle(
    store: WorkbenchStore,
    case_slug: str,
    rec_slug: str,
    run_name: str,
) -> dict[str, Any] | None:
    """Load run bundle reliably whenever a run exists on disk (never silently drop a committed MVCC run during GC!)."""
    rdir = store.run_dir(case_slug, rec_slug, run_name)
    exists_on_disk = (
        (rdir / "current").is_symlink()
        or (rdir / "CURRENT.json").is_file()
        or (rdir / "run_meta.json").is_file()
    )
    if not exists_on_disk:
        return None
    return store.load_run(case_slug, rec_slug, run_name)


def _verify_run_recording_binding(
    rec: RecordingSpec,
    run_bundle: dict[str, Any] | None,
    run_name: str,
) -> None:
    """Refuse to display a modified/re-registered recording under an older run analyzed on different source bytes."""
    if run_bundle is None:
        return
    saved_rfp = (run_bundle.get("meta") or {}).get("recording_fingerprint") or (
        run_bundle.get("summary") or {}
    ).get("recording_fingerprint")
    if saved_rfp and saved_rfp != rec.fingerprint:
        raise ToolkitError(
            "run_recording_mismatch",
            f"Run {run_name!r} was analyzed on recording fingerprint {saved_rfp[:12]}, "
            f"but registered recording {rec.slug!r} now has fingerprint {rec.fingerprint[:12]}; "
            "refusing to display a modified recording under an earlier run.",
            recording=rec.slug,
            run_name=run_name,
            run_recording_fingerprint=saved_rfp,
            current_recording_fingerprint=rec.fingerprint,
        )


def _extract_run_lowpass(run_bundle: dict[str, Any] | None) -> float | None:
    if run_bundle is None:
        return None
    cfg = run_bundle.get("config") or {}
    acfg = cfg.get("analysis_config") if isinstance(cfg.get("analysis_config"), dict) else cfg
    flts = acfg.get("filters") if isinstance(acfg.get("filters"), dict) else {}
    lp = flts.get("lowpass")
    if lp is not None and float(lp) > 0:
        return float(lp)
    return None


@dataclass
class _RawEntry:
    raw: Any
    tmp_obj: Any
    full_sig: tuple[Any, ...]
    stat_sig: tuple[Any, ...]
    settled: bool
    verified_mono: float


# Small recordings use private snapshots. Large source reads bracket every window
# with stat and sampled-content checks and reopen the handle on identity changes.
_FULL_REVERIFY_INTERVAL_S = 60.0


class _LazyRawLRU:
    def __init__(self, cap: int = 6):
        self._lock = threading.RLock()
        self._map: dict[str, _RawEntry] = {}
        self._order: list[str] = []
        self._cap = cap
        atexit.register(self.clear)

    def clear(self) -> None:
        with self._lock:
            for k in list(self._order):
                self._evict(k)

    def _evict(self, key: str) -> None:
        old = self._map.pop(key, None)
        if key in self._order:
            self._order.remove(key)
        if old is not None:
            try:
                old.raw.close()
            except Exception:
                pass
            try:
                old.tmp_obj.close()
            except Exception:
                pass

    def _touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def _fast_path_ok(self, rec: RecordingSpec, entry: _RawEntry) -> bool:
        if recording_is_large(rec) or not entry.settled or time.monotonic() - entry.verified_mono > _FULL_REVERIFY_INTERVAL_S:
            return False
        return quick_source_signature(rec, entry.full_sig) == entry.stat_sig

    def _full_verify(self, rec: RecordingSpec, key: str) -> tuple[tuple[Any, ...], int]:
        verified_at_ns = time.time_ns()
        try:
            return verify_source_recording_integrity(rec), verified_at_ns
        except ToolkitError:
            self._evict(key)
            raise

    def _open_snapshot(self, rec: RecordingSpec, sig_before: tuple[Any, ...]) -> tuple[Any, Any]:
        tmp_obj = ExitStack()
        snap_dir = None
        snap_main = Path(rec.path)
        raw = None
        try:
            if recording_is_large(rec):
                snap_main, _bp = tmp_obj.enter_context(pin_window_files(rec, sig_before))
                snap_dir = None if snap_main == Path(rec.path) else snap_main.parent
                sig_snap = sig_before
            else:
                snap_main, private_dir, _bp_snap, sig_snap = create_verified_snapshot_dir(rec)
                tmp_obj.callback(private_dir.cleanup)
                snap_dir = Path(private_dir.name).resolve()
            if sig_snap != sig_before:
                raise ToolkitError(
                    "source_recording_modified",
                    f"Source recording {Path(rec.path).name!r} changed on disk while snapshotting; refusing to display modified signal.",
                    recording=rec.slug,
                )
            raw = open_mne_raw(snap_main, preload=False)
            verify_raw_filenames(rec, raw.filenames, snap_dir, sig_before, opened_stat=getattr(raw, "_brainlens_fd_stat", None))
            if int(raw.n_times) != int(rec.n_samples) or abs(float(raw.info["sfreq"]) - float(rec.sfreq)) > 1e-2:
                raise ToolkitError(
                    "source_recording_modified",
                    f"Source recording {Path(rec.path).name!r} geometry changed on disk "
                    f"(samples {rec.n_samples}->{raw.n_times}, sfreq {rec.sfreq}->{raw.info['sfreq']}); "
                    "refusing to display modified signal under an earlier run.",
                    recording=rec.slug,
                )
        except Exception:
            if raw is not None:
                try:
                    raw.close()
                except Exception:
                    pass
            try:
                tmp_obj.close()
            except Exception:
                pass
            raise
        return raw, tmp_obj

    def read(self, rec: RecordingSpec, picks: list[int], s0: int, s1: int) -> np.ndarray:
        key = f"{rec.path}:{rec.fingerprint}"
        with self._lock:
            entry = self._map.get(key)
            if entry is None or not self._fast_path_ok(rec, entry):
                sig_before, verified_at_ns = self._full_verify(rec, key)
                if entry is not None and entry.full_sig != sig_before:
                    self._evict(key)
                    entry = None
                if entry is None:
                    raw, tmp_obj = self._open_snapshot(rec, sig_before)
                    entry = _RawEntry(raw, tmp_obj, sig_before, (), False, 0.0)
                    self._map[key] = entry
                entry.stat_sig = stat_signature(sig_before)
                entry.settled = stat_signature_is_settled(entry.stat_sig, verified_at_ns)
                entry.verified_mono = time.monotonic()
            self._touch(key)
            while len(self._order) > self._cap:
                self._evict(self._order[0])
            # A large recording uses the live verified handle; bracket its read even on cache hits.
            try:
                out = read_scaled_volts(entry.raw, rec, s0, s1, picks=picks)
            except Exception:
                self._evict(key)
                # A short read can be the first observable sign of a concurrent truncate.
                verify_source_recording_integrity(rec)
                raise
            # Post-read bracket: catches the source being replaced/rewritten during the read. Settled files are
            # re-stat'ed (inode/size/mtime/ctime + ancestor dirs); otherwise (recently modified file, or the stat
            # check failed) the source is re-hashed in full so the precise error code is preserved.
            if not recording_is_large(rec) and entry.settled and quick_source_signature(rec, entry.full_sig) == entry.stat_sig:
                return out
            try:
                sig_after = verify_source_recording_integrity(rec)
            except ToolkitError:
                self._evict(key)
                raise
            if sig_after != entry.full_sig:
                self._evict(key)
                raise ToolkitError(
                    "source_recording_modified",
                    f"Source recording {Path(rec.path).name!r} changed on disk during read; refusing to return unverified waveform.",
                    recording=rec.slug,
                )
            return out


_READER_CACHE = _LazyRawLRU()


def _adjacent_neighbor_index(rec: RecordingSpec, ch_idx: int, order: ChannelDisplayOrder) -> int | None:
    if rec.reference == "bipolar":
        return None
    target = rec.channels[ch_idx]
    shaft = order.depth_shaft(target.id, target.name)
    if shaft is None or shaft != target.shaft:
        return None
    mates = [
        (c.contact_index, i)
        for i, c in enumerate(rec.channels)
        if c.unit == "V" and c.shaft == shaft and order.depth_shaft(c.id, c.name) == shaft
    ]
    mates.sort()
    pos = next((k for k, (_, i) in enumerate(mates) if i == ch_idx), None)
    if pos is None:
        return None
    if pos + 1 < len(mates):
        return mates[pos + 1][1]
    if pos - 1 >= 0:
        return mates[pos - 1][1]
    return None


def _decimate_extrema(times: np.ndarray, values: np.ndarray, max_pts: int = 2200) -> tuple[list[float], list[float | None]]:
    n = len(values)
    if n <= max_pts:
        return (
            [round(float(t), 6) for t in times],
            [round(float(v), 4) if math.isfinite(float(v)) else None for v in values],
        )
    buckets = max(1, max_pts // 2)
    edges = np.linspace(0, n, buckets + 1, dtype=int)
    ot: list[float] = []
    ov: list[float | None] = []
    for b in range(buckets):
        i0, i1 = int(edges[b]), int(edges[b + 1])
        if i1 <= i0:
            continue
        seg = values[i0:i1]
        tseg = times[i0:i1]
        fin = np.flatnonzero(np.isfinite(seg))
        if fin.size == 0:
            ot.append(round(float(tseg[0]), 6))
            ov.append(None)
            continue
        lo = int(fin[int(np.argmin(seg[fin]))])
        hi = int(fin[int(np.argmax(seg[fin]))])
        for k in sorted({lo, hi}):
            ot.append(round(float(tseg[k]), 6))
            ov.append(round(float(seg[k]), 4))
    return ot, ov


def _morlet_power(sig_uv: np.ndarray, sfreq: float, freqs: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(sig_uv - np.nanmedian(sig_uv), nan=0.0)
    n = len(clean)
    out = np.zeros((len(freqs), n), dtype=np.float64)
    for fi, f in enumerate(freqs):
        ncyc = float(np.clip(f / 18.0, 5.0, 10.0))
        sigma_t = ncyc / (2.0 * math.pi * float(f))
        half = min(max(4, int(math.ceil(3.5 * sigma_t * sfreq))), max(4, n // 2))
        t = np.arange(-half, half + 1, dtype=np.float64) / sfreq
        wav = np.exp(-0.5 * (t / sigma_t) ** 2) * np.exp(2j * math.pi * float(f) * t)
        norm = np.sum(np.abs(wav)) / 2.0
        if norm > 0:
            wav /= norm
        out[fi] = np.abs(fftconvolve(clean, wav, mode="same")) ** 2
    return out


def _load_run_channel_pool(
    rec: RecordingSpec,
    bundle: dict[str, Any] | None,
    store: WorkbenchStore | None = None,
) -> list[ChannelSpec]:
    if bundle is not None:
        raw_prep = (bundle.get("meta") or {}).get("prepared_json_path")
        prep_p = (
            store.resolve_workspace_artifact_path(raw_prep)
            if store is not None
            else (Path(raw_prep) if raw_prep else None)
        )
        if prep_p is not None and prep_p.is_file():
            try:
                pdoc = read_json(prep_p)
                chs = [ChannelSpec.model_validate(c) for c in pdoc.get("channels", [])]
                if chs:
                    return chs
            except Exception:
                pass
        # NEVER silently fall back to unipolar acquired channels when a run used derived/re-referenced channels!
        cfg_ref_mode = (
            ((bundle.get("config") or {}).get("analysis_config") or {}).get("reference") or {}
        ).get("mode")
        eff_ref_mode = ((bundle.get("summary") or {}).get("effective_reference") or {}).get("mode")
        if (cfg_ref_mode and cfg_ref_mode != "original") or (eff_ref_mode and eff_ref_mode != "original"):
            rn = (bundle.get("meta") or {}).get("run_name") or "unknown"
            raise ToolkitError(
                "incomplete_run_artifacts",
                f"Run {rn!r} for {rec.slug!r} uses derived reference mode ({eff_ref_mode or cfg_ref_mode}) "
                f"but prepared channel definitions ({raw_prep}) are missing on disk; restore the complete workspace or re-run analysis.",
                recording=rec.slug,
                run_name=rn,
            )
    return list(rec.channels)


def _resolve_from_pool(
    pool: list[ChannelSpec],
    rec: RecordingSpec,
    token: str,
    *,
    strict_pool: bool = False,
) -> ChannelSpec:
    t = token.strip()
    for ch in pool:
        if t in {ch.id, ch.name, ch.original_label}:
            return ch
    for ch in pool:
        if t.lower() in {ch.id.lower(), ch.name.lower(), ch.original_label.lower()}:
            return ch
    if strict_pool:
        avail = ", ".join(c.name for c in pool[:8])
        raise ToolkitError(
            "channel_not_in_run",
            f"Channel {token!r} is not in this derived run's channel montage (available: {avail}).",
            channel=token,
        )
    return resolve_channel(rec, t)


def _read_channel_list_uv(rec: RecordingSpec, targets: list[ChannelSpec], rs0: int, rs1: int) -> np.ndarray:
    raw_idx_by_id = {c.id: i for i, c in enumerate(rec.channels)}
    needed_raw: set[int] = set()
    for ch in targets:
        if ch.id in raw_idx_by_id:
            needed_raw.add(raw_idx_by_id[ch.id])
        else:
            for pid in ch.parents:
                if pid in raw_idx_by_id:
                    needed_raw.add(raw_idx_by_id[pid])
    needed_sorted = sorted(needed_raw)
    pos_map = {orig_i: k for k, orig_i in enumerate(needed_sorted)}
    raw_v = _READER_CACHE.read(rec, needed_sorted, rs0, rs1) if needed_sorted else np.zeros((0, max(0, rs1 - rs0)))
    out = np.zeros((len(targets), max(0, rs1 - rs0)), dtype=np.float64)
    for ti, ch in enumerate(targets):
        if ch.id in raw_idx_by_id:
            out[ti] = raw_v[pos_map[raw_idx_by_id[ch.id]]] * 1e6
        elif ch.parents and len(ch.parents) == len(ch.weights):
            acc = np.zeros(max(0, rs1 - rs0), dtype=np.float64)
            for pid, w in zip(ch.parents, ch.weights):
                acc += float(w) * (raw_v[pos_map[raw_idx_by_id[pid]]] * 1e6)
            out[ti] = acc
    return out


@dataclass
class _DisplaySlice:
    piece: RecordingPiece
    start: int
    stop: int
    read_start: int
    read_stop: int

    @property
    def crop(self) -> slice:
        return slice(self.start - self.read_start, self.stop - self.read_start)

    def times(self, sfreq: float) -> np.ndarray:
        return self.piece.time_at(self.start, sfreq) + np.arange(self.stop - self.start) / sfreq


def _display_slices(rec: RecordingSpec, t0: float, t1: float, pad: int, minimum: int) -> list[_DisplaySlice]:
    if rec.pieces:
        ranges = rec.time_slices(t0, t1)
    else:
        # Preserve the original nearest-sample policy for continuous reader pages and cards.
        p = RecordingPiece(start_sample=0, stop_sample=rec.n_samples, onset_seconds=0)
        a = max(0, round(t0 * rec.sfreq))
        ranges = [(p, a, min(rec.n_samples, max(a + minimum, round(t1 * rec.sfreq))))]
    return [_DisplaySlice(p, a, b, max(p.start_sample, a - pad), min(p.stop_sample, b + pad)) for p, a, b in ranges]


def _visible_gaps(rec: RecordingSpec, t0: float, t1: float) -> list[dict[str, float]]:
    return [
        {"start_seconds": a, "end_seconds": b, "duration_seconds": b - a}
        for a, b in rec.gaps() if a < t1 and b > t0
    ]


def _display_cutoff_hz(value: float | str | None, name: str) -> float | None:
    if value is None:
        return None
    try:
        hz = float(value)
    except (TypeError, ValueError, OverflowError):
        hz = math.nan
    if isinstance(value, bool) or not math.isfinite(hz) or hz <= 0:
        raise ToolkitError("invalid_display_filter", f"{name} must be a finite positive frequency in Hz.")
    return hz


def compute_multichannel_batch(
    store: WorkbenchStore,
    case_slug: str,
    rec_slug: str,
    run_name: str = "baseline",
    shaft: str | None = None,
    focus_channel: str | None = None,
    page: int = 0,
    page_size: int = 16,
    start_seconds: float = 0.0,
    duration_seconds: float = 8.0,
    montage: str = "acquired",
    display_filter: str = "broadband",
    display_notch: bool = False,
    display_lff: float | str | None = None,
    display_hff: float | str | None = None,
    source_only: bool = False,
) -> dict[str, Any]:
    """Return display traces. Explicit cutoffs override the preset and an omitted edge stays off."""
    lff = _display_cutoff_hz(display_lff, "LFF")
    hff = _display_cutoff_hz(display_hff, "HFF")
    custom = lff is not None or hff is not None
    if lff is not None and hff is not None and lff >= hff:
        raise ToolkitError("invalid_display_filter", "LFF must be below HFF.")
    if custom:
        display_filter = "custom"
    rec = store.load_recording(case_slug, rec_slug)
    sfreq = rec.sfreq
    dur = float(np.clip(duration_seconds, 0.25, min(30.0, rec.duration)))
    t0 = float(np.clip(start_seconds, 0.0, max(0.0, rec.duration - dur)))
    t1 = min(rec.duration, t0 + dur)
    pad = int(round(0.35 * sfreq)) if (display_filter != "broadband" or display_notch) else 0
    slices = _display_slices(rec, t0, t1, pad, minimum=4)
    gaps = _visible_gaps(rec, t0, t1)

    run_bundle = None if source_only else _load_optional_run_bundle(store, case_slug, rec_slug, run_name)
    _verify_run_recording_binding(rec, run_bundle, run_name)
    eff_ref_mode = ((run_bundle or {}).get("summary", {}).get("effective_reference") or {}).get("mode", "original")
    use_run_pool = run_bundle is not None and eff_ref_mode != "original" and montage == "acquired"
    ch_pool = _load_run_channel_pool(rec, run_bundle, store) if use_run_pool else list(rec.channels)
    order = ChannelDisplayOrder(rec)

    eeg_idxs = [i for i, c in enumerate(ch_pool) if c.unit == "V" and not c.is_auxiliary]
    if source_only and focus_channel:
        eeg_idxs.extend(i for i, c in enumerate(ch_pool)
                        if c.is_auxiliary and c.unit == "V" and focus_channel in {c.id, c.name})
    if not eeg_idxs:
        eeg_idxs = [i for i, c in enumerate(ch_pool) if c.unit == "V"]
    eeg_idxs.sort(key=lambda i: order.position(ch_pool[i].id, ch_pool[i].name, ch_pool[i].parents))

    if shaft and shaft not in {"ALL", "ACTIVE", "NEIGHBORHOOD"}:
        req_shafts = [s.strip() for s in shaft.split(",") if s.strip()]
        shaft_rank = {s: idx for idx, s in enumerate(req_shafts)}
        depth_shafts = {i: order.depth_shaft(ch_pool[i].id, ch_pool[i].name, ch_pool[i].parents) for i in eeg_idxs}
        selected = [i for i in eeg_idxs if (depth_shafts[i] or ch_pool[i].shaft) in shaft_rank]
        if all(depth_shafts[i] for i in selected):
            selected.sort(key=lambda i: shaft_rank[depth_shafts[i]])
        total_pages = 1
    elif shaft == "ALL":
        selected = eeg_idxs
        total_pages = 1
    elif focus_channel and page == 0 and shaft == "NEIGHBORHOOD":
        f_ch = _resolve_from_pool(ch_pool, rec, focus_channel)
        f_idx = next(i for i, c in enumerate(ch_pool) if c.id == f_ch.id)
        pos = eeg_idxs.index(f_idx) if f_idx in eeg_idxs else 0
        lo = max(0, min(pos - page_size // 2, len(eeg_idxs) - page_size))
        selected = eeg_idxs[lo : lo + page_size]
        total_pages = max(1, math.ceil(len(eeg_idxs) / page_size))
    else:
        total_pages = max(1, math.ceil(len(eeg_idxs) / page_size))
        eff_page = max(0, min(total_pages - 1, page))
        selected = eeg_idxs[eff_page * page_size : (eff_page + 1) * page_size]

    neighbors: dict[int, int] = {}
    target_objs = [ch_pool[idx] for idx in selected]
    nb_objs: list[ChannelSpec] = []
    already_bipolar_guard = montage == "adjacent_bipolar" and not use_run_pool and rec.reference == "bipolar"
    eff_montage = "acquired" if already_bipolar_guard else montage
    if eff_montage == "adjacent_bipolar" and not use_run_pool:
        for idx in selected:
            nb = _adjacent_neighbor_index(rec, idx, order)
            if nb is not None:
                neighbors[idx] = len(target_objs) + len(nb_objs)
                nb_objs.append(rec.channels[nb])
        if not neighbors:
            eff_montage = "acquired"

    blocks = [_read_channel_list_uv(rec, target_objs + nb_objs, sl.read_start, sl.read_stop) for sl in slices]
    if not blocks:
        verify_source_recording_integrity(rec)

    nyq = 0.5 * sfreq
    run_lp = _extract_run_lowpass(run_bundle)
    raw_by_id = {c.id: c for c in rec.channels}
    line_hz = float(rec.line_freq or 60.0)
    sos_cache: dict[tuple[Any, ...], Any] = {}

    def _trace_filter_plan(
        contrib_objs: list[ChannelSpec],
    ) -> tuple[str, bool, list[float | None] | None, float, float | None, list[float], Any]:
        tr_rates = [float(c.source_sfreq) for c in contrib_objs if c.source_sfreq is not None and float(c.source_sfreq) > 0]
        tr_eff_nyq = 0.5 * min([sfreq, *tr_rates]) if tr_rates else nyq
        tr_ch_lps = [float(c.source_lowpass) for c in contrib_objs if c.source_lowpass is not None and float(c.source_lowpass) > 0]
        tr_lp_cands = [float(v) for v in (rec.lowpass, run_lp) if v is not None and float(v) > 0]
        if tr_ch_lps:
            tr_lp_cands.append(min(tr_ch_lps))
        tr_acq_lp = min(tr_lp_cands) if tr_lp_cands else None
        tr_bw_ceil_92 = min(tr_eff_nyq * 0.92, tr_acq_lp) if tr_acq_lp is not None else (tr_eff_nyq * 0.92)

        tr_eff_filter = "broadband"
        tr_avail = True
        tr_band_hz: list[float | None] | None = None
        if custom:
            if lff is not None and lff >= tr_bw_ceil_92:
                tr_avail = False
            else:
                tr_eff_filter = "custom"
                tr_band_hz = [lff, min(hff, tr_bw_ceil_92) if hff is not None else None]
        elif display_filter == "eeg_1_70":
            hi_eeg = min(70.0, tr_bw_ceil_92)
            if tr_eff_nyq > 75.0 and hi_eeg >= 30.0:
                tr_eff_filter = "eeg_1_70"
                tr_band_hz = [1.0, round(hi_eeg, 1)]
            else:
                tr_avail = False
        elif display_filter == "ripple_80_250":
            hi_r = min(250.0, tr_bw_ceil_92)
            if hi_r >= 95.0:
                tr_eff_filter = "ripple_80_250"
                tr_band_hz = [80.0, round(hi_r, 1)]
            else:
                tr_avail = False
        elif display_filter == "fast_ripple_250_450":
            hi_fr = min(450.0, tr_bw_ceil_92)
            if hi_fr >= 265.0:
                tr_eff_filter = "fast_ripple_250_450"
                tr_band_hz = [250.0, round(hi_fr, 1)]
            else:
                tr_avail = False

        notch_hz = [hz for hz in (line_hz, 2.0 * line_hz) if display_notch and hz < tr_eff_nyq * 0.9]
        ckey = (tr_eff_filter, tuple(tr_band_hz) if tr_band_hz else None, tuple(notch_hz))
        if ckey not in sos_cache:
            parts = []
            for hz in notch_hz:
                b_n, a_n = iirnotch(hz, Q=30.0, fs=sfreq)
                parts.append(tf2sos(b_n, a_n))
            if tr_band_hz is not None:
                low, high = tr_band_hz
                if low is None:
                    parts.append(butter(4, high, btype="lowpass", fs=sfreq, output="sos"))
                elif high is None:
                    parts.append(butter(4, low, btype="highpass", fs=sfreq, output="sos"))
                else:
                    parts.append(butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos"))
            sos_cache[ckey] = np.vstack(parts) if parts else None
        return tr_eff_filter, tr_avail, tr_band_hz, tr_eff_nyq, tr_acq_lp, notch_hz, sos_cache[ckey]

    invalid_by_cid: dict[str, list[list[float]]] = {}
    events_in_win: list[dict[str, Any]] = []
    anatomy = AnatomyMatcher(load_anatomy_labels(store, case_slug))
    if run_bundle is not None:
        selected_channels = {ch_pool[i].id: ch_pool[i] for i in selected}
        qc_inv = run_bundle["qc"].get("invalid_intervals") or {}
        for idx in selected:
            ch = ch_pool[idx]
            raw_spans: list[tuple[int, int]] = []
            for a, b_s in qc_inv.get(ch.id, []):
                raw_spans.append((int(a), int(b_s)))
            # Propagate parent QC bad intervals to derived bipolar/average channels
            for pid in ch.parents:
                for a, b_s in qc_inv.get(pid, []):
                    raw_spans.append((int(a), int(b_s)))
            if eff_montage == "adjacent_bipolar" and idx in neighbors:
                nb_ch = nb_objs[neighbors[idx] - len(target_objs)]
                for a, b_s in qc_inv.get(nb_ch.id, []):
                    raw_spans.append((int(a), int(b_s)))
            if raw_spans:
                clipped = [
                    [round(max(t0, sl.piece.time_at(a, sfreq)), 4), round(min(t1, sl.piece.time_at(b, sfreq)), 4)]
                    for sl in slices for a, b in merge_intervals(raw_spans, sl.start, sl.stop)
                ]
                if clipped:
                    invalid_by_cid[ch.id] = clipped
        for ev in run_bundle["events"]:
            if ev["channel_id"] in selected_channels and ev["end_seconds"] >= t0 and ev["start_seconds"] <= t1:
                events_in_win.append({
                    "id": ev["id"],
                    "channel_id": ev["channel_id"],
                    "channel_name": ev["channel_name"],
                    "anatomy": anatomy.match(ev["channel_name"], selected_channels[ev["channel_id"]].contacts),
                    "start_seconds": ev["start_seconds"],
                    "end_seconds": ev["end_seconds"],
                    "duration_ms": round((ev["end_seconds"] - ev["start_seconds"]) * 1000.0, 2),
                    "synchrony_class": ev.get("synchrony_class", "focal"),
                    "coactive_shaft_count": ev.get("coactive_shaft_count", 1),
                    "verdict": ev.get("clinician_verdict"),
                    "rehearsal_verdict": ev.get("rehearsal_verdict"),
                    "clipped": ev.get("clipped", False),
                })

    traces: list[dict[str, Any]] = []
    min_eff_nyq = nyq
    min_acq_lp: float | None = None
    for pos_k, idx in enumerate(selected):
        ch = ch_pool[idx]
        lbl = ch.name
        trace_anatomy = anatomy.match(ch.name, ch.contacts)
        contrib_objs: list[ChannelSpec] = [ch]
        for pid in ch.parents:
            if pid in raw_by_id:
                contrib_objs.append(raw_by_id[pid])
        if eff_montage == "adjacent_bipolar" and idx in neighbors:
            nb_ch = nb_objs[neighbors[idx] - len(target_objs)]
            contrib_objs.append(nb_ch)
            lbl = f"{ch.name}-{nb_ch.name}"
            trace_anatomy = anatomy.match_pair(ch, nb_ch)
        try:
            tr_eff_filter, tr_avail, tr_band_hz, tr_eff_nyq, tr_acq_lp, notch_hz, tr_sos = _trace_filter_plan(contrib_objs)
            dt, dv = [], []
            for sl, block in zip(slices, blocks):
                sig_uv = block[pos_k]
                if eff_montage == "adjacent_bipolar" and idx in neighbors:
                    sig_uv = sig_uv - block[neighbors[idx]]
                if tr_sos is not None and len(sig_uv) > 32:
                    clean = np.nan_to_num(sig_uv - np.nanmedian(sig_uv), nan=0.0)
                    padlen = min(len(clean) - 1, 3 * (2 * len(tr_sos) + 1))
                    sig_uv = sosfiltfilt(tr_sos, clean, padlen=padlen)
                elif tr_sos is not None:
                    if tr_band_hz is not None:
                        tr_eff_filter, tr_band_hz, tr_avail = "broadband", None, False
                    notch_hz = []
                ts, vs = _decimate_extrema(sl.times(sfreq), sig_uv[sl.crop])
                if dt:
                    dt.append((dt[-1] + ts[0]) / 2)
                    dv.append(None)
                dt.extend(ts)
                dv.extend(vs)
        except (ValueError, np.linalg.LinAlgError, OverflowError) as exc:
            if not custom:
                raise
            raise ToolkitError("invalid_display_filter", "Cutoffs cannot be resolved at this sample rate.") from exc
        min_eff_nyq = min(min_eff_nyq, tr_eff_nyq)
        if tr_acq_lp is not None:
            min_acq_lp = tr_acq_lp if min_acq_lp is None else min(min_acq_lp, tr_acq_lp)
        traces.append({
            "channel_id": ch.id,
            "channel_name": ch.name,
            "display_label": lbl,
            "anatomy": trace_anatomy,
            "shaft": ch.shaft,
            "display_shaft": order.depth_shaft(ch.id, ch.name, ch.parents),
            "times": dt,
            "values_uv": dv,
            "invalid_seconds": invalid_by_cid.get(ch.id, []),
            "display_filter": tr_eff_filter,
            "effective_display_filter": tr_eff_filter,
            "display_filter_available": tr_avail,
            "display_filter_band_hz": tr_band_hz,
            "effective_nyquist_hz": round(tr_eff_nyq, 2),
            "effective_notch_hz": notch_hz,
        })

    avail_traces = [t for t in traces if t["display_filter_available"]]
    unavail_traces = [t for t in traces if not t["display_filter_available"]]
    if display_filter == "broadband":
        eff_display_filter = "broadband"
        display_filter_available = True
        display_filter_band_hz: list[float | None] | None = None
    elif traces and not unavail_traces:
        eff_display_filter = display_filter
        display_filter_available = True
        bands_present = [t["display_filter_band_hz"] for t in avail_traces if t["display_filter_band_hz"]]
        display_filter_band_hz = [
            max((b[0] for b in bands_present if b[0] is not None), default=None),
            min((b[1] for b in bands_present if b[1] is not None), default=None),
        ] if bands_present else None
    elif traces and avail_traces and unavail_traces:
        eff_display_filter = "partial"
        display_filter_available = False
        display_filter_band_hz = None
    else:
        eff_display_filter = "broadband"
        display_filter_available = False
        display_filter_band_hz = None

    notices: list[str] = []
    if already_bipolar_guard:
        notices.append("Recording is already bipolar; adjacent within-shaft subtraction skipped to avoid double re-referencing.")
    elif montage == "adjacent_bipolar" and not neighbors:
        notices.append("Showing acquired channels. No adjacent depth contacts on this page.")
    elif eff_montage == "adjacent_bipolar" and not use_run_pool:
        notices.append(
            "Display montage: adjacent within-shaft subtraction (acquired reference is 'unknown'; saved pipeline counts stay on acquired channels)."
            if rec.reference == "unknown"
            else f"Display montage: adjacent within-shaft bipolar (acquired reference: {rec.reference})."
        )
    elif use_run_pool:
        notices.append(
            f"Showing analyzed {eff_ref_mode} montage from run '{run_name}' (acquired reference: {rec.reference})."
        )
    if display_filter != "broadband":
        if not display_filter_available:
            bw_desc = (
                f"effective lowpass {min_acq_lp:g} Hz / Nyquist {min_eff_nyq:g} Hz"
                if (min_acq_lp is not None and min_acq_lp < min_eff_nyq * 0.92)
                else (f"channel native Nyquist {min_eff_nyq:g} Hz" if min_eff_nyq < nyq else f"Nyquist {nyq:g} Hz")
            )
            shortest = min((sl.read_stop - sl.read_start for sl in slices), default=0)
            if shortest <= 32:
                bw_desc = f"short window with {shortest} samples"
            if avail_traces and unavail_traces:
                unav_lbls = ", ".join(t["display_label"] for t in unavail_traces[:4]) + ("…" if len(unavail_traces) > 4 else "")
                notices.append(
                    f"Requested display filter '{display_filter}' unavailable on {len(unavail_traces)}/{len(traces)} trace(s) "
                    f"({unav_lbls}; {bw_desc}) — shown broadband; applied to eligible traces."
                )
            else:
                notices.append(f"Requested display filter '{display_filter}' unavailable at {sfreq:g} Hz ({bw_desc}); showing broadband.")
        elif display_filter == "fast_ripple_250_450" and display_filter_band_hz:
            notices.append(f"Display filter: Fast Ripple {display_filter_band_hz[0]:g}–{display_filter_band_hz[1]:g} Hz (display-only; saved pipeline counts unchanged).")
    if custom and hff is not None:
        clamped = [
            t["display_filter_band_hz"][1] for t in avail_traces
            if t["display_filter_band_hz"][1] < hff
        ]
        if clamped:
            low, high = min(clamped), max(clamped)
            limits = f"{low:g}" if low == high else f"{low:g}–{high:g}"
            notices.append(
                f"HFF clamped to {limits} Hz on {len(clamped)}/{len(traces)} trace(s) "
                "by Nyquist or acquisition low-pass."
            )
    if display_notch and traces:
        n_line = sum(bool(t["effective_notch_hz"]) for t in traces)
        n_second = sum(len(t["effective_notch_hz"]) == 2 for t in traces)
        if n_second == len(traces):
            notices.append(f"Display notch ({line_hz:g} Hz + 2nd harmonic) active; saved pipeline counts unchanged.")
        else:
            notices.append(
                f"Notch {line_hz:g} Hz active on {n_line}/{len(traces)} traces "
                f"(2nd harmonic on {n_second}/{len(traces)})."
            )
    if not slices:
        notices.append("Recording gap: no samples in this window.")
    notice = " · ".join(notices) if notices else None

    rmeta = (run_bundle or {}).get("meta") or {}
    run_asig = str(rmeta.get("analysis_signature") or run_analysis_signature(run_bundle)) if run_bundle is not None else None
    run_agen = (str(rmeta.get("analysis_generation_id") or rmeta.get("generation_id") or "") or None) if run_bundle is not None else None

    return {
        "case": case_slug,
        "recording": rec_slug,
        "recording_fingerprint": rec.fingerprint,
        "analysis_signature": run_asig,
        "analysis_generation_id": run_agen,
        "run_name": run_name,
        "sfreq_hz": sfreq,
        "line_freq_hz": line_hz,
        "start_seconds": round(t0 if rec.pieces else slices[0].start / sfreq, 5),
        "end_seconds": round(t1 if rec.pieces else slices[0].stop / sfreq, 5),
        "gaps": gaps,
        "total_duration_seconds": round(rec.duration, 3),
        "page": page,
        "total_pages": total_pages,
        "montage": eff_montage,
        "requested_montage": montage,
        "effective_reference_mode": eff_ref_mode,
        "display_filter": eff_display_filter,
        "requested_display_filter": display_filter,
        "requested_display_lff": lff,
        "requested_display_hff": hff,
        "effective_display_filter": eff_display_filter,
        "display_filter_available": display_filter_available,
        "display_filter_band_hz": display_filter_band_hz,
        "display_notch": display_notch,
        "montage_notice": notice,
        "traces": traces,
        "events": events_in_win,
    }


def compute_triad_view(
    target: str,
    workspace_dir: str | Path | None = None,
    recording: str | None = None,
    run_name: str = "baseline",
    channel: str | None = None,
    event_id: str | None = "top_focal",
    center_seconds: float | None = None,
    window_seconds: float = 0.6,
    *,
    allow_filesystem_persist: bool = True,
    source_only: bool = False,
) -> dict[str, Any]:
    store = WorkbenchStore(workspace_dir, create_dirs=allow_filesystem_persist)
    spec = resolve_case_target(target, store)
    if spec["kind"] == "filesystem":
        if not allow_filesystem_persist:
            raise ToolkitError(
                "case_not_registered",
                f"Target {target!r} is not a registered case/recording in this workspace.",
            )
        from .io_readers import inspect_case
        insp = inspect_case(str(spec["path"]), workspace_dir=store.root, persist=True)
        cslug = insp["case"]
        rslug = sanitize_slug(recording) if recording else insp["recordings"][0]["recording"]
    elif spec["kind"] == "registered_recording":
        cslug = spec["case"]
        rslug = sanitize_slug(recording) if recording else spec["recording"]
    else:
        cslug = spec["case"]
        rslug = sanitize_slug(recording) if recording else store.list_recordings(cslug)[0]

    rec = store.load_recording(cslug, rslug)
    sfreq = rec.sfreq
    all_events: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    run_bundle: dict[str, Any] | None = None
    run_bundle = None if source_only else _load_optional_run_bundle(store, cslug, rslug, run_name)
    if run_bundle is not None:
        _verify_run_recording_binding(rec, run_bundle, run_name)
        all_events = run_bundle["events"]
        summary = run_bundle["summary"]
    ch_pool = _load_run_channel_pool(rec, run_bundle, store)
    cfg_ref_mode = (
        (((run_bundle or {}).get("config") or {}).get("analysis_config") or {}).get("reference") or {}
    ).get("mode")
    eff_ref_mode = (((run_bundle or {}).get("summary") or {}).get("effective_reference") or {}).get("mode")
    is_derived_run = bool(
        run_bundle is not None and ((cfg_ref_mode and cfg_ref_mode != "original") or (eff_ref_mode and eff_ref_mode != "original"))
    )

    focus_event: dict[str, Any] | None = None
    view_mode = "candidate"
    view_notice: str | None = None

    win_s = float(np.clip(window_seconds, 0.2, min(30.0, rec.duration)))

    if center_seconds is not None and event_id in {None, "top_focal", "top_raw"}:
        ch_obj = _resolve_from_pool(ch_pool, rec, channel, strict_pool=is_derived_run) if channel else next(c for c in ch_pool if c.unit == "V" and not c.is_auxiliary)
        center = float(center_seconds)
        view_mode = "cursor_window"
    elif event_id == "control":
        if run_bundle is None or summary.get("detector_status") != "complete":
            raise ToolkitError(
                "control_requires_detection",
                f"Quiet background control requires a completed HFO detection run "
                f"(run {run_name!r} is missing or detector_status={summary.get('detector_status', 'missing')!r}).",
            )
        if center_seconds is not None:
            ch_obj = _resolve_from_pool(ch_pool, rec, channel, strict_pool=is_derived_run) if channel else next(c for c in ch_pool if c.unit == "V" and not c.is_auxiliary)
            center = float(center_seconds)
            ct0 = float(np.clip(center - 0.5 * win_s, 0.0, max(0.0, rec.duration - win_s)))
            ct1 = min(rec.duration, ct0 + win_s)
            control_slices = _display_slices(rec, ct0, ct1, 0, minimum=8)
            if len(control_slices) != 1 or _visible_gaps(rec, ct0, ct1):
                raise ToolkitError("control_window_invalid_coverage", "A background control must lie entirely within recorded data.")
            cs0, cs1 = control_slices[0].start, control_slices[0].stop
            overlaps = [
                e for e in all_events
                if (e["channel_id"] == ch_obj.id or e.get("synchrony_class") == "widespread_burst")
                and e["end_seconds"] >= ct0
                and e["start_seconds"] <= ct1
            ]
            if overlaps:
                raise ToolkitError(
                    "control_window_not_clean",
                    f"Requested control window [{ct0:.3f}, {ct1:.3f}]s on {ch_obj.name} overlaps {len(overlaps)} detected candidate/burst event(s).",
                    overlapping_event_ids=[e["id"] for e in overlaps[:10]],
                )
            det_p = store.resolve_workspace_artifact_path((run_bundle.get("meta") or {}).get("detection_json_path"))
            if det_p is None or not det_p.is_file():
                raise ToolkitError(
                    "control_window_invalid_coverage",
                    f"Detection coverage artifact is missing for run {run_name!r}; cannot certify quiet control on {ch_obj.name}.",
                )
            det_doc = read_json(det_p)
            valid_spans = (det_doc.get("valid_intervals") or {}).get(ch_obj.id, [])
            if not valid_spans or not any(int(a) <= cs0 and cs1 <= int(b) for a, b in valid_spans):
                raise ToolkitError(
                    "control_window_invalid_coverage",
                    f"Requested control window [{ct0:.3f}, {ct1:.3f}]s on {ch_obj.name} falls outside valid detector-analyzed coverage.",
                )
            view_mode = "background_control"
        else:
            ctrls = select_control_windows(store, cslug, rslug, run_name=run_name, channel=channel, count=1, window_seconds=win_s)
            if not ctrls:
                raise ToolkitError("no_control_window", f"No candidate-free valid control window found on {channel or 'leading channel'}.")
            c0 = ctrls[0]
            ch_obj = _resolve_from_pool(ch_pool, rec, c0["channel_id"], strict_pool=is_derived_run)
            center = c0["center_seconds"]
            view_mode = "background_control"
    elif event_id in {"top_focal", "top", "top_raw", "top_burst"} and all_events:
        if event_id == "top_burst":
            pool = [e for e in all_events if e.get("synchrony_class") == "widespread_burst"]
            if not pool:
                raise ToolkitError("no_burst_events", "No widespread multi-shaft burst events found in this run.")
            view_mode = "widespread_burst"
        elif event_id in {"top_focal", "top"}:
            pool = [e for e in all_events if e.get("synchrony_class") != "widespread_burst"] or all_events
        else:
            pool = all_events

        if channel:
            ch_req = _resolve_from_pool(ch_pool, rec, channel, strict_pool=is_derived_run)
            ch_pool_ev = [e for e in pool if e["channel_id"] == ch_req.id]
            if ch_pool_ev:
                focus_event = ch_pool_ev[0]
                ch_obj = ch_req
                center = 0.5 * (focus_event["start_seconds"] + focus_event["end_seconds"])
            else:
                # Never silently jump to a different channel when the clinician explicitly requested one!
                ch_obj = ch_req
                focus_event = None
                view_mode = "cursor_window"
                center = float(center_seconds) if center_seconds is not None else min(5.0, rec.duration * 0.5)
                view_notice = f"No matching {event_id} events on {ch_req.name}; showing signal window on {ch_req.name}."
        else:
            lead = (
                summary.get("leading_raw_channels")
                if event_id in {"top_raw", "top_burst"}
                else (summary.get("leading_focal_channels") or summary.get("leading_raw_channels"))
            ) or []
            top_cid = lead[0]["channel_id"] if lead else pool[0]["channel_id"]
            ch_pool_ev = [e for e in pool if e["channel_id"] == top_cid]
            focus_event = ch_pool_ev[0] if ch_pool_ev else pool[0]
            ch_obj = _resolve_from_pool(ch_pool, rec, focus_event["channel_id"], strict_pool=is_derived_run)
            center = 0.5 * (focus_event["start_seconds"] + focus_event["end_seconds"])
    elif event_id and event_id not in {"top_focal", "top", "top_raw", "top_burst"}:
        focus_event = next((e for e in all_events if e["id"] == event_id), None)
        if focus_event is None:
            raise ToolkitError("event_not_found", f"Event {event_id!r} not found.")
        ch_obj = _resolve_from_pool(ch_pool, rec, focus_event["channel_id"], strict_pool=is_derived_run)
        center = 0.5 * (focus_event["start_seconds"] + focus_event["end_seconds"])
    else:
        ch_obj = _resolve_from_pool(ch_pool, rec, channel, strict_pool=is_derived_run) if channel else next(c for c in ch_pool if c.unit == "V" and not c.is_auxiliary)
        center = float(center_seconds) if center_seconds is not None else min(5.0, rec.duration * 0.5)
        view_mode = "cursor_window"

    if ch_obj.unit != "V":
        raise ToolkitError(
            "non_voltage_channel",
            f"Channel {ch_obj.name!r} has non-voltage unit ({ch_obj.source_unit or ch_obj.unit}); "
            "HFO Triad waveforms, µV scaling, and time-frequency evidence cards require verified voltage EEG/SEEG channels.",
            channel=ch_obj.name,
            unit=ch_obj.unit,
            source_unit=ch_obj.source_unit,
        )

    t0 = float(np.clip(center - 0.5 * win_s, 0.0, max(0.0, rec.duration - win_s)))
    t1 = min(rec.duration, t0 + win_s)
    slices = _display_slices(rec, t0, t1, round(0.5 * sfreq), minimum=8)
    gaps = _visible_gaps(rec, t0, t1)
    raw_idx_by_id = {c.id: i for i, c in enumerate(rec.channels)}
    order = ChannelDisplayOrder(rec)
    nb_idx = _adjacent_neighbor_index(rec, raw_idx_by_id[ch_obj.id], order) if ch_obj.id in raw_idx_by_id else None
    read_targets = [ch_obj] + ([rec.channels[nb_idx]] if nb_idx is not None else [])
    blocks = [_read_channel_list_uv(rec, read_targets, sl.read_start, sl.read_stop) for sl in slices]
    if not blocks:
        verify_source_recording_integrity(rec)
        view_notice = "Recording gap: no samples in this window."
    matcher = AnatomyMatcher(load_anatomy_labels(store, cslug))
    channel_anatomy = matcher.match(ch_obj.name, ch_obj.contacts)

    # A single missing point separates adjacent runs in waveform clients. It is never
    # passed to a filter, Hilbert transform or wavelet convolution.
    def joined(values):
        if not values:
            return [None, None]
        out = []
        for i, arr in enumerate(values):
            if i:
                out.append(None)
            out.extend(round(float(v), 4) if math.isfinite(float(v)) else None for v in arr)
        return out

    time_blocks = [sl.times(sfreq) for sl in slices]
    times = []
    for ts in time_blocks:
        if times:
            times.append((times[-1] + ts[0]) / 2)
        times.extend(ts)
    if not times:
        times = [t0, t1]
    broad_uv = joined([block[0, sl.crop] for sl, block in zip(slices, blocks)])
    adj_diff = None
    if nb_idx is not None:
        nb_ch = rec.channels[nb_idx]
        adj_diff = {
            "label": f"{ch_obj.name}-{nb_ch.name}",
            "anatomy": matcher.match_pair(ch_obj, nb_ch),
            "values_uv": joined([block[0, sl.crop] - block[1, sl.crop] for sl, block in zip(slices, blocks)]),
        }

    nyq = 0.5 * sfreq
    raw_by_id_tr = {c.id: c for c in rec.channels}
    tr_contrib = [ch_obj] + [raw_by_id_tr[pid] for pid in ch_obj.parents if pid in raw_by_id_tr]
    tr_src_rates = [float(c.source_sfreq) for c in tr_contrib if c.source_sfreq is not None and float(c.source_sfreq) > 0]
    eff_nyq = 0.5 * min([sfreq, *tr_src_rates]) if tr_src_rates else nyq
    run_lp = _extract_run_lowpass(run_bundle)
    triad_lps = [float(v) for v in (rec.lowpass, run_lp) if v is not None and float(v) > 0] + [
        float(c.source_lowpass) for c in tr_contrib if c.source_lowpass is not None and float(c.source_lowpass) > 0
    ]
    acq_lp = min(triad_lps) if triad_lps else None
    bands: dict[str, Any] = {}
    clean_blocks = [np.nan_to_num(block[0] - np.nanmedian(block[0]), nan=0.0) for block in blocks]
    for bkey, (lo, hi) in (("ripple", (80.0, 250.0)), ("fast_ripple", (250.0, 450.0))):
        eff_hi = min(hi, eff_nyq * 0.94, acq_lp) if acq_lp is not None else min(hi, eff_nyq * 0.94)
        if lo >= eff_hi - 10.0:
            if acq_lp is not None and acq_lp < eff_nyq * 0.94:
                reason = f"{lo:g}–{hi:g} Hz exceeds effective lowpass ({acq_lp:g} Hz)"
            elif eff_nyq < nyq:
                reason = f"{lo:g}–{hi:g} Hz exceeds channel native Nyquist ({eff_nyq:.0f} Hz)"
            else:
                reason = f"{lo:g}–{hi:g} Hz exceeds Nyquist ({nyq:.0f} Hz)"
            bands[bkey] = {"available": False, "reason": reason}
            continue
        sos = butter(4, [lo, eff_hi], btype="bandpass", fs=sfreq, output="sos")
        if not blocks:
            bands[bkey] = {"available": False, "reason": view_notice}
            continue
        filtered = [
            sosfiltfilt(sos, clean, padlen=min(len(clean) - 1, 3 * (2 * len(sos) + 1)))
            for clean in clean_blocks
        ]
        bands[bkey] = {
            "available": True,
            "band_hz": [round(lo, 1), round(eff_hi, 1)],
            "values_uv": joined([f[sl.crop] for f, sl in zip(filtered, slices)]),
            "envelope_uv": joined([np.abs(hilbert(f))[sl.crop] for f, sl in zip(filtered, slices)]),
        }

    eff_bw_hi = min(450.0, eff_nyq * 0.92)
    if acq_lp is not None and acq_lp > 0.0:
        eff_bw_hi = min(eff_bw_hi, acq_lp)
    tf_hi = max(1e-3, eff_bw_hi)
    tf_lo = min(40.0, tf_hi * 0.25)
    if tf_lo >= tf_hi:
        tf_lo = tf_hi * 0.2
    freqs = np.linspace(tf_lo, tf_hi, 42)
    tf_times, db_blocks, tf_pieces = [], [], []
    for sl, block, ts in zip(slices, blocks, time_blocks):
        power = _morlet_power(block[0], sfreq, freqs)
        window = power[:, sl.crop]
        columns = np.linspace(0, window.shape[1] - 1, min(180, window.shape[1]), dtype=int)
        background = np.maximum(np.median(power, axis=1, keepdims=True), 1e-12)
        db_blocks.append(10.0 * np.log10(np.maximum(window[:, columns], 1e-12) / background))
        first = len(tf_times)
        tf_times.extend(ts[columns])
        tf_pieces.append({
            "first_column": first, "end_column": len(tf_times),
            "start_seconds": sl.piece.time_at(sl.start, sfreq), "end_seconds": sl.piece.time_at(sl.stop, sfreq),
        })
    db_sub = np.hstack(db_blocks) if blocks else np.empty((len(freqs), 0))
    t_sub = np.asarray(tf_times)

    ev_t0, ev_t1 = (focus_event["start_seconds"], focus_event["end_seconds"]) if focus_event else (center - 0.04, center + 0.04)
    in_cols = np.flatnonzero((t_sub >= ev_t0) & (t_sub <= ev_t1))
    if in_cols.size == 0:
        in_cols = np.array([len(t_sub) // 2])
    has_hfo_band = bool(bands["ripple"].get("available") or bands["fast_ripple"].get("available"))
    hfo_mask = (freqs >= 80.0) & (freqs <= eff_bw_hi)
    peak_hz = None
    peak_db = None
    if view_mode != "background_control" and has_hfo_band and np.any(hfo_mask) and not any(
        g["start_seconds"] <= center < g["end_seconds"] for g in gaps
    ):
        sub = db_sub[np.ix_(hfo_mask, in_cols)]
        f_hfo = freqs[hfo_mask]
        best = int(np.argmax(sub))
        fi, ti = divmod(best, sub.shape[1])
        peak_hz = round(float(f_hfo[fi]), 1)
        peak_db = round(float(sub[fi, ti]), 2)

    # Compute window RMS / P95 metrics (critical for background control floor & candidate SNR)
    rip_vals = np.asarray(bands["ripple"]["values_uv"], dtype=float) if bands["ripple"].get("available") else np.array([])
    frip_vals = np.asarray(bands["fast_ripple"]["values_uv"], dtype=float) if bands["fast_ripple"].get("available") else np.array([])
    rip_vals = rip_vals[np.isfinite(rip_vals)]
    frip_vals = frip_vals[np.isfinite(frip_vals)]
    rip_rms = round(float(np.sqrt(np.mean(rip_vals ** 2))), 2) if rip_vals.size else None
    rip_p95 = round(float(np.percentile(np.abs(rip_vals), 95)), 2) if rip_vals.size else None
    frip_rms = round(float(np.sqrt(np.mean(frip_vals ** 2))), 2) if frip_vals.size else None

    eff_ref_label = (
        summary.get("effective_reference_label")
        or (f"original ({rec.reference})" if not ch_obj.parents else "derived")
    )

    in_win_events = [
        e for e in all_events
        if e["channel_id"] == ch_obj.id and e["end_seconds"] >= t0 and e["start_seconds"] <= t1
    ]

    rmeta_tr = (run_bundle or {}).get("meta") or {}
    tr_asig = str(rmeta_tr.get("analysis_signature") or run_analysis_signature(run_bundle)) if run_bundle is not None else None
    tr_agen = (str(rmeta_tr.get("analysis_generation_id") or rmeta_tr.get("generation_id") or "") or None) if run_bundle is not None else None

    return {
        "case": cslug,
        "recording": rslug,
        "recording_fingerprint": rec.fingerprint,
        "analysis_signature": tr_asig,
        "analysis_generation_id": tr_agen,
        "run_name": run_name,
        "view_mode": view_mode,
        "view_notice": view_notice,
        "channel_id": ch_obj.id,
        "channel_name": ch_obj.name,
        "derived": bool(ch_obj.parents),
        "display_shaft": order.depth_shaft(ch_obj.id, ch_obj.name, ch_obj.parents),
        "anatomy": channel_anatomy,
        "shaft": ch_obj.shaft,
        "sfreq_hz": sfreq,
        "acquired_reference": rec.reference,
        "effective_reference_label": eff_ref_label,
        "provenance_origin": rec.provenance.origin_kind,
        "window_seconds": [round(t0 if rec.pieces else slices[0].start / sfreq, 5),
                           round(t1 if rec.pieces else slices[0].stop / sfreq, 5)],
        "gaps": gaps,
        "focus_event": {**focus_event, "anatomy": channel_anatomy} if focus_event is not None else None,
        "overlapping_events": [{**event, "anatomy": channel_anatomy} for event in in_win_events],
        "peak_hfo_frequency_hz": peak_hz,
        "peak_island_db": peak_db,
        "signal_metrics": {
            "ripple_rms_uv": rip_rms,
            "ripple_p95_uv": rip_p95,
            "fast_ripple_rms_uv": frip_rms,
        },
        "times": [round(float(t), 6) for t in times],
        "broadband_uv": broad_uv,
        "adjacent_bipolar": adj_diff,
        "bands": bands,
        "tfr": {
            "pieces": tf_pieces,
            "times": [round(float(t), 6) for t in t_sub],
            "frequencies_hz": [round(float(f), 2) for f in freqs],
            "db_above_median": [[round(float(v), 2) for v in row] for row in db_sub],
        },
    }


_UNSET: Any = object()


def _encode_figure(fig: plt.Figure, path: Path | None) -> bytes:
    """Encode the same pixels for embedding and, for local cards, the on-disk PNG."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        png = buf.getvalue()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(png)
        return png
    finally:
        plt.close(fig)


_RENDER_LOCK = threading.RLock()


def _serialized_figure(render):
    """HTTP jobs and MCP requests can share the same process and pyplot state."""
    @wraps(render)
    def locked(*args, **kwargs):
        with _RENDER_LOCK:
            return render(*args, **kwargs)
    return locked


@_serialized_figure
def render_spatial_figure(
    store: WorkbenchStore,
    case_slug: str,
    rec_slug: str,
    run_name: str = "baseline",
    return_base64: bool = False,
    *,
    overlay: dict[str, Any] | None = _UNSET,
    bundle: dict[str, Any] | None = None,
    display_text: Callable[[str], str] | None = None,
    display_names: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Render a 2-panel (Isometric 3D + Axial Top-Down) SEEG shaft projection PNG for MDT sheets."""
    cslug = sanitize_slug(case_slug)
    rslug = sanitize_slug(rec_slug)
    label = display_text or str
    if bundle is None:
        bundle = store.load_run(cslug, rslug, run_name)
    s = bundle["summary"]
    sp = s.get("spatial") or {}
    if not sp.get("available") or not sp.get("contacts"):
        raise ToolkitError("spatial_unavailable", f"No 3D electrode coordinates available for {cslug}/{rslug}.")

    if overlay is _UNSET:
        overlay = store.load_clinical_overlay(cslug)
    soz_set = {c.upper() for c in (overlay or {}).get("soz_channels", [])}
    res_set = {c.upper() for c in (overlay or {}).get("resected_channels", [])}
    contacts = sp["contacts"]

    xs = np.array([c["xyz"][0] for c in contacts], dtype=float)
    ys = np.array([c["xyz"][1] for c in contacts], dtype=float)
    zs = np.array([c["xyz"][2] for c in contacts], dtype=float)
    cx, cy, cz = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max()), 0.5 * (zs.min() + zs.max())

    def _proj(xyz: list[float], yaw: float, pitch: float) -> tuple[float, float, float]:
        dx, dy, dz = xyz[0] - cx, xyz[1] - cy, xyz[2] - cz
        rx = dx * math.cos(yaw) - dy * math.sin(yaw)
        ry = dx * math.sin(yaw) + dy * math.cos(yaw)
        return rx, ry * math.sin(pitch) - dz * math.cos(pitch), ry * math.cos(pitch) + dz * math.sin(pitch)

    order = ChannelDisplayOrder(store.load_recording(cslug, rslug))
    parents = {row["channel_id"]: row.get("parents") or () for row in s["channels"]}
    by_shaft: dict[str, list[dict[str, Any]]] = {}
    for c in contacts:
        shaft = order.depth_shaft(c["channel_id"], c["channel_name"], parents.get(c["channel_id"], ()))
        if shaft:
            by_shaft.setdefault(shaft, []).append(c)
    for arr in by_shaft.values():
        arr.sort(key=lambda c: order.position(c["channel_id"], c["channel_name"], parents.get(c["channel_id"], ())))

    max_rate = max([float(c.get("focal_rate_per_min") or 0.0) for c in contacts] + [0.1])

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), dpi=160, facecolor="#ffffff")
    views = [
        (axes[0], "Isometric electrode view", -0.62, 0.34),
        (axes[1], "Axial Top-Down Projection (L ← → R)", 0.0, math.pi / 2.0),
    ]

    for ax, title, yaw, pitch in views:
        ax.set_facecolor("#f8fafc")
        for arr in by_shaft.values():
            if len(arr) > 1:
                pts = [_proj(c["xyz"], yaw, pitch) for c in arr]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#94a3b8", lw=1.1, alpha=0.65, zorder=1)

        proj_items = []
        for c in contacts:
            px, py, pz = _proj(c["xyz"], yaw, pitch)
            proj_items.append((pz, px, py, c))
        proj_items.sort(key=lambda t: t[0])

        span_x = max(1.0, max(t[1] for t in proj_items) - min(t[1] for t in proj_items))
        span_y = max(1.0, max(t[2] for t in proj_items) - min(t[2] for t in proj_items))

        for _pz, px, py, c in proj_items:
            rate = float(c.get("focal_rate_per_min") or 0.0)
            is_soz = c["channel_name"].upper() in soz_set
            is_res = c["channel_name"].upper() in res_set
            if rate > 0:
                u = min(1.0, 0.3 + 0.7 * (rate / max_rate))
                col = plt.cm.magma(0.35 + 0.55 * u)
                sz = 46 + 95 * (rate / max_rate)
            else:
                col = "#cbd5e1"
                sz = 18
            ax.scatter([px], [py], s=sz, color=[col], edgecolors="#0f172a" if rate > 0 else "#ffffff", linewidths=0.8, zorder=3 if rate > 0 else 2)
            if is_soz:
                ax.scatter([px], [py], s=sz + 90, facecolors="none", edgecolors="#0284c7", linewidths=1.8, linestyles="--", zorder=4)
            if is_res:
                ax.scatter([px], [py], s=sz + 145, facecolors="none", edgecolors="#059669", linewidths=1.5, linestyles=":", zorder=4)

        # Collision-free labels prioritizing active contacts and SOZ contacts
        placed: list[tuple[float, float]] = []
        cands = sorted(
            proj_items,
            key=lambda t: (
                -(float(t[3].get("focal_rate_per_min") or 0.0) * 10 + (5 if t[3]["channel_name"].upper() in soz_set else 0)),
                t[3].get("contact_index", 99),
            ),
        )
        for _pz, px, py, c in cands:
            hot = float(c.get("focal_rate_per_min") or 0.0) > 0 or c["channel_name"].upper() in (soz_set | res_set)
            if not hot and c.get("contact_index") != 1:
                continue
            dx_off = span_x * 0.025
            for dy_mult in (0.0, 0.045, -0.045, 0.08):
                lx, ly = px + dx_off, py + dy_mult * span_y
                if all(abs(lx - ox) > span_x * 0.075 or abs(ly - oy) > span_y * 0.048 for ox, oy in placed):
                    placed.append((lx, ly))
                    ax.text(
                        lx, ly, label(c["channel_name"]),
                        fontsize=7.6 if hot else 6.6,
                        fontweight="bold" if hot else "normal",
                        color="#0f172a" if hot else "#64748b",
                        va="center",
                        zorder=5,
                    )
                    break

        ax.set_title(title, fontsize=9.0, fontweight="bold", color="#1e293b", pad=6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        for spn in ax.spines.values():
            spn.set_color("#e2e8f0")

    tpl_tag = " · TEMPLATE FRAME (not patient MRI)" if sp.get("template") else ""
    source_label = label(overlay["source"]) if overlay else ""
    ov_tag = f" · Dashed rings = external overlay ({source_label[:48]})" if overlay else ""
    case_label, rec_label = display_names or (label(cslug), label(rslug))
    fig.suptitle(
        f"{case_label} / {rec_label} · {label(run_name)} · Focal HFO distribution",
        fontsize=10.0, fontweight="bold", color="#0f172a", y=0.97,
    )
    fig.text(
        0.5, 0.905,
        f"Coord: {label(str(sp.get('coordinate_system')))} ({label(str(sp.get('units')))}){tpl_tag} · Circle size/color = Focal HFO Rate (/min){ov_tag}",
        ha="center", fontsize=7.8, color="#475569",
    )
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.88])

    cards_dir = store.case_dir(cslug) / "cards"
    out_path = cards_dir / sanitize_slug(f"{rslug}_{run_name}_spatial_map.png")
    png_bytes = _encode_figure(fig, out_path if display_names is None else None)

    res: dict[str, Any] = {
        "image_path": str(out_path) if display_names is None else None,
        "case": cslug,
        "recording": rslug,
        "run_name": run_name,
        "view_mode": "spatial_projection",
        "coordinate_system": sp.get("coordinate_system"),
        "template": bool(sp.get("template")),
        "mapped_contacts": len(contacts),
        "has_external_overlay": overlay is not None,
    }
    if return_base64:
        res["png_base64"] = base64.b64encode(png_bytes).decode("ascii")
    return res


@_serialized_figure
def render_evidence_card(
    target: str,
    workspace_dir: str | Path | None = None,
    recording: str | None = None,
    run_name: str = "baseline",
    channel: str | None = None,
    event_id: str | None = "top_focal",
    center_seconds: float | None = None,
    window_seconds: float = 0.6,
    return_base64: bool = False,
    highlight_interval: list[float] | tuple[float, float] | None = None,
    *,
    display_text: Callable[[str], str] | None = None,
    display_names: tuple[str, str] | None = None,
    evidence_snapshot: dict[str, Any] | None = None,
    recording_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    store = WorkbenchStore(workspace_dir)
    if event_id in {"evidence_map", "epileptogenicity"}:
        from .evidence_cards import render_ei_card, render_map_card

        spec = resolve_case_target(target, store)
        if spec["kind"] == "filesystem":
            from .io_readers import inspect_case

            spec = inspect_case(str(spec["path"]), workspace_dir=store.root, persist=True)
        rec = recording or spec.get("recording")
        if event_id == "evidence_map":
            return render_map_card(store, spec["case"], recording=rec, run_name=run_name,
                                   return_base64=return_base64, snapshot=evidence_snapshot,
                                   display_text=display_text, recording_labels=recording_labels)
        return render_ei_card(store, spec["case"], recording=rec, run_name=run_name,
                              return_base64=return_base64)

    if event_id == "spatial":
        spec = resolve_case_target(target, store)
        cslug = spec["case"] if spec["kind"] != "filesystem" else sanitize_slug(Path(spec["path"]).stem)
        rslug = sanitize_slug(recording) if recording else (spec.get("recording") or store.list_recordings(cslug)[0])
        return render_spatial_figure(
            store, cslug, rslug, run_name=run_name, return_base64=return_base64,
            display_text=display_text, display_names=display_names,
        )

    triad = compute_triad_view(
        target=target,
        workspace_dir=store.root,
        recording=recording,
        run_name=run_name,
        channel=channel,
        event_id=event_id,
        center_seconds=center_seconds,
        window_seconds=window_seconds,
    )

    times = np.asarray(triad["times"], dtype=float)
    raw_uv = np.asarray([np.nan if v is None else v for v in triad["broadband_uv"]], dtype=float)
    ev = triad["focus_event"]
    score_lines = model_score_labels(ev) if ev else []

    fig = plt.figure(figsize=(9.0, 6.2), dpi=160, facecolor="#ffffff")
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.12, 1.05, 1.38],
        width_ratios=[1.0, 0.022],
        hspace=0.25, wspace=0.02,
        left=0.09, right=0.93, top=0.87 - 0.030 * len(score_lines), bottom=0.09,
    )
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_band = fig.add_subplot(gs[1, 0], sharex=ax_raw)
    ax_tfr = fig.add_subplot(gs[2, 0], sharex=ax_raw)
    cax = fig.add_subplot(gs[2, 1])

    label = display_text or str
    raw_lbl = f"{label(triad['channel_name'])} ({'derived' if triad['derived'] else 'acquired'})"
    ax_raw.plot(times, raw_uv, color="#0f172a", lw=1.15, label=raw_lbl)
    if triad["adjacent_bipolar"]:
        bp = triad["adjacent_bipolar"]
        bp_uv = np.asarray([np.nan if v is None else v for v in bp["values_uv"]], dtype=float)
        ax_raw.plot(times, bp_uv, color="#64748b", lw=0.85, linestyle="--", alpha=0.82, label=f"{label(bp['label'])} (display diff)")
    ax_raw.set_ylabel("Raw (µV)", fontsize=8.5, fontweight="bold", color="#1e293b")
    ax_raw.legend(loc="upper right", frameon=True, facecolor="#ffffff", framealpha=0.86, edgecolor="none", fontsize=7.5)
    ax_raw.grid(True, alpha=0.22, linestyle=":")

    rip = triad["bands"]["ripple"]
    frip = triad["bands"]["fast_ripple"]
    if rip.get("available"):
        b0, b1 = rip["band_hz"]
        ax_band.plot(times, rip["values_uv"], color="#0284c7", lw=1.05, label=f"Ripple {b0:g}–{b1:g} Hz")
        ax_band.plot(times, rip["envelope_uv"], color="#0284c7", lw=0.7, linestyle=":", alpha=0.45)
    if frip.get("available"):
        b0, b1 = frip["band_hz"]
        ax_band.plot(times, frip["values_uv"], color="#dc2626", lw=0.9, alpha=0.85, label=f"Fast Ripple {b0:g}–{b1:g} Hz")
    ax_band.set_ylabel("Bandpass (µV)", fontsize=8.5, fontweight="bold", color="#1e293b")
    if rip.get("available") or frip.get("available"):
        ax_band.legend(loc="upper right", frameon=True, facecolor="#ffffff", framealpha=0.86, edgecolor="none", fontsize=7.5)
    else:
        ax_band.text(
            0.5, 0.5,
            label(rip.get("reason") or "HFO bands unavailable (exceeds effective bandwidth)"),
            transform=ax_band.transAxes, ha="center", va="center", fontsize=8.0, color="#64748b",
        )
    ax_band.grid(True, alpha=0.22, linestyle=":")

    norm_hl: tuple[float, float] | None = None
    if highlight_interval is not None and len(highlight_interval) == 2:
        h0, h1 = float(highlight_interval[0]), float(highlight_interval[1])
        if h0 > h1:
            h0, h1 = h1, h0
        if h1 > h0:
            norm_hl = (h0, h1)

    for item in triad["overlapping_events"]:
        is_f = ev is not None and item["id"] == ev["id"]
        col = ("#ef4444" if item.get("synchrony_class") == "widespread_burst" else "#10b981") if is_f else "#94a3b8"
        for ax in (ax_raw, ax_band):
            ax.axvspan(item["start_seconds"], item["end_seconds"], color=col, alpha=0.18 if is_f else 0.10)
        ax_tfr.axvline(item["start_seconds"], color="#fbbf24" if is_f else "#cbd5e1", lw=0.95, linestyle="--")
        ax_tfr.axvline(item["end_seconds"], color="#fbbf24" if is_f else "#cbd5e1", lw=0.95, linestyle="--")

    if norm_hl is not None:
        h0, h1 = norm_hl
        for ax in (ax_raw, ax_band):
            ax.axvspan(h0, h1, color="#2563eb", alpha=0.14)
            ax.axvline(h0, color="#2563eb", lw=1.1, linestyle="--", alpha=0.85)
            ax.axvline(h1, color="#2563eb", lw=1.1, linestyle="--", alpha=0.85)
        ax_tfr.axvline(h0, color="#38bdf8", lw=1.1, linestyle="--")
        ax_tfr.axvline(h1, color="#38bdf8", lw=1.1, linestyle="--")

    tf = np.asarray(triad["tfr"]["frequencies_hz"], dtype=float)
    tdb = np.asarray(triad["tfr"]["db_above_median"], dtype=float)
    im = None
    vmax = max(6.0, float(np.percentile(tdb, 99.2))) if tdb.size else 6.0
    for part in triad["tfr"]["pieces"]:
        cols = slice(part["first_column"], part["end_column"])
        im = ax_tfr.imshow(
            tdb[:, cols], origin="lower", aspect="auto", interpolation="nearest", cmap="magma",
            vmin=-2.0, vmax=vmax, extent=(part["start_seconds"], part["end_seconds"], tf[0], tf[-1]),
        )
    if tf[-1] >= 80.0:
        ax_tfr.axhline(80.0, color="#38bdf8", lw=0.7, linestyle=":", alpha=0.8)
    if tf[-1] >= 250.0:
        ax_tfr.axhline(250.0, color="#f87171", lw=0.7, linestyle=":", alpha=0.8)
    ax_tfr.set_ylabel("Freq (Hz)", fontsize=8.5, fontweight="bold", color="#1e293b")
    ax_tfr.set_xlabel("Original Recording Time (s)", fontsize=9, fontweight="bold", color="#1e293b")

    if im is not None:
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label("dB vs bg", fontsize=7.5)
        cbar.ax.tick_params(labelsize=7)
    else:
        cax.set_visible(False)

    for ax in (ax_raw, ax_band):
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in (ax_raw, ax_band, ax_tfr):
        ax.tick_params(labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    w0, w1 = triad["window_seconds"]
    for gap in triad["gaps"]:
        a, b = max(w0, gap["start_seconds"]), min(w1, gap["end_seconds"])
        for ax in (ax_raw, ax_band, ax_tfr):
            ax.axvspan(a, b, color="#94a3b8", alpha=0.28)
        ax_raw.text((a + b) / 2, 0.04, f"Gap {gap['duration_seconds']:g} s", ha="center", va="bottom",
                    fontsize=7, transform=ax_raw.get_xaxis_transform())
    ax_raw.set_xlim(w0, w1)

    if triad["view_mode"] == "background_control":
        mode_tag = f"Quiet control · {times[0]:.2f}–{times[-1]:.2f} s · 0 candidates"
    elif ev is not None:
        dur_ms = (ev["end_seconds"] - ev["start_seconds"]) * 1000.0
        scls = ev.get("synchrony_class", "focal").replace("_", " ")
        shafts = ev.get("coactive_shaft_count", 1)
        co_info = f"{ev.get('coactive_channel_count', 1)} ch"
        if triad["display_shaft"]:
            co_info += f" / {shafts} {'shaft' if shafts == 1 else 'shafts'}"
        mode_tag = f"{scls.capitalize()} · {ev['start_seconds']:.3f}–{ev['end_seconds']:.3f} s · {dur_ms:.1f} ms · {co_info}"
    elif norm_hl is not None:
        h0, h1 = norm_hl
        mode_tag = f"Clip {h0:.3f}–{h1:.3f} s · window {times[0]:.3f}–{times[-1]:.3f} s"
    else:
        mode_tag = f"Window {times[0]:.3f}–{times[-1]:.3f} s"

    sm = triad.get("signal_metrics") or {}
    if triad["view_mode"] == "background_control":
        r_rms = sm.get("ripple_rms_uv")
        r_p95 = sm.get("ripple_p95_uv")
        detail_txt = (
            f" · Ripple RMS {r_rms:.1f} µV · P95 {r_p95:.1f} µV"
            if r_rms is not None and r_p95 is not None
            else " · Quiet background floor"
        )
    else:
        detail_txt = (f" · Peak ~{triad['peak_hfo_frequency_hz']} Hz (+{triad['peak_island_db']} dB)"
                      if triad["peak_hfo_frequency_hz"] else "")

    case_label, rec_label = display_names or (label(triad["case"]), label(triad["recording"]))
    rec_short = rec_label if len(rec_label) <= 22 else rec_label[:19] + "…"
    suptitle_str = f"{case_label} / {rec_short} · {label(triad['channel_name'])} · {mode_tag}"
    fig.suptitle(
        suptitle_str,
        fontsize=8.6 if len(suptitle_str) > 84 else 9.6,
        fontweight="bold",
        color="#0f172a",
        y=0.965,
    )
    ref_label = label(triad.get("effective_reference_label", triad["acquired_reference"]))
    sub_str = f"{label(triad['run_name'])} · {triad['sfreq_hz']:g} Hz · Reference: {ref_label} · {triad['provenance_origin']}{detail_txt}"
    fig.text(
        0.5, 0.918,
        sub_str,
        ha="center",
        fontsize=7.1 if len(sub_str) > 108 else 7.8,
        color="#475569",
    )
    for i, score in enumerate(score_lines):
        fig.text(0.5, 0.887 - i * 0.030, score, ha="center", fontsize=8, color="#334155")

    cards_dir = store.case_dir(triad["case"]) / "cards"
    fname = sanitize_slug(f"{triad['recording']}_{triad['run_name']}_{triad['channel_name']}_{triad['view_mode']}_{times[0]:.2f}s_{times[-1]:.2f}s") + ".png"
    out_path = cards_dir / fname

    png_bytes = _encode_figure(fig, out_path if display_names is None else None)

    result: dict[str, Any] = {
        "image_path": str(out_path) if display_names is None else None,
        "case": triad["case"],
        "recording": triad["recording"],
        "run_name": triad["run_name"],
        "view_mode": triad["view_mode"],
        "channel": triad["channel_name"],
        "window_seconds": triad["window_seconds"],
        "focus_event": ev,
        "peak_hfo_frequency_hz": triad["peak_hfo_frequency_hz"],
        "peak_island_db": triad["peak_island_db"],
        "gaps": triad["gaps"],
    }
    if return_base64:
        result["png_base64"] = base64.b64encode(png_bytes).decode("ascii")
    return result
