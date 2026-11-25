"""Source recording SHA-256 integrity verification, fingerprinting, and atomic snapshot staging."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterator, NamedTuple

from .io_signals import _extract_fiff_next_filename, _extract_vhdr_component_names, _parse_brainvision_header
from .io_vendor import is_vendor_recording_suffix, verify_vendor_companion_links
from .models import ChannelSpec, MarkerSpec, RecordingSpec, ToolkitError, canonical_digest, canonical_json, file_sha256


LARGE_FILE_THRESHOLD_BYTES = int(os.environ.get("BRAINLENS_LARGE_FILE_THRESHOLD_BYTES", 1024**3))
_SAMPLE_BLOCK_BYTES = 1024**2
_INTERIOR_BLOCKS = 8
ProgressCallback = Callable[[str, float, str], None]
_Stat = tuple[int, int, int, int, int]

class _HashIdentity(NamedTuple):
    stat: _Stat
    sha256: str
    sample: str
    settled: bool


# One current content identity per resolved file. A stripe serializes the initial full read and
# any forced refresh, including concurrent inspect, display, and analysis requests.
_HASH_LOCKS = tuple(threading.Lock() for _ in range(64))
_HASH_CACHE: dict[Path, _HashIdentity] = {}


def _large_file(size: int) -> bool:
    return size >= LARGE_FILE_THRESHOLD_BYTES


def recording_is_large(rec: RecordingSpec) -> bool:
    """Include signal companions: a BrainVision .vhdr itself may be only a few KiB."""
    size = rec.source_file_size
    if size is None:
        size = Path(rec.path).stat().st_size
    if _large_file(size) or any(_large_file(int(meta.get("size") or 0)) for meta in rec.companion_files.values()):
        return True
    # Older registrations can lack the entire companion manifest. Route by actual
    # geometry; _verify_or_stage_recording still checks the legacy content fingerprint.
    main = Path(rec.path)
    if not rec.source_sha256 and not rec.companion_files and main.suffix.lower() == ".vhdr":
        parts = _parse_brainvision_header(main).components.values()
        return any(_large_file(Path(part).stat().st_size) for part in parts)
    return False


def _sample_ranges(size: int) -> list[tuple[int, int]]:
    last = max(0, size - _SAMPLE_BLOCK_BYTES)
    spans: list[tuple[int, int]] = []
    for i in range(_INTERIOR_BLOCKS + 2):
        start = last * i // (_INTERIOR_BLOCKS + 1)
        stop = min(size, start + _SAMPLE_BLOCK_BYTES)
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], stop)
        else:
            spans.append((start, stop))
    return spans


def _sample_digest(fh: BinaryIO, size: int) -> str:
    digest = hashlib.blake2b(digest_size=32)
    for start, stop in _sample_ranges(size):
        fh.seek(start)
        digest.update(fh.read(stop - start))
    return digest.hexdigest()


def _stream_digests(fh: BinaryIO, size: int, name: str, progress_cb: ProgressCallback | None,
                    dest: BinaryIO | None = None) -> tuple[str, str]:
    """Bind the sampled baseline to the exact same stream as the full SHA-256."""
    full, sampled = hashlib.sha256(), hashlib.blake2b(digest_size=32)
    spans = iter(_sample_ranges(size))
    span = next(spans, None)
    offset = 0
    next_report = 0.0
    fh.seek(0)
    while chunk := fh.read(_SAMPLE_BLOCK_BYTES):
        stop = offset + len(chunk)
        full.update(chunk)
        if dest is not None:
            dest.write(chunk)
        while span is not None and span[0] < stop:
            sampled.update(memoryview(chunk)[max(0, span[0] - offset):min(len(chunk), span[1] - offset)])
            if span[1] > stop:
                break
            span = next(spans, None)
        offset = stop
        if progress_cb is not None and (time.monotonic() >= next_report or offset == size):
            fraction = offset / max(1, size)
            progress_cb("hash", fraction, f"SHA-256 {name}: {offset:,} / {size:,} bytes")
            next_report = time.monotonic() + 0.25
    if offset != size:
        raise ToolkitError("source_recording_modified", f"Source {name!r} changed size during verification.")
    return full.hexdigest(), sampled.hexdigest()


def _full_large_hash(fh: BinaryIO, size: int, name: str,
                     progress_cb: ProgressCallback | None) -> tuple[str, str]:
    return _stream_digests(fh, size, name, progress_cb)


def _stat_tuple(path: Path) -> tuple[int, int, int, int, int]:
    return _stat_fields(path.stat())


def _stat_fields(st: os.stat_result) -> _Stat:
    return (
        int(getattr(st, "st_dev", 0)),
        int(getattr(st, "st_ino", 0)),
        int(st.st_size),
        int(st.st_mtime_ns),
        int(st.st_ctime_ns),
    )


def _canonical_entry_path(path: Path) -> Path:
    """Resolve ancestor directories without dereferencing a final file symlink leaf."""
    if path.is_dir() and not path.is_symlink():
        return path.resolve()
    try:
        if path.is_dir():
            return path.resolve()
    except OSError:
        pass
    return path.parent.resolve() / path.name


def _ancestor_chain_sig(path: Path) -> tuple[tuple[str, int, int], ...]:
    """Snapshot (path, dev, ino) for every ancestor directory up to root."""
    out: list[tuple[str, int, int]] = []
    cur = path.parent.resolve()
    while True:
        st = cur.stat()
        out.append((str(cur), int(getattr(st, "st_dev", 0)), int(getattr(st, "st_ino", 0))))
        if cur.parent == cur:
            break
        cur = cur.parent
    return tuple(out)


def _copy_and_hash_file(src: Path, dest: Path) -> tuple[str, int, tuple[int, int, int, int, int]]:
    """Verify bytes captured in a private file, sharing settled large-source identities with window readers."""
    src = src.resolve()
    with _HASH_LOCKS[hash(src) % len(_HASH_LOCKS)]:
        before = _stat_tuple(src)
        settled = stat_signature_is_settled(((), before, ()), time.time_ns())
        cached = _HASH_CACHE.get(src)
        fd = os.open(dest, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w+b") as output, src.open("rb") as source:
                if _stat_fields(os.fstat(source.fileno())) != before:
                    raise ToolkitError("source_recording_modified", f"Source {src.name!r} changed before snapshotting.")
                if cached and cached.stat == before and cached.settled and settled and _large_file(before[2]):
                    # Explicit full exports need a byte-verified private copy. Hash the new
                    # destination in full, independent of the cached source SHA-256.
                    shutil.copyfileobj(source, output, length=_SAMPLE_BLOCK_BYTES)
                    output.flush()
                    if file_sha256(dest) != cached.sha256:
                        _HASH_CACHE.pop(src, None)
                        raise ToolkitError("source_recording_modified", f"Private copy of {src.name!r} differs from its verified source.")
                    sha, sampled = cached.sha256, cached.sample
                else:
                    sha, sampled = _stream_digests(source, before[2], src.name, None, dest=output)
                if before != _stat_fields(os.fstat(source.fileno())) or before != _stat_tuple(src):
                    raise ToolkitError("source_recording_modified", f"Source {src.name!r} changed during snapshotting.")
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        _HASH_CACHE[src] = _HashIdentity(before, sha, sampled, settled)
        return sha, before[2], before


def _hash_in_place(src: Path, *, recording: str, label: str = "Source recording",
                   progress_cb: ProgressCallback | None = None, sampled_checks: bool = False) -> tuple[str, int, _Stat]:
    src = src.resolve()
    with _HASH_LOCKS[hash(src) % len(_HASH_LOCKS)]:
        before = _stat_tuple(src)
        error = ToolkitError(
            "source_recording_modified",
            f"{label} {src.name!r} changed on disk during verification; refusing to read modified signal.",
            recording=recording,
        )
        size = before[2]
        if not sampled_checks and not _large_file(size):
            sha = file_sha256(src)
            if before != _stat_tuple(src):
                raise error
            return sha, size, before

        cached = _HASH_CACHE.get(src)
        settled = stat_signature_is_settled(((), before, ()), time.time_ns())
        with src.open("rb") as fh:
            if _stat_fields(os.fstat(fh.fileno())) != before:
                raise error
            if (cached is not None and cached.stat == before and cached.settled and settled
                    and _sample_digest(fh, size) == cached.sample):
                sha, sampled = cached.sha256, cached.sample
            else:
                sha, sampled = _full_large_hash(fh, size, src.name, progress_cb)
            if _stat_fields(os.fstat(fh.fileno())) != before or _stat_tuple(src) != before:
                raise error
        _HASH_CACHE[src] = _HashIdentity(before, sha, sampled, settled)
        return sha, size, before


def _compute_v2_fingerprint(
    rec_sha256: str,
    comp_hashes: dict[str, str],
    modality: str,
    reference: str,
    line_freq: float | None,
    channels: list[ChannelSpec],
    markers: list[MarkerSpec],
    bp_sha: str | None,
) -> str:
    return canonical_digest({
        "recording_sha256": rec_sha256,
        "companions": comp_hashes,
        "modality": modality,
        "reference": reference,
        "line_freq": float(line_freq) if line_freq is not None else None,
        "channels": [
            (c.id, c.name, c.kind, c.unit, c.source_sfreq, c.source_lowpass, tuple(c.contacts))
            for c in channels
        ],
        "markers": sorted((m.model_dump(mode="json") for m in markers), key=lambda d: canonical_json(d)),
        "bipolar_pairs_sha256": bp_sha,
    })


def _v2_fingerprint_for_recording(
    rec: RecordingSpec,
    rec_sha256: str,
    comp_hashes: dict[str, str],
    bp_sha: str | None,
) -> str:
    return _compute_v2_fingerprint(
        rec_sha256, comp_hashes, rec.modality, rec.reference, rec.line_freq, rec.channels, rec.markers, bp_sha
    )


def _legacy_fingerprint_candidates(
    rec: RecordingSpec,
    rec_sha256: str,
    comp_hashes: dict[str, str],
    bp_sha: str | None = None,
) -> set[str]:
    """Compute SHA-256-backed fingerprint candidates for legacy recording.json files lacking source_sha256."""
    if bp_sha is None and rec.published_bipolar_pairs_path:
        bp_p = Path(rec.published_bipolar_pairs_path)
        if bp_p.is_file():
            bp_sha = file_sha256(bp_p)
    lf = float(rec.line_freq) if rec.line_freq is not None else None
    ch_v1 = [(c.id, c.name, c.kind, c.unit, c.source_sfreq, c.source_lowpass) for c in rec.channels]
    markers_raw = [m.model_dump(mode="json") for m in rec.markers]
    return {
        _v2_fingerprint_for_recording(rec, rec_sha256, comp_hashes, bp_sha),
        canonical_digest({
            "recording_sha256": rec_sha256,
            "companions": comp_hashes,
            "modality": rec.modality,
            "reference": rec.reference,
            "line_freq": lf,
            "channels": ch_v1,
            "markers": markers_raw,
            "bipolar_pairs_sha256": bp_sha,
        }),
        canonical_digest({"recording_sha256": rec_sha256, "companions": comp_hashes, "modality": rec.modality}),
    }


def _verify_or_stage_recording(
    rec: RecordingSpec,
    *,
    snap_dir: Path | None = None,
    raw_filenames: tuple[Any, ...] | list[Any] | None = None,
    copy_fn=_copy_and_hash_file,
    progress_cb: ProgressCallback | None = None,
) -> tuple[Path | None, Path | None, tuple[Any, ...]]:
    """Bind an in-place read or private snapshot to the registered content identity and path topology."""
    p = Path(rec.path)
    if not p.is_file():
        raise ToolkitError(
            "source_recording_missing",
            f"Source recording file no longer exists on disk: {rec.path}",
            recording=rec.slug,
        )
    p_entry = _canonical_entry_path(p)
    p_res = p_entry.resolve()
    anc_pre = (_ancestor_chain_sig(p_entry), _ancestor_chain_sig(p_res))
    use_samples = recording_is_large(rec)

    def verify_file(path: Path, label: str = "Source recording") -> tuple[str, int, _Stat]:
        return _hash_in_place(path, recording=rec.slug, label=label, progress_cb=progress_cb, sampled_checks=use_samples)

    snap_main: Path | None = None
    if snap_dir is not None:
        snap_main = snap_dir / p_entry.name
        actual_sha, actual_size, st_main = copy_fn(p_res, snap_main)
    else:
        st_check = _stat_tuple(p_res)
        if rec.source_file_size is not None and st_check[2] != int(rec.source_file_size):
            raise ToolkitError(
                "source_recording_modified",
                f"Source recording {p.name!r} changed size on disk ({rec.source_file_size} -> {st_check[2]} bytes); "
                "refusing to read modified signal under an earlier registered recording/run.",
                recording=rec.slug,
            )
        actual_sha, actual_size, st_main = verify_file(p_res)

    if rec.source_file_size is not None and actual_size != int(rec.source_file_size):
        raise ToolkitError(
            "source_recording_modified",
            f"Source recording {p.name!r} changed size on disk ({rec.source_file_size} -> {actual_size} bytes); "
            "refusing to read modified signal under an earlier registered recording/run.",
            recording=rec.slug,
        )

    verified_companions: dict[str, tuple[Any, ...]] = {}
    signal_comp_hashes: dict[str, str] = {}
    bp_resolved_str = str(Path(rec.published_bipolar_pairs_path).resolve()) if rec.published_bipolar_pairs_path else None
    bp_snap_path: Path | None = None

    for comp_path_str, comp_meta in (rec.companion_files or {}).items():
        cp = Path(comp_path_str)
        if not cp.is_file():
            raise ToolkitError(
                "source_recording_missing",
                f"Companion recording data file missing on disk: {cp.name}",
                recording=rec.slug,
            )
        cp_res = cp.resolve()
        logical_str = comp_meta.get("logical_path")
        lp_entry: Path | None = None
        lp_anc_pre: tuple[Any, ...] | None = None
        if logical_str:
            lp_entry = _canonical_entry_path(Path(logical_str))
            if not lp_entry.is_file():
                raise ToolkitError(
                    "source_recording_modified",
                    f"Companion file {lp_entry.name!r} was removed or broken on disk; refusing to read under an earlier registered recording/run.",
                    recording=rec.slug,
                )
            if lp_entry.resolve() != cp_res:
                raise ToolkitError(
                    "source_recording_modified",
                    f"Companion file {lp_entry.name!r} was repointed to a different file on disk; refusing to read under an earlier registered recording/run.",
                    recording=rec.slug,
                )
            lp_anc_pre = _ancestor_chain_sig(lp_entry)

        is_bp = str(cp_res) == bp_resolved_str
        exp_size, exp_sha = comp_meta.get("size"), comp_meta.get("sha256")
        if snap_dir is not None:
            stage_name = str(comp_meta.get("stage_name") or cp.name)
            dest_comp = snap_dir / (f"__brainlens_bp_{cp_res.name}" if is_bp else stage_name)
            actual_csha, csize, cst = copy_fn(cp_res, dest_comp)
            if is_bp:
                bp_snap_path = dest_comp
        else:
            cst_check = _stat_tuple(cp_res)
            if exp_size is not None and cst_check[2] != int(exp_size):
                raise ToolkitError(
                    "source_recording_modified",
                    f"Companion data file {cp.name!r} changed size on disk; refusing to read modified signal.",
                    recording=rec.slug,
                )
            actual_csha, csize, cst = verify_file(cp_res, "Companion data file")

        if lp_entry is not None and (lp_entry.resolve() != cp_res or _ancestor_chain_sig(lp_entry) != lp_anc_pre):
            raise ToolkitError(
                "source_recording_modified",
                f"Companion file {lp_entry.name!r} changed target during verification; refusing unverified read.",
                recording=rec.slug,
            )
        if exp_size is not None and csize != int(exp_size):
            raise ToolkitError(
                "source_recording_modified",
                f"Companion data file {cp.name!r} changed size on disk; refusing to read modified signal.",
                recording=rec.slug,
            )
        if exp_sha and actual_csha != exp_sha:
            raise ToolkitError(
                "source_recording_modified",
                f"Companion data file {cp.name!r} content (SHA-256) changed on disk; refusing to read modified signal.",
                recording=rec.slug,
            )
        verified_companions[str(cp_res)] = (*cst, actual_csha)
        if not is_bp:
            signal_comp_hashes[str(cp_res)] = actual_csha

    vhdr_src = snap_main if snap_main is not None else p_res
    if p_entry.suffix.lower() == ".vhdr" and rec.source_sha256:
        for cname in _extract_vhdr_component_names(vhdr_src):
            lcand = p_entry.parent / cname
            if not lcand.is_file() or str(lcand.resolve()) not in verified_companions:
                raise ToolkitError(
                    "source_recording_modified",
                    f"BrainVision header component {cname!r} changed target on disk; refusing to read modified signal.",
                    recording=rec.slug,
                )
    elif rec.source_sha256 and is_vendor_recording_suffix(p_entry):
        verify_vendor_companion_links(p_entry, vhdr_src, verified_companions, rec.slug)

    if rec.source_sha256:
        if actual_sha != rec.source_sha256:
            raise ToolkitError(
                "source_recording_modified",
                f"Source recording {p.name!r} content (SHA-256) changed on disk since registration; "
                "refusing to read modified signal under an earlier registered recording/run.",
                recording=rec.slug,
            )
        if bp_resolved_str and bp_resolved_str not in verified_companions:
            bp_p = Path(bp_resolved_str)
            if not bp_p.is_file():
                raise ToolkitError(
                    "source_recording_missing",
                    f"Registered published bipolar pairs sidecar missing on disk: {bp_p.name}",
                    recording=rec.slug,
                )
            if snap_dir is not None:
                bp_snap_path = snap_dir / f"__brainlens_bp_{bp_p.name}"
                bp_sha, _bp_sz, bst = copy_fn(bp_p, bp_snap_path)
            else:
                bp_sha, _bp_sz, bst = verify_file(bp_p, "Published bipolar pairs sidecar")
            if _v2_fingerprint_for_recording(rec, actual_sha, signal_comp_hashes, bp_sha) != rec.fingerprint:
                raise ToolkitError(
                    "source_recording_modified",
                    f"Published bipolar pairs sidecar {bp_p.name!r} content (SHA-256) changed on disk since registration; "
                    "refusing to derive altered montage under an earlier recording fingerprint.",
                    recording=rec.slug,
                )
            verified_companions[bp_resolved_str] = (*bst, bp_sha)
    else:
        legacy_comp_hashes: dict[str, str] = {}
        if p_res.suffix.lower() == ".vhdr":
            if snap_dir is not None:
                orig_parent = p_res.parent
                for cname in _extract_vhdr_component_names(vhdr_src):
                    cp_r = (orig_parent / cname).resolve()
                    if not cp_r.is_file() or cp_r.parent != orig_parent:
                        raise ToolkitError("source_recording_missing", f"BrainVision companion missing: {cname}", recording=rec.slug)
                    c_sha, _c_sz, cst = copy_fn(cp_r, snap_dir / cp_r.name)
                    legacy_comp_hashes[str(cp_r)] = c_sha
                    verified_companions[str(cp_r)] = (*cst, c_sha)
            else:
                hdr = _parse_brainvision_header(p_res)
                for cp in hdr.components.values():
                    cp_r = Path(cp).resolve()
                    c_sha, _c_sz, cst = verify_file(cp_r, "Companion file")
                    legacy_comp_hashes[str(cp_r)] = c_sha
                    verified_companions[str(cp_r)] = (*cst, c_sha)
        elif p_res.name.lower().endswith((".fif", ".fif.gz")):
            orig_parent = p_res.parent
            seen_leg: set[Path] = {p_res}
            cur_leg_snap = snap_main or p_res
            while cur_leg_snap is not None:
                nxt_name = _extract_fiff_next_filename(cur_leg_snap)
                if not nxt_name:
                    break
                cp_r = (orig_parent / Path(nxt_name).name).resolve()
                if cp_r in seen_leg:
                    break
                if not cp_r.is_file() or cp_r.parent != orig_parent:
                    raise ToolkitError("source_recording_missing", f"Split FIFF companion missing: {nxt_name}", recording=rec.slug)
                seen_leg.add(cp_r)
                if snap_dir is None:
                    cur_leg_snap = cp_r
                    c_sha, _c_sz, cst = verify_file(cp_r, "Companion file")
                else:
                    cur_leg_snap = snap_dir / cp_r.name
                    c_sha, _c_sz, cst = copy_fn(cp_r, cur_leg_snap)
                legacy_comp_hashes[str(cp_r)] = c_sha
                verified_companions[str(cp_r)] = (*cst, c_sha)

        bp_sha_leg: str | None = None
        if bp_resolved_str and Path(bp_resolved_str).is_file():
            bp_p = Path(bp_resolved_str)
            if snap_dir is not None:
                bp_snap_path = snap_dir / f"__brainlens_bp_{bp_p.name}"
                bp_sha_leg, _bp_sz, bst = copy_fn(bp_p, bp_snap_path)
            else:
                bp_sha_leg, _bp_sz, bst = verify_file(bp_p, "Sidecar")
            verified_companions[bp_resolved_str] = (*bst, bp_sha_leg)

        if rec.fingerprint not in _legacy_fingerprint_candidates(rec, actual_sha, legacy_comp_hashes, bp_sha=bp_sha_leg):
            raise ToolkitError(
                "source_recording_modified",
                f"Legacy recording {p.name!r} content changed on disk (SHA-256 fingerprint mismatch); "
                "refusing to read modified signal under an earlier registered recording/run.",
                recording=rec.slug,
            )

    if raw_filenames:
        for fn in raw_filenames:
            if not fn:
                continue
            try:
                fp_cand = Path(str(fn)).resolve()
            except OSError:
                continue
            if fp_cand != p_res and str(fp_cand) not in verified_companions:
                raise ToolkitError(
                    "source_recording_modified",
                    f"Unverified multi-part companion file {fp_cand.name!r} is not bound to registered recording {rec.slug!r}; "
                    "re-inspect or analyze under a new run_name.",
                    recording=rec.slug,
                )

    p_res_post = p_entry.resolve()
    anc_post = (_ancestor_chain_sig(p_entry), _ancestor_chain_sig(p_res_post))
    if anc_pre != anc_post or p_res_post != p_res:
        raise ToolkitError(
            "source_recording_modified",
            f"Source directory hierarchy for {p.name!r} changed during verification; refusing unverified read.",
            recording=rec.slug,
        )

    sig = (anc_post, (*st_main, actual_sha), tuple(sorted(verified_companions.items())))
    return snap_main, bp_snap_path, sig


def verify_source_recording_integrity(
    rec: RecordingSpec,
    *,
    raw_filenames: tuple[Any, ...] | list[Any] | None = None,
    progress_cb: ProgressCallback | None = None,
) -> tuple[Any, ...]:
    """Verify the registered content identity, logical paths, and ancestor chains of all signal inputs."""
    try:
        _, _, sig = _verify_or_stage_recording(rec, snap_dir=None, raw_filenames=raw_filenames, progress_cb=progress_cb)
    except OSError as exc:
        raise ToolkitError("source_recording_modified", "Source or companion became unavailable during verification.",
                           recording=rec.slug) from exc
    return sig


def stat_signature(sig: tuple[Any, ...]) -> tuple[Any, ...]:
    """Hash-free projection of a full verification signature: ancestor chains plus (dev, ino, size, mtime_ns, ctime_ns) per file."""
    anc, main, comps = sig
    return (anc, tuple(main[:5]), tuple((k, tuple(v[:5])) for k, v in comps))


def quick_source_signature(rec: RecordingSpec, verified_sig: tuple[Any, ...]) -> tuple[Any, ...] | None:
    """Re-derive `stat_signature(verified_sig)` from the filesystem without hashing any bytes.

    Returns None when a file is missing/unreadable or a registered companion symlink was repointed, so callers
    fall back to full SHA-256 verification. Equality with the stored projection means no verified file was
    replaced, rewritten or renamed (inode, size, mtime and ctime all unchanged) since the full verification.
    """
    try:
        p_entry = _canonical_entry_path(Path(rec.path))
        p_res = p_entry.resolve()
        anc = (_ancestor_chain_sig(p_entry), _ancestor_chain_sig(p_res))
        main = _stat_tuple(p_res)
        comps = tuple((k, _stat_tuple(Path(k))) for k, _v in verified_sig[2])
        for comp_path_str, comp_meta in (rec.companion_files or {}).items():
            logical = comp_meta.get("logical_path")
            if logical and _canonical_entry_path(Path(logical)).resolve() != Path(comp_path_str).resolve():
                return None
    except OSError:
        return None
    return (anc, main, comps)


def stat_signature_is_settled(stat_sig: tuple[Any, ...], verified_at_ns: int, margin_ns: int = 2_000_000_000) -> bool:
    """Racy-timestamp guard (as in git): trust stat equality only if every file's mtime/ctime predates verification by `margin_ns`.

    A write landing within the filesystem's timestamp granularity of the last modification could otherwise
    leave (size, mtime, ctime) unchanged; files touched that recently always get full SHA-256 verification.
    """
    _anc, main, comps = stat_sig
    newest = max([int(main[3]), int(main[4]), *(int(v[3]) for _k, v in comps), *(int(v[4]) for _k, v in comps)])
    return int(verified_at_ns) - newest > int(margin_ns)


def create_verified_snapshot_dir(
    rec: RecordingSpec,
) -> tuple[Path, tempfile.TemporaryDirectory[str], Path | None, tuple[Any, ...]]:
    """Create a private 0700 verified temp snapshot directory of `rec` & ALL companions (caller owns cleanup)."""
    tmp_obj = tempfile.TemporaryDirectory(prefix="brainlens_snap_")
    try:
        snap_dir = Path(tmp_obj.name).resolve()
        try:
            os.chmod(snap_dir, 0o700)
        except OSError:
            pass
        snap_main, bp_snap_path, sig = _verify_or_stage_recording(rec, snap_dir=snap_dir)
        assert snap_main is not None
        return snap_main, tmp_obj, bp_snap_path, sig
    except Exception:
        tmp_obj.cleanup()
        raise


@contextmanager
def verified_source_snapshot(
    rec: RecordingSpec,
) -> Iterator[tuple[Path, Path, Path | None, tuple[Any, ...]]]:
    """Capture the source recording and all verified companions in a private 0700 directory."""
    snap_main, tmp_obj, bp_snap_path, sig = create_verified_snapshot_dir(rec)
    try:
        yield snap_main, Path(tmp_obj.name).resolve(), bp_snap_path, sig
    finally:
        tmp_obj.cleanup()


def verify_raw_filenames(
    rec: RecordingSpec, filenames: tuple[Any, ...] | list[Any], snapshot_dir: Path | None,
    signature: tuple[Any, ...], *, opened_stat: _Stat | None = None,
) -> None:
    """Check exactly which files a lazy reader opened before it serves samples."""
    if snapshot_dir is None:
        allowed = {signature[1][:5], *(v[:5] for _, v in signature[2])}
        valid = (opened_stat is None or opened_stat == signature[1][:5]) and all(
            _stat_tuple(Path(str(f))) in allowed for f in filenames if f
        )
    else:
        directory = snapshot_dir.resolve()
        allowed = {signature[1][:5], *(v[:5] for _, v in signature[2])}
        valid = all(
            p.resolve().is_relative_to(directory) or (
                recording_is_large(rec) and _canonical_entry_path(p).parent == directory and _stat_tuple(p) in allowed
            ) for p in (Path(str(f)) for f in filenames if f)
        )
    if not valid:
        raise ToolkitError("source_recording_modified", "Signal reader opened an unverified source or companion.",
                           recording=rec.slug)


def _pin_file(stack: ExitStack, src: Path, dest: Path, expected: _Stat) -> None:
    """Give a path-based reader a stable name for the verified open inode (including multipart reads)."""
    handle = stack.enter_context(src.open("rb"))
    if _stat_fields(os.fstat(handle.fileno())) != expected:
        raise ToolkitError("source_recording_modified", f"{src.name!r} changed before opening its signal data.")
    # Linux procfs reopens an fd with an independent file position. macOS /dev/fd instead
    # shares the seek position, which multi-part readers cannot use. APFS supports a
    # metadata-only copy-on-write clone directly from our verified open file descriptor.
    target = Path("/proc/self/fd") / str(handle.fileno())
    if target.is_file() and _stat_tuple(target) == expected:
        dest.symlink_to(target)
    elif sys.platform == "darwin":
        import ctypes

        directory = os.open(dest.parent, os.O_RDONLY)
        try:
            clone = ctypes.CDLL(None, use_errno=True).fclonefileat
            if clone(handle.fileno(), directory, os.fsencode(dest.name), 0) != 0:
                raise ToolkitError(
                    "pinned_source_unavailable",
                    "Large multipart recordings need a temporary directory on the same APFS volume. "
                    "Set TMPDIR to a writable directory on the source volume.",
                    os_error=os.strerror(ctypes.get_errno()),
                )
        finally:
            os.close(directory)
    else:
        raise ToolkitError("pinned_source_unavailable", f"Cannot open a stable read handle for {src.name!r} on this filesystem.")
    if _stat_fields(os.fstat(handle.fileno())) != expected:
        raise ToolkitError("source_recording_modified", f"{src.name!r} changed while securing its signal data.")


@contextmanager
def pin_window_files(rec: RecordingSpec, signature: tuple[Any, ...]) -> Iterator[tuple[Path, Path | None]]:
    """Reconstruct the verified input filenames without copying their data or resolving mutable ancestors on read."""
    with ExitStack() as stack:
        source = Path(rec.path)
        is_edf = source.suffix.lower() in {".edf", ".bdf"}
        if is_edf and not rec.published_bipolar_pairs_path:
            # The EDF/BDF reader itself pins its open descriptor; verify_raw_filenames
            # checks that descriptor against the main source identity before any sample read.
            yield source, None
            return
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="brainlens_window_"))).resolve()
        main = source if is_edf else directory / source.name
        if not is_edf:
            _pin_file(stack, source, main, signature[1][:5])
        bp_source = Path(rec.published_bipolar_pairs_path).resolve() if rec.published_bipolar_pairs_path else None
        bp = None
        for path, identity in signature[2]:
            src = Path(path)
            name = rec.companion_files.get(path, {}).get("stage_name") or src.name
            if src == bp_source:
                name = f"__brainlens_bp_{src.name}"
                bp = directory / name
            if Path(name).name != name or (directory / name).exists():
                raise ToolkitError("source_recording_modified", "Invalid or conflicting registered signal companion filename.")
            if is_edf and not _large_file(identity[2]):
                # Small montage sidecars can sit on a different volume from the workspace.
                # Verify the private bytes before they enter a reference plan.
                private_sidecar = directory / name
                shutil.copyfile(src, private_sidecar)
                if file_sha256(private_sidecar) != identity[5] or _stat_tuple(src) != identity[:5]:
                    raise ToolkitError("source_recording_modified", f"Sidecar {name!r} changed while opening the window.")
            else:
                _pin_file(stack, src, directory / name, identity[:5])
        yield main, bp


@contextmanager
def verified_window_source(rec: RecordingSpec) -> Iterator[tuple[Path, Path | None, Path | None, tuple[Any, ...]]]:
    """Bracket window reads. Use immutable private copies for small inputs and verified live files for large ones."""
    with ExitStack() as stack:
        if recording_is_large(rec):
            before = verify_source_recording_integrity(rec)
            main, bp = stack.enter_context(pin_window_files(rec, before))
            directory = None if main == Path(rec.path) else main.parent
        else:
            main, directory, bp, before = stack.enter_context(verified_source_snapshot(rec))
        try:
            yield main, directory, bp, before
        finally:
            if verify_source_recording_integrity(rec) != before:
                raise ToolkitError("source_recording_modified", "Source or companion changed during the signal read.",
                                   recording=rec.slug)
