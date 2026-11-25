"""Low-level EEG/SEEG/iEEG file format readers, EDF/BDF continuity & mixed-rate resampling, and modality inference."""

from __future__ import annotations

import codecs
import math
import os
import re
from tempfile import TemporaryDirectory
import threading
import time
from collections import OrderedDict
from contextlib import ExitStack, nullcontext
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO

import mne
import numpy as np
from mne.io.edf.edf import RawEDF, _read_annotations_edf
try:
    from mne.io.edf.edf import RawBDF
except ImportError:  # MNE < 1.10 dispatches BDF by extension through RawEDF.
    RawBDF = RawEDF

from .io_vendor import VENDOR_REC_SUFFIXES, is_vendor_recording_file, open_vendor_raw
from .models import RecordingPiece, RecordingSpec, ToolkitError

_BASE_REC_SUFFIXES = (".edf", ".bdf", ".vhdr", ".fif", ".fif.gz")
_REC_SUFFIXES = _BASE_REC_SUFFIXES + VENDOR_REC_SUFFIXES
_VOLT_UNITS = {"V", "mV", "uV", "µV", "μV", "nV"}
_ALLOWED_KINDS = {"seeg", "ecog", "eeg", "eog", "ecg", "emg", "stim", "misc", "resp", "bio"}
_TAL_ONSET_RE = re.compile(rb"^([+-]\d+(?:\.\d+)?)\x14")


def is_recording_file(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name.lower()
    if any(name.endswith(s) for s in _BASE_REC_SUFFIXES):
        return True
    return is_vendor_recording_file(path)


def recording_stem(path: Path) -> str:
    if path.name.lower().endswith(".fif.gz"):
        return path.name[:-7]
    return path.stem


def _has_mixed_edf_bdf_rates(raw: Any) -> bool:
    """Return True when an EDF/BDF header contains heterogeneous channel sampling rates."""
    extras_list = getattr(raw, "_raw_extras", None) or []
    if not extras_list or not isinstance(extras_list[0], dict):
        return False
    ex = extras_list[0]
    n_samps = ex.get("n_samps")
    if n_samps is None:
        return False
    sel = ex.get("sel")
    idxs = list(sel) if sel is not None else list(range(len(n_samps)))
    tal_idx = ex.get("tal_idx")
    tal_set: set[int] = set()
    if tal_idx is not None:
        try:
            tal_set.update(int(x) for x in tal_idx)
        except TypeError:
            tal_set.add(int(tal_idx))
    rates = {int(n_samps[i]) for i in idxs if int(i) not in tal_set and 0 <= int(i) < len(n_samps)}
    return len(rates) > 1


def _read_edf_bdf_native_slice(
    path: Path,
    file_sig_idx: int,
    native_start: int,
    native_stop: int,
    *,
    is_bdf: bool = False,
    stream=None,
) -> tuple[Any, int]:
    """Read ONLY native samples [native_start, native_stop) of ONE EDF/BDF signal directly from disk (O(window) RAM)."""
    import numpy as np

    with (nullcontext(stream) if stream is not None else path.open("rb")) as fh:
        fh.seek(0)
        fixed = fh.read(256)
        hdr_bytes = int(fixed[184:192].decode("latin-1").strip())
        n_records = int(fixed[236:244].decode("latin-1").strip())
        ns = int(fixed[252:256].decode("latin-1").strip())
        fh.seek(256 + ns * (16 + 80))
        units = [fh.read(8).decode("latin-1").strip() for _ in range(ns)]
        pmin = [float(fh.read(8).decode("latin-1").strip()) for _ in range(ns)]
        pmax = [float(fh.read(8).decode("latin-1").strip()) for _ in range(ns)]
        dmin = [float(fh.read(8).decode("latin-1").strip()) for _ in range(ns)]
        dmax = [float(fh.read(8).decode("latin-1").strip()) for _ in range(ns)]
        fh.seek(256 + ns * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80))
        n_samps = [int(fh.read(8).decode("latin-1").strip()) for _ in range(ns)]

        bps = 3 if is_bdf else 2
        rec_bytes = sum(n_samps) * bps
        if n_records <= 0 and rec_bytes > 0:
            n_records = max(0, (path.stat().st_size - hdr_bytes) // rec_bytes)
        spr = n_samps[file_sig_idx]
        total_nat = n_records * spr
        ns0 = max(0, min(total_nat, native_start))
        ns1 = max(ns0, min(total_nat, native_stop))
        if ns1 <= ns0 or spr <= 0:
            return np.zeros(0, dtype=np.float64), ns0

        ch_offset_in_rec = sum(n_samps[:file_sig_idx]) * bps
        r0 = ns0 // spr
        r1 = (ns1 - 1) // spr + 1
        out_dig = np.empty((r1 - r0) * spr, dtype=np.int32)
        for ri in range(r0, r1):
            fh.seek(hdr_bytes + ri * rec_bytes + ch_offset_in_rec)
            buf = fh.read(spr * bps)
            if is_bdf:
                raw_u8 = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 3)
                val = (
                    raw_u8[:, 0].astype(np.int32)
                    | (raw_u8[:, 1].astype(np.int32) << 8)
                    | (raw_u8[:, 2].astype(np.int32) << 16)
                )
                val = np.where(val >= 0x800000, val - 0x1000000, val)
                out_dig[(ri - r0) * spr : (ri - r0 + 1) * spr] = val
            else:
                out_dig[(ri - r0) * spr : (ri - r0 + 1) * spr] = np.frombuffer(buf, dtype="<i2")

        sl_off = ns0 - r0 * spr
        dig = out_dig[sl_off : sl_off + (ns1 - ns0)].astype(np.float64)
        denom = dmax[file_sig_idx] - dmin[file_sig_idx]
        gain = (pmax[file_sig_idx] - pmin[file_sig_idx]) / denom if denom != 0 else 1.0
        phys = (dig - dmin[file_sig_idx]) * gain + pmin[file_sig_idx]
        u = units[file_sig_idx].lower().replace("µ", "u").replace("μ", "u")
        scale = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "nv": 1e-9}.get(u, 1.0)
        return phys * scale, ns0


def _read_resampled_edf_bdf_channel(
    path: Path,
    file_sig_idx: int,
    src_fs: float,
    dst_fs: float,
    start: int,
    stop: int,
    *,
    is_bdf: bool = False,
    half_w: int = 16,
    beta: float = 8.6,
    piece: RecordingPiece | None = None,
    stream=None,
):
    """Window-invariant, exact-at-native-samples, O(window) Kaiser-windowed sinc resampler for mixed-rate EDF/BDF."""
    import numpy as np

    n_out = max(0, stop - start)
    if n_out == 0:
        return np.zeros(0, dtype=np.float64)
    ratio = src_fs / dst_fs
    k_lo = int(math.floor(start * ratio)) - half_w
    k_hi = int(math.ceil((stop - 1) * ratio)) + half_w + 1
    if piece is not None:
        k_lo = max(k_lo, round(piece.start_sample * ratio))
        k_hi = min(k_hi, round(piece.stop_sample * ratio))
    nat, actual_lo = _read_edf_bdf_native_slice(path, file_sig_idx, k_lo, k_hi, is_bdf=is_bdf, stream=stream)
    if nat.size == 0:
        return np.zeros(n_out, dtype=np.float64)

    offsets = np.arange(-half_w + 1, half_w + 1, dtype=np.int64)
    inv_i0_beta = 1.0 / float(np.i0(beta))
    out = np.empty(n_out, dtype=np.float64)
    blk = 4096
    for off in range(0, n_out, blk):
        end_off = min(n_out, off + blk)
        s_idx = np.arange(start + off, start + end_off, dtype=np.float64)
        u = s_idx * ratio
        k0 = np.floor(u).astype(np.int64)
        ks = k0[:, None] + offsets[None, :]
        dx = u[:, None] - ks.astype(np.float64)
        w_arg = np.clip(1.0 - (dx / float(half_w)) ** 2, 0.0, 1.0)
        win = np.i0(beta * np.sqrt(w_arg)) * inv_i0_beta
        ker = np.sinc(dx) * win
        ker_sum = np.sum(ker, axis=1, keepdims=True)
        ker /= np.where(np.abs(ker_sum) > 1e-12, ker_sum, 1.0)
        rel_idx = np.clip(ks - actual_lo, 0, nat.size - 1)
        out[off:end_off] = np.sum(nat[rel_idx] * ker, axis=1)
    return out


def _scan_edf_tals(path: Path, fh: BinaryIO) -> tuple[list[RecordingPiece], bytes]:
    """Read only the annotation channels of each record. Retain piece starts and labeled TALs."""
    fh.seek(0)
    fixed = fh.read(256)
    discontinuous = fixed[192:236].strip().upper().startswith((b"EDF+D", b"BDF+D"))
    try:
        duration = Decimal(fixed[244:252].strip().decode("ascii"))
        if not duration.is_finite() or duration <= 0:
            raise ValueError
        hdr_bytes = int(fixed[184:192])
        n_records = int(fixed[236:244])
        ns = int(fixed[252:256])
        labels = [fh.read(16).strip().lower() for _ in range(ns)]
        annotations = [i for i, label in enumerate(labels) if label in (b"edf annotations", b"bdf annotations")]
        if not annotations:
            if discontinuous:
                raise ValueError
            return [], b""
        fh.seek(256 + 216 * ns)
        samples = [int(fh.read(8)) for _ in range(ns)]
        bps = 3 if path.suffix.lower() == ".bdf" else 2
        widths = [n * bps for n in samples]
        stride = sum(widths)
        spr = max(n for i, n in enumerate(samples) if i not in annotations)
        if min(stride, spr) <= 0:
            raise ValueError
        if n_records == -1:
            n_records = (path.stat().st_size - hdr_bytes) // stride
        if n_records < 1:
            raise ValueError
    except (ValueError, InvalidOperation, UnicodeError) as exc:
        raise ToolkitError("discontinuous_recording_unsupported", f"Invalid EDF/BDF record geometry or missing timekeeping TAL in {path.name!r}.") from exc

    fields = [(sum(widths[:i]), widths[i]) for i in annotations]
    starts: list[tuple[int, Decimal]] = []
    labels_tal = bytearray()
    origin = previous = Decimal(0)
    for index in range(n_records):
        for channel, (offset, length) in enumerate(fields):
            fh.seek(hdr_bytes + index * stride + offset)
            payload = fh.read(length)
            if len(payload) != length:
                raise ToolkitError("discontinuous_recording_unsupported", f"Truncated TAL in data record {index}.")
            entries = payload.rstrip(b"\x00").split(b"\x00")
            if channel == 0:
                match = _TAL_ONSET_RE.match(entries[0])
                if not match or entries[0][match.end():] != b"\x14":
                    raise ToolkitError("discontinuous_recording_unsupported", f"Missing timekeeping TAL in data record {index}.")
                onset = Decimal(match[1].decode("ascii"))
                if index == 0:
                    origin = onset
                    starts.append((0, Decimal(0)))
                    labels_tal.extend(entries[0] + b"\x00")
                elif onset < previous + duration:
                    raise ToolkitError("discontinuous_recording_unsupported", f"Overlapping EDF/BDF data record {index}.")
                elif onset > previous + duration:
                    starts.append((index * spr, onset - origin))
                previous = onset
                entries = entries[1:]
            for tal in entries:
                if tal:
                    labels_tal.extend(tal + b"\x00")
    pieces = []
    if discontinuous or len(starts) > 1:
        for k, (start, onset) in enumerate(starts):
            stop = starts[k + 1][0] if k + 1 < len(starts) else n_records * spr
            pieces.append(RecordingPiece(start_sample=start, stop_sample=stop, onset_seconds=float(onset)))
    return pieces, bytes(labels_tal)


_TAL_CACHE: OrderedDict[tuple[Any, ...], tuple[list[RecordingPiece], bytes]] = OrderedDict()
_TAL_LOCK = threading.Lock()


def _edf_clocks_and_annotations(path: Path, stream: BinaryIO) -> tuple[list[RecordingPiece], bytes]:
    # All stale/changed-source decisions remain with the integrity verifier. Cache parsed
    # metadata only for a stable open inode, including the same inode under private pins.
    from .io_integrity import _large_file, _stat_fields, stat_signature_is_settled

    signature = _stat_fields(os.fstat(stream.fileno()))
    cacheable = _large_file(signature[2]) and stat_signature_is_settled(((), signature, ()), time.time_ns())
    key = (path.suffix.lower(), signature)
    if cacheable:
        with _TAL_LOCK:
            if key in _TAL_CACHE:
                _TAL_CACHE.move_to_end(key)
                return _TAL_CACHE[key]
    metadata = _scan_edf_tals(path, stream)
    if _stat_fields(os.fstat(stream.fileno())) != signature:
        raise ToolkitError("source_recording_modified", f"{path.name!r} changed during TAL parsing.")
    if cacheable:
        with _TAL_LOCK:
            _TAL_CACHE[key] = metadata
            if len(_TAL_CACHE) > 8:
                _TAL_CACHE.popitem(last=False)
    return metadata


def read_edf_timeline(path: Path) -> list[RecordingPiece]:
    if path.suffix.lower() not in {".edf", ".bdf"}:
        return []
    with path.open("rb") as fh:
        return _edf_clocks_and_annotations(path, fh)[0]


def _check_edf_bdf_continuity(path: Path) -> None:
    """Reject a gapped source when the caller needs a regular, continuous EDF clock."""
    if read_edf_timeline(path):
        raise ToolkitError("discontinuous_recording_unsupported", "A continuous EDF export would lose the source record clock.")


class _WindowedTAL:
    """MNE's constructor asks for *every* record to extract TALs, even with preload=False."""

    def __init__(self, path: Path, preload: bool | str):
        stream = path.open("rb")
        self._brainlens_pins = ExitStack()
        try:
            pieces, tal = _edf_clocks_and_annotations(path, stream)
            first_tal = _TAL_ONSET_RE.match(tal)
            self._brainlens_first_tal_seconds = float(first_tal[1]) if first_tal else 0.0
            # MNE's decoder accepts packed 16-bit TAL words for either format.
            words = np.frombuffer(tal + (b"\x00" if len(tal) % 2 else b""), dtype="<i2")
            self._brainlens_tal_words = words
            self._brainlens_stream = stream
            st = os.fstat(stream.fileno())
            self._brainlens_fd_stat = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            stream.seek(0)
            source = stream
            if tuple(int(x) for x in mne.__version__.split(".")[:2]) < (1, 10):
                # Older MNE accepts paths only. Pin the same descriptor under a name
                # with the original suffix so its EDF/BDF header dispatch still works.
                from .io_integrity import _pin_file
                directory = self._brainlens_pins.enter_context(TemporaryDirectory(prefix="brainlens-edf-"))
                source = Path(directory) / path.name
                _pin_file(self._brainlens_pins, path, source, self._brainlens_fd_stat)
            super().__init__(source, preload=preload, stim_channel=None, infer_types=False, verbose="ERROR")
            if source is not stream:
                self._raw_extras[0]["brainlens_fd_path"] = source
                self._filenames = [path]
            del self._brainlens_tal_words
            self._brainlens_pieces = pieces
            if pieces:
                # MNE crops at packed n_times. Keep late true-time markers for registration.
                self._brainlens_annotations = _read_annotations_edf(words, ch_names=self.ch_names)
        except BaseException:
            self._brainlens_pins.close()
            stream.close()
            raise

    def set_annotations(self, annotations, *args, **kwargs):
        if annotations is not None and hasattr(self, "_brainlens_tal_words") and annotations.orig_time is None:
            # Older BaseRaw.set_annotations materializes raw.times for its final endpoint.
            # EDF samples start at zero. Reproduce that crop and time origin in constant memory.
            annotations = annotations.copy()
            annotations.crop(0, self.n_times / self.info["sfreq"])
            annotations._orig_time = self.info["meas_date"]
            self._annotations = annotations
            return self
        return super().set_annotations(annotations, *args, **kwargs)

    def close(self):
        try:
            super().close()
        finally:
            self._brainlens_pins.close()
            self._brainlens_stream.close()

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        if not isinstance(idx, slice) and len(idx) == 0:
            return [self._brainlens_tal_words]
        # MNE calls this method on a _ReadSegmentFileProtector for normal signal reads.
        # EDF and BDF share this implementation and use the file subtype to decode.
        reader = self
        if fd_path := self._raw_extras[fi].get("brainlens_fd_path"):
            reader = SimpleNamespace(_raw_extras=self._raw_extras, filenames=[fd_path], _filenames=[fd_path])
        return RawEDF._read_segment_file(reader, data, idx, fi, start, stop, cals, mult)


class _WindowedEDF(_WindowedTAL, RawEDF):
    pass


class _WindowedBDF(_WindowedTAL, RawBDF):
    pass


def open_mne_raw(path: str | Path, preload: bool | str = False):
    p = Path(path)
    low = p.name.lower()
    if low.endswith(".edf"):
        return _WindowedEDF(p, preload)
    if low.endswith(".bdf"):
        return _WindowedBDF(p, preload)
    if low.endswith((".fif", ".fif.gz")):
        return mne.io.read_raw_fif(p, preload=preload, on_split_missing="raise", verbose="ERROR")
    if low.endswith(".vhdr"):
        return mne.io.read_raw_brainvision(p, preload=preload, verbose="ERROR")
    return open_vendor_raw(p, preload=preload)


def read_scaled_volts(raw, recording: RecordingSpec, start: int, stop: int, picks: list[int] | None = None):
    import warnings
    import numpy as np

    idx = picks if picks is not None else list(range(len(recording.channels)))
    sfx = Path(recording.path).suffix.lower()
    is_edf_bdf = sfx in {".edf", ".bdf"}
    if is_edf_bdf and not getattr(raw, "preload", False) and _has_mixed_edf_bdf_rates(raw):
        extras_list = getattr(raw, "_raw_extras", None) or []
        ex = extras_list[0] if extras_list and isinstance(extras_list[0], dict) else {}
        rec_len = ex.get("record_length")
        try:
            rec_len_ok = bool(rec_len is not None and np.all(np.asarray(rec_len, dtype=np.float64) > 0))
        except Exception:
            rec_len_ok = False
        fns = getattr(raw, "filenames", None) or ()
        snap_p = Path(str(fns[0])) if fns and fns[0] else None
        if rec_len_ok and snap_p is not None and snap_p.is_file():
            sel = ex.get("sel")
            full_rows: list[tuple[int, int]] = []
            low_rows: list[tuple[int, int, float]] = []
            for row_i, ch_i in enumerate(idx):
                ch_fs = recording.channels[ch_i].source_sfreq
                if ch_fs is not None and abs(float(ch_fs) - float(recording.sfreq)) > 1e-3:
                    low_rows.append((row_i, ch_i, float(ch_fs)))
                else:
                    full_rows.append((row_i, ch_i))
            vals = np.empty((len(idx), max(0, stop - start)), dtype=np.float64)
            if full_rows:
                full_picks = [ch_i for _, ch_i in full_rows]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Loading an EDF with mixed sampling frequencies")
                    full_data = raw.get_data(picks=full_picks, start=start, stop=stop, reject_by_annotation=None)
                for k, (row_i, ch_i) in enumerate(full_rows):
                    row_v = full_data[k]
                    if recording.channels[ch_i].source_unit == "nV":
                        row_v = row_v * 1e-9
                    vals[row_i] = row_v
            for row_i, ch_i, ch_fs in low_rows:
                file_sig_idx = int(sel[ch_i]) if sel is not None and ch_i < len(sel) else ch_i
                for piece in recording.pieces or [None]:
                    a = max(start, piece.start_sample) if piece else start
                    b = min(stop, piece.stop_sample) if piece else stop
                    if a < b:
                        vals[row_i, a - start:b - start] = _read_resampled_edf_bdf_channel(
                            snap_p, file_sig_idx, ch_fs, float(recording.sfreq), a, b, is_bdf=(sfx == ".bdf"), piece=piece,
                            stream=getattr(raw, "_brainlens_stream", None),
                        )
            return vals
        raise ToolkitError("invalid_recording", "Missing EDF/BDF geometry for bounded mixed-rate reading.")

    vals = raw.get_data(picks=idx, start=start, stop=stop, reject_by_annotation=None)
    if is_edf_bdf:
        for row_i, ch_i in enumerate(idx):
            if recording.channels[ch_i].source_unit == "nV":
                vals[row_i] *= 1e-9
    return vals


def _parse_bdf_header(path: Path) -> SimpleNamespace:
    with path.open("rb") as fh:
        fixed = fh.read(256)
        if len(fixed) < 256:
            raise ToolkitError("invalid_bdf", f"BDF header truncated: {path.name}")
        record_duration = float(fixed[244:252].decode("latin-1").strip() or "1")
        ns = int(fixed[252:256].decode("latin-1").strip())
        labels = [fh.read(16).decode("latin-1").strip() for _ in range(ns)]
        fh.seek(256 + ns * (16 + 80))
        units = [fh.read(8).decode("latin-1").strip() for _ in range(ns)]
        fh.seek(256 + ns * (16 + 80 + 8 * 5))
        prefilt = [fh.read(80).decode("latin-1").strip() for _ in range(ns)]
        n_samp = [int(fh.read(8).decode("latin-1").strip()) for _ in range(ns)]
    signals = [
        SimpleNamespace(
            label=labels[i],
            physical_dimension=units[i],
            sampling_frequency=(n_samp[i] / record_duration) if record_duration > 0 else 0.0,
            prefiltering=prefilt[i],
        )
        for i in range(ns)
    ]
    return SimpleNamespace(
        signals=signals, components={},
        local_patient_identification=fixed[8:88].decode("latin-1").strip(),
        local_recording_identification=fixed[88:168].decode("latin-1").strip(),
    )


def _read_vhdr_text(path: Path) -> str:
    """Decode BrainVision .vhdr respecting Codepage=ANSI (cp1252) / UTF-8 without mangling 'µV' or accents."""
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")
    peek = raw.decode("latin-1", errors="ignore")
    m = re.search(r"(?mi)^\s*codepage\s*=\s*([^\r\n;]+)", peek)
    if m:
        cp_raw = m.group(1).strip().lower()
        if cp_raw in {"utf-8", "utf8"}:
            return raw.decode("utf-8-sig", errors="replace")
        if cp_raw in {"ansi", "windows-1252", "cp1252", "latin-1", "latin1", "iso-8859-1"}:
            return raw.decode("cp1252", errors="replace")
        try:
            return raw.decode(codecs.lookup(cp_raw).name, errors="replace")
        except LookupError:
            pass
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _parse_brainvision_header(path: Path, *, verified_components: set[Path] | None = None) -> SimpleNamespace:
    text = _read_vhdr_text(path)
    section: str | None = None
    common: dict[str, str] = {}
    channels: dict[int, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        if "=" not in line:
            continue
        k, v = (part.strip() for part in line.split("=", 1))
        if section == "common infos":
            common[k.lower()] = v
        elif section == "channel infos" and re.fullmatch(r"Ch\d+", k, re.IGNORECASE):
            channels[int(k[2:])] = v
    count = int(common["numberofchannels"])
    interval_us = float(common["samplinginterval"])
    components: dict[str, str] = {}
    for role, key in (("data", "datafile"), ("markers", "markerfile")):
        cname = common.get(key)
        if cname:
            cand = (path.parent / cname).resolve()
            if not cand.is_file() or (cand.parent != path.parent.resolve() and cand not in (verified_components or set())):
                raise ToolkitError("missing_companion", f"BrainVision {role} file missing: {cname}")
            components[role] = str(cand)
    signals = []
    for idx in range(1, count + 1):
        fields = channels[idx].split(",")
        label = fields[0].replace("\\1", ",")
        unit = fields[3].strip() if len(fields) > 3 and fields[3].strip() else "µV"
        signals.append(
            SimpleNamespace(
                label=label,
                physical_dimension=unit,
                sampling_frequency=1e6 / interval_us,
                prefiltering="",
            )
        )
    return SimpleNamespace(signals=signals, components=components)


def _extract_vhdr_component_names(vhdr_path: Path) -> list[str]:
    text = _read_vhdr_text(vhdr_path)
    section: str | None = None
    names: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        if section == "common infos" and "=" in line:
            k, v = (part.strip() for part in line.split("=", 1))
            if k.lower() in {"datafile", "markerfile"} and v:
                names.append(v)
    return names


def _extract_fiff_next_filename(fif_path: Path) -> str | None:
    """Read MNE FIFF split-continuation filename strictly from the given FIF file's FIFFB_REF tag."""
    try:
        from mne._fiff.constants import FIFF
        from mne._fiff.open import fiff_open
        from mne._fiff.tag import read_tag
        from mne._fiff.tree import dir_tree_find
    except ImportError:
        from mne.io.constants import FIFF
        from mne.io.open import fiff_open
        from mne.io.tag import read_tag
        from mne.io.tree import dir_tree_find

    fid, tree, _ = fiff_open(Path(fif_path))
    try:
        for ref in dir_tree_find(tree, FIFF.FIFFB_REF):
            role = None
            fname = None
            for ent in ref.get("directory") or ():
                if ent.kind == FIFF.FIFF_REF_ROLE:
                    role = int(read_tag(fid, ent.pos).data.item())
                elif ent.kind == FIFF.FIFF_REF_FILE_NAME:
                    fname = str(read_tag(fid, ent.pos).data)
            if role == int(FIFF.FIFFV_ROLE_NEXT_FILE) and fname:
                return fname
    finally:
        fid.close()
    return None


def _filter_split_fif_continuations(rec_files: list[Path], open_fn=open_mne_raw) -> list[Path]:
    """Exclude ONLY actual MNE FIFF split-continuation chunks discovered from FIFF header links."""
    fifs = [p for p in rec_files if p.name.lower().endswith((".fif", ".fif.gz"))]
    if len(fifs) < 2:
        return rec_files
    continuation_paths: set[Path] = set()
    for p in fifs:
        try:
            raw = open_fn(p, preload=False)
            try:
                for fn in list(getattr(raw, "filenames", None) or ())[1:]:
                    if fn:
                        continuation_paths.add(Path(str(fn)).resolve())
            finally:
                raw.close()
        except Exception:
            continue
    return [p for p in rec_files if p.resolve() not in continuation_paths] if continuation_paths else rec_files


def _prefilter_lowpass(text: str) -> float | None:
    m = re.search(r"(?:LP|lowpass)\s*:?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:Hz)?", text or "", re.IGNORECASE)
    return float(m.group(1)) if m and float(m.group(1)) > 0 else None


def _infer_declared_modality(
    sidecar: dict[str, Any],
    bids_json: dict[str, Any],
    folder_prov: dict[str, Any],
) -> str | None:
    for doc in (sidecar, sidecar.get("provenance", {}), folder_prov, folder_prov.get("same_subject_association", {})):
        if isinstance(doc, dict):
            m = str(doc.get("modality") or "").lower()
            if m in {"seeg", "ieeg", "scalp"}:
                return m
            ds_obj = doc.get("dataset") if isinstance(doc.get("dataset"), dict) else {}
            pid = doc.get("participant_identity") or ds_obj.get("participant_identity") or {}
            if isinstance(pid, dict) and str(pid.get("implant") or "").lower() in {"seeg", "ieeg", "scalp"}:
                return str(pid["implant"]).lower()
    if bids_json:
        if (bids_json.get("SEEGChannelCount") or 0) > 0:
            return "seeg"
        if (bids_json.get("ECOGChannelCount") or 0) > 0:
            return "ieeg"
        if (bids_json.get("EEGChannelCount") or 0) > 0:
            return "scalp"
    ch_meta = sidecar.get("channels")
    if isinstance(ch_meta, dict) and ch_meta:
        kinds = {str(v.get("kind", "")).lower() for v in ch_meta.values() if isinstance(v, dict)}
        if "seeg" in kinds:
            return "seeg"
        if "ecog" in kinds:
            return "ieeg"
        if kinds == {"eeg"}:
            return "scalp"
    return None


def _infer_modality(
    rec_path: Path,
    sidecar: dict[str, Any],
    bids_json: dict[str, Any],
    folder_prov: dict[str, Any],
    *,
    bids_channel_kinds: dict[str, str] | None = None,
    mne_ch_types: list[str] | None = None,
    is_fif: bool = False,
) -> str:
    declared = _infer_declared_modality(sidecar, bids_json, folder_prov)
    if declared:
        return declared
    for kind_set in (
        set(bids_channel_kinds.values()) if bids_channel_kinds else set(),
        set(mne_ch_types) if (is_fif and mne_ch_types) else set(),
    ):
        if "seeg" in kind_set:
            return "seeg"
        if "ecog" in kind_set:
            return "ieeg"
        if "eeg" in kind_set:
            return "scalp"
    low = str(rec_path).lower()
    return "scalp" if "scalp" in low or "_eeg." in low else ("ieeg" if "ecog" in low or "zurich" in low else "seeg")
