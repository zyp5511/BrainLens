"""Standalone data contracts, deterministic hashing, and SEEG shaft helpers (zero legacy imports)."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_serializer, model_validator

def _detect_source_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "workbench" / "pyproject.toml").is_file():
            return parent
    return None


_SOURCE_ROOT = _detect_source_root()
IS_SOURCE_CHECKOUT = _SOURCE_ROOT is not None
PROJECT_ROOT = _SOURCE_ROOT if _SOURCE_ROOT is not None else Path.cwd()
USER_CONFIG_DIR = Path.home() / ".brainlens"
LEGACY_USER_CONFIG_DIR = Path.home() / ".braintoolkit"
USER_ENGINE_JSON = USER_CONFIG_DIR / "engine.json"
LEGACY_USER_ENGINE_JSON = LEGACY_USER_CONFIG_DIR / "engine.json"
DEFAULT_WORKBENCH_DIR = (
    (_SOURCE_ROOT / "workbench_results")
    if _SOURCE_ROOT is not None
    else (Path.cwd() / "workbench_results")
)
LEGACY_ENGINE_JSON = (
    (_SOURCE_ROOT / "results" / "engine.json")
    if _SOURCE_ROOT is not None
    else (Path.cwd() / "results" / "engine.json")
)


def get_env(
    primary: str,
    legacy: str | tuple[str, ...] | None = None,
    default: str | None = None,
) -> str | None:
    """Read an environment variable with legacy BTK_*/BRAINTOOLKIT_* fallback."""
    val = os.environ.get(primary)
    if val is not None and val != "":
        return val
    if legacy is not None:
        names = (legacy,) if isinstance(legacy, str) else legacy
        for name in names:
            lval = os.environ.get(name)
            if lval is not None and lval != "":
                return lval
    return default


def default_workspace_dir() -> Path:
    env_dir = get_env(
        "BRAINLENS_WORKSPACE",
        ("BRAINLENS_WORKBENCH_DIR", "BRAINTOOLKIT_WORKSPACE", "BRAINTOOLKIT_WORKBENCH_DIR"),
    )
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    if _SOURCE_ROOT is not None:
        return (_SOURCE_ROOT / "workbench_results").resolve()
    return (Path.cwd() / "workbench_results").resolve()

_SHAFT_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_'\-]*?)0*([1-9][0-9]*)$")
_AUX_PREFIXES = ("EKG", "ECG", "EMG", "EOG", "RESP", "STIM", "TRIG", "DC", "PHOTIC", "REF", "GND")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ToolkitError(Exception):
    """Structured domain error with stable machine code and clear message."""

    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.details}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: str | Path, payload: Any, *, mode: int | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}.tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        if mode is not None:
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
            try:
                os.fchmod(fd, mode)
            except OSError:
                pass
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
        else:
            with tmp.open("w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
        os.replace(tmp, p)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def trust_store_dir() -> Path:
    """Out-of-workspace machine-local trust directory for cryptographic sheet verification."""
    env = (get_env("BRAINLENS_TRUST_DIR", "BTK_TRUST_DIR") or "").strip()
    base = Path(env).expanduser().resolve() if env else USER_CONFIG_DIR.resolve()
    base.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(base, 0o700)
    except OSError:
        pass
    return base


def _get_or_create_sheet_hmac_key() -> bytes:
    tdir = trust_store_dir()
    kpath = tdir / "sheet_hmac.key"
    with FileLock(str(tdir / ".trust.lock")):
        if kpath.is_file():
            raw = kpath.read_bytes().strip()
            if len(raw) >= 32:
                return raw
        key_hex = secrets.token_hex(32).encode("ascii")
        tmp = kpath.with_name(f".{kpath.name}.{os.getpid()}.tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
        with os.fdopen(fd, "wb") as fh:
            fh.write(key_hex)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, kpath)
        return key_hex


def _sheet_trust_key(workspace_id: str, case_slug: str, filename: str) -> str:
    return f"{workspace_id}|{case_slug}|{filename}"


def _sheet_hmac(key: bytes, workspace_id: str, case_slug: str, filename: str, html_sha256: str) -> str:
    msg = f"{workspace_id}|{case_slug}|{filename}|{html_sha256}".encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def record_verified_sheet(
    workspace_id: str,
    case_slug: str,
    filename: str,
    html_bytes: bytes,
    primary_run: str,
    *,
    run_signature: dict[str, Any] | None = None,
) -> str:
    """Record an HMAC-SHA256 trust entry OUTSIDE the workspace in ~/.brainlens/verified_sheets.json."""
    tdir = trust_store_dir()
    key = _get_or_create_sheet_hmac_key()
    digest = hashlib.sha256(html_bytes).hexdigest()
    sig = _sheet_hmac(key, workspace_id, case_slug, filename, digest)
    mpath = tdir / "verified_sheets.json"
    with FileLock(str(tdir / ".trust.lock")):
        manifest = read_json(mpath) if mpath.is_file() else {}
        if not isinstance(manifest, dict):
            manifest = {}
        entry: dict[str, Any] = {
            "sha256": digest,
            "hmac": sig,
            "workspace_id": workspace_id,
            "case": case_slug,
            "filename": filename,
            "primary_run": primary_run,
            "generated_at": utc_now(),
        }
        if run_signature is not None:
            entry["run_signature"] = run_signature
        manifest[_sheet_trust_key(workspace_id, case_slug, filename)] = entry
        atomic_write_json(mpath, manifest, mode=0o600)
    return digest


def get_verified_sheet_entry(
    workspace_id: str,
    case_slug: str,
    filename: str,
    html_bytes: bytes,
) -> dict[str, Any] | None:
    """Return the verified manifest entry if HTML bytes match this machine's HMAC-SHA256 trust store."""
    tdir = trust_store_dir()
    kpath = tdir / "sheet_hmac.key"
    mpath = tdir / "verified_sheets.json"
    if not (kpath.is_file() and mpath.is_file()):
        return None
    try:
        key = kpath.read_bytes().strip()
        if len(key) < 32:
            return None
        manifest = read_json(mpath)
        entry = manifest.get(_sheet_trust_key(workspace_id, case_slug, filename)) if isinstance(manifest, dict) else None
        if not isinstance(entry, dict):
            return None
        saved_sha = str(entry.get("sha256") or "")
        saved_hmac = str(entry.get("hmac") or "")
        if not saved_sha or not saved_hmac:
            return None
        actual_sha = hashlib.sha256(html_bytes).hexdigest()
        if not secrets.compare_digest(saved_sha, actual_sha):
            return None
        expected_hmac = _sheet_hmac(key, workspace_id, case_slug, filename, actual_sha)
        if not secrets.compare_digest(saved_hmac, expected_hmac):
            return None
        return entry
    except Exception:
        return None


def verify_sheet_trust(workspace_id: str, case_slug: str, filename: str, html_bytes: bytes) -> bool:
    """Verify an exported case sheet against this machine's out-of-workspace HMAC-SHA256 trust store."""
    return get_verified_sheet_entry(workspace_id, case_slug, filename, html_bytes) is not None


def parse_shaft_group(channel_name: str, channel_kind: str = "seeg") -> dict[str, Any]:
    """Group SEEG/iEEG contacts by depth-electrode shaft prefix + contact number.

    Strictly a visual display/navigation grouping; never asserts anatomy or bipolar validity.
    """
    clean = channel_name.strip()
    upper = clean.upper()
    is_aux = channel_kind not in {"seeg", "ecog", "eeg"} or any(upper.startswith(p) for p in _AUX_PREFIXES)
    if is_aux:
        return {
            "shaft": "AUX",
            "contact_index": 9999,
            "is_auxiliary": True,
            "display_group": "Auxiliary / Non-EEG",
        }
    m = _SHAFT_RE.match(clean)
    if m:
        prefix = m.group(1).rstrip("_- ")
        idx = int(m.group(2))
        if prefix:
            return {
                "shaft": prefix,
                "contact_index": idx,
                "is_auxiliary": False,
                "display_group": f"Shaft {prefix}",
            }
    return {
        "shaft": "OTHER",
        "contact_index": 9999,
        "is_auxiliary": False,
        "display_group": "Other EEG",
    }


def merge_intervals(
    spans: list[tuple[int, int]] | list[list[int]],
    clip_start: int | None = None,
    clip_end: int | None = None,
) -> list[tuple[int, int]]:
    cleaned: list[tuple[int, int]] = []
    for a_raw, b_raw in spans:
        a, b = int(a_raw), int(b_raw)
        if clip_start is not None:
            a = max(clip_start, a)
        if clip_end is not None:
            b = min(clip_end, b)
        if b > a:
            cleaned.append((a, b))
    if not cleaned:
        return []
    cleaned.sort()
    merged = [cleaned[0]]
    for a, b in cleaned[1:]:
        la, lb = merged[-1]
        if a <= lb:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def intersect_intervals(
    left: list[tuple[int, int]] | list[list[int]],
    right: list[tuple[int, int]] | list[list[int]],
) -> list[tuple[int, int]]:
    a_list = merge_intervals(left)
    b_list = merge_intervals(right)
    i = j = 0
    out: list[tuple[int, int]] = []
    while i < len(a_list) and j < len(b_list):
        lo = max(a_list[i][0], b_list[j][0])
        hi = min(a_list[i][1], b_list[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a_list[i][1] < b_list[j][1]:
            i += 1
        else:
            j += 1
    return out


def complement_intervals(
    spans: list[tuple[int, int]] | list[list[int]],
    start: int,
    stop: int,
) -> list[tuple[int, int]]:
    merged = merge_intervals(spans, start, stop)
    out: list[tuple[int, int]] = []
    cursor = start
    for a, b in merged:
        if a > cursor:
            out.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < stop:
        out.append((cursor, stop))
    return out


def true_runs_mask(mask_1d) -> list[tuple[int, int]]:
    import numpy as np

    arr = np.asarray(mask_1d, dtype=bool)
    if arr.size == 0 or not arr.any():
        return []
    padded = np.empty(arr.size + 2, dtype=bool)
    padded[0] = False
    padded[-1] = False
    padded[1:-1] = arr
    diffs = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(diffs[k]), int(diffs[k + 1])) for k in range(0, len(diffs), 2)]


def seconds_to_sample_range(start_s: float, end_s: float, sfreq: float, n_samples: int) -> tuple[int, int]:
    a = max(0, min(n_samples, int(math.floor(start_s * sfreq + 1e-9))))
    b = max(a, min(n_samples, int(math.ceil(end_s * sfreq - 1e-9))))
    return a, b


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


class MarkerSpec(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    label: str
    source: str
    channels: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ordered(self) -> MarkerSpec:
        if self.end < self.start:
            raise ValueError("marker end cannot precede start")
        return self


class SpatialSpec(StrictModel):
    coordinate_system: str = Field(min_length=1)
    units: Literal["m", "cm", "mm"]
    positions: dict[str, tuple[float, float, float]]
    source: str
    template: bool = False
    surface_path: str | None = None


class ProvenanceSpec(StrictModel):
    origin_kind: Literal["synthetic_engineering_data", "public_dataset", "local_recording"]
    is_synthetic: bool = False
    summary: str
    dataset_name: str | None = None
    participant_id: str | None = None
    participant_id_source: Literal["bids", "sidecar", "header", "withheld"] | None = None
    doi: str | None = None
    license: str | None = None
    public_url: str | None = None
    same_patient_basis: str | None = None
    source_file: str | None = None
    sidecar_path: str | None = None
    limitations: list[str] = Field(default_factory=list)


class ChannelSpec(StrictModel):
    id: str
    name: str
    original_label: str
    kind: str
    unit: Literal["V", "dimensionless", "unknown"] = "V"
    source_unit: str | None = None
    source_sfreq: float | None = Field(default=None, gt=0)
    source_lowpass: float | None = Field(default=None, gt=0)
    contacts: list[str] = Field(default_factory=list)
    parents: list[str] = Field(default_factory=list)
    weights: list[float] = Field(default_factory=list)
    shaft: str = "OTHER"
    contact_index: int = 9999
    is_auxiliary: bool = False


class RecordingPiece(StrictModel):
    """Half-open interval in packed source samples, anchored to the recording clock."""

    start_sample: int = Field(ge=0)
    stop_sample: int = Field(gt=0)
    onset_seconds: float = Field(ge=0)

    def time_at(self, sample: float, sfreq: float) -> float:
        return self.onset_seconds + (sample - self.start_sample) / sfreq

    def sample_offset(self, seconds: float, sfreq: float) -> float:
        offset = (seconds - self.onset_seconds) * sfreq
        # Subtracting a days-long onset can move a sample boundary across floor/ceil.
        # Snap only within the representation error of the true-time timestamp.
        nearest = round(offset)
        tolerance = sfreq * max(math.ulp(seconds), math.ulp(self.onset_seconds)) + math.ulp(offset)
        return float(nearest) if abs(offset - nearest) <= tolerance else offset


class RecordingSpec(StrictModel):
    # Used only for local same-subject checks, never included in model_dump() or exports.
    _subject_digest: str | None = PrivateAttr(default=None)
    _header_terms: tuple[str, ...] | None = PrivateAttr(default=None)
    _registration_path: Path | None = PrivateAttr(default=None)
    # Kept outside public spec dumps and analysis inputs. None denotes a legacy registration.
    _header_clock: dict[str, str | None] | None = PrivateAttr(default=None)

    id: str
    slug: str
    path: str
    fingerprint: str
    source_sha256: str | None = None
    source_file_size: int | None = None
    source_file_mtime_ns: int | None = None
    companion_files: dict[str, dict[str, Any]] = Field(default_factory=dict)
    modality: Literal["seeg", "ieeg", "scalp"]
    sfreq: float = Field(gt=0)
    n_samples: int = Field(gt=0)
    pieces: list[RecordingPiece] = Field(default_factory=list)
    source_first_sample: int = 0
    channels: list[ChannelSpec]
    reference: Literal["common", "bipolar", "average", "unknown"] = "unknown"
    reference_description: str | None = None
    highpass: float | None = None
    lowpass: float | None = None
    line_freq: float | None = None
    markers: list[MarkerSpec] = Field(default_factory=list)
    spatial: SpatialSpec | None = None
    provenance: ProvenanceSpec
    published_bipolar_pairs_path: str | None = None
    spatial_overlay_path: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        if self.pieces:
            return self.pieces[-1].time_at(self.n_samples, self.sfreq)
        return self.n_samples / self.sfreq

    def time_slices(self, start_seconds: float, end_seconds: float) -> list[tuple[RecordingPiece, int, int]]:
        """Intersect a true-time window with recorded data, rounding outward within each piece."""
        if end_seconds <= start_seconds:
            return []
        if not self.pieces:
            p = RecordingPiece(start_sample=0, stop_sample=self.n_samples, onset_seconds=0)
            a, b = seconds_to_sample_range(start_seconds, end_seconds, self.sfreq, self.n_samples)
            return [(p, a, b)] if a < b else []
        slices = []
        for p in self.pieces:
            if end_seconds <= p.onset_seconds:
                break
            if start_seconds >= p.time_at(p.stop_sample, self.sfreq):
                continue
            a = max(p.start_sample, p.start_sample + math.floor(p.sample_offset(start_seconds, self.sfreq)))
            b = min(p.stop_sample, p.start_sample + math.ceil(p.sample_offset(end_seconds, self.sfreq)))
            if a < b:
                slices.append((p, a, b))
        return slices

    def gaps(self) -> list[tuple[float, float]]:
        return [(p.time_at(p.stop_sample, self.sfreq), q.onset_seconds) for p, q in zip(self.pieces, self.pieces[1:])]


class BipolarPair(StrictModel):
    anode: str
    cathode: str
    name: str | None = None

    @model_validator(mode="after")
    def _distinct(self) -> BipolarPair:
        if self.anode == self.cathode:
            raise ValueError("bipolar anode and cathode must differ")
        return self


class ReferenceConfig(StrictModel):
    mode: Literal["original", "bipolar", "average"] = "original"
    pairs: list[BipolarPair] = Field(default_factory=list)
    average_channels: list[str] = Field(default_factory=list)


class QCConfig(StrictModel):
    window_seconds: float = Field(default=10.0, gt=0)
    flat_tolerance: float = Field(default=1e-12, ge=0)
    amplitude_warning_volts: float = Field(default=0.001, gt=0)
    exclude_flat: bool = True
    line_noise_enabled: bool = True
    line_noise_frequencies_hz: list[float] = Field(default_factory=lambda: [50.0, 60.0])


class ExclusionSpec(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    channels: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)
    scope: Literal["acquired", "derived"] = "acquired"

    @model_validator(mode="after")
    def _ordered(self) -> ExclusionSpec:
        if self.end <= self.start:
            raise ValueError("exclusion end must exceed start")
        return self


class FilterConfig(StrictModel):
    highpass: float | None = Field(default=None, gt=0)
    lowpass: float | None = Field(default=None, gt=0)
    notch: list[float] = Field(default_factory=list)


class DetectorConfig(StrictModel):
    enabled: bool = True
    name: Literal["STE", "MNI", "HIL"] = "STE"
    band: tuple[float, float] = (80.0, 450.0)
    parameters: dict[str, float | int | bool] = Field(default_factory=dict)
    edge_seconds: float = Field(default=0.5, ge=0)
    minimum_segment_seconds: float = Field(default=2.0, gt=0)

    @model_validator(mode="after")
    def _valid(self) -> DetectorConfig:
        if not (0 < self.band[0] < self.band[1]):
            raise ValueError("detector band requires 0 < low < high")
        if 2 * self.edge_seconds >= self.minimum_segment_seconds:
            raise ValueError("minimum_segment_seconds must exceed 2 * edge_seconds")
        return self


class ClassifierConfig(StrictModel):
    enabled: bool = False
    artifact: bool = True
    spike: bool = True
    ehfo: bool = True
    thresholds: dict[str, float] = Field(default_factory=dict)
    model_sources: dict[str, str] = Field(default_factory=dict)


class AnalysisSegment(StrictModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    label: str

    @model_validator(mode="after")
    def _ordered(self) -> AnalysisSegment:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Segment end must follow its start.")
        return self


class AnalysisConfig(StrictModel):
    protocol_name: str = "seeg-standard-v2"
    start_seconds: float = Field(default=0.0, ge=0)
    end_seconds: float | None = Field(default=None, gt=0)
    segments: list[AnalysisSegment] | None = Field(default=None, min_length=1)
    channels: list[str] = Field(default_factory=list)
    state: str | None = None
    exclusions: list[ExclusionSpec] = Field(default_factory=list)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    epoch_seconds: float = Field(default=10.0, gt=0)
    synchrony_window_ms: float = Field(default=30.0, gt=0)
    widespread_burst_min_shafts: int = Field(default=5, ge=2)

    @model_serializer(mode="wrap")
    def _serialize(self, serializer):
        config = serializer(self)
        if self.segments is None:
            # Keep legacy run signatures, protocol hashes and config files byte-compatible.
            config.pop("segments", None)
        return config

    @classmethod
    def for_modality(cls, modality: str) -> AnalysisConfig:
        if modality == "scalp":
            return cls(
                protocol_name="scalp-standard-v2",
                detector=DetectorConfig(enabled=False),
            )
        return cls()


class EngineSettings(StrictModel):
    source_root: str | None = None
    python_executable: str
    timeout_seconds: float = Field(default=1200.0, gt=0)
    model_cache: str | None = None
    max_workers: int = Field(default=2, ge=1, le=64)
    memory_budget_mb: int = Field(default=4096, ge=128)
    parallel_min_channel_samples: int = Field(default=24_000_000, ge=0)


class ProtocolSpec(StrictModel):
    """Reusable group protocol — intentionally strips case-specific channel/interval exclusions."""

    name: str
    slug: str
    description: str = ""
    modality: Literal["seeg", "ieeg", "scalp"] = "seeg"
    reference_mode: Literal["original", "bipolar", "average"] = "original"
    qc: QCConfig = Field(default_factory=QCConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    epoch_seconds: float = 10.0
    synchrony_window_ms: float = 30.0
    widespread_burst_min_shafts: int = 5
    created_at: str = Field(default_factory=utc_now)
