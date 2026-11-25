"""Native vendor EEG/iEEG recording readers, companion discovery, header & unit parsing, and identity extraction."""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import math
import re
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import mne
import numpy as np

from .models import ToolkitError

VENDOR_REC_SUFFIXES: tuple[str, ...] = (
    ".eeg",
    ".lay",
    ".set",
    ".cnt",
    ".cdt",
    ".dap",
    ".gdf",
    ".data",
    ".trc",
)

_VENDOR_FORMAT_NAMES: dict[str, str] = {
    ".eeg": "nihon_kohden",
    ".lay": "persyst",
    ".set": "eeglab",
    ".cnt": "neuroscan_cnt",
    ".cdt": "curry",
    ".dap": "curry",
    ".gdf": "gdf",
    ".data": "nicolet",
    ".trc": "micromed",
}


def _vendor_format_for_path(path: Path | str) -> str:
    return _VENDOR_FORMAT_NAMES.get(Path(path).suffix.lower(), "vendor")


def _raise_vendor_header_unreadable(fmt: str, exc: BaseException, *, phase: str = "header") -> NoReturn:
    """Raise a privacy-safe ToolkitError naming only the format and exception class (never raw exception/header text)."""
    exc_type = type(exc).__name__
    raise ToolkitError(
        "vendor_header_unreadable",
        f"Could not read {fmt} recording {phase} ({exc_type}); the file header may be malformed or unsupported.",
        format=fmt,
        exception_type=exc_type,
    ) from exc


def _import_micromed_rawio() -> Any | None:
    try:
        from neo.rawio.micromedrawio import MicromedRawIO

        return MicromedRawIO
    except ImportError:
        return None


def _has_micromed_reader() -> bool:
    return _import_micromed_rawio() is not None

_NIHON_VALID_HEADERS: frozenset[bytes] = frozenset({
    b"EEG-1100A V01.00",
    b"EEG-1100B V01.00",
    b"EEG-1100C V01.00",
    b"QI-403A   V01.00",
    b"QI-403A   V02.00",
    b"EEG-2100  V01.00",
    b"EEG-2100  V02.00",
    b"DAE-2100D V01.30",
    b"DAE-2100D V02.00",
    b"EEG-1200A V01.00",
})

_VENDOR_PLACEHOLDER_TOKENS: frozenset[str] = frozenset({
    "",
    "x",
    "unknown",
    "none",
    "n/a",
    "na",
    "anon",
    "anonymous",
    "subject",
    "patient",
    "0",
    "sub-0",
    "sub-x",
    "unspecified",
    "no_name",
    "no name",
    "noname",
    "not specified",
    "blank",
    "empty",
    "null",
    "default",
    "-",
    "--",
    "?",
})

_VOLT_SCALES: dict[str, float] = {
    "v": 1.0,
    "mv": 1e-3,
    "uv": 1e-6,
    "µv": 1e-6,
    "μv": 1e-6,
    "nv": 1e-9,
}

# GDF 2.x ISO/IEEE 11073-10101 / BioSig physical dimension codes handled by MNE.
_GDF2_UNIT_MAP: dict[int, str] = {
    4256: "V",
    4274: "mV",
    4275: "uV",
    512: "dimensionless",
    0: "",
}

_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%d-%b-%Y",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%m/%d/%y",
    "%d-%m-%y",
    "%d.%m.%Y",
    "%d.%m.%y",
    "%Y%m%d",
)


def is_vendor_recording_suffix(path: Path | str) -> bool:
    return Path(path).suffix.lower() in VENDOR_REC_SUFFIXES


def _read_text_fallback(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _clean_cstr(raw: bytes, encoding: str = "latin-1") -> str:
    return raw.split(b"\x00", 1)[0].decode(encoding, errors="replace").strip()


def _is_placeholder(val: Any) -> bool:
    s = str(val or "").strip().lower()
    if s in _VENDOR_PLACEHOLDER_TOKENS:
        return True
    if s.startswith("sub-") and s[4:] in _VENDOR_PLACEHOLDER_TOKENS:
        return True
    parts = [p for p in re.split(r"[\s_-]+", s) if p]
    if parts and all(p in _VENDOR_PLACEHOLDER_TOKENS for p in parts):
        return True
    return False


def _is_real_term(val: Any, *, min_digits_only_len: int = 2) -> bool:
    s = str(val or "").strip()
    if not s or _is_placeholder(s):
        return False
    if s.isdigit() and len(s) < min_digits_only_len:
        return False
    return True


_PART_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "not", "yes", "with", "from", "this", "that", "have", "has", "was", "are", "but",
    "test", "raw", "data", "file", "head", "clip", "run", "sub", "ses", "task", "acq", "eeg", "ieeg",
    "seeg", "ecog", "demo", "sample", "recording", "channel", "chan", "stim", "bdf", "edf", "trc", "cnt",
    "gdf", "curry", "nicolet", "persyst", "nihon", "eeglab", "micromed", "brain",
})


def _add_term_variants(terms: set[str], raw_val: Any, *, split_parts: bool = False, min_digits_only_len: int = 2) -> None:
    s = str(raw_val or "").strip()
    if not _is_real_term(s, min_digits_only_len=min_digits_only_len):
        return
    terms.add(s)
    if split_parts:
        min_part_digits = max(min_digits_only_len, 3)
        for word in re.split(r"[\s,;]+", s):
            word = word.strip()
            if (
                len(word) >= 3
                and word.lower() not in _PART_STOPWORDS
                and _is_real_term(word, min_digits_only_len=min_part_digits)
            ):
                terms.add(word)
        for part in re.split(r"[\s,;_/]+", s):
            part = part.strip()
            if (
                len(part) >= 3
                and part.lower() not in _PART_STOPWORDS
                and _is_real_term(part, min_digits_only_len=min_part_digits)
            ):
                terms.add(part)


def _add_date_variants(terms: set[str], raw_val: Any) -> None:
    s = str(raw_val or "").strip()
    if not s or _is_placeholder(s):
        return
    terms.add(s)
    for fmt in _DATE_FORMATS:
        try:
            parsed = dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
        if parsed.year > 1:
            terms.update((
                parsed.isoformat(),
                parsed.strftime("%Y/%m/%d"),
                parsed.strftime("%Y.%m.%d"),
                parsed.strftime("%d-%b-%Y").upper(),
                parsed.strftime("%d.%m.%y"),
            ))
        break


def _add_edf_style_patient_terms(terms: set[str], text: str) -> None:
    s = text.strip()
    if not s:
        return
    fields = s.split()
    if len(fields) >= 4 and fields[1].upper() in {"M", "F", "X"}:
        _add_term_variants(terms, fields[0], min_digits_only_len=2)
        _add_date_variants(terms, fields[2])
        for slot in fields[3:]:
            _add_term_variants(terms, slot, split_parts=True, min_digits_only_len=3)
    else:
        _add_term_variants(terms, s, split_parts=True, min_digits_only_len=3)


def _add_edf_style_recording_terms(terms: set[str], text: str) -> None:
    fields = text.strip().split()
    if len(fields) >= 3 and fields[0].lower() == "startdate":
        _add_date_variants(terms, fields[1])
        for slot in fields[2:4]:
            _add_term_variants(terms, slot, min_digits_only_len=3)


def _find_sibling_ci(folder: Path, names: tuple[str, ...] | list[str]) -> Path | None:
    """Locate a companion file strictly inside `folder` (exact match first, then case-insensitive)."""
    if not folder.is_dir():
        return None
    folder_res = folder.resolve()
    for name in names:
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            continue
        cand = folder / name
        try:
            if cand.is_file() and cand.resolve().is_relative_to(folder_res):
                return cand
        except OSError:
            continue
    lowered = {name.lower(): idx for idx, name in enumerate(names) if name and "/" not in name and "\\" not in name}
    if not lowered:
        return None
    hits: list[tuple[int, str, Path]] = []
    try:
        entries = sorted(folder.iterdir(), key=lambda p: p.name)
    except OSError:
        return None
    for entry in entries:
        rank = lowered.get(entry.name.lower())
        if rank is None:
            continue
        try:
            if entry.is_file() and entry.resolve().is_relative_to(folder_res):
                hits.append((rank, entry.name, entry))
        except OSError:
            continue
    if not hits:
        return None
    hits.sort(key=lambda item: (item[0], item[1]))
    return hits[0][2]


def is_nihon_kohden_eeg(path: Path) -> bool:
    """Return True only for genuine Nihon Kohden .EEG/.eeg files (excluding BrainVision .eeg data files)."""
    if not path.is_file() or path.suffix.lower() != ".eeg":
        return False
    try:
        with path.open("rb") as fh:
            head = fh.read(16)
            if head not in _NIHON_VALID_HEADERS:
                return False
            fh.seek(0x0081)
            ctl = fh.read(16)
            if len(ctl) == 16 and ctl not in _NIHON_VALID_HEADERS:
                return False
            return True
    except OSError:
        return False


def is_nicolet_data_file(path: Path) -> bool:
    """Return True only for Nicolet .data files paired with a valid Nicolet .head header."""
    if not path.is_file() or path.suffix.lower() != ".data":
        return False
    head = _find_sibling_ci(path.parent, [f"{path.stem}.head"])
    if head is None:
        return False
    try:
        snippet = head.read_bytes()[:4096].decode("latin-1", errors="ignore")
    except OSError:
        return False
    return "sample_freq=" in snippet and "elec_names=" in snippet


def is_vendor_recording_file(path: Path) -> bool:
    if not path.is_file():
        return False
    sfx = path.suffix.lower()
    if sfx not in VENDOR_REC_SUFFIXES:
        return False
    if sfx == ".eeg":
        return is_nihon_kohden_eeg(path)
    if sfx == ".data":
        return is_nicolet_data_file(path)
    if sfx == ".trc":
        return _has_micromed_reader()
    return True


def check_unsupported_vendor_target(path: Path) -> None:
    """Raise a clear ToolkitError('unsupported_format') with install/export guidance for unsupported targets."""
    low = path.name.lower()
    if low.endswith(".mff"):
        raise ToolkitError(
            "unsupported_format",
            f"EGI NetStation .mff directory bundle {path.name!r} is not supported natively "
            "(MFF is a directory bundle requiring the optional mffpy package). "
            "Export the recording as EDF+ from EGI NetStation.",
            format="egi_mff",
            suggested_export="EDF+",
        )
    if low.endswith(".erd"):
        raise ToolkitError(
            "unsupported_format",
            f"Natus/XLTEK .erd recording {path.name!r} uses a proprietary compressed format and is not supported natively. "
            "Export the recording as EDF+ from Natus NeuroWorks.",
            format="natus_erd",
            suggested_export="EDF+",
        )
    if low.endswith(".trc") and not _has_micromed_reader():
        raise ToolkitError(
            "unsupported_format",
            f"Micromed .trc recording {path.name!r} requires the optional `neo` package. "
            'Install it with `pip install "brainlens[micromed]"` (or export EDF+ from Micromed SystemPlus).',
            format="micromed_trc",
            install_hint='pip install "brainlens[micromed]"',
            suggested_export="EDF+",
        )
    if low.endswith(".e") and path.is_file():
        raise ToolkitError(
            "unsupported_format",
            f"Native NicoletOne .e recording {path.name!r} is not supported by MNE "
            "(only Nicolet .data/.head exports are supported natively). "
            "Export the recording as EDF+ or Nicolet .data/.head.",
            format="nicolet_e",
            suggested_export="EDF+",
        )
    if low.endswith(".eeg") and path.is_file() and not is_nihon_kohden_eeg(path):
        raise ToolkitError(
            "unsupported_format",
            f"{path.name!r} is not a valid Nihon Kohden .EEG file "
            "(if this is a BrainVision data file, open its paired .vhdr header file instead).",
            format="non_nihon_eeg",
        )


def check_unsupported_vendor_folder(folder: Path) -> None:
    """When folder discovery finds no supported recordings, surface specific guidance if known unsupported formats exist."""
    if not folder.is_dir():
        return
    try:
        entries = sorted(folder.rglob("*"))
    except OSError:
        return
    for entry in entries:
        if any(part.startswith(".") for part in entry.relative_to(folder).parts):
            continue
        low = entry.name.lower()
        if low.endswith((".mff", ".erd", ".trc")) or (low.endswith(".e") and entry.is_file()):
            check_unsupported_vendor_target(entry)


# ---------------------------------------------------------------------------
# Format-specific header & companion parsers
# ---------------------------------------------------------------------------

def _parse_nihon_pnt_fields(pnt_path: Path) -> list[tuple[int, str, str]]:
    """Parse Nihon Kohden .PNT self-describing field table at offset 0x0400."""
    try:
        blob = pnt_path.read_bytes()
    except OSError:
        return []
    if len(blob) < 0x0412 or blob[:16] not in _NIHON_VALID_HEADERS:
        return []
    n_fields = int(blob[0x0411])
    pos = 0x0412 + n_fields * 4
    if pos > len(blob):
        return []
    out: list[tuple[int, str, str]] = []
    for i in range(n_fields):
        l_len, v_len, _flags = struct.unpack_from("<BBH", blob, 0x0412 + i * 4)
        if pos + l_len + v_len > len(blob):
            break
        lbl = _clean_cstr(blob[pos : pos + l_len])
        val = _clean_cstr(blob[pos + l_len : pos + l_len + v_len])
        pos += l_len + v_len
        out.append((i, lbl, val))
    return out


def _parse_nihon_header(eeg_path: Path) -> SimpleNamespace:
    terms: set[str] = set()
    patient_code: str | None = None
    patient_name: str | None = None

    pnt = _find_sibling_ci(eeg_path.parent, [f"{eeg_path.stem}.PNT", f"{eeg_path.stem}.pnt"])
    if pnt is not None:
        fields = _parse_nihon_pnt_fields(pnt)
        for idx, lbl, val in fields:
            if not val:
                continue
            low_lbl = lbl.lower()
            if idx == 0 or low_lbl in {"id", "patient id", "pat. id"}:
                if _is_real_term(val, min_digits_only_len=2):
                    patient_code = val
                    _add_term_variants(terms, val, min_digits_only_len=2)
            elif idx == 1 or low_lbl in {"eeg-nr.", "eeg no.", "eeg nr", "eeg number"}:
                _add_term_variants(terms, val, min_digits_only_len=3)
            elif idx == 2 or low_lbl in {"name", "patient name", "full name"}:
                if _is_real_term(val, min_digits_only_len=2):
                    patient_name = val
                    _add_term_variants(terms, val, split_parts=True, min_digits_only_len=2)
            elif idx == 4 or any(k in low_lbl for k in ("birth", "geburt", "dob")):
                _add_date_variants(terms, val)
            elif idx in {6, 7, 16, 18} or any(
                k in low_lbl for k in ("arzt", "doctor", "physician", "assistent", "operator", "tech", "dept", "station", "abtl", "comment", "kommentar")
            ):
                _add_term_variants(terms, val, split_parts=True, min_digits_only_len=3)
            elif idx == 17 or low_lbl in {"datum", "date"}:
                _add_date_variants(terms, val)

    return SimpleNamespace(
        signals=None,
        patient=SimpleNamespace(code=patient_code, name=patient_name),
        vendor_identity_terms=sorted(terms),
        vendor_format="nihon_kohden",
    )


def _parse_persyst_lay_sections(lay_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    fileinfo: dict[str, str] = {}
    patient: dict[str, str] = {}
    section = ""
    for raw_line in _read_text_fallback(lay_path).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]") and len(line) >= 3:
            section = line[1:-1].strip().lower()
            continue
        if section in {"fileinfo", "patient"} and "=" in line:
            k, v = line.split("=", 1)
            key = k.strip().lower()
            val = v.strip()
            if section == "fileinfo":
                fileinfo[key] = val
            else:
                patient[key] = val
    return fileinfo, patient


def _parse_persyst_header(lay_path: Path) -> SimpleNamespace:
    fileinfo, patient = _parse_persyst_lay_sections(lay_path)
    cal_raw = fileinfo.get("calibration")
    if cal_raw is not None:
        try:
            if float(cal_raw) <= 0:
                raise ValueError("non-positive calibration")
        except ValueError as exc:
            _raise_vendor_header_unreadable("persyst", exc)

    terms: set[str] = set()
    first = patient.get("first", "").strip()
    middle = patient.get("middle", "").strip()
    last = patient.get("last", "").strip()
    full = patient.get("fullname", "").strip()
    joined_name = " ".join(p for p in (first, middle, last) if p).strip()

    patient_code: str | None = None
    for k in ("id", "medicalrecordn", "mrn", "patientid"):
        val = patient.get(k, "").strip()
        if _is_real_term(val, min_digits_only_len=2):
            if patient_code is None:
                patient_code = val
            _add_term_variants(terms, val, min_digits_only_len=2)

    patient_name: str | None = None
    for cand in (full, joined_name, first, middle, last):
        if _is_real_term(cand, min_digits_only_len=2):
            if patient_name is None:
                patient_name = cand
            _add_term_variants(terms, cand, split_parts=True, min_digits_only_len=2)

    for k in ("birthdate", "dob", "testdate"):
        if patient.get(k):
            _add_date_variants(terms, patient[k])

    for k in ("physician", "technician", "medications", "history"):
        if patient.get(k):
            _add_term_variants(terms, patient[k], split_parts=True, min_digits_only_len=3)

    if patient.get("comments1"):
        _add_edf_style_patient_terms(terms, patient["comments1"])
    if patient.get("comments2"):
        _add_edf_style_recording_terms(terms, patient["comments2"])

    return SimpleNamespace(
        signals=None,
        patient=SimpleNamespace(code=patient_code, name=patient_name),
        vendor_identity_terms=sorted(terms),
        vendor_format="persyst",
    )


def _read_eeglab_meta(set_path: Path) -> dict[str, Any]:
    from mne.io.eeglab._eeglab import _readmat

    try:
        mat = _readmat(str(set_path), uint16_codec=None, preload=False)
    except NotImplementedError as exc:
        msg = str(exc)
        if "v7.3" in msg.lower() or "hdf" in msg.lower():
            raise ToolkitError(
                "unsupported_format",
                f"EEGLAB MATLAB v7.3 (HDF5) file {set_path.name!r} requires pymatreader/h5py (not installed). "
                "Re-save as MATLAB v5 .set or export EDF+.",
                format="eeglab_v73",
                suggested_export="EDF+",
            ) from exc
        _raise_vendor_header_unreadable("eeglab", exc)
    except Exception as exc:
        _raise_vendor_header_unreadable("eeglab", exc)

    if isinstance(mat, dict) and "ALLEEG" in mat:
        raise ToolkitError("unsupported_format", f"EEGLAB ALLEEG container {set_path.name!r} is not supported; save a single continuous EEG dataset.")
    eeg = mat.get("EEG", mat) if isinstance(mat, dict) else {}
    if isinstance(eeg, dict) and "EEG" in eeg and isinstance(eeg["EEG"], dict):
        eeg = eeg["EEG"]
    if not isinstance(eeg, dict):
        _raise_vendor_header_unreadable("eeglab", ValueError("missing EEG struct"))
    trials = int(eeg.get("trials", 1) or 1)
    if trials != 1:
        raise ToolkitError(
            "unsupported_format",
            f"Epoched EEGLAB .set file {set_path.name!r} (trials={trials}) is not supported for continuous HFO review; export continuous raw data.",
            format="eeglab_epoched",
            trials=trials,
        )
    return eeg


def _as_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("latin-1", errors="replace").strip()
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return ""
        if val.ndim == 0:
            return _as_str(val.item())
    return ""


def _parse_eeglab_header(set_path: Path) -> SimpleNamespace:
    eeg = _read_eeglab_meta(set_path)
    terms: set[str] = set()
    subject = _as_str(eeg.get("subject"))
    patient_code = subject if _is_real_term(subject, min_digits_only_len=2) else None
    if patient_code:
        _add_term_variants(terms, patient_code, split_parts=True, min_digits_only_len=2)

    filepath = _as_str(eeg.get("filepath"))
    if filepath and len(filepath) >= 3:
        terms.add(filepath)

    etc = eeg.get("etc")
    if isinstance(etc, dict):
        for k, v in etc.items():
            kl = str(k).lower()
            sv = _as_str(v)
            if not sv:
                continue
            if any(tok in kl for tok in ("birth", "dob")):
                _add_date_variants(terms, sv)
            elif any(tok in kl for tok in ("patient", "subject", "name", "mrn", "his_id", "physician", "doctor", "hospital")):
                _add_term_variants(terms, sv, split_parts=True, min_digits_only_len=3)

    return SimpleNamespace(
        signals=None,
        patient=SimpleNamespace(code=patient_code, name=None),
        vendor_identity_terms=sorted(terms),
        vendor_format="eeglab",
    )


def _parse_cnt_header(cnt_path: Path) -> SimpleNamespace:
    try:
        with cnt_path.open("rb") as fh:
            hdr = fh.read(900)
    except OSError as exc:
        _raise_vendor_header_unreadable("neuroscan_cnt", exc)
    if len(hdr) < 900:
        _raise_vendor_header_unreadable("neuroscan_cnt", EOFError("truncated header"))

    terms: set[str] = set()
    raw_pid = _clean_cstr(hdr[21:41])
    raw_doc = _clean_cstr(hdr[41:61])
    raw_ref = _clean_cstr(hdr[61:81])
    raw_hosp = _clean_cstr(hdr[81:101])
    raw_pat_slot = _clean_cstr(hdr[101:121])
    raw_name = _clean_cstr(hdr[121:141])
    raw_session = _clean_cstr(hdr[205:225])
    raw_date = _clean_cstr(hdr[225:235])

    patient_code = raw_pid if _is_real_term(raw_pid, min_digits_only_len=2) else None
    patient_name = raw_name if _is_real_term(raw_name, min_digits_only_len=2) else None
    if patient_code:
        _add_term_variants(terms, patient_code, min_digits_only_len=2)
    if patient_name:
        _add_term_variants(terms, patient_name, split_parts=True, min_digits_only_len=2)
    for slot in (raw_doc, raw_ref, raw_hosp, raw_pat_slot, raw_session):
        _add_term_variants(terms, slot, split_parts=True, min_digits_only_len=3)
    if raw_date:
        _add_date_variants(terms, raw_date)

    return SimpleNamespace(
        signals=None,
        patient=SimpleNamespace(code=patient_code, name=patient_name),
        vendor_identity_terms=sorted(terms),
        vendor_format="neuroscan_cnt",
    )


def _parse_gdf_header(gdf_path: Path) -> SimpleNamespace:
    """Parse GDF 1.x or 2.x binary header for verified physical units, per-channel rates/filters, and patient identity."""
    with gdf_path.open("rb") as fh:
        fixed = fh.read(256)
        if len(fixed) < 256:
            _raise_vendor_header_unreadable("gdf", EOFError("truncated header"))
        ver_str = fixed[:8].decode("latin-1", errors="replace")
        if not ver_str.startswith("GDF"):
            _raise_vendor_header_unreadable("gdf", ValueError("invalid GDF magic"))
        try:
            vnum = float(ver_str[4:].strip())
        except ValueError as exc:
            _raise_vendor_header_unreadable("gdf", exc)

        terms: set[str] = set()
        patient_code: str | None = None
        patient_name: str | None = None
        signals: list[SimpleNamespace] = []

        if vnum < 1.9:
            pid_raw = _clean_cstr(fixed[8:88])
            rid_raw = _clean_cstr(fixed[88:168])
            _add_edf_style_patient_terms(terms, pid_raw)
            _add_edf_style_recording_terms(terms, rid_raw)
            pid_parts = pid_raw.split(" ", 2)
            if pid_parts and _is_real_term(pid_parts[0], min_digits_only_len=2):
                patient_code = pid_parts[0]
            if len(pid_parts) >= 2 and _is_real_term(pid_parts[1], min_digits_only_len=2):
                patient_name = pid_parts[1]
            # Fixed header offsets in GDF 1.x:
            # 168: date(16), 184: hdr_nbytes(i64), 192: equip(8), 200: hosp(8), 208: tech(8), 216: res(20),
            # 236: n_records(i64), 244: rec_dur(2*u32), 252: nchan(u32)
            rec_num, rec_den = struct.unpack_from("<II", fixed, 244)
            rec_sec = (float(rec_num) / float(rec_den)) if (rec_num > 0 and rec_den > 0) else 1.0
            nchan = int(struct.unpack_from("<I", fixed, 252)[0])
            labels = [_clean_cstr(fh.read(16)) for _ in range(nchan)]
            fh.seek(80 * nchan, 1)  # transducer
            raw_units = [_clean_cstr(fh.read(8)) for _ in range(nchan)]
            fh.seek((8 + 8 + 8 + 8) * nchan, 1)  # phys min/max (f64), dig min/max (i64)
            prefilts = [_clean_cstr(fh.read(80)) for _ in range(nchan)]
            n_samps = struct.unpack(f"<{nchan}i", fh.read(4 * nchan))
            for i in range(nchan):
                u = raw_units[i].strip()
                # MNE GDF 1.x only scales uV (1e-6) and V (1.0); other units like mV are NOT scaled by MNE.
                if u[:2] == "uV":
                    eff_unit = "uV"
                elif u == "V":
                    eff_unit = "V"
                elif u in {"1", "dimensionless"}:
                    eff_unit = "dimensionless"
                elif u:
                    eff_unit = f"{u} (unverified in GDF 1.x)"
                else:
                    eff_unit = ""
                signals.append(
                    SimpleNamespace(
                        label=labels[i],
                        physical_dimension=eff_unit,
                        sampling_frequency=(float(n_samps[i]) / rec_sec) if (rec_sec > 0 and n_samps[i] > 0) else 0.0,
                        prefiltering=prefilts[i],
                    )
                )
        else:
            pid_raw = _clean_cstr(fixed[8:74])
            rid_raw = _clean_cstr(fixed[88:152])
            _add_edf_style_patient_terms(terms, pid_raw)
            _add_edf_style_recording_terms(terms, rid_raw)
            if rid_raw and not rid_raw.lower().startswith("startdate"):
                _add_term_variants(terms, rid_raw, split_parts=True, min_digits_only_len=3)
            pid_parts = pid_raw.split(" ", 2)
            if pid_parts and _is_real_term(pid_parts[0], min_digits_only_len=2):
                patient_code = pid_parts[0]
            if len(pid_parts) >= 2 and _is_real_term(pid_parts[1], min_digits_only_len=2):
                patient_name = pid_parts[1]
            # GDF 2.x birthday at offset 176 (u64)
            bday_raw = struct.unpack_from("<Q", fixed, 176)[0]
            if bday_raw != 0:
                try:
                    bdt = dt.datetime(1, 1, 1, tzinfo=dt.UTC) + dt.timedelta(bday_raw * (2.0**-32) - 367.0)
                    if bdt.year > 1:
                        _add_date_variants(terms, bdt.date().isoformat())
                except Exception:
                    pass
            rec_num, rec_den = struct.unpack_from("<II", fixed, 244)
            rec_sec = (float(rec_num) / float(rec_den)) if (rec_num > 0 and rec_den > 0) else 1.0
            nchan = int(struct.unpack_from("<H", fixed, 252)[0])
            labels = [_clean_cstr(fh.read(16)) for _ in range(nchan)]
            fh.seek((80 + 6) * nchan, 1)  # transducer + obsolete phys_dim string
            ucodes = struct.unpack(f"<{nchan}H", fh.read(2 * nchan))
            fh.seek((8 + 8 + 8 + 8 + 68) * nchan, 1)  # phys/dig min/max + obsolete
            lowpasses = struct.unpack(f"<{nchan}f", fh.read(4 * nchan))
            highpasses = struct.unpack(f"<{nchan}f", fh.read(4 * nchan))
            _notches = struct.unpack(f"<{nchan}f", fh.read(4 * nchan))
            n_samps = struct.unpack(f"<{nchan}i", fh.read(4 * nchan))
            for i in range(nchan):
                code = int(ucodes[i])
                eff_unit = _GDF2_UNIT_MAP.get(code, f"gdf_unit_{code}")
                lp = float(lowpasses[i])
                hp = float(highpasses[i])
                pf_parts: list[str] = []
                if math.isfinite(hp) and hp > 0:
                    pf_parts.append(f"HP:{hp:g}Hz")
                if math.isfinite(lp) and lp > 0:
                    pf_parts.append(f"LP:{lp:g}Hz")
                signals.append(
                    SimpleNamespace(
                        label=labels[i],
                        physical_dimension=eff_unit,
                        sampling_frequency=(float(n_samps[i]) / rec_sec) if (rec_sec > 0 and n_samps[i] > 0) else 0.0,
                        prefiltering=" ".join(pf_parts),
                    )
                )

    return SimpleNamespace(
        signals=signals,
        patient=SimpleNamespace(code=patient_code, name=patient_name),
        vendor_identity_terms=sorted(terms),
        vendor_format="gdf",
    )


def _parse_nicolet_head_kv(head_path: Path) -> dict[str, str]:
    kv: dict[str, str] = {}
    for raw_line in _read_text_fallback(head_path).splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        kv[k.strip().lower()] = v.strip()
    return kv


def _parse_nicolet_header(data_path: Path) -> SimpleNamespace:
    head = _find_sibling_ci(data_path.parent, [f"{data_path.stem}.head"])
    if head is None:
        raise ToolkitError("missing_companion", f"Nicolet header file (.head) missing for {data_path.name}")
    kv = _parse_nicolet_head_kv(head)
    conv_raw = kv.get("conversion_factor")
    if conv_raw is not None:
        try:
            if float(conv_raw) <= 0:
                raise ValueError("non-positive conversion_factor")
        except ValueError as exc:
            _raise_vendor_header_unreadable("nicolet", exc)

    terms: set[str] = set()
    patient_code: str | None = None
    patient_name: str | None = None

    pat_id = kv.get("pat_id", "")
    if _is_real_term(pat_id, min_digits_only_len=2):
        patient_code = pat_id
        _add_term_variants(terms, pat_id, min_digits_only_len=2)
    for k in ("adm_id", "rec_id"):
        if kv.get(k):
            _add_term_variants(terms, kv[k], min_digits_only_len=3)
    for k in ("pat_name", "patient_name", "name", "fullname"):
        if kv.get(k) and _is_real_term(kv[k], min_digits_only_len=2):
            if patient_name is None:
                patient_name = kv[k]
            _add_term_variants(terms, kv[k], split_parts=True, min_digits_only_len=2)
    for k in ("dob", "birthdate", "birthday"):
        if kv.get(k):
            _add_date_variants(terms, kv[k])
    if kv.get("start_ts"):
        date_part = kv["start_ts"].split()[0]
        _add_date_variants(terms, date_part)

    return SimpleNamespace(
        signals=None,
        patient=SimpleNamespace(code=patient_code, name=patient_name),
        vendor_identity_terms=sorted(terms),
        vendor_format="nicolet",
    )


# ---------------------------------------------------------------------------
# Compumedics Curry parser & pure-Python BaseRaw fallback (when curryreader is absent)
# ---------------------------------------------------------------------------

def _curry_extract_block_kv(text: str, section_name: str) -> dict[str, str]:
    pattern = re.compile(
        rf"^\s*{re.escape(section_name)}\s+START\b(.*?)^\s*{re.escape(section_name)}\s+END\b",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        return {}
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _curry_extract_all_lists(text: str, list_name: str) -> list[list[str]]:
    pattern = re.compile(
        rf"^\s*{re.escape(list_name)}\s+START_LIST[^\n]*\n(.*?)^\s*{re.escape(list_name)}\s+END_LIST\b",
        re.MULTILINE | re.DOTALL,
    )
    return [m.group(1).splitlines() for m in pattern.finditer(text)]


def _curry_resolve_files(main_path: Path) -> dict[str, Path | None]:
    folder = main_path.parent
    sfx = main_path.suffix.lower()
    if sfx == ".cdt":
        hdr = _find_sibling_ci(folder, [f"{main_path.name}.dpa", f"{main_path.name}.dpo"])
        if hdr is None:
            raise ToolkitError(
                "missing_companion",
                f"Compumedics Curry header file (.cdt.dpa/.cdt.dpo) missing for {main_path.name}",
            )
        events = _find_sibling_ci(folder, [f"{main_path.name}.cef", f"{main_path.name}.ceo"])
        hpi = _find_sibling_ci(folder, [f"{main_path.name}.hpi"])
        return {"data": main_path, "header": hdr, "labels": hdr, "events": events, "hpi": hpi}
    if sfx == ".dap":
        kv = _curry_extract_block_kv(_read_text_fallback(main_path), "DATA_PARAMETERS")
        data_ext = (kv.get("DataExtension") or "dat").lstrip(".").strip() or "dat"
        data_p = _find_sibling_ci(folder, [f"{main_path.stem}.{data_ext}", f"{main_path.stem}.dat"])
        if data_p is None:
            raise ToolkitError("missing_companion", f"Compumedics Curry 7 data file (.{data_ext}) missing for {main_path.name}")
        rs3 = _find_sibling_ci(folder, [f"{main_path.stem}.rs3"])
        if rs3 is None:
            raise ToolkitError("missing_companion", f"Compumedics Curry 7 labels file (.rs3) missing for {main_path.name}")
        events = _find_sibling_ci(folder, [f"{main_path.stem}.cef", f"{main_path.stem}.ceo"])
        hpi = _find_sibling_ci(folder, [f"{main_path.stem}.hpi"])
        return {"data": data_p, "header": main_path, "labels": rs3, "events": events, "hpi": hpi}
    raise ToolkitError("unsupported_format", f"Unsupported Compumedics Curry file: {main_path.name}")


def _parse_curry_metadata(main_path: Path) -> dict[str, Any]:
    files = _curry_resolve_files(main_path)
    hdr_path = files["header"]
    lbl_path = files["labels"]
    data_path = files["data"]
    assert hdr_path is not None and lbl_path is not None and data_path is not None

    hdr_text = _read_text_fallback(hdr_path)
    lbl_text = hdr_text if lbl_path == hdr_path else _read_text_fallback(lbl_path)
    dp = _curry_extract_block_kv(hdr_text, "DATA_PARAMETERS")

    n_trials = int(float(dp.get("NumTrials", "1") or "1"))
    if n_trials > 1:
        raise ToolkitError(
            "unsupported_format",
            f"Epoched Compumedics Curry file {main_path.name!r} (NumTrials={n_trials}) is not supported; export continuous raw data.",
            format="curry_epoched",
            trials=n_trials,
        )

    stime_us = float(dp.get("SampleTimeUsec", "0") or "0")
    sfreq_hdr = float(dp.get("SampleFreqHz", "0") or "0")
    if stime_us <= 0 and sfreq_hdr <= 0:
        _raise_vendor_header_unreadable("curry", ValueError("non-positive sampling frequency"))
    sfreq = (1e6 / stime_us) if stime_us > 0 else sfreq_hdr

    n_channels = int(float(dp.get("NumChannels", "0") or "0"))
    n_samples_hdr = int(float(dp.get("NumSamples", "0") or "0"))
    byteorder = (dp.get("DataByteOrder") or "INTEL").strip().upper()
    samp_order = (dp.get("DataSampOrder") or "SAMP").strip().upper()
    data_format = (dp.get("DataFormat") or "FLOAT").strip().upper()

    is_ascii = byteorder == "ASCII"
    endian = ">" if byteorder == "MOTOROLA" else "<"
    if data_format in {"INT16", "0", "2"}:
        np_dtype = np.dtype(f"{endian}i2")
    elif data_format in {"INT32", "3", "4"}:
        np_dtype = np.dtype(f"{endian}i4")
    else:
        np_dtype = np.dtype(f"{endian}f4")

    # Parse DEVICE_PARAMETERS blocks (one per channel group).
    ch_types: list[str] = []
    units: list[str] = []
    group_scales: list[float] = []
    group_offsets: list[float] = []
    dev_pat = re.compile(
        r"^\s*DEVICE_PARAMETERS([^\n]*)\bSTART\b(.*?)^\s*DEVICE_PARAMETERS[^\n]*\bEND\b",
        re.MULTILINE | re.DOTALL,
    )
    for m in dev_pat.finditer(hdr_text):
        group_tag = m.group(1).strip().upper()
        kv: dict[str, str] = {}
        for line in m.group(2).splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                kv[k.strip()] = v.strip()
        n_grp = int(float(kv.get("NumChanThisGroup", "0") or "0"))
        if n_grp <= 0:
            continue
        unit = (kv.get("DataUnit") or "").strip()
        scale = float(kv.get("CommonScale", "1") or "1")
        offset = float(kv.get("CommonOffset", "0") or "0")
        ctype = "mag" if "MAG" in group_tag else ("misc" if "OTHER" in group_tag else "eeg")
        ch_types.extend([ctype] * n_grp)
        units.extend([unit] * n_grp)
        group_scales.extend([scale] * n_grp)
        group_offsets.extend([offset] * n_grp)

    ch_names: list[str] = []
    for block in _curry_extract_all_lists(lbl_text, "LABELS"):
        for line in block:
            name = line.strip()
            if name:
                ch_names.append(name)

    if not (len(ch_names) == len(ch_types) == len(units) == n_channels):
        chan_in_file: list[int] = []
        for block in _curry_extract_all_lists(hdr_text, "CHAN_IN_FILE"):
            for line in block:
                s = line.strip()
                if s:
                    chan_in_file.append(int(s))
        if chan_in_file and len(chan_in_file) == len(ch_names):
            order = [int(i) for i in np.argsort(chan_in_file) if chan_in_file[int(i)] != 0]
            ch_names = [ch_names[i] for i in order]
            if len(ch_types) == len(chan_in_file):
                ch_types = [ch_types[i] for i in order]
                units = [units[i] for i in order]
                group_scales = [group_scales[i] for i in order]
                group_offsets = [group_offsets[i] for i in order]

    if n_channels <= 0:
        n_channels = len(ch_names)
    if len(ch_names) != n_channels:
        _raise_vendor_header_unreadable("curry", ValueError("channel count mismatch"))
    if len(units) != n_channels:
        units = ["uV"] * n_channels
        ch_types = ["eeg"] * n_channels
        group_scales = [1.0] * n_channels
        group_offsets = [0.0] * n_channels

    if is_ascii:
        n_samples = n_samples_hdr
    else:
        file_samps = data_path.stat().st_size // (np_dtype.itemsize * n_channels)
        n_samples = file_samps if (n_samples_hdr <= 0 or file_samps != n_samples_hdr) else n_samples_hdr

    meas_date: dt.datetime | None = None
    try:
        meas_date = dt.datetime(
            int(dp["StartYear"]),
            int(dp["StartMonth"]),
            int(dp["StartDay"]),
            int(dp["StartHour"]),
            int(dp["StartMin"]),
            int(dp["StartSec"]),
            int(dp.get("StartMillisec", "0")) * 1000,
            tzinfo=dt.UTC,
        )
    except Exception:
        meas_date = None

    # Parse optional event annotations (.cef / .ceo)
    ev_onsets: list[float] = []
    ev_durations: list[float] = []
    ev_descriptions: list[str] = []
    ev_path = files.get("events")
    if ev_path is not None and ev_path.is_file():
        ev_text = _read_text_fallback(ev_path)
        num_blocks = _curry_extract_all_lists(ev_text, "NUMBER_LIST")
        rem_blocks = _curry_extract_all_lists(ev_text, "REMARK_LIST")
        num_rows = [ln.strip() for blk in num_blocks for ln in blk if ln.strip()]
        rem_rows = [ln for blk in rem_blocks for ln in blk]
        for idx, row in enumerate(num_rows):
            parts = row.split()
            if len(parts) < 3:
                continue
            try:
                sample = int(parts[0])
                etype = int(parts[2])
                s0 = int(parts[4]) if len(parts) >= 6 else sample
                s1 = int(parts[5]) if len(parts) >= 6 else sample
            except ValueError:
                continue
            remark = rem_rows[idx].strip() if idx < len(rem_rows) else ""
            ev_onsets.append(float(s0) / sfreq)
            ev_durations.append(max(0.0, float(s1 - s0) / sfreq))
            ev_descriptions.append(remark if remark else str(etype))

    return {
        "files": files,
        "dp": dp,
        "sfreq": float(sfreq),
        "n_channels": n_channels,
        "n_samples": int(n_samples),
        "ch_names": ch_names,
        "ch_types": ch_types,
        "units": units,
        "scales": group_scales,
        "offsets": group_offsets,
        "is_ascii": is_ascii,
        "samp_order": samp_order,
        "dtype": np_dtype,
        "meas_date": meas_date,
        "events": (ev_onsets, ev_durations, ev_descriptions),
    }


def _parse_curry_header(main_path: Path) -> SimpleNamespace:
    meta = _parse_curry_metadata(main_path)
    dp = meta["dp"]
    terms: set[str] = set()
    patient_code: str | None = None
    patient_name: str | None = None

    for k, v in dp.items():
        kl = k.lower()
        sv = v.strip()
        if not sv:
            continue
        if any(tok in kl for tok in ("birth", "dob")):
            _add_date_variants(terms, sv)
        elif kl in {"patientid", "patient_id", "subjectid", "subject_id", "mrn"}:
            if _is_real_term(sv, min_digits_only_len=2):
                patient_code = sv
                _add_term_variants(terms, sv, min_digits_only_len=2)
        elif kl in {"patientname", "patient_name", "subjectname", "subject_name"}:
            if _is_real_term(sv, min_digits_only_len=2):
                patient_name = sv
                _add_term_variants(terms, sv, split_parts=True, min_digits_only_len=2)
        elif kl in {"currentsessionlabel", "comments", "amplifierinfo"}:
            _add_term_variants(terms, sv, min_digits_only_len=3)
        elif kl == "filehistory":
            # FileHistory often embeds workstation file paths (e.g. C:\Users\<user>\...)
            for m in re.finditer(r"(?:[A-Za-z]:\\|/)[^\s¶\t]+", sv):
                path_tok = m.group(0).strip()
                if len(path_tok) >= 4:
                    terms.add(path_tok)
                for um in re.finditer(r"(?:Users|home)[\\/]([^\\/\s¶]+)", path_tok, re.IGNORECASE):
                    _add_term_variants(terms, um.group(1), min_digits_only_len=3)

    signals = [
        SimpleNamespace(
            label=meta["ch_names"][i],
            physical_dimension=meta["units"][i],
            sampling_frequency=meta["sfreq"],
            prefiltering="",
        )
        for i in range(meta["n_channels"])
    ]
    return SimpleNamespace(
        signals=signals,
        patient=SimpleNamespace(code=patient_code, name=patient_name),
        vendor_identity_terms=sorted(terms),
        vendor_format="curry",
    )


class _RawCurryFallback(mne.io.BaseRaw):
    """Pure-Python lazy BaseRaw reader for continuous Compumedics Curry 7/8/9 when `curryreader` is unavailable."""

    def __init__(self, main_path: Path, preload: bool | str = False) -> None:
        from mne._fiff.constants import FIFF
        from mne._fiff.meas_info import create_info

        try:
            meta = _parse_curry_metadata(main_path)
            data_path: Path = meta["files"]["data"]
            sfreq: float = meta["sfreq"]
            ch_names: list[str] = meta["ch_names"]
            ch_types: list[str] = meta["ch_types"]
            units: list[str] = meta["units"]
            scales: list[float] = meta["scales"]
            offsets: list[float] = meta["offsets"]
            n_samples: int = meta["n_samples"]

            info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
            cals: list[float] = []
            with info._unlock():
                for i, u in enumerate(units):
                    ul = u.strip().lower()
                    if ul in _VOLT_SCALES:
                        info["chs"][i]["unit"] = FIFF.FIFF_UNIT_V
                        cals.append(float(scales[i]) * _VOLT_SCALES[ul])
                    elif ul == "ft":
                        info["chs"][i]["unit"] = FIFF.FIFF_UNIT_T
                        cals.append(float(scales[i]) * 1e-15)
                    else:
                        info["chs"][i]["unit"] = FIFF.FIFF_UNIT_NONE
                        cals.append(float(scales[i]))
                if meta["meas_date"] is not None:
                    info["meas_date"] = meta["meas_date"]

            raw_extra = {
                "is_ascii": meta["is_ascii"],
                "samp_order": meta["samp_order"],
                "dtype": meta["dtype"],
                "n_channels": meta["n_channels"],
                "n_samples": n_samples,
                "offsets": np.asarray(offsets, dtype=np.float64),
                "max_block_samples": max(1, 16 * 1024**2 // max(1, meta["dtype"].itemsize * meta["n_channels"])),
            }
            super().__init__(
                info,
                preload=False,
                filenames=[data_path],
                last_samps=[n_samples - 1],
                orig_format="single" if meta["dtype"].kind == "f" else "int",
                raw_extras=[raw_extra],
                verbose="ERROR",
            )
            self._cals = np.asarray(cals, dtype=np.float64)
            if preload:
                self._preload_data(preload)
            onsets, durations, descriptions = meta["events"]
            if onsets:
                self.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=descriptions, orig_time=None))
        except ToolkitError:
            raise
        except Exception as exc:
            _raise_vendor_header_unreadable("curry", exc)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        from mne._fiff.utils import _mult_cal_one, _read_segments_file

        ex = self._raw_extras[fi]
        fname = self.filenames[fi]
        assert fname is not None
        offsets = ex["offsets"]
        has_offset = bool(np.any(offsets != 0.0))

        if ex["is_ascii"]:
            if isinstance(idx, slice):
                idx_arr = np.arange(idx.start or 0, idx.stop or ex["n_channels"], idx.step or 1)
            else:
                idx_arr = np.asarray(idx, dtype=int)
            block = np.loadtxt(fname, skiprows=start, max_rows=stop - start, ndmin=2).T.astype(np.float64)
            if has_offset:
                block -= offsets[:, None]
            _mult_cal_one(data, block, idx_arr, cals, mult)
            return

        if ex["samp_order"] == "CHAN":
            dtype = ex["dtype"]
            n_samps_tot = ex["n_samples"]
            n_ch = ex["n_channels"]
            n_read = stop - start
            if isinstance(idx, slice):
                idx_arr = np.arange(idx.start or 0, idx.stop or n_ch, idx.step or 1)
            else:
                idx_arr = np.asarray(idx, dtype=int)
            block = np.empty((n_ch, n_read), dtype=np.float64)
            with open(fname, "rb") as fh:
                for ch_i in idx_arr:
                    fh.seek((int(ch_i) * n_samps_tot + start) * dtype.itemsize)
                    block[int(ch_i)] = np.fromfile(fh, dtype=dtype, count=n_read).astype(np.float64)
            if has_offset:
                block -= offsets[:, None]
            _mult_cal_one(data, block, idx_arr, cals, mult)
            return

        if has_offset:
            dtype = ex["dtype"]
            n_ch = ex["n_channels"]
            n_read = stop - start
            with open(fname, "rb") as fh:
                fh.seek(start * n_ch * dtype.itemsize)
                raw_blk = np.fromfile(fh, dtype=dtype, count=n_read * n_ch).reshape(n_read, n_ch).T.astype(np.float64)
            raw_blk -= offsets[:, None]
            _mult_cal_one(data, raw_blk, idx, cals, mult)
            return

        _read_segments_file(
            self,
            data,
            idx,
            fi,
            start,
            stop,
            cals,
            mult,
            dtype=ex["dtype"],
            n_channels=ex["n_channels"],
            max_block_samples=ex["max_block_samples"],
        )


# ---------------------------------------------------------------------------
# Micromed .TRC reader via optional python-neo (BSD-3-Clause)
# ---------------------------------------------------------------------------

def _read_micromed_fixed_header_identity(trc_path: Path) -> dict[str, Any]:
    """Extract patient/hospital/date identity fields from Micromed TRC fixed binary header (bytes 0..176)."""
    with trc_path.open("rb") as fh:
        hdr = fh.read(176)
    if len(hdr) < 176:
        _raise_vendor_header_unreadable("micromed", EOFError("truncated header"))

    laboratory = _clean_cstr(hdr[32:64], encoding="cp1252")
    surname = _clean_cstr(hdr[64:86], encoding="cp1252")
    firstname = _clean_cstr(hdr[86:106], encoding="cp1252")
    dob_m, dob_d, dob_y = struct.unpack_from("<BBB", hdr, 106)
    reserved_str = _clean_cstr(hdr[109:128], encoding="cp1252")
    rec_d, rec_m, rec_y = struct.unpack_from("<BBB", hdr, 128)

    terms: set[str] = set()
    patient_code: str | None = None
    patient_name: str | None = None

    if reserved_str and all(32 <= ord(ch) < 127 for ch in reserved_str) and _is_real_term(reserved_str, min_digits_only_len=2):
        patient_code = reserved_str
        _add_term_variants(terms, reserved_str, split_parts=True, min_digits_only_len=2)

    real_first = firstname if _is_real_term(firstname, min_digits_only_len=2) else ""
    real_sur = surname if _is_real_term(surname, min_digits_only_len=2) else ""
    if real_first and real_sur:
        patient_name = real_sur if real_first == real_sur else f"{real_first} {real_sur}"
    elif real_sur or real_first:
        patient_name = real_sur or real_first

    for cand in (
        patient_name,
        f"{real_sur} {real_first}".strip() if (real_first and real_sur and real_first != real_sur) else "",
        real_first,
        real_sur,
    ):
        if cand:
            _add_term_variants(terms, cand, split_parts=True, min_digits_only_len=2)

    if _is_real_term(laboratory, min_digits_only_len=3):
        _add_term_variants(terms, laboratory, split_parts=True, min_digits_only_len=3)

    dob: dt.date | None = None
    if 1 <= dob_m <= 12 and 1 <= dob_d <= 31:
        try:
            dob = dt.date(1900 + dob_y, dob_m, dob_d)
        except ValueError:
            dob = None
    elif 1 <= dob_d <= 12 and 1 <= dob_m <= 31:
        try:
            dob = dt.date(1900 + dob_y, dob_d, dob_m)
        except ValueError:
            dob = None
    if dob is not None and 1900 <= dob.year <= 2100:
        _add_date_variants(terms, dob.isoformat())

    if 1 <= rec_m <= 12 and 1 <= rec_d <= 31:
        try:
            rec_date = dt.date(1900 + rec_y, rec_m, rec_d)
        except ValueError:
            rec_date = None
        if rec_date is not None and 1900 <= rec_date.year <= 2100:
            _add_date_variants(terms, rec_date.isoformat())

    return {
        "patient_code": patient_code,
        "patient_name": patient_name,
        "first_name": real_first or None,
        "last_name": real_sur or None,
        "birthday": dob,
        "terms": sorted(terms),
    }


def _open_neo_micromed_reader(trc_path: Path) -> Any:
    """Open and parse a Micromed .TRC header via neo.rawio.MicromedRawIO with privacy-safe errors."""
    MicromedRawIO = _import_micromed_rawio()
    if MicromedRawIO is None:
        check_unsupported_vendor_target(trc_path)
        raise ToolkitError(
            "unsupported_format",
            f"Micromed .trc recording {trc_path.name!r} requires the optional `neo` package. "
            'Install it with `pip install "brainlens[micromed]"` (or export EDF+ from Micromed SystemPlus).',
            format="micromed_trc",
            install_hint='pip install "brainlens[micromed]"',
            suggested_export="EDF+",
        )
    reader = MicromedRawIO(filename=str(trc_path))
    try:
        reader.parse_header()
    except ToolkitError:
        raise
    except NotImplementedError as exc:
        if "header_version" in str(exc):
            raise ToolkitError(
                "unsupported_format",
                f"Micromed .trc recording {trc_path.name!r} uses an unsupported TRC header version "
                "(neo supports header version 4 only). Export EDF+ from Micromed SystemPlus.",
                format="micromed_header_version",
                suggested_export="EDF+",
            ) from exc
        _raise_vendor_header_unreadable("micromed", exc)
    except Exception as exc:
        low_msg = str(exc).lower()
        if "sampling rates must be the same" in low_msg:
            raise ToolkitError(
                "unsupported_format",
                f"Micromed .trc recording {trc_path.name!r} contains mixed per-channel sampling rates, "
                "which is not supported natively. Export a uniform-rate EDF+ from Micromed SystemPlus.",
                format="micromed_mixed_rates",
                suggested_export="EDF+",
            ) from exc
        if "do not have the same" in low_msg and "units" in low_msg:
            raise ToolkitError(
                "unsupported_format",
                f"Micromed .trc recording {trc_path.name!r} contains mixed channel units across its signal stream, "
                "which is not supported natively. Export EDF+ from Micromed SystemPlus.",
                format="micromed_mixed_units",
                suggested_export="EDF+",
            ) from exc
        _raise_vendor_header_unreadable("micromed", exc)
    return reader


def _micromed_channel_names(signal_channels: Any) -> list[str]:
    names: list[str] = []
    seen: dict[str, int] = {}
    for idx, ch in enumerate(signal_channels):
        base = str(ch["name"]).strip("\x00 ").strip() or f"CH{idx + 1}"
        if base in seen:
            seen[base] += 1
            cand = f"{base}-{seen[base]}"
            while cand in seen:
                seen[base] += 1
                cand = f"{base}-{seen[base]}"
            names.append(cand)
            seen[cand] = 1
        else:
            seen[base] = 1
            names.append(base)
    return names


def _micromed_channel_type(name: str, *, is_volt: bool) -> str:
    if not is_volt:
        return "misc"
    up = name.strip().upper()
    if up.startswith(("ECG", "EKG")):
        return "ecg"
    if up.startswith(("EOG", "VEOG", "HEOG")):
        return "eog"
    if up.startswith("EMG"):
        return "emg"
    if up.startswith(("TRIG", "STIM", "MKR")):
        return "stim"
    if up.startswith(("RESP", "THOR", "ABDO")):
        return "resp"
    if up.startswith(("SPO2", "PULS", "BEAT", "DC", "XYZ", "AUX")):
        return "misc"
    return "eeg"


def _micromed_segment_table(reader: Any, trc_path: Path, sfreq: float) -> tuple[list[tuple[int, int, int]], int]:
    """Validate single-block Micromed segments and return ([(cum_start, cum_stop, seg_idx), ...], total_samples)."""
    if int(reader.block_count()) != 1 or int(reader.signal_streams_count()) != 1:
        _raise_vendor_header_unreadable("micromed", ValueError("unexpected block or stream count"))
    nb_seg = int(reader.segment_count(0))
    if nb_seg <= 0:
        _raise_vendor_header_unreadable("micromed", ValueError("no signal segments"))
    info_segments = getattr(reader, "info_segments", None)
    seg_table: list[tuple[int, int, int]] = []
    cum = 0
    prev_len = 0
    for seg_i in range(nb_seg):
        seg_len = int(reader.get_signal_size(block_index=0, seg_index=seg_i, stream_index=0))
        if seg_len < 0:
            _raise_vendor_header_unreadable("micromed", ValueError("negative segment size"))
        if seg_i > 0:
            if isinstance(info_segments, (list, tuple)) and len(info_segments) == nb_seg:
                gap_samples = int(info_segments[seg_i][0]) - (int(info_segments[seg_i - 1][0]) + prev_len)
            else:
                prev_t_stop = float(reader.segment_t_stop(0, seg_i - 1))
                cur_t_start = float(reader.get_signal_t_start(0, seg_i, 0))
                gap_samples = int(round((cur_t_start - prev_t_stop) * sfreq))
            if gap_samples != 0:
                raise ToolkitError(
                    "discontinuous_recording_unsupported",
                    f"Discontinuous multi-segment Micromed .trc recording {trc_path.name!r} has time gaps between TRONCA segments; "
                    "export continuous segments as EDF+ before clinical HFO review.",
                    format="micromed_discontinuous",
                    segment_count=nb_seg,
                )
        seg_table.append((cum, cum + seg_len, seg_i))
        cum += seg_len
        prev_len = seg_len
    if cum <= 0:
        _raise_vendor_header_unreadable("micromed", ValueError("empty signal"))
    return seg_table, cum


def _parse_micromed_header(trc_path: Path) -> SimpleNamespace:
    ident = _read_micromed_fixed_header_identity(trc_path)
    reader = _open_neo_micromed_reader(trc_path)
    signal_channels = reader.header["signal_channels"]
    if len(signal_channels) == 0:
        _raise_vendor_header_unreadable("micromed", ValueError("no signal channels"))
    sfreq = float(reader.get_signal_sampling_rate(stream_index=0))
    if not math.isfinite(sfreq) or sfreq <= 0:
        _raise_vendor_header_unreadable("micromed", ValueError("non-positive sampling rate"))
    _micromed_segment_table(reader, trc_path, sfreq)
    ch_names = _micromed_channel_names(signal_channels)
    signals = [
        SimpleNamespace(
            label=ch_names[i],
            physical_dimension=str(signal_channels["units"][i]).strip(),
            sampling_frequency=float(signal_channels["sampling_rate"][i]),
            prefiltering="",
        )
        for i in range(len(signal_channels))
    ]
    return SimpleNamespace(
        signals=signals,
        patient=SimpleNamespace(code=ident["patient_code"], name=ident["patient_name"]),
        vendor_identity_terms=ident["terms"],
        vendor_format="micromed",
    )


class _MicromedReaderHandle:
    """Deepcopy-safe lazy handle around neo.rawio.MicromedRawIO (prevents pickling open BufferedReaders)."""

    def __init__(self, path: Path, reader: Any | None = None) -> None:
        self.path = Path(path)
        self.reader = reader

    def get(self) -> Any:
        if self.reader is None:
            self.reader = _open_neo_micromed_reader(self.path)
        return self.reader

    def close(self) -> None:
        reader = self.reader
        self.reader = None
        if reader is not None:
            bufs = getattr(reader, "_memmap_analogsignal_buffers", None)
            if isinstance(bufs, dict):
                for segs in bufs.values():
                    if isinstance(segs, dict):
                        for chs in segs.values():
                            if isinstance(chs, dict):
                                for fh in chs.values():
                                    try:
                                        fh.close()
                                    except Exception:
                                        pass
                bufs.clear()

    def __deepcopy__(self, memo: dict[int, Any]) -> _MicromedReaderHandle:
        return _MicromedReaderHandle(self.path, reader=None)


class _RawMicromed(mne.io.BaseRaw):
    """Lazy chunked MNE BaseRaw adapter for Micromed .TRC recordings backed by neo.rawio.MicromedRawIO."""

    def __init__(self, trc_path: Path, preload: bool | str = False) -> None:
        from mne._fiff.constants import FIFF
        from mne._fiff.meas_info import create_info

        try:
            ident = _read_micromed_fixed_header_identity(trc_path)
            reader = _open_neo_micromed_reader(trc_path)

            signal_channels = reader.header["signal_channels"]
            n_channels = int(len(signal_channels))
            if n_channels <= 0:
                _raise_vendor_header_unreadable("micromed", ValueError("no signal channels"))
            sfreq = float(reader.get_signal_sampling_rate(stream_index=0))
            if not math.isfinite(sfreq) or sfreq <= 0:
                _raise_vendor_header_unreadable("micromed", ValueError("non-positive sampling rate"))

            seg_table, total_samples = _micromed_segment_table(reader, trc_path, sfreq)
            ch_names = _micromed_channel_names(signal_channels)
            ch_types: list[str] = []
            cals: list[float] = []
            units_fiff: list[int] = []
            for i in range(n_channels):
                unit_str = str(signal_channels["units"][i]).strip()
                ul = unit_str.lower()
                if ul in _VOLT_SCALES:
                    units_fiff.append(FIFF.FIFF_UNIT_V)
                    cals.append(_VOLT_SCALES[ul])
                    ch_types.append(_micromed_channel_type(ch_names[i], is_volt=True))
                else:
                    units_fiff.append(FIFF.FIFF_UNIT_NONE)
                    cals.append(1.0)
                    ch_types.append(_micromed_channel_type(ch_names[i], is_volt=False))

            info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
            with info._unlock():
                for i in range(n_channels):
                    info["chs"][i]["unit"] = units_fiff[i]
                bl_ann = (reader.raw_annotations or {}).get("blocks", [{}])[0]
                rec_dt = bl_ann.get("rec_datetime")
                if isinstance(rec_dt, dt.datetime) and 1900 <= rec_dt.year <= 2100:
                    info["meas_date"] = rec_dt.replace(tzinfo=dt.UTC)
                subj: dict[str, Any] = {}
                if ident["patient_code"]:
                    subj["his_id"] = ident["patient_code"]
                if ident["first_name"]:
                    subj["first_name"] = ident["first_name"]
                if ident["last_name"]:
                    subj["last_name"] = ident["last_name"]
                if ident["birthday"] is not None:
                    subj["birthday"] = ident["birthday"]
                if subj:
                    info["subject_info"] = subj

            itemsize = max(1, int(np.dtype(signal_channels["dtype"][0]).itemsize))
            raw_extra = {
                "reader_handle": _MicromedReaderHandle(trc_path, reader=reader),
                "n_channels": n_channels,
                "n_samples": total_samples,
                "seg_table": seg_table,
                "max_block_samples": max(1, 16 * 1024**2 // max(1, itemsize * n_channels)),
            }
            super().__init__(
                info,
                preload=False,
                filenames=[trc_path],
                last_samps=[total_samples - 1],
                orig_format="int",
                raw_extras=[raw_extra],
                verbose="ERROR",
            )
            self._cals = np.asarray(cals, dtype=np.float64)
            if preload:
                self._preload_data(preload)

            # Import Micromed triggers, notes, and epochs (Event A / Event B) as MNE Annotations.
            onsets: list[float] = []
            durations: list[float] = []
            descriptions: list[str] = []
            t_start0 = float(reader.get_signal_t_start(block_index=0, seg_index=0, stream_index=0))
            ev_channels = reader.header["event_channels"]
            nb_seg = len(seg_table)
            for ev_i in range(int(reader.event_channels_count())):
                ev_ch_name = str(ev_channels["name"][ev_i]).strip()
                for seg_i in range(nb_seg):
                    cnt = int(reader.event_count(block_index=0, seg_index=seg_i, event_channel_index=ev_i))
                    if cnt <= 0:
                        continue
                    ts, raw_durs, raw_lbls = reader.get_event_timestamps(
                        block_index=0, seg_index=seg_i, event_channel_index=ev_i
                    )
                    times = reader.rescale_event_timestamp(ts, dtype="float64", event_channel_index=ev_i)
                    durs_s = (
                        reader.rescale_epoch_duration(raw_durs, dtype="float64", event_channel_index=ev_i)
                        if raw_durs is not None
                        else None
                    )
                    for k in range(len(times)):
                        onset = max(0.0, float(times[k]) - t_start0)
                        dur = max(0.0, float(durs_s[k])) if durs_s is not None else 0.0
                        lbl_txt = str(raw_lbls[k]).strip("\x00 ").strip()
                        if ev_ch_name.lower() == "note":
                            desc = lbl_txt or "Note"
                        else:
                            desc = f"{ev_ch_name} {lbl_txt}".strip() if lbl_txt else ev_ch_name
                        onsets.append(onset)
                        durations.append(dur)
                        descriptions.append(desc)
            if onsets:
                self.set_annotations(
                    mne.Annotations(onset=onsets, duration=durations, description=descriptions, orig_time=None)
                )
        except ToolkitError:
            raise
        except Exception as exc:
            _raise_vendor_header_unreadable("micromed", exc)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        ex = self._raw_extras[fi]
        handle = ex.get("reader_handle")
        if handle is None:
            raise RuntimeError("Micromed reader is closed")
        reader = handle.get()
        n_ch = ex["n_channels"]
        if isinstance(idx, slice):
            ch_start, ch_stop, ch_step = idx.indices(n_ch)
            ch_sel: Any = slice(ch_start, ch_stop) if ch_step == 1 else np.arange(ch_start, ch_stop, ch_step, dtype=int)
        else:
            ch_sel = np.asarray(idx, dtype=int)

        max_block = ex["max_block_samples"]
        out_pos = 0
        for cum_start, cum_stop, seg_idx in ex["seg_table"]:
            if cum_stop <= start:
                continue
            if cum_start >= stop:
                break
            local_start = max(start, cum_start) - cum_start
            local_stop = min(stop, cum_stop) - cum_start
            cur = local_start
            while cur < local_stop:
                nxt = min(local_stop, cur + max_block)
                raw_chunk = reader.get_analogsignal_chunk(
                    block_index=0,
                    seg_index=seg_idx,
                    i_start=cur,
                    i_stop=nxt,
                    stream_index=0,
                    channel_indexes=ch_sel,
                    prefer_slice=True,
                )
                scaled = reader.rescale_signal_raw_to_float(
                    raw_chunk,
                    dtype="float64",
                    stream_index=0,
                    channel_indexes=ch_sel,
                ).T
                n_chunk = nxt - cur
                view = data[:, out_pos : out_pos + n_chunk]
                if mult is not None:
                    view[:] = mult @ scaled
                else:
                    np.multiply(scaled, cals.reshape(-1, 1), out=view)
                out_pos += n_chunk
                cur = nxt

    def close(self) -> None:
        for ex in getattr(self, "_raw_extras", ()) or ():
            handle = ex.get("reader_handle") if isinstance(ex, dict) else None
            if handle is not None:
                handle.close()
        super().close()


# ---------------------------------------------------------------------------
# Companion discovery & integrity verification hooks
# ---------------------------------------------------------------------------

def discover_vendor_companions(rec_entry: Path, staged_main: Path) -> list[tuple[Path, str]]:
    """Return [(orig_companion_path, stage_name), ...] for a vendor recording.

    Required companions raise ToolkitError('missing_companion') when absent.
    Optional companions are returned whenever present on disk so they are bound and SHA-256 verified.
    """
    folder = rec_entry.parent
    sfx = rec_entry.suffix.lower()
    fmt = _vendor_format_for_path(rec_entry)
    out: list[tuple[Path, str]] = []

    try:
        if sfx == ".eeg":
            if not is_nihon_kohden_eeg(staged_main):
                check_unsupported_vendor_target(rec_entry)
                raise ToolkitError("unsupported_format", f"{rec_entry.name!r} is not a valid Nihon Kohden .EEG file.")
            for ext_upper in ("PNT", "21E", "LOG"):
                cand = _find_sibling_ci(folder, [f"{rec_entry.stem}.{ext_upper}", f"{rec_entry.stem}.{ext_upper.lower()}"])
                if cand is not None:
                    out.append((cand, f"{rec_entry.stem}.{ext_upper}"))
            return out

        if sfx == ".lay":
            fileinfo, _ = _parse_persyst_lay_sections(staged_main)
            raw_dat = fileinfo.get("file", "").strip()
            dat_name = raw_dat.replace("\\", "/").rsplit("/", 1)[-1].strip() if raw_dat else f"{rec_entry.stem}.dat"
            cand = _find_sibling_ci(folder, [dat_name, f"{rec_entry.stem}.dat"])
            if cand is None:
                raise ToolkitError("missing_companion", f"Persyst data file (.dat) missing for {rec_entry.name}")
            return [(cand, dat_name)]

        if sfx == ".set":
            eeg = _read_eeglab_meta(staged_main)
            data_ref = eeg.get("data")
            if isinstance(data_ref, str):
                ref_name = data_ref.replace("\\", "/").rsplit("/", 1)[-1].strip()
                ref_ext = Path(ref_name).suffix.lower()
                if ref_ext == ".dat":
                    raise ToolkitError(
                        "unsupported_format",
                        f"Legacy EEGLAB .dat companion is not supported by MNE for {rec_entry.name!r}; re-save as .set/.fdt or export EDF+.",
                    )
                if ref_name and ref_name.lower() != rec_entry.name.lower() and ref_ext != ".set":
                    cand = _find_sibling_ci(folder, [ref_name])
                    if cand is not None:
                        return [(cand, ref_name)]
                    fallback_name = f"{rec_entry.stem}.fdt"
                    cand_fb = _find_sibling_ci(folder, [fallback_name])
                    if cand_fb is not None:
                        return [(cand_fb, fallback_name)]
                    raise ToolkitError("missing_companion", f"EEGLAB .fdt data file missing for {rec_entry.name}")
            return []

        if sfx == ".cdt":
            hdr = _find_sibling_ci(folder, [f"{rec_entry.name}.dpa", f"{rec_entry.name}.dpo"])
            if hdr is None:
                raise ToolkitError("missing_companion", f"Compumedics Curry header (.cdt.dpa/.cdt.dpo) missing for {rec_entry.name}")
            hdr_ext = hdr.suffix.lower()
            out.append((hdr, f"{rec_entry.name}{hdr_ext}"))
            for opt_ext in (".cef", ".ceo", ".hpi"):
                cand = _find_sibling_ci(folder, [f"{rec_entry.name}{opt_ext}"])
                if cand is not None:
                    out.append((cand, f"{rec_entry.name}{opt_ext}"))
            return out

        if sfx == ".dap":
            kv = _curry_extract_block_kv(_read_text_fallback(staged_main), "DATA_PARAMETERS")
            data_ext = (kv.get("DataExtension") or "dat").lstrip(".").strip() or "dat"
            data_p = _find_sibling_ci(folder, [f"{rec_entry.stem}.{data_ext}", f"{rec_entry.stem}.dat"])
            if data_p is None:
                raise ToolkitError("missing_companion", f"Compumedics Curry 7 data file missing for {rec_entry.name}")
            rs3 = _find_sibling_ci(folder, [f"{rec_entry.stem}.rs3"])
            if rs3 is None:
                raise ToolkitError("missing_companion", f"Compumedics Curry 7 labels file (.rs3) missing for {rec_entry.name}")
            out.append((data_p, f"{rec_entry.stem}.dat"))
            out.append((rs3, f"{rec_entry.stem}.rs3"))
            for opt_ext in (".cef", ".ceo", ".hpi"):
                cand = _find_sibling_ci(folder, [f"{rec_entry.stem}{opt_ext}"])
                if cand is not None:
                    out.append((cand, f"{rec_entry.stem}{opt_ext}"))
            return out

        if sfx == ".data":
            head = _find_sibling_ci(folder, [f"{rec_entry.stem}.head"])
            if head is None:
                raise ToolkitError("missing_companion", f"Nicolet header file (.head) missing for {rec_entry.name}")
            return [(head, f"{rec_entry.stem}.head")]

        # .cnt, .gdf, and .trc are single-file formats.
        return []
    except ToolkitError:
        raise
    except Exception as exc:
        _raise_vendor_header_unreadable(fmt, exc)


def verify_vendor_companion_links(
    p_entry: Path,
    main_src: Path,
    verified_companions: dict[str, tuple[Any, ...]],
    rec_slug: str,
) -> None:
    """Re-verify vendor header companion references and guard against newly added or repointed companions."""
    try:
        discovered = discover_vendor_companions(p_entry, main_src)
    except ToolkitError as exc:
        if exc.code == "missing_companion":
            raise ToolkitError(
                "source_recording_missing",
                f"Companion recording file missing on disk for {p_entry.name!r}: {exc.message}",
                recording=rec_slug,
            ) from exc
        raise
    for comp_orig, _stage_name in discovered:
        if not comp_orig.is_file() or str(comp_orig.resolve()) not in verified_companions:
            raise ToolkitError(
                "source_recording_modified",
                f"Vendor companion file {comp_orig.name!r} changed target or appeared after registration; "
                "refusing to read modified recording under an earlier registration.",
                recording=rec_slug,
            )


def parse_vendor_header(main_path: Path) -> SimpleNamespace | None:
    sfx = main_path.suffix.lower()
    fmt = _vendor_format_for_path(main_path)
    try:
        if sfx == ".eeg":
            return _parse_nihon_header(main_path)
        if sfx == ".lay":
            return _parse_persyst_header(main_path)
        if sfx == ".set":
            return _parse_eeglab_header(main_path)
        if sfx == ".cnt":
            return _parse_cnt_header(main_path)
        if sfx in {".cdt", ".dap"}:
            return _parse_curry_header(main_path)
        if sfx == ".gdf":
            return _parse_gdf_header(main_path)
        if sfx == ".data":
            return _parse_nicolet_header(main_path)
        if sfx == ".trc":
            return _parse_micromed_header(main_path)
        return None
    except ToolkitError:
        raise
    except Exception as exc:
        _raise_vendor_header_unreadable(fmt, exc)


def extract_vendor_header_identity_terms(raw: Any = None, header: Any = None) -> list[str]:
    """Collect privacy-sensitive identity terms from parsed vendor headers and raw.info."""
    terms: set[str] = set()
    if header is not None:
        for t in getattr(header, "vendor_identity_terms", ()) or ():
            if _is_real_term(t, min_digits_only_len=2):
                terms.add(str(t).strip())
    if raw is not None and hasattr(raw, "info"):
        subject = raw.info.get("subject_info") or {}
        if isinstance(subject, dict):
            num_id = subject.get("id")
            if isinstance(num_id, (int, np.integer)) and int(num_id) > 0:
                _add_term_variants(terms, str(int(num_id)), min_digits_only_len=2)
    return sorted(terms)


# ---------------------------------------------------------------------------
# Vendor MNE reader dispatch
# ---------------------------------------------------------------------------

def _open_cnt(path: Path, preload: bool | str) -> Any:
    try:
        return mne.io.read_raw_cnt(path, preload=preload, data_format="auto", verbose="ERROR")
    except RuntimeError as first_err:
        msg = str(first_err)
        if "Could not automatically compute" not in msg and "data_format" not in msg:
            if "ANT Neuro" in msg:
                raise ToolkitError(
                    "unsupported_format",
                    f"Could not read {path.name!r} as Neuroscan CNT (if this is an ANT Neuro CNT file, export EDF+).",
                ) from first_err
            _raise_vendor_header_unreadable("neuroscan_cnt", first_err)

    # Fallback when header numsamples==0 (classic Scan 4.1) or clipped data bytes break MNE's auto divisor check.
    with path.open("rb") as fh:
        hdr = fh.read(900)
    if len(hdr) < 900:
        _raise_vendor_header_unreadable("neuroscan_cnt", EOFError("truncated header"))
    n_channels = int(struct.unpack_from("<H", hdr, 370)[0])
    n_samples_hdr = int(struct.unpack_from("<i", hdr, 864)[0])
    evpos = int(struct.unpack_from("<i", hdr, 886)[0])
    flag_353 = int(hdr[353])
    data_bytes = evpos - (900 + 75 * n_channels)
    if n_channels <= 0 or data_bytes <= 0:
        _raise_vendor_header_unreadable("neuroscan_cnt", ValueError("invalid header geometry"))

    candidates: list[str] = ["int16", "int32"]
    if n_samples_hdr > 0 and data_bytes < n_samples_hdr * n_channels * 4:
        candidates = ["int16"]

    scored: list[tuple[float, str]] = []
    for fmt in candidates:
        try:
            probe = mne.io.read_raw_cnt(path, preload=False, data_format=fmt, verbose="ERROR")
            try:
                stop = min(256, int(probe.n_times))
                if stop < 2:
                    continue
                seg = probe.get_data(start=0, stop=stop)
                if not np.all(np.isfinite(seg)):
                    continue
                max_abs = float(np.max(np.abs(seg)))
                std_val = float(np.std(seg))
                diff_std = float(np.std(np.diff(seg, axis=1)))
                smoothness = (diff_std / std_val) if std_val > 1e-12 else 1.0
                # Strongly penalize unphysiological multi-volt garbage (typically int32 on 16-bit data).
                amp_penalty = 10.0 if max_abs > 0.25 else 0.0
                tie = 0.0 if ((fmt == "int16") == (flag_353 == 0)) else 0.05
                scored.append((amp_penalty + smoothness + tie, fmt))
            finally:
                probe.close()
        except Exception:
            continue

    if not scored:
        _raise_vendor_header_unreadable("neuroscan_cnt", ValueError("undetermined sample format"))
    scored.sort(key=lambda item: item[0])
    best_fmt = scored[0][1]
    return mne.io.read_raw_cnt(path, preload=preload, data_format=best_fmt, verbose="ERROR")


def _open_curry(path: Path, preload: bool | str) -> Any:
    # Validate continuous recording & companions before opening.
    _parse_curry_metadata(path)
    try:
        import curryreader  # noqa: F401
    except ImportError:
        return _RawCurryFallback(path, preload=preload)
    raw = mne.io.read_raw_curry(path, preload=preload, verbose="ERROR")
    if not isinstance(raw, mne.io.BaseRaw):
        raise ToolkitError(
            "unsupported_format",
            f"Compumedics Curry file {path.name!r} is epoched/evoked (not continuous raw); export continuous raw data.",
        )
    return raw


def _open_persyst(path: Path, preload: bool | str) -> Any:
    # Guard MNE's case-sensitive `fname.endswith('.lay')` check when running on case-insensitive FS.
    eff_path = path
    if not path.name.endswith(".lay"):
        lowered = path.with_suffix(".lay")
        if lowered.is_file():
            eff_path = lowered
    return mne.io.read_raw_persyst(eff_path, preload=preload, verbose="ERROR")


def _open_eeglab(path: Path, preload: bool | str) -> Any:
    # Pre-check MATLAB v7.3 and epoched datasets for clean ToolkitError messages.
    _read_eeglab_meta(path)
    return mne.io.read_raw_eeglab(path, preload=preload, verbose="ERROR")


def _open_micromed(path: Path, preload: bool | str) -> Any:
    return _RawMicromed(path, preload=preload)


def open_vendor_raw(path: Path, preload: bool | str = False):
    """Open a supported vendor recording via MNE (preload=False by default), suppressing stray MNE stdout."""
    check_unsupported_vendor_target(path)
    sfx = path.suffix.lower()
    if sfx not in VENDOR_REC_SUFFIXES:
        raise ToolkitError("unsupported_format", f"Unsupported EEG file format: {path.name}")
    fmt = _vendor_format_for_path(path)

    # MNE's Persyst reader prints birthdate errors directly to stdout and Nicolet logs info without @verbose;
    # silence stdout + MNE info logs so MCP JSON-RPC and CLI JSON stdout stay clean.
    try:
        with contextlib.redirect_stdout(io.StringIO()), mne.utils.use_log_level("ERROR"):
            if sfx == ".eeg":
                return mne.io.read_raw_nihon(path, preload=preload, verbose="ERROR")
            if sfx == ".lay":
                return _open_persyst(path, preload=preload)
            if sfx == ".set":
                return _open_eeglab(path, preload=preload)
            if sfx == ".cnt":
                return _open_cnt(path, preload=preload)
            if sfx in {".cdt", ".dap"}:
                return _open_curry(path, preload=preload)
            if sfx == ".gdf":
                return mne.io.read_raw_gdf(path, preload=preload, stim_channel=None, verbose="ERROR")
            if sfx == ".data":
                eff_path = path
                if not path.name.endswith(".data") and path.with_suffix(".data").is_file():
                    eff_path = path.with_suffix(".data")
                return mne.io.read_raw_nicolet(eff_path, ch_type="eeg", preload=preload, verbose="ERROR")
            if sfx == ".trc":
                return _open_micromed(path, preload=preload)
    except ToolkitError:
        raise
    except Exception as exc:
        _raise_vendor_header_unreadable(fmt, exc)
    raise ToolkitError("unsupported_format", f"Unsupported EEG file format: {path.name}")
