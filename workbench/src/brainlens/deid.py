"""Identity policy for local registrations, shared exports and MCP responses."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import ProvenanceSpec, RecordingSpec, ToolkitError, canonical_digest, read_json

if TYPE_CHECKING:
    from .storage import WorkbenchStore

HEADER_SUBJECT_WITHHELD = "header subject ID withheld"
SHEET_DEIDENTIFICATION_META = '<meta name="brainlens-deidentification" content="share-v1" />'

_GENERIC_SUBJECT_IDS = frozenset({
    "", "x", "unknown", "none", "n/a", "na", "anon", "anonymous", "subject", "patient", "0", "sub-0", "sub-x",
    "unspecified", "unspecified unspecified", "no_name", "no name", "noname",
    "surrogate", "surrogate name", "surrogate_name",
})
# A registration or analysis can fail before recording.json exists. Keep identity
# discovered in the verified snapshot available just for this particular MCP call.
_AGENT_TERMS: ContextVar[set[str] | None] = ContextVar("brainlens_agent_header_terms", default=None)
_Identity = tuple[tuple[str, ...], str | None]
_AGENT_READS: ContextVar[dict[tuple[str, str], set[_Identity]] | None] = ContextVar("brainlens_agent_identity_reads", default=None)


@contextmanager
def agent_identity_scope() -> Iterator[None]:
    token = _AGENT_TERMS.set(set())
    reads_token = _AGENT_READS.set({})
    try:
        yield
    finally:
        _AGENT_READS.reset(reads_token)
        _AGENT_TERMS.reset(token)


def _norm_subject_token(val: Any) -> str | None:
    s = str(val or "").strip().lower()
    if s in _GENERIC_SUBJECT_IDS:
        return None
    if s.startswith("sub-"):
        s = s[4:]
    return s if s not in _GENERIC_SUBJECT_IDS else None


def subject_digest(label: str | None) -> str | None:
    token = _norm_subject_token(label)
    return canonical_digest(token) if token else None


def share_labels(case: str, recordings: list[str], pseudonym: str | None) -> tuple[str, dict[str, str]]:
    """Return filename-safe display labels within one case, independent of source names."""
    label = "shared" if pseudonym is None else pseudonym.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", label):
        raise ToolkitError(
            "invalid_pseudonym",
            "Use a pseudonym of 1 to 64 ASCII letters, digits, periods, underscores or hyphens, starting with a letter or digit.",
        )
    ordered = sorted(set(recordings))
    names = {rec: label if len(ordered) == 1 else f"{label}-{i:02d}" for i, rec in enumerate(ordered, 1)}
    return label, names


def registration_provenance(
    prov: ProvenanceSpec, *, sidecar_label: str | None, bids_label: str | None, native_label: str | None,
) -> ProvenanceSpec:
    """Preserve the local clinical label and remember which source authorized it."""
    pid = sidecar_label or bids_label or native_label
    if pid is None:
        return prov
    source = "sidecar" if sidecar_label else "bids" if bids_label else "header"
    summary = prov.summary
    if not prov.is_synthetic and pid not in summary:
        summary = f"{summary} · {pid}"
    return prov.model_copy(update={"participant_id": pid, "participant_id_source": source, "summary": summary})


def native_header_terms(raw: Any = None, edf: Any = None) -> list[str]:
    """Collect identity strings while reading a verified EDF/BDF/FIF header."""
    terms: set[str] = set()
    subject = (raw.info.get("subject_info") or {}) if raw is not None else {}
    for key in ("his_id", "first_name", "middle_name", "last_name"):
        if subject.get(key):
            terms.add(str(subject[key]))
    if subject.get("first_name") and subject.get("last_name"):
        terms.add(f"{subject['first_name']} {subject['last_name']}")
    birthday = subject.get("birthday")
    if isinstance(birthday, dt.date) and birthday.year > 1:
        terms.update((birthday.isoformat(), birthday.strftime("%d-%b-%Y").upper(), birthday.strftime("%d.%m.%y")))

    patient = str(getattr(edf, "local_patient_identification", "") or "").strip()
    if patient and _norm_subject_token(patient) is not None:
        fields = patient.split()
        # The four EDF+ slots are patient code, sex, birthdate and name.
        if len(fields) >= 4 and fields[1].upper() in {"M", "F", "X"}:
            terms.update([fields[0], *fields[2:]])
            terms.update(fields[3].split("_"))
        else:
            terms.add(patient)
            terms.update(fields)
    recording = str(getattr(edf, "local_recording_identification", "") or "").split()
    if len(recording) >= 3 and recording[0].lower() == "startdate":
        terms.update(recording[1:4])  # start date, hospital administration and technician codes
    from .io_vendor import extract_vendor_header_identity_terms

    terms.update(extract_vendor_header_identity_terms(raw=raw, header=edf))
    cleaned = sorted({s.strip() for s in terms if _norm_subject_token(s)})
    transient = _AGENT_TERMS.get()
    if transient is not None:
        transient.update(cleaned)
    return cleaned


def _snapshot_identity(rec: RecordingSpec) -> tuple[str | None, list[str]]:
    from .io_integrity import verified_source_snapshot
    from .io_sidecars import _extract_native_subject_id
    from .io_signals import _parse_bdf_header, open_mne_raw
    from .io_vendor import is_vendor_recording_suffix, parse_vendor_header

    import edfio

    with verified_source_snapshot(rec) as (snapshot, *_):
        edf = None
        if snapshot.suffix.lower() == ".edf":
            edf = edfio.read_edf(snapshot, lazy_load_data=True)
        elif snapshot.suffix.lower() == ".bdf":
            edf = _parse_bdf_header(snapshot)
        elif is_vendor_recording_suffix(snapshot):
            edf = parse_vendor_header(snapshot)
        with open_mne_raw(snapshot, preload=False) as raw:
            return _extract_native_subject_id(raw, edf), native_header_terms(raw, edf)


def _trusted_label(rec: RecordingSpec, terms: list[str]) -> str | None:
    prov = rec.provenance
    if not prov.participant_id or prov.participant_id_source in {"header", "withheld"}:
        return None
    if prov.participant_id_source in {"bids", "sidecar"}:
        return prov.participant_id
    # Legacy registrations recorded the chosen label, but not its provenance.
    from .io_sidecars import _bids_subject_ids_from_path, _extract_prov_participant_id

    pid = _norm_subject_token(prov.participant_id)
    for label in _bids_subject_ids_from_path(Path(rec.path)):
        if pid == _norm_subject_token(label):
            return label
    if prov.sidecar_path:
        try:
            declared = _extract_prov_participant_id(read_json(Path(prov.sidecar_path)))
        except (OSError, ValueError):
            declared = None
        if declared and pid == _norm_subject_token(declared):
            return declared
    # A label absent from the verified native header came from the other
    # registration sources (e.g. the case's explicit PROVENANCE.json).
    if pid not in {_norm_subject_token(t) for t in terms}:
        return prov.participant_id
    return None


@lru_cache(maxsize=1024)
def _stored_identity(
    path: Path, _file_signature: tuple[int, ...], expected_fp: str | None,
) -> tuple[str | None, tuple[str, ...], str | None]:
    doc = read_json(path)
    if expected_fp is not None and doc["spec"]["fingerprint"] != expected_fp:
        raise ToolkitError("run_recording_mismatch", "Recording identity changed during export. Retry with the current run.")
    prov = doc["spec"]["provenance"]
    terms = doc.get("native_header_terms")
    source = prov.get("participant_id_source")
    native = prov.get("participant_id") if source == "header" else None
    if terms is not None and (source or not prov.get("participant_id")):
        return native, tuple(terms), prov.get("participant_id") if source in {"bids", "sidecar"} else None
    rec = RecordingSpec.model_validate(doc["spec"])
    if terms is None:
        native, terms = _snapshot_identity(rec)
    return native, tuple(terms), _trusted_label(rec, terms)


def stored_identity(path: Path, expected_fp: str | None = None) -> tuple[str | None, tuple[str, ...], str | None]:
    st = path.stat()
    return _stored_identity(path, (st.st_dev, st.st_ino, st.st_ctime_ns, st.st_mtime_ns, st.st_size), expected_fp)


def restore_withheld_provenance(prov: dict[str, Any], path: Path) -> dict[str, Any]:
    """Recover a local label for workspaces created with the earlier always-on policy."""
    if prov.get("participant_id_source") != "withheld":
        return prov
    try:
        native, _, _ = stored_identity(path)
    except (ToolkitError, OSError, ValueError):
        return prov
    if not native:
        return prov
    return {**prov, "participant_id": native, "participant_id_source": "header",
            "summary": prov["summary"].replace(HEADER_SUBJECT_WITHHELD, native)}


def _text_redactor(
    terms: Iterable[str], public: Iterable[str], aliases: dict[str, str] | None = None,
) -> Callable[[str], str]:
    # One pass avoids replacing text inserted by a previous replacement.
    replacements: dict[str, str | None] = {t: HEADER_SUBJECT_WITHHELD for t in terms if t}
    replacements.update(aliases or {})
    replacements.update({p: None for p in public if p})
    patterns = []
    actions: list[str | None] = []
    ordered = sorted(replacements, key=lambda s: (-len(s), replacements[s] is not None, s not in (aliases or {})))
    for label in ordered:
        parts = [re.escape(part) for part in re.split(r"[\s_-]+", label) if part]
        if not parts:
            continue
        pattern = r"[\s_-]+".join(parts)
        if len(label) <= 3 or replacements[label] is None:
            pattern = rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])"
        patterns.append(f"({pattern})")
        actions.append(replacements[label])
    if not patterns:
        return lambda text: text
    matcher = re.compile("|".join(patterns), re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        replacement = actions[match.lastindex - 1]
        return match[0] if replacement is None else replacement

    return lambda text: matcher.sub(replace, text)


def _recording_identity(rec: RecordingSpec) -> tuple[tuple[str, ...], str | None]:
    if rec._header_terms is not None:
        return rec._header_terms, _trusted_label(rec, list(rec._header_terms))
    if rec._registration_path is not None:
        _, terms, label = stored_identity(rec._registration_path, rec.fingerprint)
        return terms, label
    _, terms = _snapshot_identity(rec)
    return tuple(terms), _trusted_label(rec, terms)


def trusted_participant_label(rec: RecordingSpec) -> str | None:
    return _recording_identity(rec)[1]


def remember_agent_read(case: str, rec: RecordingSpec) -> None:
    """Retain the identity actually read if another request replaces the registration."""
    reads = _AGENT_READS.get()
    if reads is not None:
        try:
            identity = _recording_identity(rec)
        except (ToolkitError, ValueError, OSError):
            # The response boundary handles unavailable legacy sources. A local
            # cache read itself must still finish under the normal store contract.
            identity = ((rec.provenance.participant_id,) if rec.provenance.participant_id else (), None)
        reads.setdefault((case, rec.slug), set()).add(identity)


def _disclosure_rules(
    identities: Iterable[tuple[tuple[str, ...], str | None]], *, additional_private: Iterable[str] = (),
    aliases: dict[str, str] | None = None,
) -> Callable[[str], str]:
    terms = set(additional_private)
    hidden = {_norm_subject_token(term) for term in terms}
    public: set[str] = set()
    for native_terms, label in identities:
        terms.update(native_terms)
        permitted_token = _norm_subject_token(label)
        hidden.update(token for term in native_terms if (token := _norm_subject_token(term)) != permitted_token)
        if label:
            public.add(label)
    allowed = {label for label in public if _norm_subject_token(label) not in hidden}
    return _text_redactor(terms, allowed, aliases)


def recording_redactor(
    recordings: list[RecordingSpec], *, aliases: dict[str, str] | None = None, pseudonym: str | None = None,
) -> Callable[[str], str]:
    identities = [_recording_identity(rec) for rec in recordings]
    if pseudonym is not None:
        label = pseudonym.strip()
        # A source ID explicitly declared in BIDS/sidecar provenance is still
        # public. Reject private IDs before inserting the alias into filenames.
        meaningful = [
            (tuple(t for t in terms if len(t) >= 4 or t.casefold() == label.casefold()), public)
            for terms, public in identities
        ]
        if _disclosure_rules(meaningful)(label) != label:
            raise ToolkitError("invalid_pseudonym", "Choose a pseudonym that does not contain native header identity.")
    aliases = dict(aliases or {})
    for rec in recordings:
        if rec.slug in aliases:
            alias = aliases[rec.slug]
            path = Path(rec.path)
            suffix = ".fif.gz" if path.name.lower().endswith(".fif.gz") else path.suffix
            aliases.update({str(path): alias + suffix, path.name: alias + suffix, path.name.removesuffix(suffix): alias})
    return _disclosure_rules(identities, aliases=aliases)


# Protocol keys and generated identifiers can coincide with short patient codes.
# Only maps containing user-selected labels should redact these reserved words.
_ENVELOPE_FIELDS = frozenset({
    "case", "cases", "case_slug", "recording", "recordings", "recording_id", "run", "run_name",
    "channel", "channels", "channel_name", "id", "role", "status", "code", "error", "details",
    "name", "path", "description", "summary", "mode", "kind", "message", "action", "source",
    "label", "labels", "data", "type", "value", "author", "note", "text", "counts", "shaft",
    "event_id", "participant_id", "participant_id_source", "provenance", "origin_kind",
})
_LABEL_MAPS = frozenset({
    "recordings", "by_recording", "runs_by_recording", "recording_fingerprints",
    "per_channel_tags", "shaft_groups", "by_case", "positions",
})
_DIGEST_FIELDS = ("fingerprint", "signature", "sha256", "generation_id", "export_hash")


def scrub_agent_payload(payload: dict[str, Any], store: WorkbenchStore) -> dict[str, Any]:
    """Mask header identity at the MCP response boundary with per-recording disclosure rules."""
    transient = _AGENT_TERMS.get() or set()
    registry: dict[str, dict[str, set[_Identity]]] = {}
    unavailable: set[tuple[str, str]] = set()
    for entry in (store.root / "cases").glob("*/recordings/*/recording.json"):
        if not entry.resolve().is_relative_to(store.root):
            continue
        case, rec = entry.parents[2].name, entry.parent.name
        try:
            _, native_terms, trusted_label = stored_identity(entry)
        except (ToolkitError, OSError, ValueError):
            unavailable.add((case, rec))
            # Slugs and a legacy registration's saved primary ID can occur in
            # aggregate lists and paths even when the original header is offline.
            known = {case, rec}
            try:
                spec = read_json(entry)["spec"]
                known.update({Path(spec["path"]).name, Path(spec["path"]).stem})
                if pid := spec["provenance"].get("participant_id"):
                    known.add(pid)
            except (OSError, ValueError, KeyError, TypeError):
                pass
            native_terms = tuple(known)
            trusted_label = None
        registry.setdefault(case, {})[rec] = {(native_terms, trusted_label)}
    agent_reads = _AGENT_READS.get() or {}
    for (case, rec), captured in agent_reads.items():
        registry.setdefault(case, {}).setdefault(rec, set()).update(captured)

    @lru_cache(maxsize=None)
    def redactor(case: str | None, rec: str | None, include_transient: bool = False) -> Callable[[str], str]:
        extra = transient if include_transient else ()
        rows = registry.get(case, {})
        if rec in rows:
            return _disclosure_rules(rows[rec], additional_private=extra)
        if rows:
            return _disclosure_rules((row for snapshots in rows.values() for row in snapshots), additional_private=extra)
        all_rows = (row for rs in registry.values() for snapshots in rs.values() for row in snapshots)
        return _disclosure_rules(all_rows, additional_private=transient)

    def walk(value: Any, case: str | None, rec: str | None, field: str = "", include_transient: bool = False) -> Any:
        if isinstance(value, str):
            if any(kind in field for kind in _DIGEST_FIELDS) and re.fullmatch(r"[a-fA-F0-9]{8,128}", value):
                return value
            if field in {"recording_id", "id"} and re.fullmatch(r"rec_[0-9a-f]{16}", value):
                return value
            return redactor(case, rec, include_transient)(value)
        if isinstance(value, (list, tuple)):
            return [walk(item, case, rec, field, include_transient) for item in value]
        if not isinstance(value, dict):
            return value

        row_case = value.get("case") or value.get("case_slug")
        if isinstance(row_case, str) and case != row_case:
            case, rec = row_case, None
        row_rec = value.get("recording") or value.get("active_recording")
        if isinstance(row_rec, str):
            rec = row_rec
        if value.get("error") is True and case is None:
            details = value.get("details")
            det_case = (details.get("case") or details.get("case_slug")) if isinstance(details, dict) else None
            det_rec = details.get("recording") if isinstance(details, dict) else None
            if isinstance(det_case, str):
                case = det_case
                if isinstance(det_rec, str):
                    rec = det_rec
            elif isinstance(det_rec, str):
                read_cases = [c for c, r in agent_reads if r == det_rec]
                if len(read_cases) == 1:
                    case, rec = read_cases[0], det_rec
                else:
                    reg_cases = [c for c, recs in registry.items() if det_rec in recs]
                    if len(reg_cases) == 1:
                        case, rec = reg_cases[0], det_rec
            elif len(agent_reads) == 1:
                case, rec = next(iter(agent_reads))
            if case is not None and transient:
                include_transient = True
        if (case, rec) in unavailable or (rec is None and any(c == case for c, _ in unavailable)):
            return {"error": True, "code": "source_identity_unavailable",
                    "message": "The legacy recording's source is required to verify its header identity."}
        scrub = redactor(case, rec, include_transient)
        result: dict[Any, Any] = {}
        for key, item in value.items():
            # Preserve the actual protocol envelope, scrub labels in dynamic maps
            # and any other keys carrying patient identifiers.
            safe_key = scrub(key) if isinstance(key, str) and (key not in _ENVELOPE_FIELDS or field in _LABEL_MAPS) else key
            child_rec = key if field in {"recordings", "by_recording", "runs_by_recording"} and key in registry.get(case, {}) else rec
            result[safe_key] = walk(item, case, child_rec, key if isinstance(key, str) else "", include_transient)
        return result

    return walk(payload, None, None)


# Markers and user-chosen recording/channel labels can themselves contain a date.
# Cover common clinical header, ISO/BIDS, numeric and written calendar forms.
_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE_IN_NAVIGATION = re.compile(
    rf"(?<!\d)(?:(?:19|20)\d{{2}}[-_./年]?(?:0?[1-9]|1[0-2])[-_./月]?(?:0?[1-9]|[12]\d|3[01])日?"
    rf"|\d{{1,2}}[-_./]\d{{1,2}}[-_./](?:\d{{4}}|\d{{2}})"
    rf"|\d{{1,2}}[- ./]{_MONTH}[- .,]+\d{{2,4}}|{_MONTH}[- .]+\d{{1,2}}(?:st|nd|rd|th)?[, .-]+\d{{4}})"
    r"(?!\d)(?:[T ]\d{2}:?\d{2}(?::?\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?",
    re.IGNORECASE,
)


def scrub_navigation_dates(value: Any) -> Any:
    if isinstance(value, str):
        if re.fullmatch(r"rec_[0-9a-f]{16}", value):
            return value
        return _DATE_IN_NAVIGATION.sub("[date withheld]", value)
    if isinstance(value, dict):
        return {scrub_navigation_dates(k): scrub_navigation_dates(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_navigation_dates(v) for v in value]
    return value
