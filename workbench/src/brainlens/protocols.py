"""Reusable Group Analysis Protocols (strips patient-specific exclusions & bad channels)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import AnalysisConfig, ProtocolSpec, ReferenceConfig, ToolkitError, atomic_write_json, read_json
from .storage import WorkbenchStore, sanitize_slug


def save_protocol(
    name: str,
    case_slug: str,
    recording: str | None = None,
    run_name: str = "baseline",
    description: str = "",
    workspace_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Extract reusable analysis method from a saved run WITHOUT case-specific channel/time exclusions."""
    store = WorkbenchStore(workspace_dir)
    recs = store.list_recordings(case_slug)
    rec = sanitize_slug(recording) if recording else recs[0]
    bundle = store.load_run(case_slug, rec, run_name)
    rec_model = store.load_recording(case_slug, rec)
    cfg = AnalysisConfig.model_validate(bundle["config"]["analysis_config"])

    slug = sanitize_slug(name)
    proto = ProtocolSpec(
        name=name.strip(),
        slug=slug,
        description=description or f"Saved from {case_slug}/{rec} ({run_name})",
        modality=rec_model.modality,
        reference_mode=cfg.reference.mode,
        qc=cfg.qc,
        filters=cfg.filters,
        detector=cfg.detector,
        classifier=cfg.classifier,
        epoch_seconds=cfg.epoch_seconds,
        synchrony_window_ms=cfg.synchrony_window_ms,
        widespread_burst_min_shafts=cfg.widespread_burst_min_shafts,
    )
    out_path = store.root / "protocols" / f"{slug}.json"
    atomic_write_json(out_path, proto.model_dump(mode="json"))
    return {
        "protocol_slug": slug,
        "path": str(out_path),
        "protocol": proto.model_dump(mode="json"),
        "stripped_case_specific_fields": [
            "channels",
            "exclusions",
            "start_seconds",
            "end_seconds",
            *(["segments"] if cfg.segments else []),
            "state",
            "explicit_bipolar_pairs",
            "average_channels",
        ],
    }


def list_protocols(workspace_dir: str | Path | None = None) -> list[dict[str, Any]]:
    store = WorkbenchStore(workspace_dir)
    proot = store.root / "protocols"
    out: list[dict[str, Any]] = []
    if not proot.is_dir():
        return out
    for p in sorted(proot.glob("*.json")):
        try:
            out.append(read_json(p))
        except Exception:
            continue
    return out


def load_protocol_config(
    protocol_name_or_slug: str,
    store: WorkbenchStore,
    *,
    recording_modality: str | None = None,
) -> AnalysisConfig:
    slug = sanitize_slug(protocol_name_or_slug)
    p = store.root / "protocols" / f"{slug}.json"
    if not p.is_file():
        raise ToolkitError("protocol_not_found", f"Saved protocol {protocol_name_or_slug!r} not found.")
    proto = ProtocolSpec.model_validate(read_json(p))
    if recording_modality:
        intracranial = {"seeg", "ieeg"}
        compat = (proto.modality == recording_modality) or (
            proto.modality in intracranial and recording_modality in intracranial
        )
        if not compat:
            raise ToolkitError(
                "protocol_modality_mismatch",
                f"Protocol {proto.slug!r} targets modality '{proto.modality}' and cannot be applied to '{recording_modality}' recording.",
                protocol_modality=proto.modality,
                recording_modality=recording_modality,
            )
    return AnalysisConfig(
        protocol_name=proto.slug,
        reference=ReferenceConfig(mode=proto.reference_mode),
        qc=proto.qc,
        filters=proto.filters,
        detector=proto.detector,
        classifier=proto.classifier,
        epoch_seconds=proto.epoch_seconds,
        synchrony_window_ms=proto.synchrony_window_ms,
        widespread_burst_min_shafts=proto.widespread_burst_min_shafts,
    )
