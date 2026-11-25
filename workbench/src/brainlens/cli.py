"""Command-line interface for BrainLens Neurophysiology Workbench."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

from .analytics import compute_overlay_concordance, query_case
from .anatomy import clear_anatomy_labels, read_anatomy_sidecar, recording_anatomy_coverage, save_anatomy_labels
from .bids_export import export_bids_events
from .case_sheet import export_case_sheet
from .cohort import list_cohorts, run_cohort
from .edf_export import export_edf_annotations
from .io_readers import discover_available_inputs, inspect_case
from .models import ToolkitError, default_workspace_dir, get_env, read_json
from .pipeline import analyze_case
from .protocols import list_protocols, save_protocol
from .pyhfo_runner import PyHFORunner, check_engine_health, resolve_engine_settings, save_engine_settings
from .storage import WorkbenchStore, sanitize_slug
from .triad_views import render_evidence_card
from .workbench_http import WorkbenchHTTPServer, is_loopback_http_url


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _parse_interval_spec(spec: str) -> dict[str, Any]:
    # Format: start:end or start:end:CH1,CH2
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise ToolkitError("invalid_interval", f"Expected start:end[:ch1,ch2], got {spec!r}")
    out: dict[str, Any] = {"start": float(parts[0]), "end": float(parts[1])}
    if len(parts) == 3 and parts[2].strip():
        out["channels"] = [c.strip() for c in parts[2].split(",") if c.strip()]
    return out


def _serve_workbench(
    workspace: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    daemon: bool = False,
    case: str | None = None,
    recording: str | None = None,
    view: str | None = None,
) -> int:
    store = WorkbenchStore(workspace)
    patch: dict[str, Any] = {}
    if case:
        patch["case"] = sanitize_slug(case)
    if recording:
        patch["recording"] = sanitize_slug(recording)
    if view in {"overview", "reader", "sheet"}:
        patch["view"] = view
    ui = store.update_ui_state(patch, updated_by="cli") if patch else store.load_ui_state()

    meta_file = store.root / "workbench_server.json"
    if meta_file.is_file():
        try:
            info = read_json(meta_file)
            base_u = str(info["url"]).rstrip("/")
            if is_loopback_http_url(base_u):
                with urllib.request.urlopen(base_u + "/api/health", timeout=0.8) as resp:
                    h = json.loads(resp.read().decode("utf-8"))
                    if h.get("ok") and Path(h.get("workspace", "")).resolve() == store.root:
                        base_url = base_u + "/"
                        q = {k: v for k, v in {"case": ui.get("case"), "recording": ui.get("recording"), "view": ui.get("view")}.items() if v}
                        deep = base_url + ("?" + urllib.parse.urlencode(q) if q else "")
                        if open_browser and get_env("BRAINLENS_NO_BROWSER", "BRAINTOOLKIT_NO_BROWSER") != "1":
                            webbrowser.open(deep)
                        _print_json({"url": base_url, "deep_link": deep, "reused": True, "pid": h.get("pid")})
                        return 0
        except Exception:
            pass

    if daemon:
        cmd = [
            sys.executable,
            "-m",
            "brainlens.cli",
            "--workspace",
            str(store.root),
            "workbench",
            "--host",
            host,
            "--port",
            str(port),
            "--no-open",
        ]
        env = dict(os.environ)
        src_dir = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if meta_file.is_file():
                try:
                    info = read_json(meta_file)
                    with urllib.request.urlopen(str(info["url"]).rstrip("/") + "/api/health", timeout=0.5) as resp:
                        h = json.loads(resp.read().decode("utf-8"))
                        if h.get("ok") and h.get("pid") == proc.pid:
                            base_url = str(info["url"]).rstrip("/") + "/"
                            q = {k: v for k, v in {"case": ui.get("case"), "recording": ui.get("recording"), "view": ui.get("view")}.items() if v}
                            deep = base_url + ("?" + urllib.parse.urlencode(q) if q else "")
                            if open_browser and get_env("BRAINLENS_NO_BROWSER", "BRAINTOOLKIT_NO_BROWSER") != "1":
                                webbrowser.open(deep)
                            _print_json({"url": base_url, "deep_link": deep, "reused": False, "pid": proc.pid})
                            return 0
                except Exception:
                    pass
            time.sleep(0.1)
        raise ToolkitError("workbench_start_timeout", "Timed out waiting for background Workbench server.")

    srv = WorkbenchHTTPServer(workspace_dir=store.root, host=host, port=port)
    url = srv.start()
    q = {k: v for k, v in {"case": ui.get("case"), "recording": ui.get("recording"), "view": ui.get("view")}.items() if v}
    deep = url.rstrip("/") + "/" + ("?" + urllib.parse.urlencode(q) if q else "")
    if open_browser and get_env("BRAINLENS_NO_BROWSER", "BRAINTOOLKIT_NO_BROWSER") != "1":
        try:
            webbrowser.open(deep)
        except Exception:
            pass
    print(f"BrainLens Neurophysiology Workbench running at {deep} (workspace: {store.root})", flush=True)

    stop_flag = False

    def _handle_sig(_signum: int, _frame: Any) -> None:
        nonlocal stop_flag
        stop_flag = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)
    try:
        while not stop_flag:
            time.sleep(0.3)
    finally:
        srv.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="brainlens", description="BrainLens SEEG/iEEG/EEG Neurophysiology Workbench CLI")
    p.add_argument(
        "--workspace",
        default=str(default_workspace_dir()),
        help="Path to workbench results workspace (default: ./workbench_results)",
    )
    sub = p.add_subparsers(dest="command")

    p_ins = sub.add_parser("inspect", help="Inspect workspace or a recording/folder/BIDS directory")
    p_ins.add_argument("target", nargs="?", default="*", help="File/folder path, case name, or '*'")
    p_ins.add_argument("--modality", default=None, help="Optional modality override (seeg, ieeg, scalp)")
    p_ins.add_argument("--spatial-sidecar", default=None, help="Optional spatial coordinate JSON/TSV")

    p_an = sub.add_parser("analyze", help="Run QC + PyHFO + Synchrony & optional ONNX classification")
    p_an.add_argument("target", help="File/folder path or registered case[/recording]")
    p_an.add_argument("--run-name", default=None, help="Branch/run slug (auto-named if omitted)")
    p_an.add_argument("--parent-run", default=None, help="Parent run slug for branching")
    p_an.add_argument("--protocol", default=None, help="Saved protocol name/slug")
    p_an.add_argument("--reason", default=None, help="Clinical/scientific rationale for this run")
    p_an.add_argument("--detector", choices=["STE", "MNI", "HIL"], default=None)
    p_an.add_argument("--band", nargs=2, type=float, metavar=("LOW_HZ", "HIGH_HZ"), default=None)
    p_an.add_argument("--rms-thres", type=float, default=None)
    p_an.add_argument("--reference", choices=["original", "bipolar", "average"], default=None)
    p_an.add_argument("--classifier", action="store_true", help="Enable PyHFO ONNX artifact/spike/eHFO scoring")
    p_an.add_argument("--state", default=None, help="Restrict analysis to recording marker state (e.g. SWS)")
    p_an.add_argument("--exclude-interval", action="append", default=[], help="start:end[:ch1,ch2]")
    p_an.add_argument("--exclude-channel", action="append", default=[], help="Channel name to exclude")
    p_an.add_argument("--overrides", default=None, help="Raw JSON object of config overrides")
    p_an.add_argument("--spatial-sidecar", default=None)
    p_an.add_argument("--overwrite", action="store_true")

    p_q = sub.add_parser("query", help="Query focal/burst rankings, branch sensitivity, concordance, or control windows")
    p_q.add_argument("target", help="Case or case/recording")
    p_q.add_argument("--mode", default="auto", choices=["auto", "recording_summary", "compare_runs", "compare_recordings", "control_windows", "overlay_concordance", "cohort_summary", "timeline", "trends"])
    p_q.add_argument("--run-a", default="baseline")
    p_q.add_argument("--run-b", default=None)
    p_q.add_argument("--recording", default=None)
    p_q.add_argument("--channel", default=None)
    p_q.add_argument("--measure", default="ripple", help="Quantitative navigation trend")
    p_q.add_argument("--epoch-seconds", type=float, default=10)
    p_q.add_argument("--text", help="Filter admission marker labels")

    p_ev = sub.add_parser("evidence", help="Show the per-contact evidence map for a recording and run")
    p_ev.add_argument("case", help="Registered case slug")
    p_ev.add_argument("--recording", required=True)
    p_ev.add_argument("--run-name", default="baseline")

    p_ei = sub.add_parser("ei", help="Compute ictal Epileptogenicity Index for one recording")
    p_ei.add_argument("case", help="Registered case slug")
    p_ei.add_argument("--recording", required=True)
    p_ei.add_argument("--onset-seconds", type=float, help="Seizure onset in recording seconds (default: global onset marker)")

    p_card = sub.add_parser("card", help="Render a 3-panel Clinical Triad PNG card")
    p_card.add_argument("target", help="Case or case/recording")
    p_card.add_argument("--recording", default=None)
    p_card.add_argument("--run-name", default="baseline")
    p_card.add_argument("--channel", default=None)
    p_card.add_argument("--event", default="top_focal", help="top_focal, top_burst, control, spatial, evidence_map, epileptogenicity, or event_id")
    p_card.add_argument("--center", type=float, default=None)
    p_card.add_argument("--window", type=float, default=0.6)

    p_exp = sub.add_parser("export", help="Export a print-ready Presurgical Case Discussion Sheet HTML")
    p_slides = sub.add_parser("export-slides", help="Export the MDT presentation as PowerPoint slides")
    for parser in (p_exp, p_slides):
        parser.add_argument("case", help="Registered case slug")
        parser.add_argument("--primary-run", default="baseline")
        parser.add_argument("--comparison-run", default=None)
        parser.add_argument("--question", default=None)
    p_slides.add_argument("--max-cards", type=int, default=6, help="Maximum number of Clinical Triad cards")

    p_bids = sub.add_parser("export-bids", help="Export BIDS *_events.tsv + *_events.json")
    p_bids.add_argument("case", help="Registered case slug")
    p_bids.add_argument("--recording", default=None)
    p_bids.add_argument("--run-name", default="baseline")
    p_bids.add_argument("--only-confirmed", action="store_true")
    p_bids.add_argument("--include-bursts", action="store_true")

    p_edf = sub.add_parser("export-edf", help="Export annotated EDF+ and annotation CSV for EEG readers")
    p_edf.add_argument("case", help="Registered case slug")
    p_edf.add_argument("--recording", default=None)
    p_edf.add_argument("--run-name", default="baseline")
    p_edf.add_argument("--only-confirmed", action="store_true", help="Include only clinician-confirmed HFO events")
    p_edf.add_argument("--exclude-bursts", action="store_true", help="Omit widespread bursts")

    for parser in (p_exp, p_slides, p_bids, p_edf):
        parser.add_argument("--deidentify", action="store_true", help="Remove native header identity for sharing")
        parser.add_argument("--pseudonym", help="Case label for a de-identified export (e.g. P001)")
    p_sets = sub.add_parser("review-set", help="Freeze or list independent reader event samples")
    sets_sub = p_sets.add_subparsers(dest="set_action", required=True)
    p_create = sets_sub.add_parser("create", help="Freeze all events or a stratified sample")
    p_create.add_argument("--name", required=True, help="Unique review set name within this case")
    p_create.add_argument("--sample-size", type=int, help="Number of events to sample (default: all)")
    p_create.add_argument("--seed", type=int, help="Sampling seed (default: 0)")
    p_list = sets_sub.add_parser("list", help="List frozen sets for a recording and run")
    p_agree = sub.add_parser("agreement", help="Compute inter-rater agreement for one recording")
    p_agree.add_argument("--readers", help="Comma-separated reader names (default: human readers)")
    p_agree.add_argument("--review-set", help="Restrict agreement to a frozen review set")
    for command in (p_create, p_list, p_agree):
        command.add_argument("case", help="Registered case slug")
        command.add_argument("--recording", required=True)
        command.add_argument("--run-name", default="baseline")

    p_coh = sub.add_parser("cohort", help="Run a Group Protocol across multiple cases or list cohorts")
    p_coh.add_argument("targets", nargs="*", help="Case names, paths, or '*' for all registered cases")
    p_coh.add_argument("--protocol", default=None, help="Saved protocol name/slug")
    p_coh.add_argument("--name", default=None, help="Cohort output slug")
    p_coh.add_argument("--run-name", default=None)
    p_coh.add_argument("--list", action="store_true", help="List saved cohorts")
    p_coh.add_argument("--overwrite", action="store_true")

    p_ov = sub.add_parser("overlay", help="Attach external clinical SOZ / Resected overlay (requires explicit --source)")
    p_ov.add_argument("case", help="Registered case slug")
    p_ov.add_argument("--source", required=True, help="Mandatory provenance source citation")
    p_ov.add_argument("--soz", default="", help="Comma-separated SOZ channel names")
    p_ov.add_argument("--resected", default="", help="Comma-separated Resected channel names")
    p_ov.add_argument("--irritative", default="", help="Comma-separated Irritative channel names")
    p_ov.add_argument("--notes", default="")
    p_ov.add_argument("--recording", default=None)
    p_ov.add_argument("--run-name", default="baseline")

    p_anat = sub.add_parser("anatomy", help="Import or clear external anatomical labels for a registered case")
    p_anat.add_argument("case", help="Registered case slug")
    p_anat.add_argument("file", nargs="?", help="Contact/region CSV or BIDS electrodes TSV")
    p_anat.add_argument("--source", help="Required source citation for imported labels")
    p_anat.add_argument("--patient-native", action="store_true", help="Labels come from this patient's own MRI/CT co-registration")
    p_anat.add_argument("--clear", action="store_true", help="Remove the case's external anatomical labels")

    p_proto = sub.add_parser("protocol", help="List or save reusable group analysis protocols")
    p_proto_sub = p_proto.add_subparsers(dest="proto_action", required=True)
    p_proto_sub.add_parser("list", help="List saved protocols")
    p_psave = p_proto_sub.add_parser("save", help="Save protocol from an existing run")
    p_psave.add_argument("name", help="Protocol name")
    p_psave.add_argument("--case", required=True)
    p_psave.add_argument("--recording", default=None)
    p_psave.add_argument("--run-name", default="baseline")
    p_psave.add_argument("--description", default="")

    p_wb = sub.add_parser("workbench", help="Launch the persistent Clinical SEEG Workbench UI")
    p_wb.add_argument("--host", default="127.0.0.1")
    p_wb.add_argument("--port", type=int, default=8765)
    p_wb.add_argument("--no-open", action="store_true")
    p_wb.add_argument("--daemon", action="store_true", help="Start detached background server and exit")
    p_wb.add_argument("--case", default=None)
    p_wb.add_argument("--recording", default=None)
    p_wb.add_argument("--view", choices=["overview", "reader", "sheet"], default=None)

    p_wa = sub.add_parser("watch-actions", help="Wait for clinician annotations/triggers on the interactive MDT Case Sheet (Tier 2 Co-Review Wakeup)")
    p_wa.add_argument("--case", default=None, help="Optional case slug filter")
    p_wa.add_argument("--since", type=int, default=None, help="Only wake on actions with seq > since (defaults to consumer cursor)")
    p_wa.add_argument("--timeout", type=float, default=180.0, help="Max wait seconds (up to 300s)")
    p_wa.add_argument("--trigger-only", action="store_true", help="Wait for an explicit 'Send to Agent' / 'Notify Agent' trigger")
    p_wa.add_argument("--claim", action="store_true", help="Atomically claim returned actions so subsequent waits do not spin")
    p_wa.add_argument("--consumer", default="agent_cli", help="Consumer identity for claim & per-case cursor")

    p_ra = sub.add_parser("resolve-actions", help="Resolve per-case MDT report actions with an Agent reply")
    p_ra.add_argument("case", help="Registered case slug")
    p_ra.add_argument("--id", dest="action_ids", action="append", required=True, help="Action ID to resolve (repeatable)")
    p_ra.add_argument("--reply", default="", help="Agent reply text pushed live to the report Inbox")
    p_ra.add_argument("--outcome", default="resolved", help="Outcome tag (default: resolved)")
    p_ra.add_argument("--consumer", default="agent_cli", help="Resolver identity")

    p_eng = sub.add_parser("configure-engine", help="Configure a packaged PyHFO engine or a legacy checkout")
    p_eng.add_argument("--source", help="Legacy PyHFO checkout (containing src/hfo_app.py)")
    p_eng.add_argument("--python", help="Python interpreter with PyHFO 4.x installed, or interpreter for --source")
    p_eng.add_argument("--fetch-models", action="store_true", help="Download and verify the three pinned ONNX models into the PyHFO hub cache")
    p_eng.add_argument("--model-cache", default=None, help="Optional directory containing PyHFO ONNX model_{a,s,e}/model.onnx + .spec.json")
    p_eng.add_argument("--max-workers", type=int, help="Max worker processes for PyHFO detection/classification (default: 2)")
    p_eng.add_argument("--timeout", type=float, help="Subprocess timeout in seconds (default: 1200)")
    p_eng.add_argument("--scope", choices=["user", "workspace"], help="Save to ~/.brainlens/engine.json (default) or <workspace>/engine.json")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ws = Path(args.workspace).expanduser().resolve()

    try:
        if not args.command or args.command == "workbench":
            return _serve_workbench(
                workspace=ws,
                host=getattr(args, "host", "127.0.0.1"),
                port=getattr(args, "port", 8765),
                open_browser=not getattr(args, "no_open", False),
                daemon=getattr(args, "daemon", False),
                case=getattr(args, "case", None),
                recording=getattr(args, "recording", None),
                view=getattr(args, "view", None),
            )

        if args.command == "inspect":
            if args.target in {"*", "", "list"}:
                store = WorkbenchStore(ws)
                _print_json({
                    "workspace": str(store.root),
                    "registered_cases": [c["case"] for c in store.list_cases()],
                    "available_inputs": discover_available_inputs(),
                    "protocols": [p["slug"] for p in list_protocols(ws)],
                    "engine": check_engine_health(ws),
                })
            else:
                _print_json(inspect_case(args.target, workspace_dir=ws, modality=args.modality, spatial_sidecar=args.spatial_sidecar))
            return 0

        if args.command == "analyze":
            ov: dict[str, Any] = json.loads(args.overrides) if args.overrides else {}
            if args.detector:
                ov["detector"] = args.detector
            if args.band:
                ov["band"] = list(args.band)
            if args.rms_thres is not None:
                ov["rms_thres"] = args.rms_thres
            if args.reference:
                ov["reference_mode"] = args.reference
            if args.classifier:
                ov["classifier_enabled"] = True
            if args.state:
                ov["state"] = args.state
            if args.exclude_interval:
                ov.setdefault("exclude_intervals", []).extend(_parse_interval_spec(x) for x in args.exclude_interval)
            if args.exclude_channel:
                ov.setdefault("exclude_channels", []).extend(args.exclude_channel)

            res = analyze_case(
                target=args.target,
                workspace_dir=ws,
                run_name=args.run_name,
                overrides=ov or None,
                parent_run=args.parent_run,
                protocol=args.protocol,
                reason=args.reason,
                spatial_sidecar=args.spatial_sidecar,
                overwrite=args.overwrite,
            )
            _print_json(res)
            return 0

        if args.command == "review-set":
            from .review_sets import create_review_set, list_review_sets

            store = WorkbenchStore(ws)
            if args.set_action == "create":
                _print_json(create_review_set(store, args.case, args.recording, args.run_name, args.name, args.sample_size, args.seed))
            else:
                _print_json({"review_sets": list_review_sets(store, args.case, args.recording, args.run_name)})
            return 0

        if args.command == "agreement":
            from .agreement import reader_agreement

            readers = [r.strip() for r in args.readers.split(",")] if args.readers else None
            _print_json(reader_agreement(WorkbenchStore(ws), args.case, args.recording, args.run_name, readers, args.review_set))
            return 0

        if args.command == "evidence":
            from .evidence_map import get_evidence_map

            _print_json(get_evidence_map(WorkbenchStore(ws), args.case, recording=args.recording, run_name=args.run_name))
            return 0

        if args.command == "ei":
            from .evidence_map import get_epileptogenicity

            _print_json(get_epileptogenicity(WorkbenchStore(ws), args.case, recording=args.recording, onset_seconds=args.onset_seconds))
            return 0

        if args.command == "query":
            _print_json(
                query_case(
                    target=args.target,
                    workspace_dir=ws,
                    mode=args.mode,
                    run_a=args.run_a,
                    run_b=args.run_b,
                    recording=args.recording,
                    channel=args.channel,
                    measure=args.measure,
                    epoch_seconds=args.epoch_seconds,
                    text=args.text,
                )
            )
            return 0

        if args.command == "card":
            _print_json(
                render_evidence_card(
                    target=args.target,
                    workspace_dir=ws,
                    recording=args.recording,
                    run_name=args.run_name,
                    channel=args.channel,
                    event_id=args.event,
                    center_seconds=args.center,
                    window_seconds=args.window,
                )
            )
            return 0

        if args.command in {"export", "export-slides"}:
            exporter = export_case_sheet
            options = {}
            if args.command == "export-slides":
                from .mdt_slides import export_mdt_slides

                exporter = export_mdt_slides
                options["max_cards"] = args.max_cards
            _print_json(
                exporter(
                    case_slug=args.case,
                    workspace_dir=ws,
                    primary_run=args.primary_run,
                    comparison_run=args.comparison_run,
                    clinical_question=args.question,
                    deidentify=args.deidentify,
                    pseudonym=args.pseudonym,
                    **options,
                )
            )
            return 0

        if args.command == "export-bids":
            _print_json(
                export_bids_events(
                    case_slug=args.case,
                    recording=args.recording,
                    run_name=args.run_name,
                    only_confirmed=args.only_confirmed,
                    exclude_widespread_bursts=not args.include_bursts,
                    deidentify=args.deidentify,
                    pseudonym=args.pseudonym,
                    workspace_dir=ws,
                )
            )
            return 0

        if args.command == "export-edf":
            _print_json(
                export_edf_annotations(
                    case_slug=args.case,
                    recording=args.recording,
                    run_name=args.run_name,
                    only_confirmed=args.only_confirmed,
                    exclude_widespread_bursts=args.exclude_bursts,
                    deidentify=args.deidentify,
                    pseudonym=args.pseudonym,
                    workspace_dir=ws,
                )
            )
            return 0

        if args.command == "cohort":
            if args.list:
                _print_json({"cohorts": list_cohorts(ws)})
            else:
                _print_json(
                    run_cohort(
                        targets=args.targets or "*",
                        protocol=args.protocol,
                        cohort_name=args.name,
                        run_name=args.run_name,
                        workspace_dir=ws,
                        overwrite=args.overwrite,
                    )
                )
            return 0

        if args.command == "anatomy":
            store = WorkbenchStore(ws)
            if not store.has_case(args.case):
                raise ToolkitError("case_not_found", f"Case {args.case!r} is not registered.")
            if args.clear:
                if args.file or args.source or args.patient_native:
                    raise ToolkitError("invalid_input", "Use --clear alone with the registered case name.")
                _print_json({"case": args.case, "cleared": clear_anatomy_labels(store, args.case)})
            else:
                if not args.file:
                    raise ToolkitError("missing_anatomy_file", "Provide a CSV/TSV file or use --clear.")
                payload, counts = read_anatomy_sidecar(args.file, args.source, patient_native=args.patient_native)
                saved = save_anatomy_labels(store, args.case, payload, author="cli")
                _print_json({
                    "anatomy_labels": saved, "anatomy_import": counts,
                    "coverage_by_recording": recording_anatomy_coverage(store, args.case, saved),
                })
            return 0

        if args.command == "overlay":
            store = WorkbenchStore(ws)
            saved = store.save_clinical_overlay(
                args.case,
                {
                    "source": args.source,
                    "soz_channels": args.soz,
                    "resected_channels": args.resected,
                    "irritative_channels": args.irritative,
                    "notes": args.notes,
                },
            )
            recs = store.list_recordings(args.case)
            r = args.recording or (recs[0] if recs else None)
            conc = (
                compute_overlay_concordance(store, args.case, rec_slug=r, run_name=args.run_name)
                if (r and store.has_run(args.case, sanitize_slug(r), sanitize_slug(args.run_name)))
                else None
            )
            _print_json({"overlay": saved, "concordance": conc})
            return 0

        if args.command == "watch-actions":
            import urllib.parse
            import urllib.request
            from .workbench_http import ensure_workbench_server

            store = WorkbenchStore(ws)
            srv_info = ensure_workbench_server(workspace_dir=ws)
            cslug_opt = sanitize_slug(args.case) if args.case else None
            cur = store.load_report_actions(
                case_slug=cslug_opt,
                only_pending=True,
                include_claimed=False,
                limit=1,
                consumer=args.consumer,
            )
            eff_since = args.since if args.since is not None else int(cur.get("consumer_cursor", 0))
            wparams = {
                "since": str(eff_since),
                "timeout": str(max(1.0, min(300.0, float(args.timeout)))),
                "trigger_only": "1" if args.trigger_only else "0",
                "claim": "1" if args.claim else "0",
                "consumer": args.consumer,
            }
            if cslug_opt:
                wparams["case"] = cslug_opt
            wait_url = str(srv_info["url"]).rstrip("/") + "/api/report-actions/wait?" + urllib.parse.urlencode(wparams)
            tok = srv_info.get("auth_token")
            req = urllib.request.Request(wait_url, headers={"X-BrainLens-Token": tok} if tok else {}, method="GET")
            with urllib.request.urlopen(req, timeout=max(5.0, min(310.0, float(args.timeout) + 5.0))) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            _print_json(data)
            return 0

        if args.command == "resolve-actions":
            import urllib.request

            store = WorkbenchStore(ws)
            cslug = sanitize_slug(args.case)
            meta_file = store.root / "workbench_server.json"
            resolved = None
            if meta_file.is_file():
                try:
                    info = read_json(meta_file)
                    tok = info.get("auth_token")
                    base_u = str(info["url"]).rstrip("/")
                    if tok and is_loopback_http_url(base_u):
                        req = urllib.request.Request(
                            base_u + "/api/report-actions/resolve",
                            data=json.dumps({
                                "case": cslug,
                                "action_ids": list(args.action_ids),
                                "agent_reply": args.reply,
                                "outcome": args.outcome,
                                "resolved_by": args.consumer,
                            }, ensure_ascii=False).encode("utf-8"),
                            headers={"Content-Type": "application/json", "X-BrainLens-Token": tok},
                            method="POST",
                        )
                        with urllib.request.urlopen(req, timeout=4.0) as resp:
                            resolved = json.loads(resp.read().decode("utf-8"))
                except Exception:
                    resolved = None
            if resolved is None:
                resolved = store.resolve_report_actions(
                    case_slug=cslug,
                    action_ids=list(args.action_ids),
                    agent_reply=args.reply,
                    outcome=args.outcome,
                    resolved_by=args.consumer,
                )
            _print_json(resolved)
            return 0

        if args.command == "configure-engine":
            if args.python is None:
                register_options = (args.source, args.model_cache, args.max_workers, args.timeout, args.scope)
                if not args.fetch_models or any(value is not None for value in register_options):
                    raise ToolkitError(
                        "engine_python_missing", "Supply --python to save engine settings, or use "
                        "`configure-engine --fetch-models` by itself to populate the active package engine's cache.",
                    )
                runner = PyHFORunner(resolve_engine_settings(ws))
                _print_json({"fetched_models": runner.fetch_models()})
                return 0
            _print_json(
                save_engine_settings(
                    source_root=args.source,
                    python_executable=args.python,
                    model_cache=args.model_cache,
                    max_workers=args.max_workers if args.max_workers is not None else 2,
                    timeout_seconds=args.timeout if args.timeout is not None else 1200.0,
                    scope=args.scope or "user",
                    workspace_dir=ws,
                    fetch_models=args.fetch_models,
                )
            )
            return 0

        if args.command == "protocol":
            if args.proto_action == "list":
                _print_json({"protocols": list_protocols(ws)})
            else:
                _print_json(
                    save_protocol(
                        name=args.name,
                        case_slug=args.case,
                        recording=args.recording,
                        run_name=args.run_name,
                        description=args.description,
                        workspace_dir=ws,
                    )
                )
            return 0

        parser.print_help()
        return 1
    except ToolkitError as exc:
        _print_json({"error": True, "code": exc.code, "message": exc.message, "details": exc.details})
        return 2


def workbench_main() -> None:
    """Entrypoint for `brainlens-workbench`; accepts `--workspace` before or after flags."""
    raw = list(sys.argv[1:])
    global_args: list[str] = []
    sub_args: list[str] = []
    i = 0
    while i < len(raw):
        tok = raw[i]
        if tok == "--workspace" and i + 1 < len(raw):
            global_args.extend([tok, raw[i + 1]])
            i += 2
        elif tok.startswith("--workspace="):
            global_args.append(tok)
            i += 1
        else:
            sub_args.append(tok)
            i += 1
    sys.exit(main([*global_args, "workbench", *sub_args]))


if __name__ == "__main__":
    sys.exit(main())
