"""Tests for native vendor EEG/iEEG format support (discovery, integrity, units, markers, de-id, reader & PyHFO)."""

from __future__ import annotations

import asyncio
import datetime as dt
import importlib.util
import json
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest
import scipy.io

WORKBENCH_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKBENCH_DIR.parent
SRC_DIR = WORKBENCH_DIR / "src"
if importlib.util.find_spec("brainlens") is None and SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlens import mcp_server
from brainlens.bids_export import export_bids_events
from brainlens.case_sheet import export_case_sheet
from brainlens.edf_export import export_edf_annotations
from brainlens.io_integrity import verify_source_recording_integrity
from brainlens.io_readers import _discover_folder_recordings, inspect_case, is_recording_file, open_mne_raw
from brainlens.mdt_slides import export_mdt_slides
from brainlens.models import ToolkitError, read_json
from brainlens.pipeline import analyze_case
from brainlens.pyhfo_runner import check_engine_health
from brainlens.storage import WorkbenchStore
from brainlens.triad_views import compute_multichannel_batch, compute_triad_view

_REPO_SYNTH = PROJECT_ROOT / "inputs" / "synthetic_seeg.edf"
_FIXTURE_SYNTH = Path(__file__).resolve().parent / "fixtures" / "synthetic_seeg.edf"
SYNTHETIC_EDF = _REPO_SYNTH if _REPO_SYNTH.is_file() else _FIXTURE_SYNTH


# ---------------------------------------------------------------------------
# Synthetic vendor recording builders (pure numpy / scipy / struct)
# ---------------------------------------------------------------------------

def _write_synthetic_eeglab(
    folder: Path,
    stem: str,
    *,
    sfreq: float = 500.0,
    n_samples: int = 1500,
    separate_fdt: bool = True,
    subject: str = "MRN_EEG_7741",
    filepath: str = "/Users/secret_clinician/case_alpha",
    trials: int = 1,
) -> tuple[Path, Path | None]:
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(n_samples, dtype=np.float64) / sfreq
    # EEGLAB stores data in uV
    sig_uv = np.vstack([
        50.0 * np.sin(2.0 * np.pi * 10.0 * t),
        35.0 * np.cos(2.0 * np.pi * 14.0 * t),
        np.ones(n_samples, dtype=np.float64) * 5.0,
    ]).astype(np.float32)

    chanlocs_dtype = np.dtype([("labels", "O"), ("type", "O")])
    chanlocs = np.empty(3, dtype=chanlocs_dtype)
    chanlocs[0] = ("LA1", "SEEG")
    chanlocs[1] = ("LA2", "SEEG")
    chanlocs[2] = ("MISC1", "MISC")

    event_dtype = np.dtype([("type", "O"), ("latency", "f8"), ("duration", "f8")])
    events = np.empty(2, dtype=event_dtype)
    events[0] = ("seizure_mark", 251.0, 50.0)  # 1-based sample index in EEGLAB
    events[1] = ("spike_mark", 1001.0, 0.0)

    set_path = folder / f"{stem}.set"
    fdt_path: Path | None = None
    if separate_fdt:
        fdt_name = f"{stem}.fdt"
        fdt_path = folder / fdt_name
        # MNE reads .fdt as little-endian float32 interleaved (samples x channels -> Fortran reshape)
        np.ascontiguousarray(sig_uv.T, dtype="<f4").tofile(fdt_path)
        data_field: Any = fdt_name
    else:
        data_field = sig_uv

    eeg_struct = {
        "setname": "Synthetic EEGLAB",
        "filename": set_path.name,
        "filepath": filepath,
        "subject": subject,
        "group": "",
        "condition": "",
        "session": "",
        "comments": "",
        "nbchan": float(sig_uv.shape[0]),
        "trials": float(trials),
        "pnts": float(n_samples),
        "srate": float(sfreq),
        "xmin": 0.0,
        "xmax": float((n_samples - 1) / sfreq),
        "times": (t * 1000.0).astype(np.float64),
        "data": data_field,
        "chanlocs": chanlocs,
        "event": events,
    }
    scipy.io.savemat(set_path, {"EEG": eeg_struct}, oned_as="row")
    return set_path, fdt_path


def _write_synthetic_persyst(
    folder: Path,
    stem: str,
    *,
    sfreq: int = 500,
    n_samples: int = 1500,
    calibration: float = 0.5,
) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    dat_path = folder / f"{stem}.dat"
    lay_path = folder / f"{stem}.lay"
    t = np.arange(n_samples, dtype=np.float64) / float(sfreq)
    # 3 channels: LA1, LA2, EKG1; stored as interleaved int16 in units of `calibration` uV
    ch0 = np.round((40.0 * np.sin(2.0 * np.pi * 12.0 * t)) / calibration).astype("<i2")
    ch1 = np.round((30.0 * np.cos(2.0 * np.pi * 18.0 * t)) / calibration).astype("<i2")
    ch2 = np.round((100.0 * np.sin(2.0 * np.pi * 1.2 * t)) / calibration).astype("<i2")
    interleaved = np.column_stack([ch0, ch1, ch2]).astype("<i2")
    interleaved.tofile(dat_path)

    lay_text = "\n".join([
        "[FileInfo]",
        f"File={dat_path.name}",
        "FileType=Interleaved",
        f"SamplingRate={sfreq}",
        "HeaderLength=0",
        f"Calibration={calibration}",
        "WaveformCount=3",
        "DataType=0",
        "",
        "[Patient]",
        "First=Jane",
        "Middle=Marie",
        "Last=DoeSecret",
        "FullName=DoeSecret Jane",
        "Sex=F",
        "Hand=R",
        "BirthDate=07/06/74",
        "ID=MRN55821",
        "MedicalRecordN=REC99310",
        "TestDate=2024.05.12",
        "TestTime=10:20:30",
        "Physician=DrHouse",
        "Technician=TechAlex",
        "Medications=Keppra",
        "History=FocalEpilepsy",
        "Comments1=MRN55821 F 06-JUL-1974 DOE_JANE",
        "Comments2=Startdate 12-MAY-2024 HOSP983 TECH77 PERSYST",
        "",
        "[ChannelMap]",
        "LA1-Ref=1",
        "LA2-Ref=2",
        "EKG1-Ref=3",
        "",
        "[Comments]",
        "0.500,0.250,0,65543,seizure onset",
        "1.750,0.100,0,65542,spike burst",
        "",
        "[SampleTimes]",
        "0=0.000",
        "",
    ])
    lay_path.write_text(lay_text, encoding="utf-8")
    return lay_path, dat_path


def _write_synthetic_curry8(
    folder: Path,
    stem: str,
    *,
    sfreq: float = 500.0,
    n_samples: int = 1500,
    n_trials: int = 1,
    include_cef: bool = True,
) -> tuple[Path, Path, Path | None]:
    folder.mkdir(parents=True, exist_ok=True)
    cdt_path = folder / f"{stem}.cdt"
    dpa_path = folder / f"{stem}.cdt.dpa"
    cef_path = folder / f"{stem}.cdt.cef" if include_cef else None

    t = np.arange(n_samples, dtype=np.float32) / float(sfreq)
    data = np.column_stack([
        (45.0 * np.sin(2.0 * np.pi * 10.0 * t)).astype(np.float32),
        (30.0 * np.cos(2.0 * np.pi * 15.0 * t)).astype(np.float32),
        np.ones(n_samples, dtype=np.float32) * 3.0,
    ])
    data.astype("<f4").tofile(cdt_path)

    stime_us = int(round(1e6 / sfreq))
    dpa_text = "\n".join([
        "DATA_PARAMETERS START",
        "   DataByteOrder        = INTEL",
        "   DataSampOrder        = SAMP",
        "   DataFormat           = 6",
        "   FileVersion          = 800",
        f"   NumSamples           = {n_samples}",
        f"   NumTrials            = {n_trials}",
        "   NumChannels          = 3",
        f"   SampleTimeUsec       = {stime_us}",
        f"   SampleFreqHz         = {sfreq:g}",
        "   StartYear            = 2024",
        "   StartMonth           = 5",
        "   StartDay             = 12",
        "   StartHour            = 9",
        "   StartMin             = 30",
        "   StartSec             = 0",
        "   StartMillisec        = 0",
        "   PatientID            = CURRYPAT42",
        "   PatientName          = CurrySecretName",
        r"   FileHistory          = C:\Users\curryuser\clinical\patient.bdf",
        "DATA_PARAMETERS END",
        "",
        "DEVICE_PARAMETERS START",
        "   DataUnit             = uV",
        "   NumChanThisGroup     = 2",
        "   CommonOffset         = 0",
        "   CommonScale          = 1",
        "DEVICE_PARAMETERS END",
        "",
        "DEVICE_PARAMETERS OTHER START",
        "   DataUnit             = 1",
        "   NumChanThisGroup     = 1",
        "   CommonOffset         = 0",
        "   CommonScale          = 1",
        "DEVICE_PARAMETERS OTHER END",
        "",
        "LABELS START_LIST\t# Do not edit!",
        "LA1",
        "LA2",
        "AUX_RAW",
        "LABELS END_LIST",
        "",
    ])
    dpa_path.write_text(dpa_text, encoding="utf-8")

    if cef_path is not None:
        cef_text = "\n".join([
            "NUMBER_LIST START_LIST\t# Do not edit!",
            "  250\t 0\t 11\t-1\t 250\t 300\t 0\t 0\t 0\t 0\t 0",
            "  800\t 0\t 22\t-1\t 800\t 800\t 0\t 0\t 0\t 0\t 0",
            "NUMBER_LIST END_LIST",
            "REMARK_LIST START_LIST\t# Do not edit!",
            "curry_event_alpha",
            "",
            "REMARK_LIST END_LIST",
            "",
        ])
        cef_path.write_text(cef_text, encoding="utf-8")
    return cdt_path, dpa_path, cef_path


def _write_synthetic_curry7(folder: Path, stem: str, *, sfreq: float = 500.0, n_samples: int = 1000) -> tuple[Path, Path, Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    dap_path = folder / f"{stem}.dap"
    dat_path = folder / f"{stem}.dat"
    rs3_path = folder / f"{stem}.rs3"
    cef_path = folder / f"{stem}.cef"

    t = np.arange(n_samples, dtype=np.float32) / float(sfreq)
    data = np.column_stack([
        (25.0 * np.sin(2.0 * np.pi * 8.0 * t)).astype(np.float32),
        (20.0 * np.cos(2.0 * np.pi * 12.0 * t)).astype(np.float32),
    ])
    data.astype("<f4").tofile(dat_path)

    dap_path.write_text(
        "\n".join([
            "DATA_PARAMETERS START",
            "   DataFormat           = FLOAT",
            "   DataExtension        = dat",
            "   DataByteOrder        = INTEL",
            "   DataSampOrder        = SAMP",
            f"   NumSamples           = {n_samples}",
            "   NumTrials            = 1",
            "   NumChannels          = 2",
            f"   SampleTimeUsec       = {int(round(1e6 / sfreq))}",
            f"   SampleFreqHz         = {sfreq:g}",
            "   StartYear            = 2024",
            "   StartMonth           = 5",
            "   StartDay             = 12",
            "   StartHour            = 9",
            "   StartMin             = 0",
            "   StartSec             = 0",
            "   StartMillisec        = 0",
            "DATA_PARAMETERS END",
            "DEVICE_PARAMETERS START",
            "   DataUnit             = uV",
            "   NumChanThisGroup     = 2",
            "   CommonOffset         = 0",
            "   CommonScale          = 1",
            "DEVICE_PARAMETERS END",
            "",
        ]),
        encoding="utf-8",
    )
    rs3_path.write_text(
        "LABELS START_LIST\t# Do not edit!\nLA1\nLA2\nLABELS END_LIST\n",
        encoding="utf-8",
    )
    cef_path.write_text(
        "NUMBER_LIST START_LIST\n  100\t 0\t 5\t-1\t 100\t 120\nNUMBER_LIST END_LIST\n"
        "REMARK_LIST START_LIST\ncurry7_mark\nREMARK_LIST END_LIST\n",
        encoding="utf-8",
    )
    return dap_path, dat_path, rs3_path, cef_path


def _write_synthetic_nicolet(folder: Path, stem: str, *, sfreq: int = 256, n_samples: int = 512) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    data_path = folder / f"{stem}.data"
    head_path = folder / f"{stem}.head"
    t = np.arange(n_samples, dtype=np.float64) / float(sfreq)
    conv = 0.5
    ch0 = np.round(40.0 * np.sin(2.0 * np.pi * 10.0 * t) / conv).astype("<i2")
    ch1 = np.round(30.0 * np.cos(2.0 * np.pi * 15.0 * t) / conv).astype("<i2")
    ch2 = np.round(90.0 * np.sin(2.0 * np.pi * 1.5 * t) / conv).astype("<i2")
    np.column_stack([ch0, ch1, ch2]).astype("<i2").tofile(data_path)
    head_path.write_text(
        "\n".join([
            "start_ts=2023-08-19 14:30:00.000",
            f"num_samples={n_samples}",
            f"sample_freq={sfreq}",
            f"conversion_factor={conv:.6f}",
            "num_channels=3",
            "elec_names=[LA1,LA2,ECG]",
            "pat_id=88421",
            "adm_id=991122",
            "rec_id=775533",
            f"duration_in_sec={n_samples // sfreq}",
            "sample_bytes=2",
            "",
        ]),
        encoding="utf-8",
    )
    return data_path, head_path


def _write_synthetic_gdf2(folder: Path, name: str = "synth_v2.gdf", *, sfreq: int = 250, n_records: int = 3) -> Path:
    """Write a minimal valid GDF 2.20 binary file with 1 uV channel, 1 dimensionless channel, lowpass & 1 event."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    nchan = 2
    hdr_blocks = 1 + nchan  # 256 + 256 * nchan
    hdr_nbytes = hdr_blocks * 256

    fixed = bytearray(256)
    fixed[0:8] = b"GDF 2.20"
    fixed[8:74] = b"MRNGDF77 DOE_ALEX".ljust(66)
    fixed[88:152] = b"Startdate 14-MAR-2024 HOSP44 TECH8".ljust(64)
    # Birthday: 1982-03-14 encoded as GDF 2.x fixed-point days since 0000-01-01
    bday_days = (dt.datetime(1982, 3, 14, tzinfo=dt.UTC) - dt.datetime(1, 1, 1, tzinfo=dt.UTC)).days + 367
    struct.pack_into("<Q", fixed, 176, int(round(bday_days * (2**32))))
    struct.pack_into("<H", fixed, 184, hdr_blocks)
    struct.pack_into("<q", fixed, 236, n_records)
    struct.pack_into("<II", fixed, 244, 1, 1)  # 1 second per record
    struct.pack_into("<H", fixed, 252, nchan)

    var = bytearray()
    # Labels (16 bytes each)
    for lbl in ("LA1", "RAW_AUX"):
        var.extend(lbl.encode("ascii").ljust(16, b" "))
    var.extend(b" " * (80 * nchan))  # transducer
    var.extend(b" " * (6 * nchan))  # obsolete phys_dim
    # Unit codes: 4275 (uV), 512 (dimensionless)
    var.extend(struct.pack("<2H", 4275, 512))
    # phys_min, phys_max, dig_min, dig_max (float64 each)
    var.extend(struct.pack("<2d", -1000.0, -100.0))
    var.extend(struct.pack("<2d", 1000.0, 100.0))
    var.extend(struct.pack("<2d", -32768.0, -32768.0))
    var.extend(struct.pack("<2d", 32767.0, 32767.0))
    var.extend(b"\x00" * (68 * nchan))  # obsolete
    # lowpass, highpass, notch (float32 each)
    var.extend(struct.pack("<2f", 100.0, 100.0))
    var.extend(struct.pack("<2f", 0.5, 0.5))
    var.extend(struct.pack("<2f", 0.0, 0.0))
    # n_samps per record (int32)
    var.extend(struct.pack("<2i", sfreq, sfreq))
    # data type: 3 = INT16
    var.extend(struct.pack("<2i", 3, 3))
    # xyz (3 * float32 per channel)
    var.extend(struct.pack(f"<{3 * nchan}f", *([0.0] * (3 * nchan))))
    # GDF >= 2.19: 5 * float32 per channel (20 bytes per channel)
    var.extend(struct.pack(f"<{5 * nchan}f", *([0.0] * (5 * nchan))))
    assert len(fixed) + len(var) == hdr_nbytes

    body = bytearray()
    for ri in range(n_records):
        t = (ri * sfreq + np.arange(sfreq)) / float(sfreq)
        c0 = np.round(50.0 * np.sin(2.0 * np.pi * 10.0 * t) * 32767.0 / 1000.0).astype("<i2")
        c1 = np.round(10.0 * np.cos(2.0 * np.pi * 5.0 * t) * 32767.0 / 100.0).astype("<i2")
        body.extend(c0.tobytes())
        body.extend(c1.tobytes())

    # Event table (mode 1: pos + typ, 1 event at sample 125, code 777)
    ev = bytearray()
    ev.extend(struct.pack("<B", 1))  # etmode = 1
    ev.extend(struct.pack("<3B", 1, 0, 0))  # n_events = 1 (24-bit LE)
    ev.extend(struct.pack("<f", float(sfreq)))  # event_sr
    ev.extend(struct.pack("<I", 126))  # 1-based pos -> sample 125
    ev.extend(struct.pack("<H", 777))  # typ = 777

    path.write_bytes(bytes(fixed) + bytes(var) + bytes(body) + bytes(ev))
    return path


def _write_synthetic_cnt(folder: Path, name: str = "synth_cnt.cnt", *, sfreq: int = 500, n_samples: int = 1000) -> Path:
    """Write a minimal valid 16-bit Neuroscan CNT binary file with patient header fields, filters & 1 event."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    nchan = 2
    hdr = bytearray(900)
    hdr[0:11] = b"Version 3.0"
    hdr[21:29] = b"MRNCNT91"
    hdr[41:48] = b"DrNeuro"
    hdr[81:89] = b"CityHosp"
    hdr[121:130] = b"Smith_Pat"
    hdr[143:144] = b"M"
    hdr[144:145] = b"R"
    hdr[205:213] = b"SessAlfa"
    hdr[225:233] = b"05/12/24"
    hdr[235:243] = b"11:22:33"
    struct.pack_into("<H", hdr, 370, nchan)
    struct.pack_into("<H", hdr, 376, sfreq)
    struct.pack_into("<b", hdr, 438, 1)  # lowpass_toggle
    struct.pack_into("<b", hdr, 439, 1)  # highpass_toggle
    struct.pack_into("<i", hdr, 864, n_samples)
    struct.pack_into("<f", hdr, 869, 1.0)  # lowcutoff (highpass)
    struct.pack_into("<f", hdr, 875, 100.0)  # highcutoff (lowpass)
    data_start = 900 + 75 * nchan
    event_pos = data_start + n_samples * nchan * 2
    struct.pack_into("<i", hdr, 886, event_pos)
    struct.pack_into("<f", hdr, 890, float(n_samples) / float(sfreq))
    struct.pack_into("<i", hdr, 894, 1)  # channel_offset = 1

    ch_hdrs = bytearray()
    for idx, ch_name in enumerate(("LA1", "LA2")):
        ch = bytearray(75)
        ch[0 : len(ch_name)] = ch_name.encode("ascii")
        struct.pack_into("<ff", ch, 19, -0.05 + idx * 0.1, 0.05)  # x, y
        struct.pack_into("<h", ch, 47, 0)  # baseline
        struct.pack_into("<f", ch, 59, 1.0)  # sensitivity
        struct.pack_into("<f", ch, 71, 204.8)  # cal -> 1 uV/bit
        ch_hdrs.extend(ch)

    t = np.arange(n_samples, dtype=np.float64) / float(sfreq)
    c0 = np.round(40.0 * np.sin(2.0 * np.pi * 10.0 * t)).astype("<i2")
    c1 = np.round(30.0 * np.cos(2.0 * np.pi * 15.0 * t)).astype("<i2")
    body = np.column_stack([c0, c1]).astype("<i2").tobytes()

    # TEEG type 1 with 1 event (StimType=42 at sample 250)
    ev_byte_offset = data_start + 251 * nchan * 2
    ev_entry = struct.pack("<HBcl", 42, 0, b"\x00", ev_byte_offset)
    teeg = struct.pack("<Bll", 1, len(ev_entry), 0)

    path.write_bytes(bytes(hdr) + bytes(ch_hdrs) + body + teeg + ev_entry)
    return path


def _write_synthetic_nihon(
    folder: Path,
    stem: str,
    *,
    sfreq: int = 200,
    duration_sec: int = 3,
    include_log: bool = True,
) -> tuple[Path, Path, Path, Path | None]:
    """Write a minimal valid Nihon Kohden EEG-1100C (.EEG + .PNT + .21E + optional .LOG)."""
    folder.mkdir(parents=True, exist_ok=True)
    version = b"EEG-1100C V01.00"
    eeg_path = folder / f"{stem}.EEG"
    pnt_path = folder / f"{stem}.PNT"
    e21_path = folder / f"{stem}.21E"
    log_path = folder / f"{stem}.LOG" if include_log else None

    # .EEG binary layout (1 control block, 1 data block, 2 EEG channels)
    ctl_addr = 0x0100
    data_addr = 0x0200
    n_ch = 2
    n_samples = duration_sec * sfreq
    data_start = data_addr + 0x27 + n_ch * 10
    total_size = max(0x1800, data_start + n_samples * (n_ch + 1) * 2)
    buf = bytearray(total_size)
    buf[0:16] = version
    buf[0x0081 : 0x0081 + 16] = version
    buf[0x0091] = 1  # n_ctlblocks
    struct.pack_into("<I", buf, 0x0092, ctl_addr)
    buf[ctl_addr + 17] = 1  # n_datablocks
    struct.pack_into("<I", buf, ctl_addr + 18, data_addr)
    struct.pack_into("<H", buf, data_addr + 0x1A, sfreq)
    struct.pack_into("<I", buf, data_addr + 0x1C, duration_sec * 10)  # tenths of a second
    buf[data_addr + 0x26] = n_ch
    buf[data_addr + 0x27 + 0 * 10] = 0  # channel index 0 -> LA1 via .21E
    buf[data_addr + 0x27 + 1 * 10] = 1  # channel index 1 -> LA2 via .21E
    buf[0x17FE] = 1  # waveform_sign (must remain 1; put data after 0x1800 if overlapping)
    # Place data block safely after 0x1800 so waveform_sign at 0x17FE is never overwritten
    data_addr = 0x1800
    data_start = data_addr + 0x27 + n_ch * 10
    total_size = data_start + n_samples * (n_ch + 1) * 2
    buf = bytearray(total_size)
    buf[0:16] = version
    buf[0x0081 : 0x0081 + 16] = version
    buf[0x0091] = 1
    struct.pack_into("<I", buf, 0x0092, ctl_addr)
    buf[ctl_addr + 17] = 1
    struct.pack_into("<I", buf, ctl_addr + 18, data_addr)
    struct.pack_into("<H", buf, data_addr + 0x1A, sfreq)
    struct.pack_into("<I", buf, data_addr + 0x1C, duration_sec * 10)
    buf[data_addr + 0x26] = n_ch
    buf[data_addr + 0x27 + 0 * 10] = 0
    buf[data_addr + 0x27 + 1 * 10] = 1
    buf[0x17FE] = 1

    t = np.arange(n_samples, dtype=np.float64) / float(sfreq)
    # Encode as offset-binary uint16 (value + 0x8000)
    s0 = (np.round(300.0 * np.sin(2.0 * np.pi * 10.0 * t)).astype(np.int32) + 0x8000).astype("<u2")
    s1 = (np.round(250.0 * np.cos(2.0 * np.pi * 12.0 * t)).astype(np.int32) + 0x8000).astype("<u2")
    s_aux = np.full(n_samples, 0x8000, dtype="<u2")
    interleaved = np.column_stack([s0, s1, s_aux]).astype("<u2").tobytes()
    buf[data_start : data_start + len(interleaved)] = interleaved
    eeg_path.write_bytes(bytes(buf))

    # .PNT with self-describing patient table at 0x0400
    pnt = bytearray(0x0800)
    pnt[0:16] = version
    pnt[0x40 : 0x40 + 14] = b"20240512103000"
    pnt[0x0400] = 3
    pnt[0x0401 : 0x0401 + 16] = version
    pnt_entries = [
        ("ID", "NK9021"),
        ("EEG No.", "1"),
        ("Name", "Keller_Eva"),
        ("Sex", "F"),
        ("Date of Birth", "1991/05/14"),
        ("Age", "33"),
        ("Refer Dept", "EpilepsyWard7"),
        ("Physician", "DrVogel"),
    ]
    pnt[0x0411] = len(pnt_entries)
    pos = 0x0412 + len(pnt_entries) * 4
    for idx, (lbl, val) in enumerate(pnt_entries):
        lb = lbl.encode("latin-1") + b"\x00"
        vb = val.encode("latin-1") + b"\x00"
        struct.pack_into("<BBH", pnt, 0x0412 + idx * 4, len(lb), len(vb), 5)
        pnt[pos : pos + len(lb)] = lb
        pos += len(lb)
        pnt[pos : pos + len(vb)] = vb
        pos += len(vb)
    pnt_path.write_bytes(bytes(pnt))

    # .21E electrode label table
    e21_path.write_text("[ELECTRODE]\n0=LA1\n1=LA2\n", encoding="utf-8")

    # .LOG event table (1 log block with 1 event at t=1s)
    if log_path is not None:
        log = bytearray(0x0300)
        log[0:16] = version
        log[0x91] = 1  # n_logblocks
        blk_addr = 0x0100
        struct.pack_into("<I", log, 0x92, blk_addr)
        log[blk_addr + 1 : blk_addr + 17] = version
        log[blk_addr + 0x12] = 1  # n_logs
        entry = b"NK_MARKER_ALPHA".ljust(20, b"\x00") + b"000001" + b"\x00" * 19
        log[blk_addr + 0x14 : blk_addr + 0x14 + 45] = entry
        log_path.write_bytes(bytes(log))

    return eeg_path, pnt_path, e21_path, log_path


def _write_synthetic_micromed(
    folder: Path,
    name: str = "synth_micromed.TRC",
    *,
    sfreq_base: int = 256,
    rate_multipliers: tuple[int, ...] = (1, 1, 1),
    n_samples: int = 512,
    header_version: int = 4,
    surname: str = "RossiSecret",
    firstname: str = "MarcoTest",
    laboratory: str = "MilanEpilepsyLab",
    reserved_id: str = "MRN_MM_4419",
    dob: tuple[int, int, int] = (7, 19, 85),  # month, day, year-1900 -> 1985-07-19
    rec_dt: tuple[int, int, int, int, int, int] = (14, 6, 124, 9, 15, 0),  # 2024-06-14
    segments: list[tuple[int, int]] | None = None,
    unit_code: int = 0,  # 0 = uV
) -> Path:
    """Write a minimal valid Micromed TRC v4 file with patient header, triggers, notes, and Event A epochs."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    n_ch = len(rate_multipliers)
    ch_labels = [f"LA{i + 1}" if i < n_ch - 1 else "ECG1" for i in range(n_ch)]

    zone_names = [
        "ORDER", "LABCOD", "NOTE", "FLAGS", "TRONCA", "IMPED_B", "IMPED_E",
        "MONTAGE", "COMPRESS", "AVERAGE", "HISTORY", "DVIDEO", "EVENT A", "EVENT B", "TRIGGER",
    ]

    order_bytes = struct.pack(f"<{n_ch}H", *range(n_ch))

    labcod_bytes = bytearray(n_ch * 128)
    for c in range(n_ch):
        base = c * 128
        lbl_b = ch_labels[c].encode("ascii")[:6].ljust(6, b"\x00")
        labcod_bytes[base + 2 : base + 8] = lbl_b
        labcod_bytes[base + 8 : base + 14] = b"G2\x00\x00\x00\x00"
        # logical_min=0, logical_max=65535, logical_ground=32768, physical_min=-32768, physical_max=32768 -> 1 uV/bit
        struct.pack_into("<iiiii", labcod_bytes, base + 14, 0, 65535, 32768, -32768, 32768)
        struct.pack_into("<h", labcod_bytes, base + 34, unit_code)
        struct.pack_into("<H", labcod_bytes, base + 44, int(rate_multipliers[c]))

    note_bytes = bytearray()
    for samp_pos, txt in [(64, "seizure note alpha"), (256, "clinical mark beta")]:
        note_bytes.extend(struct.pack("<I40s", samp_pos, txt.encode("ascii").ljust(40, b"\x00")))

    tronca_bytes = bytearray()
    for seg_start, trace_off in (segments or []):
        tronca_bytes.extend(struct.pack("<II", seg_start, trace_off))
    tronca_bytes.extend(struct.pack("<II", 0, 0))

    event_a_bytes = struct.pack("<III", 7, 128, 192)
    trigger_bytes = struct.pack("<IH", 32, 5) + struct.pack("<IH", 384, 9)

    zone_blobs: dict[str, bytes] = {
        "ORDER": order_bytes,
        "LABCOD": bytes(labcod_bytes),
        "NOTE": bytes(note_bytes),
        "FLAGS": b"",
        "TRONCA": bytes(tronca_bytes),
        "IMPED_B": b"",
        "IMPED_E": b"",
        "MONTAGE": b"",
        "COMPRESS": b"",
        "AVERAGE": b"",
        "HISTORY": b"",
        "DVIDEO": b"",
        "EVENT A": event_a_bytes,
        "EVENT B": b"",
        "TRIGGER": trigger_bytes,
    }

    zone_table_start = 176
    cursor = zone_table_start + len(zone_names) * 16
    zone_table = bytearray()
    zone_payload = bytearray()
    for zname in zone_names:
        blob = zone_blobs[zname]
        zpos = cursor
        zlen = len(blob)
        zone_table.extend(struct.pack("<8sII", zname.encode("ascii").ljust(8, b" "), zpos, zlen))
        zone_payload.extend(blob)
        cursor += zlen

    data_start_offset = cursor
    hdr = bytearray(176)
    hdr[0:32] = b"* MICROMED  Brain-Quick file *\x00\x1a"
    hdr[32:64] = laboratory.encode("cp1252")[:32].ljust(32, b" ")
    hdr[64:86] = surname.encode("cp1252")[:22].ljust(22, b" ")
    hdr[86:106] = firstname.encode("cp1252")[:20].ljust(20, b" ")
    struct.pack_into("<BBB", hdr, 106, dob[0], dob[1], dob[2])
    hdr[109:128] = reserved_id.encode("cp1252")[:19].ljust(19, b"\x00")
    struct.pack_into("<bbbbbb", hdr, 128, *rec_dt)
    struct.pack_into("<IHHHH", hdr, 138, data_start_offset, n_ch, n_ch * 2, sfreq_base, 2)
    hdr[175] = header_version & 0xFF

    t = np.arange(n_samples, dtype=np.float64) / float(sfreq_base)
    cols = []
    for c in range(n_ch):
        freq = 8.0 + 4.0 * c
        amp = 45.0 - 5.0 * c
        sig_dig = np.round(amp * np.sin(2.0 * np.pi * freq * t)).astype(np.int32) + 32768
        cols.append(np.clip(sig_dig, 0, 65535).astype("<u2"))
    signal_bytes = np.column_stack(cols).astype("<u2").tobytes()

    path.write_bytes(bytes(hdr) + bytes(zone_table) + bytes(zone_payload) + signal_bytes)
    return path


def _assert_terms_masked(payload: dict[str, Any] | bytes | str, forbidden: list[str]) -> None:
    text = payload.decode("utf-8", errors="ignore") if isinstance(payload, bytes) else (
        payload if isinstance(payload, str) else json.dumps(payload)
    )
    low = text.lower()
    for term in forbidden:
        assert term.lower() not in low, f"Identity term {term!r} leaked into output!"


# ---------------------------------------------------------------------------
# Synthetic end-to-end tests (always executed)
# ---------------------------------------------------------------------------

def test_synthetic_eeglab_fdt_and_singlefile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    ws = tmp_path / "ws"
    mcp_server.set_workspace_root(ws)

    # 1) .set + .fdt pair
    d_pair = tmp_path / "eeglab_pair"
    set_p, fdt_p = _write_synthetic_eeglab(d_pair, "rec_pair", separate_fdt=True)
    assert fdt_p is not None
    discovered = _discover_folder_recordings(d_pair)
    assert [p.name for p in discovered] == ["rec_pair.set"]

    insp = inspect_case(str(d_pair), workspace_dir=ws, case_name="case_eeglab_pair")
    store = WorkbenchStore(ws)
    rec = store.load_recording("case_eeglab_pair", insp["recordings"][0]["recording"])

    with mne.io.read_raw_eeglab(set_p, preload=False, verbose="ERROR") as mraw:
        assert len(rec.channels) == len(mraw.ch_names) == 3
        assert abs(rec.sfreq - float(mraw.info["sfreq"])) < 1e-6
        assert rec.n_samples == int(mraw.n_times) == 1500
        expected_v = mraw.get_data(picks=[0], start=0, stop=250)[0]

    assert str(fdt_p.resolve()) in rec.companion_files
    assert rec.channels[0].kind == "seeg" and rec.channels[0].unit == "V"
    assert rec.channels[2].kind == "misc" and rec.channels[2].unit == "unknown"
    marker_labels = {m.label for m in rec.markers}
    assert {"seizure_mark", "spike_mark"}.issubset(marker_labels)

    batch = compute_multichannel_batch(store, "case_eeglab_pair", rec.slug, start_seconds=0.0, duration_seconds=0.5, page_size=4)
    assert len(batch["traces"]) > 0
    np.testing.assert_allclose(np.asarray(batch["traces"][0]["values_uv"]), expected_v * 1e6, rtol=1e-3, atol=1e-2)

    # MCP output must scrub EEGLAB subject and filepath
    mcp_resp = mcp_server.inspect_case(target="case_eeglab_pair")
    _assert_terms_masked(mcp_resp, ["MRN_EEG_7741", "secret_clinician"])

    # Modifying companion .fdt must be detected
    raw_bytes = bytearray(fdt_p.read_bytes())
    raw_bytes[0] ^= 0xFF
    fdt_p.write_bytes(bytes(raw_bytes))
    with pytest.raises(ToolkitError) as exc_mod:
        verify_source_recording_integrity(rec)
    assert exc_mod.value.code == "source_recording_modified"

    # Deleting companion .fdt must be detected
    fdt_p.unlink()
    with pytest.raises(ToolkitError) as exc_miss:
        verify_source_recording_integrity(rec)
    assert exc_miss.value.code == "source_recording_missing"

    # 2) Single-file embedded .set (no .fdt) + epoched rejection
    d_one = tmp_path / "eeglab_one"
    set_one, _ = _write_synthetic_eeglab(d_one, "rec_one", separate_fdt=False)
    insp_one = inspect_case(str(set_one), workspace_dir=ws, case_name="case_eeglab_one")
    rec_one = store.load_recording("case_eeglab_one", insp_one["recordings"][0]["recording"])
    assert rec_one.companion_files == {}
    assert rec_one.n_samples == 1500

    d_ep = tmp_path / "eeglab_epoched"
    set_ep, _ = _write_synthetic_eeglab(d_ep, "rec_ep", separate_fdt=False, trials=2)
    with pytest.raises(ToolkitError) as exc_ep:
        inspect_case(str(set_ep), workspace_dir=ws)
    assert exc_ep.value.code == "unsupported_format"


def test_synthetic_persyst_companions_markers_deid_and_exports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    ws = tmp_path / "ws"
    mcp_server.set_workspace_root(ws)

    d = tmp_path / "persyst_dir"
    lay_p, dat_p = _write_synthetic_persyst(d, "clinical_clip")
    assert [p.name for p in _discover_folder_recordings(d)] == ["clinical_clip.lay"]

    insp = inspect_case(str(d), workspace_dir=ws, case_name="case_persyst")
    store = WorkbenchStore(ws)
    rec = store.load_recording("case_persyst", insp["recordings"][0]["recording"])

    assert str(dat_p.resolve()) in rec.companion_files
    assert [c.name for c in rec.channels] == ["LA1", "LA2", "EKG1"]
    assert all(c.unit == "V" for c in rec.channels)
    assert rec.channels[0].shaft == "LA" and rec.channels[2].is_auxiliary is True
    assert {"seizure onset", "spike burst"}.issubset({m.label for m in rec.markers})

    batch = compute_multichannel_batch(store, "case_persyst", rec.slug, start_seconds=0.0, duration_seconds=1.0, page_size=4)
    assert [tr["channel_name"] for tr in batch["traces"]] == ["LA1", "LA2"]

    # MCP output masks every Persyst [Patient] identity field
    mcp_resp = mcp_server.inspect_case(target="case_persyst")
    forbidden = ["MRN55821", "REC99310", "DoeSecret", "Jane", "DrHouse", "TechAlex", "1974-07-06", "06-JUL-1974", "HOSP983", "2024.05.12"]
    _assert_terms_masked(mcp_resp, forbidden)

    # Downstream pipeline + de-identified exports work unchanged
    analyze_case("case_persyst", workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    sheet = export_case_sheet("case_persyst", workspace_dir=ws, primary_run="baseline", max_cards=0, deidentify=True, pseudonym="PP01")
    slides = export_mdt_slides("case_persyst", workspace_dir=ws, primary_run="baseline", deidentify=True, pseudonym="PP01")
    bids = export_bids_events("case_persyst", workspace_dir=ws, run_name="baseline", deidentify=True, pseudonym="PP01")
    edfa = export_edf_annotations("case_persyst", workspace_dir=ws, run_name="baseline", deidentify=True, pseudonym="PP01")
    assert Path(slides["slides_path"]).is_file()
    for p_str in (sheet["report_path"], bids["json_path"], edfa["json_path"], edfa["csv_path"]):
        _assert_terms_masked(Path(p_str).read_bytes(), forbidden)

    # Tampering with .dat detected on both reader and integrity check
    b = bytearray(dat_p.read_bytes())
    b[4] ^= 0x55
    dat_p.write_bytes(bytes(b))
    with pytest.raises(ToolkitError) as exc_mod:
        compute_multichannel_batch(store, "case_persyst", rec.slug, start_seconds=0.0, duration_seconds=0.5)
    assert exc_mod.value.code == "source_recording_modified"


def test_synthetic_curry8_and_curry7(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    ws = tmp_path / "ws"
    mcp_server.set_workspace_root(ws)

    # Curry 8 (.cdt + .cdt.dpa + .cdt.cef)
    d8 = tmp_path / "curry8_dir"
    cdt_p, dpa_p, cef_p = _write_synthetic_curry8(d8, "rec8")
    assert cef_p is not None
    assert [p.name for p in _discover_folder_recordings(d8)] == ["rec8.cdt"]

    insp8 = inspect_case(str(d8), workspace_dir=ws, case_name="case_curry8")
    store = WorkbenchStore(ws)
    rec8 = store.load_recording("case_curry8", insp8["recordings"][0]["recording"])
    assert str(dpa_p.resolve()) in rec8.companion_files
    assert str(cef_p.resolve()) in rec8.companion_files
    assert rec8.channels[0].unit == "V" and rec8.channels[1].unit == "V"
    assert rec8.channels[2].unit == "dimensionless"  # DataUnit = 1 -> unverified
    assert any("unverified" in w for w in rec8.warnings)
    assert {"curry_event_alpha", "22"}.issubset({m.label for m in rec8.markers})

    batch8 = compute_multichannel_batch(store, "case_curry8", rec8.slug, start_seconds=0.0, duration_seconds=1.0, page_size=4)
    assert [tr["channel_name"] for tr in batch8["traces"]] == ["LA1", "LA2"]

    mcp8 = mcp_server.inspect_case(target="case_curry8")
    _assert_terms_masked(mcp8, ["CURRYPAT42", "CurrySecretName", "curryuser"])

    # Changing .cdt.cef companion detected
    cef_p.write_text(cef_p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ToolkitError) as exc_cef:
        verify_source_recording_integrity(rec8)
    assert exc_cef.value.code == "source_recording_modified"

    # Epoched Curry 8 rejected cleanly
    d8_ep = tmp_path / "curry8_ep"
    cdt_ep, _, _ = _write_synthetic_curry8(d8_ep, "ep8", n_trials=2)
    with pytest.raises(ToolkitError) as exc_ep:
        inspect_case(str(cdt_ep), workspace_dir=ws)
    assert exc_ep.value.code == "unsupported_format"

    # Curry 7 (.dap + .dat + .rs3 + .cef)
    d7 = tmp_path / "curry7_dir"
    dap_p, dat_p, rs3_p, cef7_p = _write_synthetic_curry7(d7, "rec7")
    assert [p.name for p in _discover_folder_recordings(d7)] == ["rec7.dap"]
    insp7 = inspect_case(str(d7), workspace_dir=ws, case_name="case_curry7")
    rec7 = store.load_recording("case_curry7", insp7["recordings"][0]["recording"])
    for comp in (dat_p, rs3_p, cef7_p):
        assert str(comp.resolve()) in rec7.companion_files
    assert {"curry7_mark"}.issubset({m.label for m in rec7.markers})
    batch7 = compute_multichannel_batch(store, "case_curry7", rec7.slug, start_seconds=0.0, duration_seconds=1.0, page_size=4)
    assert len(batch7["traces"]) == 2
    rs3_p.unlink()
    with pytest.raises(ToolkitError) as exc_rs3:
        verify_source_recording_integrity(rec7)
    assert exc_rs3.value.code == "source_recording_missing"


def test_synthetic_nicolet_gdf_cnt_and_nihon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    ws = tmp_path / "ws"
    mcp_server.set_workspace_root(ws)
    store = WorkbenchStore(ws)

    # 1) Nicolet (.data + .head) and unrelated .data ignored
    d_nic = tmp_path / "nicolet_dir"
    data_p, head_p = _write_synthetic_nicolet(d_nic, "nic_rec")
    (d_nic / "unrelated.data").write_bytes(b"not nicolet")
    assert [p.name for p in _discover_folder_recordings(d_nic)] == ["nic_rec.data"]
    insp_nic = inspect_case(str(d_nic), workspace_dir=ws, case_name="case_nicolet")
    rec_nic = store.load_recording("case_nicolet", insp_nic["recordings"][0]["recording"])
    assert str(head_p.resolve()) in rec_nic.companion_files
    assert [c.name for c in rec_nic.channels] == ["LA1", "LA2", "ECG"]
    batch_nic = compute_multichannel_batch(store, "case_nicolet", rec_nic.slug, start_seconds=0.0, duration_seconds=1.0)
    assert [tr["channel_name"] for tr in batch_nic["traces"]] == ["LA1", "LA2"]
    _assert_terms_masked(mcp_server.inspect_case(target="case_nicolet"), ["88421", "991122", "775533"])
    head_p.write_text(head_p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ToolkitError) as exc_head:
        verify_source_recording_integrity(rec_nic)
    assert exc_head.value.code == "source_recording_modified"

    # 2) GDF 2.x (verified uV vs dimensionless unit skipping + lowpass + events + de-id)
    d_gdf = tmp_path / "gdf_dir"
    gdf_p = _write_synthetic_gdf2(d_gdf)
    insp_gdf = inspect_case(str(gdf_p), workspace_dir=ws, case_name="case_gdf")
    rec_gdf = store.load_recording("case_gdf", insp_gdf["recordings"][0]["recording"])
    assert rec_gdf.channels[0].unit == "V" and rec_gdf.channels[0].source_lowpass == pytest.approx(100.0)
    assert rec_gdf.channels[1].unit == "dimensionless"
    assert any("unverified" in w for w in rec_gdf.warnings)
    assert any(m.label == "777" for m in rec_gdf.markers)
    batch_gdf = compute_multichannel_batch(store, "case_gdf", rec_gdf.slug, start_seconds=0.0, duration_seconds=1.0)
    assert [tr["channel_name"] for tr in batch_gdf["traces"]] == ["LA1"]
    _assert_terms_masked(mcp_server.inspect_case(target="case_gdf"), ["MRNGDF77", "DOE_ALEX", "1982-03-14", "HOSP44"])

    # 3) Neuroscan CNT (filters + alphanumeric patient_id + events + de-id)
    d_cnt = tmp_path / "cnt_dir"
    cnt_p = _write_synthetic_cnt(d_cnt)
    insp_cnt = inspect_case(str(cnt_p), workspace_dir=ws, case_name="case_cnt")
    rec_cnt = store.load_recording("case_cnt", insp_cnt["recordings"][0]["recording"])
    assert rec_cnt.highpass == pytest.approx(1.0) and rec_cnt.lowpass == pytest.approx(100.0)
    assert any(m.label == "42" for m in rec_cnt.markers)
    batch_cnt = compute_multichannel_batch(store, "case_cnt", rec_cnt.slug, start_seconds=0.0, duration_seconds=1.0)
    assert len(batch_cnt["traces"]) == 2
    _assert_terms_masked(mcp_server.inspect_case(target="case_cnt"), ["MRNCNT91", "Smith_Pat", "DrNeuro", "CityHosp"])

    # 4) Nihon Kohden (.EEG + .PNT + .21E + .LOG) + BrainVision .eeg non-collision + new-companion guard
    d_nk = tmp_path / "nihon_dir"
    eeg_p, pnt_p, e21_p, log_p = _write_synthetic_nihon(d_nk, "nk_rec", include_log=True)
    assert log_p is not None
    # Put a fake BrainVision .eeg next to it — must NOT be discovered as Nihon Kohden!
    bv_eeg = d_nk / "brainvision_data.eeg"
    bv_eeg.write_bytes(b"\x01\x02\x03\x04" * 64)
    assert is_recording_file(eeg_p) is True
    assert is_recording_file(bv_eeg) is False
    assert [p.name for p in _discover_folder_recordings(d_nk)] == ["nk_rec.EEG"]

    insp_nk = inspect_case(str(d_nk), workspace_dir=ws, case_name="case_nihon")
    rec_nk = store.load_recording("case_nihon", insp_nk["recordings"][0]["recording"])
    for comp in (pnt_p, e21_p, log_p):
        assert str(comp.resolve()) in rec_nk.companion_files
    assert [c.name for c in rec_nk.channels] == ["LA1", "LA2"]
    assert any(m.label == "NK_MARKER_ALPHA" for m in rec_nk.markers)
    batch_nk = compute_multichannel_batch(store, "case_nihon", rec_nk.slug, start_seconds=0.0, duration_seconds=1.0)
    assert len(batch_nk["traces"]) == 2
    _assert_terms_masked(
        mcp_server.inspect_case(target="case_nihon"),
        ["NK9021", "Keller_Eva", "Keller", "1991/05/14", "1991-05-14", "EpilepsyWard7", "DrVogel"],
    )

    # Tampering with .21E detected
    e21_p.write_text("[ELECTRODE]\n0=CHANGED1\n1=LA2\n", encoding="utf-8")
    with pytest.raises(ToolkitError) as exc_21e:
        verify_source_recording_integrity(rec_nk)
    assert exc_21e.value.code == "source_recording_modified"

    # Adding a brand-new .LOG after registration (when absent initially) is also detected
    d_nk2 = tmp_path / "nihon_nolog"
    eeg2_p, _, _, _ = _write_synthetic_nihon(d_nk2, "nk2", include_log=False)
    insp_nk2 = inspect_case(str(eeg2_p), workspace_dir=ws, case_name="case_nk2")
    rec_nk2 = store.load_recording("case_nk2", insp_nk2["recordings"][0]["recording"])
    (d_nk2 / "nk2.LOG").write_bytes(b"EEG-1100C V01.00" + b"\x00" * 256)
    with pytest.raises(ToolkitError) as exc_newlog:
        verify_source_recording_integrity(rec_nk2)
    assert exc_newlog.value.code == "source_recording_modified"


def test_sidecar_and_bids_channel_overrides_on_vendor_format(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    d = tmp_path / "sub-09" / "ieeg"
    lay_p, _ = _write_synthetic_persyst(d, "sub-09_task-rest_ieeg")
    # Strip patient IDs from synthetic .lay so BIDS sub-09 has no subject conflict
    txt = (
        lay_p.read_text(encoding="utf-8")
        .replace("ID=MRN55821", "ID=0")
        .replace("MedicalRecordN=REC99310", "MedicalRecordN=")
        .replace("FullName=DoeSecret Jane", "FullName=No_Name")
        .replace("First=Jane", "First=")
        .replace("Last=DoeSecret", "Last=")
        .replace("Middle=Marie", "Middle=")
    )
    lay_p.write_text(txt, encoding="utf-8")

    (d / "sub-09_task-rest_channels.tsv").write_text(
        "name\ttype\tstatus\nLA1\tECOG\tgood\nLA2\tSEEG\tbad\nEKG1\tECG\tgood\n",
        encoding="utf-8",
    )
    insp = inspect_case(str(d), workspace_dir=ws, case_name="case_bids_vendor")
    store = WorkbenchStore(ws)
    rec = store.load_recording("case_bids_vendor", insp["recordings"][0]["recording"])
    by_name = {c.name: c for c in rec.channels}
    assert by_name["LA1"].kind == "ecog"
    assert by_name["LA2"].kind == "seeg"
    assert by_name["EKG1"].kind == "ecg"
    assert "LA2" in insp["recordings"][0]["declared_bad_channels"]


def test_unsupported_vendor_formats_clear_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = tmp_path / "ws"

    # EGI .mff directory (direct & inside parent folder)
    mff_dir = tmp_path / "study.mff"
    mff_dir.mkdir()
    (mff_dir / "info.xml").write_text("<fileInfo/>", encoding="utf-8")
    with pytest.raises(ToolkitError) as exc_mff:
        inspect_case(str(mff_dir), workspace_dir=ws)
    assert exc_mff.value.code == "unsupported_format"
    assert "EDF+" in exc_mff.value.message

    parent_mff = tmp_path / "only_mff"
    (parent_mff / "rec.mff").mkdir(parents=True)
    with pytest.raises(ToolkitError) as exc_mff_dir:
        inspect_case(str(parent_mff), workspace_dir=ws)
    assert exc_mff_dir.value.code == "unsupported_format"
    assert "EDF+" in exc_mff_dir.value.message

    with pytest.raises(ToolkitError) as exc_mff_raw:
        open_mne_raw(mff_dir)
    assert exc_mff_raw.value.code == "unsupported_format"

    # Natus/XLTEK .erd, NicoletOne .e, non-Nihon .eeg
    for fname, expected_substr in [
        ("natus_rec.erd", "NeuroWorks"),
        ("nicolet_native.e", "EDF+"),
        ("brainvision_raw.eeg", ".vhdr"),
    ]:
        fp = tmp_path / fname
        fp.write_bytes(b"placeholder bytes")
        with pytest.raises(ToolkitError) as exc:
            inspect_case(str(fp), workspace_dir=ws)
        assert exc.value.code == "unsupported_format"
        assert expected_substr in exc.value.message

    # Micromed .trc without optional `neo` extra must refuse with install hint
    monkeypatch.setitem(sys.modules, "neo", None)
    monkeypatch.setitem(sys.modules, "neo.rawio", None)
    monkeypatch.setitem(sys.modules, "neo.rawio.micromedrawio", None)
    trc_fp = tmp_path / "micromed_rec.trc"
    trc_fp.write_bytes(b"placeholder bytes")
    with pytest.raises(ToolkitError) as exc_trc:
        inspect_case(str(trc_fp), workspace_dir=ws)
    assert exc_trc.value.code == "unsupported_format"
    assert 'pip install "brainlens[micromed]"' in exc_trc.value.message
    assert "EDF+" in exc_trc.value.message

    trc_dir = tmp_path / "only_trc"
    trc_dir.mkdir()
    (trc_dir / "rec.TRC").write_bytes(b"placeholder bytes")
    with pytest.raises(ToolkitError) as exc_trc_dir:
        inspect_case(str(trc_dir), workspace_dir=ws)
    assert exc_trc_dir.value.code == "unsupported_format"
    assert 'pip install "brainlens[micromed]"' in exc_trc_dir.value.message


def test_malformed_vendor_headers_hide_header_and_exception_text(tmp_path: Path) -> None:
    """Malformed headers with planted identity must return `vendor_header_unreadable` without leaking field values."""
    ws = tmp_path / "ws"
    mcp_server.set_workspace_root(ws)

    # 1) Nicolet .head with non-integer pat_id and conversion_factor containing planted identity
    d_nic = tmp_path / "bad_nicolet"
    nic_data, nic_head = _write_synthetic_nicolet(d_nic, "bad_nic")
    nic_head.write_text(nic_head.read_text(encoding="utf-8").replace("pat_id=88421", "pat_id=PX77812"), encoding="utf-8")
    resp_nic = mcp_server.inspect_case(target=str(nic_data))
    assert resp_nic.get("error") is True
    assert resp_nic["code"] == "vendor_header_unreadable"
    assert "nicolet" in resp_nic["message"].lower()
    assert "ValueError" in resp_nic["message"]
    _assert_terms_masked(resp_nic, ["PX77812", "invalid literal"])

    nic_head.write_text(
        nic_head.read_text(encoding="utf-8").replace("conversion_factor=0.500000", "conversion_factor=PX99104"),
        encoding="utf-8",
    )
    resp_nic2 = mcp_server.inspect_case(target=str(nic_data))
    assert resp_nic2["code"] == "vendor_header_unreadable"
    _assert_terms_masked(resp_nic2, ["PX99104", "could not convert"])

    # 2) Persyst .lay with planted identity in Calibration and SamplingRate
    d_per = tmp_path / "bad_persyst"
    lay_p, _ = _write_synthetic_persyst(d_per, "bad_per")
    orig_lay = lay_p.read_text(encoding="utf-8")
    lay_p.write_text(orig_lay.replace("Calibration=0.5", "Calibration=PERSYST_ID_6621"), encoding="utf-8")
    resp_per1 = mcp_server.inspect_case(target=str(lay_p))
    assert resp_per1["code"] == "vendor_header_unreadable"
    assert "persyst" in resp_per1["message"].lower()
    assert "ValueError" in resp_per1["message"]
    _assert_terms_masked(resp_per1, ["PERSYST_ID_6621", "could not convert"])

    lay_p.write_text(orig_lay.replace("SamplingRate=500", "SamplingRate=PERSYST_RATE_4412"), encoding="utf-8")
    resp_per2 = mcp_server.inspect_case(target=str(lay_p))
    assert resp_per2["code"] == "vendor_header_unreadable"
    assert "persyst" in resp_per2["message"].lower()
    _assert_terms_masked(resp_per2, ["PERSYST_RATE_4412", "could not convert", "invalid literal"])

    # 3) Curry 8 fallback with planted identity in NumSamples
    d_cur = tmp_path / "bad_curry"
    cdt_p, dpa_p, _ = _write_synthetic_curry8(d_cur, "bad_cur")
    dpa_p.write_text(
        dpa_p.read_text(encoding="utf-8").replace("NumSamples           = 1500", "NumSamples           = CURRY_ID_8831"),
        encoding="utf-8",
    )
    resp_cur = mcp_server.inspect_case(target=str(cdt_p))
    assert resp_cur["code"] == "vendor_header_unreadable"
    assert "curry" in resp_cur["message"].lower()
    assert "ValueError" in resp_cur["message"]
    _assert_terms_masked(resp_cur, ["CURRY_ID_8831", "could not convert"])

    # 4) GDF with planted identity in magic header
    bad_gdf = tmp_path / "bad_gdf.gdf"
    bad_gdf.write_bytes(b"GDF_ID99" + b"\x00" * 512)
    resp_gdf = mcp_server.inspect_case(target=str(bad_gdf))
    assert resp_gdf["code"] == "vendor_header_unreadable"
    assert "gdf" in resp_gdf["message"].lower()
    _assert_terms_masked(resp_gdf, ["GDF_ID99"])

    # 5) Micromed .TRC with planted identity in malformed header (tested when optional neo is installed)
    if importlib.util.find_spec("neo") is not None:
        d_mm = tmp_path / "bad_micromed"
        mm_p = _write_synthetic_micromed(d_mm, "bad_mm.TRC", surname="MICRO_ID_7741")
        raw_b = bytearray(mm_p.read_bytes())
        # Corrupt the first zone name ("ORDER   ") with planted identity bytes
        raw_b[176:184] = b"MMID7741"
        mm_p.write_bytes(bytes(raw_b))
        resp_mm = mcp_server.inspect_case(target=str(mm_p))
        assert resp_mm["code"] == "vendor_header_unreadable"
        assert "micromed" in resp_mm["message"].lower()
        assert "NeoReadWriteError" in resp_mm["message"]
        _assert_terms_masked(resp_mm, ["MICRO_ID_7741", "MMID7741", "zone name"])

        # Also corrupt recording month byte (offset 129 = 99) so datetime raises ValueError
        mm_p2 = _write_synthetic_micromed(d_mm, "bad_mm_date.TRC", surname="MICRO_DATE_5512", rec_dt=(14, 99, 124, 9, 0, 0))
        resp_mm2 = mcp_server.inspect_case(target=str(mm_p2))
        assert resp_mm2["code"] == "vendor_header_unreadable"
        assert "micromed" in resp_mm2["message"].lower()
        assert "ValueError" in resp_mm2["message"]
        _assert_terms_masked(resp_mm2, ["MICRO_DATE_5512", "month must be"])


def test_companion_change_error_preserves_non_identifying_filenames_and_masks_identifying_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Companion errors must keep real filenames unless the recording's own header identity appears in the filename."""
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    ws = tmp_path / "ws"
    mcp_server.set_workspace_root(ws)

    # Register an unrelated Nihon Kohden case whose PNT Name contains "Test_no_cap Asdf" and another with surname "Chan"
    d_nk = tmp_path / "nk_case"
    eeg_p, pnt_p, _, _ = _write_synthetic_nihon(d_nk, "MB0400FU")
    # Replace Name="Keller_Eva" in synthetic PNT with same-length "Test_no_cap" style + register second case named "Chan"
    pnt_bytes = pnt_p.read_bytes().replace(b"Keller_Eva", b"Test_alpha")
    pnt_p.write_bytes(pnt_bytes)
    assert not mcp_server.inspect_case(target=str(eeg_p)).get("error")

    d_chan = tmp_path / "chan_case"
    lay_chan, _ = _write_synthetic_persyst(d_chan, "unrelated_rec")
    lay_chan.write_text(
        lay_chan.read_text(encoding="utf-8").replace("DoeSecret", "Chan"),
        encoding="utf-8",
    )
    assert not mcp_server.inspect_case(target=str(lay_chan)).get("error")

    # Register EEGLAB test_raw_chanloc.set, Curry 8 test_bdf_stim_channel Curry 8.cdt, Nicolet test_nicolet_raw.data
    d_ee = tmp_path / "ee_case"
    set_p, fdt_p = _write_synthetic_eeglab(d_ee, "test_raw_chanloc", separate_fdt=True)
    assert fdt_p is not None
    r_ee = mcp_server.inspect_case(target=str(set_p))
    assert not r_ee.get("error")

    d_c8 = tmp_path / "c8_case"
    cdt_p, dpa_p, _ = _write_synthetic_curry8(d_c8, "test_bdf_stim_channel Curry 8")
    r_c8 = mcp_server.inspect_case(target=str(cdt_p))
    assert not r_c8.get("error")

    d_ni = tmp_path / "ni_case"
    data_p, head_p = _write_synthetic_nicolet(d_ni, "test_nicolet_raw")
    r_ni = mcp_server.inspect_case(target=str(data_p))
    assert not r_ni.get("error")

    # Register a Persyst case whose OWN companion filename contains its OWN patient surname ("DoeSecret")
    d_id = tmp_path / "id_in_fname"
    lay_id, dat_id = _write_synthetic_persyst(d_id, "DoeSecret_clip")
    inspect_case(str(lay_id), workspace_dir=ws, case_name="case_id_in_fname")

    # Mutate each non-identifying companion: MCP error MUST keep the real companion filename (no cross-case over-masking)
    for comp_p, case_slug in [
        (fdt_p, r_ee["case"]),
        (dpa_p, r_c8["case"]),
        (head_p, r_ni["case"]),
    ]:
        b = bytearray(comp_p.read_bytes())
        b[-1] ^= 0xFF
        comp_p.write_bytes(bytes(b))
        err = mcp_server.analyze_case(target=case_slug)
        assert err.get("error") is True and err["code"] == "source_recording_modified"
        assert comp_p.name in err["message"], f"Expected real filename {comp_p.name!r}, got: {err['message']!r}"
        assert "withheld" not in err["message"], f"Over-masked filename in: {err['message']!r}"

    # Mutate the companion whose filename DOES contain its own recording's identity: MUST mask the identity
    b_id = bytearray(dat_id.read_bytes())
    b_id[-1] ^= 0xFF
    dat_id.write_bytes(bytes(b_id))
    err_id = mcp_server.analyze_case(target="case_id_in_fname")
    assert err_id.get("error") is True and err_id["code"] == "source_recording_modified"
    assert "DoeSecret" not in json.dumps(err_id)
    assert "header subject ID withheld" in err_id["message"]


def test_synthetic_micromed_lazy_reader_markers_deid_and_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    neo = pytest.importorskip("neo")
    from neo.rawio.micromedrawio import MicromedRawIO

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    ws = tmp_path / "ws"
    mcp_server.set_workspace_root(ws)
    store = WorkbenchStore(ws)

    # 1) Continuous single-segment synthetic TRC with identity, triggers, notes, and Event A epochs
    d_mm = tmp_path / "mm_synth"
    trc_p = _write_synthetic_micromed(d_mm, "clinical_seeg.TRC")
    assert [p.name for p in _discover_folder_recordings(d_mm)] == ["clinical_seeg.TRC"]

    with open_mne_raw(trc_p, preload=False) as raw:
        assert raw.preload is False
        full_v = raw.get_data()
        sub_v = raw.get_data(picks=[0, 2], start=50, stop=200)
        np.testing.assert_allclose(sub_v, full_v[[0, 2], 50:200], rtol=0, atol=0)
        # Verify projection (`mult is not None`) works cleanly through _read_segment_file
        raw_proj = raw.copy()
        raw_proj.set_eeg_reference("average", projection=True, verbose="ERROR")
        proj_data = raw_proj.get_data(start=0, stop=64)
        assert proj_data.shape == (3, 64) and np.all(np.isfinite(proj_data))

    r_neo = MicromedRawIO(str(trc_p))
    r_neo.parse_header()
    raw_chunk = r_neo.get_analogsignal_chunk(block_index=0, seg_index=0, stream_index=0)
    ref_v = (r_neo.rescale_signal_raw_to_float(raw_chunk, dtype="float64", stream_index=0) * 1e-6).T
    np.testing.assert_allclose(full_v, ref_v, rtol=1e-12, atol=1e-15)
    blk = neo.MicromedIO(str(trc_p)).read_block()
    np.testing.assert_allclose(full_v, blk.segments[0].analogsignals[0].rescale("V").magnitude.T, rtol=1e-6, atol=1e-9)

    insp = inspect_case(str(d_mm), workspace_dir=ws, case_name="case_micromed")
    rec = store.load_recording("case_micromed", insp["recordings"][0]["recording"])
    assert rec.companion_files == {}
    assert [c.name for c in rec.channels] == ["LA1", "LA2", "ECG1"]
    assert all(c.unit == "V" for c in rec.channels)
    assert rec.channels[0].shaft == "LA" and rec.channels[2].is_auxiliary is True
    marker_labels = {m.label for m in rec.markers}
    assert {"Trigger 5", "Trigger 9", "seizure note alpha", "clinical mark beta", "Event A 7"}.issubset(marker_labels)

    batch = compute_multichannel_batch(store, "case_micromed", rec.slug, start_seconds=0.0, duration_seconds=1.0, page_size=4)
    assert [tr["channel_name"] for tr in batch["traces"]] == ["LA1", "LA2"]
    np.testing.assert_allclose(np.asarray(batch["traces"][0]["values_uv"]), full_v[0, :256] * 1e6, rtol=1e-3, atol=1e-2)

    forbidden = ["RossiSecret", "MarcoTest", "MilanEpilepsyLab", "MRN_MM_4419", "1985-07-19", "2024-06-14"]
    _assert_terms_masked(mcp_server.inspect_case(target="case_micromed"), forbidden)

    # De-identified exports mask all Micromed identity terms
    analyze_case("case_micromed", workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    sheet = export_case_sheet("case_micromed", workspace_dir=ws, primary_run="baseline", max_cards=0, deidentify=True, pseudonym="MM01")
    bids = export_bids_events("case_micromed", workspace_dir=ws, run_name="baseline", deidentify=True, pseudonym="MM01")
    edfa = export_edf_annotations("case_micromed", workspace_dir=ws, run_name="baseline", deidentify=True, pseudonym="MM01")
    for p_str in (sheet["report_path"], bids["json_path"], edfa["json_path"], edfa["csv_path"]):
        _assert_terms_masked(Path(p_str).read_bytes(), forbidden)

    # 2) Mixed per-channel sampling rates refused clearly
    mm_mixed = _write_synthetic_micromed(tmp_path / "mm_mixed", "mixed.TRC", rate_multipliers=(1, 2, 1))
    with pytest.raises(ToolkitError) as exc_mixed:
        inspect_case(str(mm_mixed), workspace_dir=ws)
    assert exc_mixed.value.code == "unsupported_format"
    assert "sampling rate" in exc_mixed.value.message.lower()

    # 3) Unsupported TRC header version refused clearly
    mm_v3 = _write_synthetic_micromed(tmp_path / "mm_v3", "v3.TRC", header_version=3)
    with pytest.raises(ToolkitError) as exc_v3:
        inspect_case(str(mm_v3), workspace_dir=ws)
    assert exc_v3.value.code == "unsupported_format"
    assert "header version" in exc_v3.value.message.lower()

    # 4) Discontinuous multi-segment TRONCA refused clearly (note: (0, 0) is TRONCA end-of-table sentinel)
    mm_gap = _write_synthetic_micromed(tmp_path / "mm_gap", "gap.TRC", n_samples=512, segments=[(5, 0), (400, 256)])
    with pytest.raises(ToolkitError) as exc_gap:
        inspect_case(str(mm_gap), workspace_dir=ws)
    assert exc_gap.value.code == "discontinuous_recording_unsupported"

    # 5) Contiguous multi-segment TRONCA opens seamlessly
    mm_contig = _write_synthetic_micromed(tmp_path / "mm_contig", "contig.TRC", n_samples=512, segments=[(5, 0), (261, 256)])
    with open_mne_raw(mm_contig, preload=False) as raw_c:
        assert raw_c.n_times == 512
        seg_cross = raw_c.get_data(start=240, stop=272)
        assert seg_cross.shape == (3, 32)


def test_existing_formats_and_seven_mcp_tools_unchanged(tmp_path: Path) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    ws = tmp_path / "ws"
    insp = inspect_case(str(SYNTHETIC_EDF), workspace_dir=ws)
    store = WorkbenchStore(ws)
    rec = store.load_recording(insp["case"], insp["recordings"][0]["recording"])
    assert rec.modality == "seeg" and len(rec.channels) > 0 and rec.source_sha256

    async def list_tools():
        async with create_connected_server_and_client_session(mcp_server.mcp) as client:
            return (await client.list_tools()).tools

    tools = asyncio.run(list_tools())
    assert len(tools) == 7
    assert {t.name for t in tools} == {
        "inspect_case",
        "analyze_case",
        "query_evidence",
        "render_evidence_card",
        "open_workbench",
        "record_judgment",
        "export_or_protocol",
    }


# ---------------------------------------------------------------------------
# Sample-based tests (read from BRAINLENS_VENDOR_SAMPLES; skip cleanly if unset/missing)
# ---------------------------------------------------------------------------

def _vendor_sample_root() -> Path:
    env = (os.environ.get("BRAINLENS_VENDOR_SAMPLES") or "").strip()
    if not env:
        pytest.skip("BRAINLENS_VENDOR_SAMPLES is not set")
    p = Path(env).expanduser().resolve()
    if not p.is_dir():
        pytest.skip(f"BRAINLENS_VENDOR_SAMPLES directory missing: {p}")
    return p


def _require_sample(rel_path: str) -> Path:
    root = _vendor_sample_root()
    p = root / rel_path
    if not p.is_file():
        pytest.skip(f"Vendor sample not present: {rel_path}")
    return p


@pytest.mark.parametrize(
    ("rel_path", "reader_kind", "min_companions", "expect_masked_terms"),
    [
        ("nihon/MB0400FU.EEG", "nihon", 3, ["6416", "Test_no_cap", "Asdf", "2019/01/01"]),
        ("persyst/sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.lay", "persyst", 1, ["19-NOV-1997", "2014.12.19"]),
        ("eeglab/test_raw_chanloc.set", "eeglab", 1, ["/Users/alex/Dropbox/sample_eeglab"]),
        ("eeglab/test_raw_event_duration.set", "eeglab", 0, ["/home/edvz/brunnerc"]),
        ("cnt/scan41_short.cnt", "cnt_int16", 0, []),
        ("cnt/test_CNT_events_mne_JWoess_clipped.cnt", "cnt_int32", 0, []),
        ("curry8/test_bdf_stim_channel Curry 8.cdt", "curry", 2, ["wagner"]),
        ("curry7/test_bdf_stim_channel Curry 7.dap", "curry", 3, []),
        ("gdf/test_gdf_1.99.gdf", "gdf", 0, ["B01", "1984-12-01"]),
        ("gdf/test_1ch.gdf", "gdf", 0, []),
        ("nicolet/test_nicolet_raw.data", "nicolet", 1, ["102", "1102", "100102"]),
    ],
)
def test_real_vendor_sample_inspect_reader_companions_and_mcp_masking(
    rel_path: str,
    reader_kind: str,
    min_companions: int,
    expect_masked_terms: list[str],
    tmp_path: Path,
) -> None:
    sample_file = _require_sample(rel_path)
    # Copy sample folder to tmp_path so tamper checks never touch build/vendor-samples/
    work_dir = tmp_path / "sample_copy"
    shutil.copytree(sample_file.parent, work_dir)
    copied_file = work_dir / sample_file.name

    ws = tmp_path / "ws"
    mcp_server.set_workspace_root(ws)
    store = WorkbenchStore(ws)

    # Folder discovery must pick the primary recording (and never companion-only files)
    discovered = _discover_folder_recordings(work_dir)
    assert copied_file.resolve() in {p.resolve() for p in discovered}

    insp = inspect_case(str(copied_file), workspace_dir=ws, case_name="sample_case")
    rb = insp["recordings"][0]
    rec = store.load_recording("sample_case", rb["recording"])
    assert len(rec.companion_files) >= min_companions

    # Compare geometry directly against MNE where MNE reads natively without optional deps
    mne_raw = None
    if reader_kind == "nihon":
        mne_raw = mne.io.read_raw_nihon(copied_file, preload=False, verbose="ERROR")
    elif reader_kind == "persyst":
        mne_raw = mne.io.read_raw_persyst(copied_file, preload=False, verbose="ERROR")
    elif reader_kind == "eeglab":
        mne_raw = mne.io.read_raw_eeglab(copied_file, preload=False, verbose="ERROR")
    elif reader_kind == "cnt_int16":
        mne_raw = mne.io.read_raw_cnt(copied_file, data_format="int16", preload=False, verbose="ERROR")
    elif reader_kind == "cnt_int32":
        mne_raw = mne.io.read_raw_cnt(copied_file, data_format="int32", preload=False, verbose="ERROR")
    elif reader_kind == "gdf":
        mne_raw = mne.io.read_raw_gdf(copied_file, stim_channel=None, preload=False, verbose="ERROR")
    elif reader_kind == "nicolet":
        mne_raw = mne.io.read_raw_nicolet(copied_file, ch_type="eeg", preload=False, verbose="ERROR")
    elif reader_kind == "curry":
        try:
            import curryreader  # noqa: F401
        except ImportError:
            assert len(rec.channels) == 3 and rec.sfreq == pytest.approx(500.0) and rec.n_samples == 5000
        else:
            mne_raw = mne.io.read_raw_curry(copied_file, preload=False, verbose="ERROR")

    if mne_raw is not None:
        try:
            assert len(rec.channels) == len(mne_raw.ch_names)
            assert rec.sfreq == pytest.approx(float(mne_raw.info["sfreq"]))
            assert rec.n_samples == int(mne_raw.n_times)
            assert rec.duration == pytest.approx(mne_raw.n_times / float(mne_raw.info["sfreq"]))
        finally:
            mne_raw.close()

    # One reader window per format
    batch = compute_multichannel_batch(
        store, "sample_case", rec.slug, start_seconds=0.0, duration_seconds=min(2.0, rec.duration), page_size=4
    )
    assert len(batch["traces"]) > 0
    assert any(v is not None for v in batch["traces"][0]["values_uv"])

    # MCP inspect_case response must not leak vendor header identity fields
    mcp_resp = mcp_server.inspect_case(target="sample_case")
    assert not mcp_resp.get("error"), mcp_resp
    if expect_masked_terms:
        _assert_terms_masked(mcp_resp, expect_masked_terms)

    # For multi-file formats, mutating any bound companion must be detected
    if rec.companion_files:
        comp_path = Path(next(iter(rec.companion_files)))
        raw_b = bytearray(comp_path.read_bytes())
        raw_b[-1] ^= 0xFF
        comp_path.write_bytes(bytes(raw_b))
        with pytest.raises(ToolkitError) as exc_mod:
            verify_source_recording_integrity(rec)
        assert exc_mod.value.code == "source_recording_modified"


@pytest.mark.parametrize(
    ("rel_path", "channels"),
    [
        ("persyst/sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.lay", ["FP1", "FP2"]),
        ("nihon/MB0400FU.EEG", ["Fp1", "Fp2"]),
    ],
)
def test_real_vendor_samples_pyhfo_analyze_and_exports(
    rel_path: str,
    channels: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_file = _require_sample(rel_path)
    if not check_engine_health().get("available"):
        pytest.skip("PyHFO engine is not available in this environment")
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    ws = tmp_path / "ws"
    store = WorkbenchStore(ws)

    insp = inspect_case(str(sample_file), workspace_dir=ws, case_name="vendor_pyhfo")
    an = analyze_case(
        "vendor_pyhfo",
        workspace_dir=ws,
        run_name="pyhfo_real",
        overrides={"band": [40, 90], "channels": channels},
    )
    rec_slug = an["runs"][0]["recording"]
    bundle = store.load_run("vendor_pyhfo", rec_slug, "pyhfo_real")
    det = read_json(bundle["meta"]["detection_json_path"])
    assert det["status"] == "complete"
    assert len(det["segments"]) >= 1

    triad = compute_triad_view(
        "vendor_pyhfo",
        workspace_dir=ws,
        recording=rec_slug,
        run_name="pyhfo_real",
        channel=channels[0],
        event_id=None,
        center_seconds=1.5,
    )
    assert triad["channel_name"] == channels[0]
    sheet = export_case_sheet("vendor_pyhfo", workspace_dir=ws, primary_run="pyhfo_real", max_cards=0, deidentify=True, pseudonym="PV01")
    slides = export_mdt_slides("vendor_pyhfo", workspace_dir=ws, primary_run="pyhfo_real", deidentify=True, pseudonym="PV01")
    bids = export_bids_events("vendor_pyhfo", workspace_dir=ws, run_name="pyhfo_real", deidentify=True, pseudonym="PV01")
    edfa = export_edf_annotations("vendor_pyhfo", workspace_dir=ws, run_name="pyhfo_real", deidentify=True, pseudonym="PV01")
    assert Path(sheet["report_path"]).is_file()
    assert Path(slides["slides_path"]).is_file()
    assert Path(bids["tsv_path"]).is_file()
    assert Path(edfa["csv_path"]).is_file()


@pytest.mark.parametrize(
    ("rel_path", "expect_markers", "expect_masked_terms"),
    [
        ("micromed/File_micromed_1.TRC", ["Trigger 1"], ["Tp_metrologie", "metrologie", "2007-11-15"]),
        ("micromed/File_mircomed2.TRC", ["DRAPEAU 1", "DRAPEAU 2", "DRAPEUX 3"], ["2024-10-11"]),
        ("micromed/File_mircomed2_2segments.TRC", ["* * * Part 1 * * *", "DRAPEAU 1", "* * * Part 2 * * *", "DRAPEUX 3"], ["2024-10-11"]),
    ],
)
def test_real_micromed_samples_data_equality_markers_and_masking(
    rel_path: str,
    expect_markers: list[str],
    expect_masked_terms: list[str],
    tmp_path: Path,
) -> None:
    neo = pytest.importorskip("neo")
    from neo.rawio.micromedrawio import MicromedRawIO

    sample_file = _require_sample(rel_path)
    ws = tmp_path / "ws"
    mcp_server.set_workspace_root(ws)
    store = WorkbenchStore(ws)

    # 1) Direct adapter verification: lazy read + exact data equality for every channel vs neo full read
    with open_mne_raw(sample_file, preload=False) as raw:
        assert raw.preload is False
        adapter_v = raw.get_data()
        mid = min(raw.n_times - 1, 24832) if "2segments" in sample_file.name else raw.n_times // 2
        picks = [0, 1, raw.info["nchan"] // 2, raw.info["nchan"] - 1]
        s0, s1 = max(0, mid - 64), min(raw.n_times, mid + 64)
        np.testing.assert_allclose(raw.get_data(picks=picks, start=s0, stop=s1), adapter_v[picks, s0:s1], rtol=0, atol=0)

    r_neo = MicromedRawIO(str(sample_file))
    r_neo.parse_header()
    raw_chunks = []
    for seg_i in range(r_neo.segment_count(0)):
        c_raw = r_neo.get_analogsignal_chunk(block_index=0, seg_index=seg_i, stream_index=0)
        raw_chunks.append(r_neo.rescale_signal_raw_to_float(c_raw, dtype="float64", stream_index=0))
    ref_rawio_v = (np.vstack(raw_chunks) * 1e-6).T
    np.testing.assert_allclose(adapter_v, ref_rawio_v, rtol=1e-12, atol=1e-15)

    blk = neo.MicromedIO(str(sample_file)).read_block()
    ref_io_v = np.vstack([seg.analogsignals[0].rescale("V").magnitude for seg in blk.segments]).T
    np.testing.assert_allclose(adapter_v, ref_io_v, rtol=1e-6, atol=1e-9)

    # 2) End-to-end inspect_case + reader + markers + MCP identity masking
    insp = inspect_case(str(sample_file), workspace_dir=ws, case_name="micromed_case")
    rec = store.load_recording("micromed_case", insp["recordings"][0]["recording"])
    assert rec.companion_files == {}
    assert len(rec.channels) == adapter_v.shape[0]
    assert rec.n_samples == adapter_v.shape[1]
    marker_labels = {m.label for m in rec.markers}
    for expected_m in expect_markers:
        assert expected_m in marker_labels, f"Missing marker {expected_m!r} in {marker_labels}"

    batch = compute_multichannel_batch(
        store, "micromed_case", rec.slug, start_seconds=0.0, duration_seconds=min(2.0, rec.duration), page_size=4
    )
    assert len(batch["traces"]) > 0
    assert any(v is not None for v in batch["traces"][0]["values_uv"])

    mcp_resp = mcp_server.inspect_case(target="micromed_case")
    assert not mcp_resp.get("error"), mcp_resp
    _assert_terms_masked(mcp_resp, expect_masked_terms)


def test_real_samples_shared_workspace_companion_errors_keep_real_names(tmp_path: Path) -> None:
    """Shared workspace with Nihon MB0400FU must not over-mask EEGLAB/Curry8/Nicolet companion change errors."""
    samples = [
        ("nihon", "nihon/MB0400FU.EEG", None),
        ("eeglab", "eeglab/test_raw_chanloc.set", "test_raw_chanloc.fdt"),
        ("curry8", "curry8/test_bdf_stim_channel Curry 8.cdt", "test_bdf_stim_channel Curry 8.cdt.dpa"),
        ("nicolet", "nicolet/test_nicolet_raw.data", "test_nicolet_raw.head"),
    ]
    ws = tmp_path / "ws"
    mcp_server.set_workspace_root(ws)
    registered: list[tuple[Path, str]] = []
    for tag, rel_path, comp_name in samples:
        src = _require_sample(rel_path)
        work = tmp_path / tag
        shutil.copytree(src.parent, work)
        resp = mcp_server.inspect_case(target=str(work / src.name))
        assert not resp.get("error"), resp
        if comp_name is not None:
            registered.append((work / comp_name, resp["case"]))

    for comp_p, case_slug in registered:
        raw_b = bytearray(comp_p.read_bytes())
        raw_b[-1] ^= 0xFF
        comp_p.write_bytes(bytes(raw_b))
        err = mcp_server.analyze_case(target=case_slug)
        assert err.get("error") is True and err["code"] == "source_recording_modified"
        assert comp_p.name in err["message"], f"Expected {comp_p.name!r} in {err['message']!r}"
        assert "withheld" not in err["message"], f"Over-masked companion name in {err['message']!r}"
