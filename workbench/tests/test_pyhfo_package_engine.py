"""Package-engine configuration, provenance, and offline model resolution."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

import packaging
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if importlib.util.find_spec("brainlens") is None and SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlens import cli, pyhfo_runner as runner
from brainlens.models import ClassifierConfig, EngineSettings, ToolkitError


# Installed into a temporary engine interpreter. Its calls log whether network access was
# permitted and emulate the two-file snapshots returned by PyHFO's pinned registry.
_LOCAL_REGISTRY = """
import json
import os
from pathlib import Path

def default_hf_onnx_sources(tags):
    return {tag: 'hf://' + tag for tag in tags}

def resolve_hf_onnx_uri(uri, *, local_files_only):
    hub = Path(os.environ['HF_HUB_CACHE'])
    hub.mkdir(parents=True, exist_ok=True)
    tag = uri.removeprefix('hf://')
    with (hub / 'requests.jsonl').open('a') as log:
        print(json.dumps([tag, local_files_only, os.environ.get('HF_HUB_OFFLINE')]), file=log)
    snapshot = hub / tag / 'snapshots' / 'pinned'
    model = snapshot / 'model.onnx'
    spec = snapshot / 'model.spec.json'
    if not local_files_only:
        snapshot.mkdir(parents=True, exist_ok=True)
        model.write_bytes(tag.encode())
        spec.write_text('{}')
    if not (model.is_file() and spec.is_file()):
        raise ValueError('pinned model missing from local hub cache')
    return str(model)
"""


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    for key in os.environ.copy():
        if key.startswith(("BRAINLENS_PYHFO_", "BTK_PYHFO_", "PYHFO_", "HF_", "HUGGINGFACE_")):
            monkeypatch.delenv(key)
    for name in ("BRAINLENS_ALLOW_WORKSPACE_ENGINE", "BTK_ALLOW_WORKSPACE_ENGINE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runner, "USER_ENGINE_JSON", tmp_path / "user" / "engine.json")
    monkeypatch.setattr(runner, "LEGACY_USER_ENGINE_JSON", tmp_path / "legacy-user.json")
    monkeypatch.setattr(runner, "LEGACY_ENGINE_JSON", tmp_path / "legacy-checkout.json")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))


@pytest.fixture
def installed_package(tmp_path):
    # A real interpreter with isolated import and distribution metadata, without pip/network.
    env = tmp_path / "engine"
    builder = venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt")
    builder.create(env)
    python = Path(builder.ensure_directories(env).env_exe)
    site = Path(subprocess.check_output(
        [python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
    ).strip())
    shutil.copytree(Path(packaging.__file__).parent, site / "packaging")
    package = site / "pyhfo"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "4.0.0"\n')
    metadata = site / "pyhfo-4.0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: pyhfo\nVersion: 4.0.0\n")
    (metadata / "RECORD").write_bytes(b"pyhfo/__init__.py,,\r\npyhfo-4.0.0.dist-info/RECORD,,\r\n")
    # Satisfy the worker import probe without native or scientific dependencies.
    for rel in ("numpy", "scipy", "mne", "pandas", "p_tqdm", "HFODetector", "onnxruntime",
                "pyhfo/hfo_app", "pyhfo/param/param_detector", "pyhfo/param/param_filter", "pyhfo/utils/utils_feature"):
        module = site / f"{rel}.py"
        module.parent.mkdir(parents=True, exist_ok=True)
        module.touch()
    (package / "utils/default_onnx_models.py").write_text(_LOCAL_REGISTRY)
    return python, site


def test_installed_package_is_last_fallback(tmp_path, monkeypatch, installed_package):
    python, _ = installed_package
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setattr(runner, "find_spec", lambda _: object())
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def config(path, source):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"python_executable": str(python), "source_root": source}))

    config(workspace / "engine.json", "/workspace-source")
    automatic = runner.resolve_engine_settings(workspace)
    assert automatic.python_executable == str(python)
    assert automatic.source_root is None

    monkeypatch.setattr(runner, "IS_SOURCE_CHECKOUT", True)
    config(runner.LEGACY_ENGINE_JSON, "/legacy-source")
    assert runner.resolve_engine_settings(workspace).source_root == "/legacy-source"
    monkeypatch.setenv("BRAINLENS_ALLOW_WORKSPACE_ENGINE", "1")
    assert runner.resolve_engine_settings(workspace).source_root == "/workspace-source"
    config(runner.USER_ENGINE_JSON, "/user-source")
    assert runner.resolve_engine_settings(workspace).source_root == "/user-source"
    monkeypatch.setenv("BRAINLENS_PYHFO_PYTHON", str(python))
    assert runner.resolve_engine_settings(workspace).source_root is None


def test_configure_package_and_digest(tmp_path, capsys, installed_package):
    python, site = installed_package
    rc = cli.main(["configure-engine", "--python", str(python)])
    result = json.loads(capsys.readouterr().out)
    assert rc == 0, result
    settings = runner.resolve_engine_settings()
    assert not settings.source_root and settings.python_executable == str(python)
    backend = runner.PyHFORunner(settings).source_identity()
    assert backend["distribution"].lower() == "pyhfo"
    assert backend["version"] == "4.0.0"
    assert backend["python_version"] == subprocess.check_output(
        [python, "-c", "import platform; print(platform.python_version())"], text=True
    ).strip()
    worker = Path(runner.__file__).with_name("_pyhfo_worker.py")
    assert backend["worker_sha256"] == hashlib.sha256(worker.read_bytes()).hexdigest()
    record = site / "pyhfo-4.0.0.dist-info" / "RECORD"
    assert backend["record_sha256"] == hashlib.sha256(record.read_bytes()).hexdigest()
    (site / "other_package.py").write_text("unrelated = 42\n")
    assert runner.PyHFORunner(settings).source_identity()["record_sha256"] == backend["record_sha256"]


def test_missing_record_hashes_only_package_files(installed_package):
    python, site = installed_package
    (site / "pyhfo-4.0.0.dist-info/RECORD").unlink()
    settings = EngineSettings(python_executable=str(python))
    digest = runner.PyHFORunner(settings).source_identity()["package_sha256"]
    (site / "other_package.py").write_text("unrelated = 42\n")
    assert runner.PyHFORunner(settings).source_identity()["package_sha256"] == digest
    (site / "pyhfo/hfo_app.py").write_text("# Changed engine code\n")
    assert runner.PyHFORunner(settings).source_identity()["package_sha256"] != digest


def test_failed_registry_fetch_preserves_existing_configuration(installed_package, capsys):
    python, site = installed_package
    original = '{"source_root": "/previous", "python_executable": "/previous/bin/python"}'
    runner.USER_ENGINE_JSON.parent.mkdir(parents=True)
    runner.USER_ENGINE_JSON.write_text(original)
    (site / "pyhfo/utils/default_onnx_models.py").write_text(
        _LOCAL_REGISTRY + "\ndef resolve_hf_onnx_uri(*args, **kwargs):\n    raise ValueError('SHA256 mismatch')\n"
    )
    assert cli.main(["configure-engine", "--python", str(python), "--fetch-models"]) != 0
    result = json.loads(capsys.readouterr().out)
    assert result["code"] == "engine_models_unavailable" and "SHA256 mismatch" in result["message"]
    assert runner.USER_ENGINE_JSON.read_text() == original


@pytest.mark.parametrize("unsupported", ["3.99", "5.0.0", "unimportable"])
def test_configure_rejects_invalid_package(installed_package, capsys, unsupported):
    python, site = installed_package
    if unsupported == "unimportable":
        (site / "pyhfo" / "__init__.py").write_text('raise ImportError("broken pyhfo installation")\n')
    else:
        (site / "pyhfo-4.0.0.dist-info" / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: pyhfo\nVersion: {unsupported}\n"
        )
    rc = cli.main(["configure-engine", "--python", str(python)])
    result = json.loads(capsys.readouterr().out)
    assert rc != 0
    assert result["error"] and "pyhfo" in result["message"].lower()
    assert not runner.USER_ENGINE_JSON.exists()


def test_legacy_source_identity(tmp_path):
    for rel in runner.REQUIRED_PYHFO_MODULES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    settings = EngineSettings(source_root=str(tmp_path), python_executable=sys.executable)
    backend = runner.PyHFORunner(settings).source_identity()
    assert backend["source_root"] == str(tmp_path)
    assert len(bytes.fromhex(backend["source_sha256"])) == 32
    assert "record_sha256" not in backend


def test_package_worker_env_filters_and_preserves_cache_paths(tmp_path, monkeypatch):
    override = tmp_path / "override"
    monkeypatch.setenv("PYHFO_ONNX_DIR", str(override))
    monkeypatch.setenv("PYHFO_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("HF_TOKEN", "private-token")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "untrusted"))
    env = runner._sanitized_worker_env(None, None)
    assert env["BRAINLENS_PYHFO_MODE"] == "package"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["PYHFO_ONNX_DIR"] == str(override)
    assert env["PYHFO_HOME"] == str(tmp_path / "state")
    assert env["HF_HUB_CACHE"] == str(tmp_path / "hub")
    assert "HF_TOKEN" not in env and "PYTHONPATH" not in env
    assert runner._sanitized_worker_env(None, str(tmp_path / "explicit"))["PYHFO_ONNX_DIR"] == str(tmp_path / "explicit")


def test_fetch_models_populates_offline_cache(installed_package, tmp_path, monkeypatch, capsys):
    python, _ = installed_package
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setattr(runner, "find_spec", lambda _: object())
    settings = runner.resolve_engine_settings()
    config = ClassifierConfig(enabled=True)
    with pytest.raises(ToolkitError, match="configure-engine --fetch-models"):
        runner.PyHFORunner(settings).resolve_models(config)

    rc = cli.main(["configure-engine", "--fetch-models"])
    response = json.loads(capsys.readouterr().out)
    assert rc == 0, response
    assert not runner.USER_ENGINE_JSON.exists()
    paths = runner.PyHFORunner(runner.resolve_engine_settings()).resolve_models(config)
    tags = {"artifact": "a", "spike": "s", "ehfo": "e"}
    assert paths.keys() == tags.keys()
    for stage, path in paths.items():
        assert Path(path).read_text() == tags[stage]
        assert Path(path).with_suffix(".spec.json").is_file()

    health = runner.check_engine_health()
    assert health["available"] and health["mode"] == "package"
    assert health["pyhfo_version"] == "4.0.0"
    assert set(health["local_onnx_models"]) == set(tags)
    requests = [json.loads(line) for line in (tmp_path / "hub" / "requests.jsonl").read_text().splitlines()]
    assert {tag for tag, local_only, _ in requests if not local_only} == set(tags.values())
    assert all((offline == "1") == local_only for _, local_only, offline in requests)


def test_package_local_onnx_override(installed_package, tmp_path, monkeypatch):
    python, site = installed_package
    root = tmp_path / "transferred-models"
    tags = {"artifact": "a", "spike": "s", "ehfo": "e"}
    for tag in tags.values():
        folder = root / f"model_{tag}"
        folder.mkdir(parents=True)
        (folder / "model.onnx").write_bytes(tag.encode())
        (folder / "model.spec.json").write_text("{}")
    (site / "pyhfo/utils/onnx_inference.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "def discover_local_onnx_sources():\n"
        "    root = Path(os.environ['PYHFO_ONNX_DIR'])\n"
        "    return {tag: str(root / f'model_{tag}') for tag in ('a', 's', 'e')}\n"
    )
    monkeypatch.setenv("PYHFO_ONNX_DIR", str(root))
    settings = EngineSettings(python_executable=str(python))
    paths = runner.PyHFORunner(settings).resolve_models(ClassifierConfig(enabled=True))
    assert paths == {stage: str(root / f"model_{tag}" / "model.onnx") for stage, tag in tags.items()}
    assert not (tmp_path / "hub" / "requests.jsonl").exists()
