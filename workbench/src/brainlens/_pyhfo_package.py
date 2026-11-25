"""Package metadata and pinned ONNX resolution, run by the selected PyHFO Python."""

import hashlib
import json
import os
import platform
import sys
from importlib.metadata import distribution
from pathlib import Path


def describe() -> dict[str, str]:
    import pyhfo
    from packaging.specifiers import SpecifierSet

    dist = distribution("pyhfo")
    if dist.version not in SpecifierSet(">=4,<5"):
        raise ValueError(f"Installed pyhfo {dist.version} is incompatible. Install pyhfo>=4,<5.")
    result = {
        "distribution": dist.metadata["Name"],
        "version": dist.version,
        "python_version": platform.python_version(),
    }
    record = next(
        (path for path in dist.files or () if path.name == "RECORD" and path.parent.name.endswith(".dist-info")),
        None,
    )
    if record is not None:
        result["record_sha256"] = hashlib.sha256(record.locate().read_bytes()).hexdigest()
    else:
        module = Path(pyhfo.__file__).resolve()
        root = module.parent
        paths = root.rglob("*") if module.name == "__init__.py" and root.name == "pyhfo" else (module,)
        files = {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file() and "__pycache__" not in p.parts
        }
        result["package_sha256"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return result


def models(tags: list[str], fetch: bool) -> dict[str, dict[str, str]]:
    from pyhfo.utils.default_onnx_models import default_hf_onnx_sources, resolve_hf_onnx_uri

    sources = default_hf_onnx_sources(tuple(tags))
    if not fetch and os.environ.get("PYHFO_ONNX_DIR"):
        from pyhfo.utils.onnx_inference import discover_local_onnx_sources

        # PyHFO treats the three local overrides as a bundle.
        sources.update(discover_local_onnx_sources())
    paths, errors = {}, {}
    for tag in tags:
        try:
            source = sources[tag]
            if source.startswith("hf://"):
                source = resolve_hf_onnx_uri(source, local_files_only=not fetch)
            path = Path(source)
            paths[tag] = str(path / "model.onnx" if path.is_dir() else path)
        except Exception as exc:
            errors[tag] = str(exc)
    return {"paths": paths, "errors": errors}


if __name__ == "__main__":
    try:
        action = sys.argv[1]
        result = describe() if action == "describe" else models(sys.argv[2:], action == "fetch")
        print(json.dumps(result))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
