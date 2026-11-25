#!/usr/bin/env bash
# Build an offline wheelhouse bundle for installing BrainLens on an air-gapped
# hospital or clinical research workstation.
#
# Run this script on a connected machine, then transfer the output directory
# (default: dist/offline-bundle/) to the target workstation and run
# ./install_offline.sh inside the bundle.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALLER_SRC="${SCRIPT_DIR}/install_offline.sh"

OUTPUT_DIR="${PROJECT_ROOT}/dist/offline-bundle"
HOST_PYTHON="${BRAINLENS_BUILD_PYTHON:-python3}"
TARGET_PYTHON_VERSION=""
TARGET_ABI=""
INCLUDE_PPTX=1
INCLUDE_PYHFO=0
PYHFO_SOURCE=""
ONNX_MODELS=""
CLEAN_OUTPUT=0
declare -a TARGET_PLATFORMS=()

usage() {
  cat <<'EOF'
Usage: build_offline_bundle.sh [OPTIONS]

Build a self-contained offline wheelhouse bundle (with SHA256SUMS and
install_offline.sh) on a connected machine for air-gapped installation.

Options:
  -o, --output DIR             Output bundle directory (default: dist/offline-bundle)
      --python EXE             Build-host Python executable (default: python3)
      --python-version VER     Target Python version for pip download (3.11, 3.12, or 3.13)
      --platform PLAT          Target wheel platform tag (repeatable, e.g. manylinux_2_28_x86_64,
                               manylinux2014_x86_64, win_amd64, macosx_14_0_arm64)
      --abi ABI                Optional target ABI tag (usually omit so pip matches cpXY/abi3/none)
      --with-pptx              Include python-pptx>=1.0,<2 for MDT slides export (default)
      --no-pptx                Exclude python-pptx from the wheelhouse
      --with-pyhfo             Also download isolated PyHFO engine wheels into wheels-pyhfo/
      --pyhfo-source DIR       Include a pre-fetched upstream PyHFO checkout in bundle/pyhfo/
      --onnx-models DIR        Include a pre-fetched ONNX model cache in bundle/onnx-models/
      --clean                  Remove existing output directory before building
  -h, --help                   Show this help message and exit

Examples:
  # Native Linux workstation (same OS/arch/Python as build host):
  ./scripts/build_offline_bundle.sh --clean

  # Cross-target Linux x86_64 Python 3.12 with PyHFO engine requirements:
  ./scripts/build_offline_bundle.sh \
    --python-version 3.12 \
    --platform manylinux_2_28_x86_64 \
    --with-pyhfo \
    --pyhfo-source ~/pyHFO \
    --onnx-models ~/pyhfo-onnx-models \
    --clean
EOF
}

die() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

hash_file() {
  local rel_path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${rel_path}"
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${rel_path}"
  else
    die "Neither sha256sum nor shasum is available to generate SHA256SUMS."
  fi
}

verify_checksums() {
  local bundle_dir="$1"
  (
    cd "${bundle_dir}"
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum -c SHA256SUMS >/dev/null
    elif command -v shasum >/dev/null 2>&1; then
      shasum -a 256 -c SHA256SUMS >/dev/null
    else
      die "Neither sha256sum nor shasum is available to verify SHA256SUMS."
    fi
  )
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)
      [[ $# -ge 2 ]] || die "--output requires a directory path."
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires an executable path."
      HOST_PYTHON="$2"
      shift 2
      ;;
    --python-version)
      [[ $# -ge 2 ]] || die "--python-version requires a version (3.11, 3.12, or 3.13)."
      TARGET_PYTHON_VERSION="$2"
      shift 2
      ;;
    --platform)
      [[ $# -ge 2 ]] || die "--platform requires a wheel platform tag."
      TARGET_PLATFORMS+=("$2")
      shift 2
      ;;
    --abi)
      [[ $# -ge 2 ]] || die "--abi requires an ABI tag."
      TARGET_ABI="$2"
      shift 2
      ;;
    --with-pptx)
      INCLUDE_PPTX=1
      shift
      ;;
    --no-pptx)
      INCLUDE_PPTX=0
      shift
      ;;
    --with-pyhfo)
      INCLUDE_PYHFO=1
      shift
      ;;
    --pyhfo-source)
      [[ $# -ge 2 ]] || die "--pyhfo-source requires a directory path."
      PYHFO_SOURCE="$2"
      shift 2
      ;;
    --onnx-models)
      [[ $# -ge 2 ]] || die "--onnx-models requires a directory path."
      ONNX_MODELS="$2"
      shift 2
      ;;
    --clean)
      CLEAN_OUTPUT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1 (use --help for usage)"
      ;;
  esac
done

if [[ -n "${TARGET_PYTHON_VERSION}" ]]; then
  case "${TARGET_PYTHON_VERSION}" in
    3.11|3.12|3.13) ;;
    *)
      die "Unsupported --python-version '${TARGET_PYTHON_VERSION}'. BrainLens requires Python 3.11, 3.12, or 3.13."
      ;;
  esac
fi

if ! command -v "${HOST_PYTHON}" >/dev/null 2>&1 && [[ ! -x "${HOST_PYTHON}" ]]; then
  die "Build-host Python interpreter not found: ${HOST_PYTHON}"
fi

[[ -f "${PROJECT_ROOT}/workbench/pyproject.toml" ]] || die "Cannot locate workbench/pyproject.toml under ${PROJECT_ROOT}"
[[ -f "${INSTALLER_SRC}" ]] || die "Cannot locate installer script at ${INSTALLER_SRC}"

if [[ -n "${PYHFO_SOURCE}" ]]; then
  for req in src/hfo_app.py src/param/param_detector.py src/param/param_filter.py src/utils/utils_feature.py; do
    [[ -f "${PYHFO_SOURCE}/${req}" ]] || die "PyHFO checkout at ${PYHFO_SOURCE} is missing ${req} (expected commit c1c7d3fec990e8661604080664766dc1297cbfdc)."
  done
fi

if [[ -n "${ONNX_MODELS}" && ! -d "${ONNX_MODELS}" ]]; then
  die "ONNX model cache directory not found: ${ONNX_MODELS}"
fi

if [[ "${CLEAN_OUTPUT}" -eq 1 && -d "${OUTPUT_DIR}" ]]; then
  rm -rf "${OUTPUT_DIR}"
fi

WHEELS_DIR="${OUTPUT_DIR}/wheels"
mkdir -p "${WHEELS_DIR}"

# Extract BrainLens version and runtime dependencies from workbench/pyproject.toml.
BRAINLENS_VERSION="$("${HOST_PYTHON}" - "${PROJECT_ROOT}/workbench/pyproject.toml" "${OUTPUT_DIR}/requirements-brainlens.txt" "${INCLUDE_PPTX}" <<'PY'
import sys, tomllib
from pathlib import Path

pyproject = tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
proj = pyproject["project"]
deps = list(proj.get("dependencies") or [])
include_pptx = sys.argv[3] == "1"
if include_pptx and not any(d.lower().startswith("python-pptx") for d in deps):
    deps.append("python-pptx>=1.0,<2")
Path(sys.argv[2]).write_text("\n".join(deps) + "\n", encoding="utf-8")
print(proj.get("version", "unknown"))
PY
)"

declare -a PIP_TARGET_ARGS=()
if [[ ${#TARGET_PLATFORMS[@]} -gt 0 || -n "${TARGET_PYTHON_VERSION}" || -n "${TARGET_ABI}" ]]; then
  PIP_TARGET_ARGS+=(--only-binary=:all: --implementation cp)
  if [[ -n "${TARGET_PYTHON_VERSION}" ]]; then
    PIP_TARGET_ARGS+=(--python-version "${TARGET_PYTHON_VERSION}")
  fi
  if [[ -n "${TARGET_ABI}" ]]; then
    PIP_TARGET_ARGS+=(--abi "${TARGET_ABI}")
  fi
  if [[ ${#TARGET_PLATFORMS[@]} -gt 0 ]]; then
    for plat in "${TARGET_PLATFORMS[@]}"; do
      PIP_TARGET_ARGS+=(--platform "${plat}")
    done
  fi
fi

printf 'Building BrainLens %s wheel...\n' "${BRAINLENS_VERSION}"
"${HOST_PYTHON}" -m pip wheel --no-deps --wheel-dir "${WHEELS_DIR}" "${PROJECT_ROOT}/workbench"

printf 'Downloading BrainLens runtime dependency wheels into %s...\n' "${WHEELS_DIR}"
if [[ ${#PIP_TARGET_ARGS[@]} -gt 0 ]]; then
  "${HOST_PYTHON}" -m pip download \
    --dest "${WHEELS_DIR}" \
    "${PIP_TARGET_ARGS[@]}" \
    -r "${OUTPUT_DIR}/requirements-brainlens.txt"
else
  "${HOST_PYTHON}" -m pip download \
    --dest "${WHEELS_DIR}" \
    -r "${OUTPUT_DIR}/requirements-brainlens.txt"
fi

if [[ "${INCLUDE_PYHFO}" -eq 1 ]]; then
  PYHFO_WHEELS_DIR="${OUTPUT_DIR}/wheels-pyhfo"
  mkdir -p "${PYHFO_WHEELS_DIR}"
  cat > "${OUTPUT_DIR}/requirements-pyhfo.txt" <<'EOF'
HFODetector>=0.0.23,<0.1
mne>=1.7,<2
numpy>=1.26,<2
scipy>=1.11,<2
pandas>=2,<3
p_tqdm>=1.4,<2
onnxruntime>=1.18,<2
matplotlib>=3.8,<4
openpyxl>=3.1,<4
EOF
  printf 'Downloading PyHFO engine dependency wheels into %s...\n' "${PYHFO_WHEELS_DIR}"
  if [[ ${#PIP_TARGET_ARGS[@]} -gt 0 ]]; then
    "${HOST_PYTHON}" -m pip download \
      --dest "${PYHFO_WHEELS_DIR}" \
      "${PIP_TARGET_ARGS[@]}" \
      -r "${OUTPUT_DIR}/requirements-pyhfo.txt"
  else
    "${HOST_PYTHON}" -m pip download \
      --dest "${PYHFO_WHEELS_DIR}" \
      -r "${OUTPUT_DIR}/requirements-pyhfo.txt"
  fi
fi

if [[ -n "${PYHFO_SOURCE}" ]]; then
  printf 'Bundling pre-fetched PyHFO checkout into %s/pyhfo...\n' "${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}/pyhfo"
  mkdir -p "${OUTPUT_DIR}/pyhfo"
  cp -a "${PYHFO_SOURCE}/." "${OUTPUT_DIR}/pyhfo/"
  rm -rf "${OUTPUT_DIR}/pyhfo/.git"
fi

if [[ -n "${ONNX_MODELS}" ]]; then
  printf 'Bundling pre-fetched ONNX models into %s/onnx-models...\n' "${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}/onnx-models"
  mkdir -p "${OUTPUT_DIR}/onnx-models"
  cp -a "${ONNX_MODELS}/." "${OUTPUT_DIR}/onnx-models/"
fi

cp "${INSTALLER_SRC}" "${OUTPUT_DIR}/install_offline.sh"
chmod 0755 "${OUTPUT_DIR}/install_offline.sh"

BUILT_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
PLAT_SUMMARY="native"
if [[ ${#TARGET_PLATFORMS[@]} -gt 0 ]]; then
  PLAT_SUMMARY="${TARGET_PLATFORMS[*]}"
fi
PY_SUMMARY="${TARGET_PYTHON_VERSION:-native}"
HAS_PYHFO_SOURCE=0
if [[ -n "${PYHFO_SOURCE}" ]]; then
  HAS_PYHFO_SOURCE=1
fi
HAS_ONNX_MODELS=0
if [[ -n "${ONNX_MODELS}" ]]; then
  HAS_ONNX_MODELS=1
fi

cat > "${OUTPUT_DIR}/BUNDLE_MANIFEST.txt" <<EOF
brainlens_version=${BRAINLENS_VERSION}
built_utc=${BUILT_UTC}
target_python=${PY_SUMMARY}
target_platforms=${PLAT_SUMMARY}
include_pptx=${INCLUDE_PPTX}
include_pyhfo_wheels=${INCLUDE_PYHFO}
include_pyhfo_source=${HAS_PYHFO_SOURCE}
include_onnx_models=${HAS_ONNX_MODELS}
pyhfo_pinned_commit=c1c7d3fec990e8661604080664766dc1297cbfdc
EOF

printf 'Generating SHA256SUMS...\n'
rm -f "${OUTPUT_DIR}/SHA256SUMS"
(
  cd "${OUTPUT_DIR}"
  while IFS= read -r rel_file; do
    rel_clean="${rel_file#./}"
    hash_file "${rel_clean}" >> SHA256SUMS
  done < <(find . -type f ! -name 'SHA256SUMS' | LC_ALL=C sort)
)

verify_checksums "${OUTPUT_DIR}"

printf 'Offline bundle ready at: %s\n' "${OUTPUT_DIR}"
printf 'Verify and install on the target workstation with:\n'
printf '  cd "%s" && ./install_offline.sh --venv /opt/brainlens-venv\n' "${OUTPUT_DIR}"
