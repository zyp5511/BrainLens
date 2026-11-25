#!/usr/bin/env bash
# Offline installer for a BrainLens wheelhouse bundle created by
# scripts/build_offline_bundle.sh.
#
# Verifies SHA256SUMS and installs BrainLens (plus optional PyHFO engine venv)
# strictly offline using pip --no-index --find-links.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUNDLE_DIR="${SCRIPT_DIR}"
VENV_DIR="./brainlens-venv"
PYTHON_EXE="python3"
WITH_PYHFO=0
PYHFO_VENV="./pyhfo-venv"
PYHFO_PYTHON=""
SKIP_VERIFY=0
VERIFY_ONLY=0

usage() {
  cat <<'EOF'
Usage: install_offline.sh [OPTIONS]

Install BrainLens from an offline wheelhouse bundle without network access.

Options:
      --bundle DIR         Path to offline bundle directory (default: script directory)
      --venv DIR           Target virtualenv path for BrainLens (default: ./brainlens-venv)
      --python EXE         Python 3.11-3.13 executable (default: python3)
      --with-pyhfo         Also install isolated PyHFO engine virtualenv from wheels-pyhfo/
      --pyhfo-venv DIR     Target virtualenv path for PyHFO (default: ./pyhfo-venv)
      --pyhfo-python EXE   Python 3.11-3.13 executable for PyHFO venv (default: --python)
      --verify-only        Verify SHA256SUMS and exit without installing
      --skip-verify        Skip SHA256SUMS verification (not recommended)
  -h, --help               Show this help message and exit
EOF
}

die() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

verify_checksums() {
  local dir="$1"
  [[ -f "${dir}/SHA256SUMS" ]] || die "Missing ${dir}/SHA256SUMS (pass --skip-verify only if you accept unverified files)."
  printf 'Verifying SHA256SUMS in %s...\n' "${dir}"
  (
    cd "${dir}"
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum -c SHA256SUMS
    elif command -v shasum >/dev/null 2>&1; then
      shasum -a 256 -c SHA256SUMS
    else
      die "Neither sha256sum nor shasum is installed; cannot verify SHA256SUMS."
    fi
  )
}

check_python_version() {
  local py="$1"
  if ! command -v "${py}" >/dev/null 2>&1 && [[ ! -x "${py}" ]]; then
    die "Python interpreter not found: ${py}"
  fi
  if ! "${py}" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)'; then
    local found_ver
    found_ver="$("${py}" -c 'import sys; print(".".join(str(x) for x in sys.version_info[:3]))' 2>/dev/null || printf 'unknown')"
    die "BrainLens requires Python 3.11-3.13, but ${py} is Python ${found_ver}."
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)
      [[ $# -ge 2 ]] || die "--bundle requires a directory path."
      BUNDLE_DIR="$2"
      shift 2
      ;;
    --venv)
      [[ $# -ge 2 ]] || die "--venv requires a directory path."
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires an executable path."
      PYTHON_EXE="$2"
      shift 2
      ;;
    --with-pyhfo)
      WITH_PYHFO=1
      shift
      ;;
    --pyhfo-venv)
      [[ $# -ge 2 ]] || die "--pyhfo-venv requires a directory path."
      PYHFO_VENV="$2"
      shift 2
      ;;
    --pyhfo-python)
      [[ $# -ge 2 ]] || die "--pyhfo-python requires an executable path."
      PYHFO_PYTHON="$2"
      shift 2
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    --skip-verify)
      SKIP_VERIFY=1
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

[[ -d "${BUNDLE_DIR}" ]] || die "Bundle directory not found: ${BUNDLE_DIR}"
BUNDLE_DIR="$(cd "${BUNDLE_DIR}" && pwd)"

if [[ "${SKIP_VERIFY}" -eq 0 ]]; then
  verify_checksums "${BUNDLE_DIR}"
fi

if [[ "${VERIFY_ONLY}" -eq 1 ]]; then
  printf 'SHA256SUMS verification succeeded.\n'
  exit 0
fi

WHEELS_DIR="${BUNDLE_DIR}/wheels"
[[ -d "${WHEELS_DIR}" ]] || die "Missing wheels directory: ${WHEELS_DIR}"

shopt -s nullglob
brainlens_wheels=("${WHEELS_DIR}"/brainlens-*.whl)
pptx_wheels=("${WHEELS_DIR}"/python_pptx-*.whl "${WHEELS_DIR}"/python-pptx-*.whl)
shopt -u nullglob

[[ ${#brainlens_wheels[@]} -ge 1 ]] || die "No brainlens-*.whl found in ${WHEELS_DIR}"
BRAINLENS_WHEEL="${brainlens_wheels[0]}"

check_python_version "${PYTHON_EXE}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Creating BrainLens virtualenv at %s...\n' "${VENV_DIR}"
  "${PYTHON_EXE}" -m venv "${VENV_DIR}"
fi

printf 'Installing BrainLens offline from %s...\n' "${WHEELS_DIR}"
"${VENV_DIR}/bin/pip" install \
  --no-index \
  --find-links "${WHEELS_DIR}" \
  "${BRAINLENS_WHEEL}"

if [[ ${#pptx_wheels[@]} -ge 1 ]]; then
  printf 'Installing python-pptx offline from %s...\n' "${WHEELS_DIR}"
  "${VENV_DIR}/bin/pip" install \
    --no-index \
    --find-links "${WHEELS_DIR}" \
    python-pptx
fi

printf 'Verifying BrainLens imports...\n'
"${VENV_DIR}/bin/python" -c 'import brainlens, anyio, edfio, filelock, matplotlib, mcp, mne, numpy, pydantic, scipy'

if [[ "${WITH_PYHFO}" -eq 1 ]]; then
  PYHFO_WHEELS_DIR="${BUNDLE_DIR}/wheels-pyhfo"
  PYHFO_REQS="${BUNDLE_DIR}/requirements-pyhfo.txt"
  [[ -d "${PYHFO_WHEELS_DIR}" ]] || die "Bundle does not include wheels-pyhfo/ (rebuild with --with-pyhfo)."
  [[ -f "${PYHFO_REQS}" ]] || die "Missing ${PYHFO_REQS} in bundle."

  EFF_PYHFO_PYTHON="${PYHFO_PYTHON:-${PYTHON_EXE}}"
  check_python_version "${EFF_PYHFO_PYTHON}"

  if [[ ! -x "${PYHFO_VENV}/bin/python" ]]; then
    printf 'Creating isolated PyHFO virtualenv at %s...\n' "${PYHFO_VENV}"
    "${EFF_PYHFO_PYTHON}" -m venv "${PYHFO_VENV}"
  fi

  printf 'Installing PyHFO engine dependencies offline from %s...\n' "${PYHFO_WHEELS_DIR}"
  "${PYHFO_VENV}/bin/pip" install \
    --no-index \
    --find-links "${PYHFO_WHEELS_DIR}" \
    -r "${PYHFO_REQS}"

  if [[ -d "${BUNDLE_DIR}/pyhfo" ]]; then
    printf '\nPyHFO checkout is bundled at %s/pyhfo.\n' "${BUNDLE_DIR}"
    if [[ -d "${BUNDLE_DIR}/onnx-models" ]]; then
      printf 'Register the engine with:\n'
      printf '  "%s/bin/brainlens" configure-engine --source "%s/pyhfo" --python "%s/bin/python" --model-cache "%s/onnx-models"\n' \
        "${VENV_DIR}" "${BUNDLE_DIR}" "${PYHFO_VENV}" "${BUNDLE_DIR}"
    else
      printf 'Register the engine with:\n'
      printf '  "%s/bin/brainlens" configure-engine --source "%s/pyhfo" --python "%s/bin/python"\n' \
        "${VENV_DIR}" "${BUNDLE_DIR}" "${PYHFO_VENV}"
    fi
  fi
fi

printf '\nBrainLens offline installation complete.\n'
printf 'Quick verification:\n'
printf '  "%s/bin/brainlens" inspect "*"\n' "${VENV_DIR}"
printf '  "%s/bin/brainlens-workbench"\n' "${VENV_DIR}"
