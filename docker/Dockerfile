# syntax=docker/dockerfile:1

# Global build arg: set --build-arg WITH_PYHFO=1 to bundle a pre-fetched
# upstream PyHFO checkout (docker/vendor/pyhfo) and ONNX models
# (docker/vendor/onnx-models) into a dedicated isolated engine venv.
# Nothing is cloned or fetched from HuggingFace at container runtime.
ARG WITH_PYHFO=0

# ---------------------------------------------------------------------------
# Stage 1: Build the BrainLens wheel and install runtime dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY workbench /build/workbench

RUN python -m venv /opt/brainlens-venv \
    && /opt/brainlens-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/brainlens-venv/bin/pip wheel --no-deps --wheel-dir /wheels /build/workbench \
    && /opt/brainlens-venv/bin/pip install --no-cache-dir \
        /wheels/brainlens-*.whl \
        "python-pptx>=1.0,<2"

# ---------------------------------------------------------------------------
# Stage 2: Optional PyHFO + ONNX engine venv (from pre-fetched build context)
# ---------------------------------------------------------------------------
FROM builder AS pyhfo-builder

ARG WITH_PYHFO

COPY docker/vendor /tmp/vendor

RUN set -eu; \
    mkdir -p /opt/pyhfo /opt/pyhfo-onnx-models /opt/pyhfo-venv; \
    if [ "${WITH_PYHFO}" = "1" ]; then \
      for req in src/hfo_app.py src/param/param_detector.py src/param/param_filter.py src/utils/utils_feature.py; do \
        if [ ! -f "/tmp/vendor/pyhfo/${req}" ]; then \
          echo "ERROR: WITH_PYHFO=1 requires pre-fetched upstream PyHFO checkout at docker/vendor/pyhfo (missing ${req})." >&2; \
          echo "See docker/vendor/README.md and docs/deployment.md." >&2; \
          exit 1; \
        fi; \
      done; \
      cp -a /tmp/vendor/pyhfo/. /opt/pyhfo/; \
      if [ -d /tmp/vendor/onnx-models ]; then \
        cp -a /tmp/vendor/onnx-models/. /opt/pyhfo-onnx-models/; \
      fi; \
      python -m venv /opt/pyhfo-venv; \
      /opt/pyhfo-venv/bin/pip install --no-cache-dir \
        "HFODetector>=0.0.23,<0.1" \
        "mne>=1.7,<2" \
        "numpy>=1.26,<2" \
        "scipy>=1.11,<2" \
        "pandas>=2,<3" \
        "p_tqdm>=1.4,<2" \
        "onnxruntime>=1.18,<2" \
        "matplotlib>=3.8,<4" \
        "openpyxl>=3.1,<4"; \
      PYTHONPATH=/opt/pyhfo /opt/pyhfo-venv/bin/python -c "\
import sys, types; \
stub = types.ModuleType('src.utils.utils_plotting'); stub.plot_feature = None; sys.modules['src.utils.utils_plotting'] = stub; \
import numpy, scipy, mne, pandas, p_tqdm, HFODetector, onnxruntime, src.hfo_app, src.param.param_detector, src.param.param_filter, src.utils.utils_feature"; \
    fi; \
    rm -rf /tmp/vendor

# ---------------------------------------------------------------------------
# Stage 3: Minimal non-root runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime-base

ENV PATH="/opt/brainlens-venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/brainlens \
    MPLBACKEND=Agg \
    BRAINLENS_WORKSPACE=/data/workbench_results \
    BRAINLENS_TRUST_DIR=/home/brainlens/.brainlens \
    BRAINLENS_NO_BROWSER=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 brainlens \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/brainlens --shell /usr/sbin/nologin brainlens \
    && mkdir -p /data/workbench_results /recordings /home/brainlens/.brainlens \
    && chown -R brainlens:brainlens /data /home/brainlens \
    && chmod 0700 /home/brainlens/.brainlens

COPY --from=builder /opt/brainlens-venv /opt/brainlens-venv

FROM runtime-base AS runtime-0

FROM runtime-base AS runtime-1
COPY --from=pyhfo-builder /opt/pyhfo /opt/pyhfo
COPY --from=pyhfo-builder /opt/pyhfo-venv /opt/pyhfo-venv
COPY --from=pyhfo-builder /opt/pyhfo-onnx-models /opt/pyhfo-onnx-models

ENV BRAINLENS_PYHFO_SOURCE=/opt/pyhfo \
    BRAINLENS_PYHFO_PYTHON=/opt/pyhfo-venv/bin/python \
    BRAINLENS_PYHFO_MODEL_CACHE=/opt/pyhfo-onnx-models

FROM runtime-${WITH_PYHFO} AS final

USER brainlens
WORKDIR /data

VOLUME ["/data"]
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["/opt/brainlens-venv/bin/python", "-c", "import json, sys, urllib.request; d = json.loads(urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=3).read()); sys.exit(0 if d.get('ok') else 1)"]

# Bind 0.0.0.0 inside the container ONLY; always publish strictly to host loopback
# on the matching port (-p 127.0.0.1:8765:8765). WorkbenchHTTPServer rejects
# non-loopback Host headers and mismatched ports by design.
ENTRYPOINT ["brainlens-workbench", "--host", "0.0.0.0", "--port", "8765"]
