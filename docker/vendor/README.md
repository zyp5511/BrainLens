# Pre-fetched PyHFO & ONNX Vendor Context (Optional)

When building the BrainLens Docker image with `--build-arg WITH_PYHFO=1`, place
the pre-fetched upstream PyHFO checkout and ONNX classifier models here before
running `docker build`:

```text
docker/vendor/
├── pyhfo/                    # git checkout of roychowdhuryresearch/pyHFO at
│   └── src/hfo_app.py        # commit c1c7d3fec990e8661604080664766dc1297cbfdc
└── onnx-models/              # optional 3-stage ONNX classifier weights
    ├── model_a/
    │   ├── model.onnx
    │   └── model.spec.json
    ├── model_s/
    │   ├── model.onnx
    │   └── model.spec.json
    └── model_e/
        ├── model.onnx
        └── model.spec.json
```

Nothing is cloned from GitHub or downloaded from HuggingFace during container
runtime. See `docs/deployment.md` for the pinned commit and download commands.
