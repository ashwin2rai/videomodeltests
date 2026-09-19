#!/usr/bin/env python3
"""Verifies the pieces `make setup` can't check from inside a shell recipe:
GPU deps import correctly, CUDA is visible, and ComfyUI imports headlessly.
Run via `make check` (also runs at the end of `make setup`). Exits non-zero
only for things that would actually block real generation — ffmpeg/ComfyUI
presence are enforced earlier in the Makefile's dependency chain, not here.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import h3

ok = True


def report(label, passed, detail=""):
    global ok
    ok = ok and passed
    status = "OK" if passed else "MISSING"
    print(f"[{status:7}] {label}" + (f" ({detail})" if detail else ""))


report("ffmpeg", shutil.which("ffmpeg") is not None)
report("ffprobe", shutil.which("ffprobe") is not None)
report("ComfyUI checkout", Path(h3.COMFYUI_ROOT).is_dir(), h3.COMFYUI_ROOT)

try:
    import torch
except ImportError:
    report("gpu dependency group (uv sync --group gpu)", False)
else:
    report("gpu dependency group (uv sync --group gpu)", True, f"torch {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    report("CUDA device", cuda_ok, torch.cuda.get_device_name(0) if cuda_ok else "no GPU visible")

try:
    h3._import_comfy()
except ImportError as e:
    report("ComfyUI headless import (comfy.sd, nodes, ...)", False, str(e))
else:
    report("ComfyUI headless import (comfy.sd, nodes, ...)", True)

print()
if ok:
    print("Environment ready. Next: make fetch-stock, then make fetch-dit DIT_URL=<url>")
else:
    print("Some checks failed above — real generation won't work until they're fixed.")
    print("(--mock and the test suite don't need any of this.)")
sys.exit(0 if ok else 1)
