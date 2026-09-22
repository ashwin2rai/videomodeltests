#!/usr/bin/env python3
"""Verifies the pieces `make setup` can't check from inside a shell recipe:
GPU deps import correctly, CUDA is visible, and ComfyUI imports headlessly.
Run via `make check` (also runs at the end of `make setup`). Exits non-zero
only for things that would actually block real generation — ffmpeg/ComfyUI
presence are enforced earlier in the Makefile's dependency chain, not here.
"""
import os
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

# Triton (pulled in by torch, and used by torch's own `_native` ops as well as any
# custom int8/ConvRot kernels a DiT checkpoint ships) JIT-compiles a small C launcher
# stub the first time a given kernel runs -- at inference time, not at process start,
# so this would otherwise go unnoticed until partway through a real generation (see
# objective/status.md). Check for a compiler the same way triton's own
# `_find_compiler()` does: $CC first, then the same fallback list it tries.
_cc_candidates = [os.environ.get("CC")] + ["cc", "gcc", "clang"]
_cc_found = next((c for c in _cc_candidates if c and shutil.which(c)), None)
report("C compiler (cc/gcc/clang, or $CC)", _cc_found is not None, _cc_found or "none found")

try:
    import torch
except ImportError as e:
    # Show the real message, not just "missing" -- an ImportError here can mean either
    # "gpu group never installed" or "installed but a system shared library it dlopens
    # is missing" (e.g. libgomp on a minimal base image), and those need different fixes.
    report("gpu dependency group (uv sync --group gpu)", False, str(e))
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
