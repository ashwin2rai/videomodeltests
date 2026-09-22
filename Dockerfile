# RunPod RTX 5090 image for the MiniMax H3 test harness (see objective/objective.md,
# objective/ui.md). Contains everything needed to serve the web UI and run real
# generation EXCEPT model weights: the fixed stock models (Qwen encoder + video/audio
# VAEs) and the swappable DiT checkpoint are fetched at container start, into the
# container's own ephemeral disk (no persistent volume by design — see
# objective/status.md's Docker planning notes). That means every fresh container start
# re-downloads them; this is a deliberate, accepted trade for a simpler deployment.
#
# The repo itself is `git clone`d inside the build (not COPYed from the local build
# context), so this single file is all you need to build the image — from any
# directory, on any machine, with no local checkout required. Pass --build-arg
# REPO_URL=... / REPO_REF=... to build from a fork or branch instead of the default.
# /app's .git is kept (not stripped) so the entrypoint can `git reset --hard` to the
# latest commit of that same branch on every container start — see
# docker-entrypoint.sh — so a plain app-code change doesn't need a rebuild. That does
# NOT cover pyproject.toml/uv.lock changes (the venv stays as baked at build time); the
# entrypoint warns loudly if it pulls a change to either. Set SKIP_GIT_PULL=1 to pin a
# container to exactly what was baked in.
#
# No nvidia/cuda base image: the `gpu` dependency group's torch wheel already bundles
# its own CUDA runtime, and RunPod's container runtime injects the host driver
# (libcuda.so.1 etc.) regardless of base image. A plain python:slim base is lighter and
# has nothing to duplicate. Revisit only if a future dependency needs the full CUDA
# toolkit (nvcc) at build or run time (e.g. a SageAttention3 source build — see
# objective/status.md's "Performance research" notes, currently paused).
#
# Runtime stage DOES need a host C/C++ toolchain, though (see "Runtime build tools"
# below) — this isn't a pure inference-only image. Triton (pulled in transitively by
# torch, and used by torch's own `_native` ops as well as any custom int8/ConvRot
# kernels a DiT checkpoint ships) JIT-compiles a small C launcher stub for every kernel
# the *first time it runs* — at inference time, not at image build time — and that
# needs a real `cc` in PATH. A minimal python:slim runtime image has none by default,
# which surfaced as `RuntimeError: Failed to find C compiler` on the first real
# generation through this image (gcc alone fixes that specific error; the fuller set
# below is deliberately more than the observed minimum — see next comment).

########################
# Stage 1: builder — anything needed to produce /opt/venv and the ComfyUI/app
# checkouts, none of which needs to survive into the runtime image (git, build tools,
# uv's own installer machinery).
########################
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# This repo. Cloned fresh rather than COPYed so the image is buildable from anywhere.
ARG REPO_URL=https://github.com/ashwin2rai/videomodeltests.git
ARG REPO_REF=main
RUN git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" /app \
    && rm -rf /app/tests /app/objective

# ComfyUI, imported headlessly as a library by the real backend — kept at the exact
# tag it's been validated against (see objective/status.md), bumped deliberately, not
# via `main` drift.
ARG COMFYUI_REF=v0.36.0
RUN git clone --depth 1 --branch "${COMFYUI_REF}" https://github.com/comfyanonymous/ComfyUI.git /ComfyUI \
    && rm -rf /ComfyUI/.git

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy

WORKDIR /app
# --no-dev drops pytest/ruff (not needed to serve or generate); --frozen refuses to
# silently re-resolve if the cloned uv.lock is out of date with pyproject.toml.
RUN uv sync --frozen --no-dev --group gpu

########################
# Stage 2: runtime — only what's needed to actually serve the UI and run generation.
########################
FROM python:3.12-slim-bookworm AS runtime

# Runtime build tools: deliberately more than the one confirmed failure (missing `cc`)
# strictly requires, since every one of these is cheap (a few tens of MB combined,
# irrelevant next to the ~20-66GB of model weights this image downloads on every
# start) and each closes off a real, plausible failure mode of the exact same shape
# — something that only bites at inference time, on a code path this repo doesn't
# control end-to-end (ComfyUI internals, a third-party DiT checkpoint's custom
# kernels, torch's own JIT paths) — rather than re-testing, hitting a new one, and
# coming back for another round-trip:
#   - build-essential: full gcc/g++/make/libc-dev toolchain, not just `gcc`. Triton
#     (and torch's own `_native` op registry) only strictly needs a C compiler for its
#     launcher-stub JIT, but g++/make cover any C++ extension a checkpoint's custom
#     kernels or a future torch.compile/inductor cpp-backend path might need — same
#     underlying "JIT-compiles at inference time" class of risk as the gcc bug already
#     hit, just a superset of it.
#   - python3-dev: guarantees Python.h/dev headers for whatever the above compiles
#     against. python:slim usually already has these (built from source), but it's a
#     few MB of insurance against that not holding on some future base image bump.
#   - libgomp1: PyTorch's CPU kernels commonly dlopen the system libgomp on Debian —
#     a classic "works on a full ML image, breaks on a minimal slim one" gap
#     (`OSError: libgomp.so.1: cannot open shared object file`) that a plain
#     python:slim base doesn't hit until the first real CPU-side tensor op needs it.
#   - ninja-build: several PyTorch/Triton JIT and CUDA-extension build paths prefer
#     ninja over make for parallel compilation; falls back gracefully if unused.
#   - git: the entrypoint's own `git reset --hard` update-on-start step needs it now
#     (see docker-entrypoint.sh and the top-of-file comment); also handy for the
#     `bash`/`check` debug modes and a future SageAttention3 source-build experiment
#     (objective/status.md, paused) that would `git clone` from inside a running
#     container.
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        ffmpeg curl ca-certificates bash git \
        build-essential python3-dev libgomp1 ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /ComfyUI /ComfyUI
COPY --from=builder /app /app

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_NO_SYNC=1 \
    COMFYUI_ROOT=/ComfyUI \
    PATH="/opt/venv/bin:${PATH}" \
    # Force stdout/stderr unbuffered. Python block-buffers stdout by default when it
    # isn't a TTY (true for `docker logs`), which can otherwise delay when a print()
    # (as opposed to `logging`, which already flushes per record) actually reaches the
    # log -- important here since the whole point of the recent h3.py/server.py logging
    # additions is real-time visibility into a long-running generation from the logs
    # alone, without being able to attach a debugger on a remote RunPod pod.
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN chmod +x scripts/fetch_models.sh scripts/docker-entrypoint.sh \
    && mkdir -p inputs outputs \
    # Pre-warm uvx's tool cache for the `hf` CLI so fetch_models.sh's first real call
    # at container start pays only for the model download, not also for installing
    # huggingface_hub[hf_xet] fresh into an ephemeral uvx env every time.
    && uvx --from 'huggingface_hub[hf_xet]' hf --version

EXPOSE 8000

ENTRYPOINT ["scripts/docker-entrypoint.sh"]
CMD ["serve"]
