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
#
# No nvidia/cuda base image: the `gpu` dependency group's torch wheel already bundles
# its own CUDA runtime, and RunPod's container runtime injects the host driver
# (libcuda.so.1 etc.) regardless of base image. A plain python:slim base is lighter and
# has nothing to duplicate. Revisit only if a future dependency needs nvcc/the CUDA
# toolkit at build or run time (e.g. a SageAttention3 source build — see
# objective/status.md's "Performance research" notes, currently paused).

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
    && rm -rf /app/.git /app/tests /app/objective

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

RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates bash \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /ComfyUI /ComfyUI
COPY --from=builder /app /app

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_NO_SYNC=1 \
    COMFYUI_ROOT=/ComfyUI \
    PATH="/opt/venv/bin:${PATH}"

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
