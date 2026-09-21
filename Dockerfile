# RunPod RTX 5090 image for the MiniMax H3 test harness (see objective/objective.md,
# objective/ui.md). Contains everything needed to serve the web UI and run real
# generation EXCEPT model weights: the fixed stock models (Qwen encoder + video/audio
# VAEs) and the swappable DiT checkpoint are fetched at container start, into the
# container's own ephemeral disk (no persistent volume by design — see
# objective/status.md's Docker planning notes). That means every fresh container start
# re-downloads them; this is a deliberate, accepted trade for a simpler deployment.
#
# No nvidia/cuda base image: the `gpu` dependency group's torch wheel already bundles
# its own CUDA runtime, and RunPod's container runtime injects the host driver
# (libcuda.so.1 etc.) regardless of base image. A plain python:slim base is lighter and
# has nothing to duplicate. Revisit only if a future dependency needs nvcc/the CUDA
# toolkit at build or run time (e.g. a SageAttention3 source build — see
# objective/status.md's "Performance research" notes, currently paused).

########################
# Stage 1: builder — anything needed to produce /opt/venv and the ComfyUI checkout,
# none of which needs to survive into the runtime image (git, build tools, uv's own
# installer machinery).
########################
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Keep in sync with the Makefile's COMFYUI_REF — the exact tag the real backend has
# been validated against (see objective/status.md). Bumped deliberately, not via `main`.
ARG COMFYUI_REF=v0.36.0
RUN git clone --depth 1 --branch "${COMFYUI_REF}" https://github.com/comfyanonymous/ComfyUI.git /ComfyUI \
    && rm -rf /ComfyUI/.git

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
# --no-dev drops pytest/ruff (not needed to serve or generate); --frozen refuses to
# silently re-resolve if uv.lock is out of date with pyproject.toml.
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

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_NO_SYNC=1 \
    COMFYUI_ROOT=/ComfyUI \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# App code + the bits fetch_models.sh/check_env.py/`make check` need. Deliberately no
# tests/ or objective/ — irrelevant at runtime, see .dockerignore.
COPY pyproject.toml uv.lock .python-version Makefile ./
COPY generate.py h3.py ./
COPY web/ web/
COPY scripts/fetch_models.sh scripts/check_env.py scripts/docker-entrypoint.sh scripts/

RUN chmod +x scripts/fetch_models.sh scripts/docker-entrypoint.sh \
    && mkdir -p inputs outputs \
    # Pre-warm uvx's tool cache for the `hf` CLI so fetch_models.sh's first real call
    # at container start pays only for the model download, not also for installing
    # huggingface_hub[hf_xet] fresh.
    && uvx --from 'huggingface_hub[hf_xet]' hf --version

EXPOSE 8000

ENTRYPOINT ["scripts/docker-entrypoint.sh"]
CMD ["serve"]
