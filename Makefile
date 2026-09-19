UV := $(shell command -v uv 2>/dev/null)
ifeq ($(UV),)
UV := $(HOME)/.local/bin/uv
endif

# Defaults to a sibling "ComfyUI" dir next to this checkout (e.g. this repo at
# /workspace/videomodeltests -> /workspace/ComfyUI, or /workspaces/foo ->
# /workspaces/ComfyUI), so it works whether the parent dir is named
# "workspace" (RunPod) or "workspaces" (Codespaces). Override if your setup
# mounts ComfyUI somewhere else.
COMFYUI_ROOT ?= $(abspath $(CURDIR)/..)/ComfyUI
# The exact tag the real backend has been validated against (see
# objective/status.md). Bump deliberately, not implicitly via `main` drift.
COMFYUI_REF ?= v0.36.0

.PHONY: help uv ffmpeg sync sync-gpu comfyui setup check test mock clean fetch-stock fetch-dit lint install-hooks

help:
	@echo "make setup       - everything needed for real generation except model weights:"
	@echo "                   uv, ffmpeg, the pinned ComfyUI checkout, the GPU dependency"
	@echo "                   group, then verifies it all actually works"
	@echo "make check       - re-run just the verification step from 'make setup'"
	@echo "make uv          - install uv if it's not already available"
	@echo "make ffmpeg      - install ffmpeg if it's not already available"
	@echo "make sync        - install/sync dependencies with uv"
	@echo "make sync-gpu    - sync dependencies including the real backend's GPU stack"
	@echo "make comfyui     - clone the pinned ComfyUI checkout (COMFYUI_REF=$(COMFYUI_REF))"
	@echo "                   the real backend imports as a library, into COMFYUI_ROOT"
	@echo "                   (default $(COMFYUI_ROOT); no-op if already present)"
	@echo "make lint        - run ruff over the whole project"
	@echo "make install-hooks - install the ruff pre-commit hook into .git/hooks/"
	@echo "make test        - run the test suite"
	@echo "make mock        - run a mock generation (no GPU/model files needed)"
	@echo "make fetch-stock - download the fixed Qwen encoder + video/audio VAEs"
	@echo "make fetch-dit DIT_URL=<url> [DIT_NAME=<filename>] - download a DiT checkpoint"
	@echo "                (add HF_TOKEN=<token> to either fetch target for authenticated"
	@echo "                 Hugging Face requests - higher rate limits, gated repo access)"
	@echo "make clean       - remove caches and build artifacts"

uv:
	@if [ ! -x "$(UV)" ]; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

ffmpeg:
	@if ! command -v ffmpeg >/dev/null 2>&1; then \
		echo "Installing ffmpeg..."; \
		SUDO=""; [ "$$(id -u)" != "0" ] && SUDO=sudo; \
		$$SUDO apt-get update -qq && $$SUDO apt-get install -y ffmpeg; \
	fi

# WARNING: `uv sync` (this target) removes any dependency-group packages not
# requested — running it on a machine that already has the `gpu` group
# installed will UNINSTALL torch/comfy-kitchen/etc. from .venv. Because the
# venv also has --system-site-packages set, that failure is silent: imports
# keep working, just against whatever unvalidated torch build the system
# happens to ship. On a GPU box, always use `sync-gpu` below instead.
sync: uv
	$(UV) sync
	mkdir -p inputs outputs

sync-gpu: uv
	$(UV) sync --group gpu
	mkdir -p inputs outputs

comfyui:
	@if [ -d "$(COMFYUI_ROOT)/.git" ]; then \
		echo "ComfyUI already present at $(COMFYUI_ROOT) ($$(git -C "$(COMFYUI_ROOT)" describe --tags --always))"; \
	else \
		echo "Cloning ComfyUI $(COMFYUI_REF) into $(COMFYUI_ROOT)..."; \
		git clone --depth 1 --branch "$(COMFYUI_REF)" https://github.com/comfyanonymous/ComfyUI.git "$(COMFYUI_ROOT)"; \
	fi

# No shell `activate` step: every target here runs through `uv run`, which
# transparently uses .venv without needing it sourced into your shell.
check: uv
	@COMFYUI_ROOT="$(COMFYUI_ROOT)" $(UV) run python3 scripts/check_env.py

setup: uv ffmpeg comfyui sync-gpu check

lint: uv
	$(UV) run ruff check .

install-hooks:
	cp scripts/pre-commit .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit
	@echo "Installed pre-commit hook (runs 'ruff check .' before each commit)."

test: uv ffmpeg
	$(UV) run pytest -q

mock: uv
	$(UV) run python generate.py \
		--mock \
		--model fake.safetensors \
		--image tests/assets/test.jpg \
		--prompt "Test prompt" \
		--output output.mp4

fetch-stock: uv
	HF_TOKEN="$(HF_TOKEN)" ./scripts/fetch_models.sh stock

fetch-dit: uv
	@[ -n "$(DIT_URL)" ] || { echo "Usage: make fetch-dit DIT_URL=<url> [DIT_NAME=<filename>]"; exit 1; }
	HF_TOKEN="$(HF_TOKEN)" ./scripts/fetch_models.sh dit "$(DIT_URL)" "$(DIT_NAME)"

clean:
	rm -rf .pytest_cache __pycache__ tests/__pycache__ outputs/
