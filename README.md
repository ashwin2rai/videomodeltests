# videomodeltests

Minimal CLI for MiniMax H3 (FL2VA) image-to-video + audio generation, targeting a single RTX 5090. See `objective/objective.md` for the full spec and `objective/status.md` for detailed progress notes. This is inference experimentation tooling, not a product.

## Status

Phases 1-3 of the objective's plan are done: local skeleton, full real backend, and a first successful real generation on the target hardware (playable MP4, video + audio, within VRAM). Phase 4 (performance) and Phase 5 (verifying more checkpoints) are in progress — see `objective/status.md`.

## Two modes

- **CPU mode** — no GPU, no model weights, no ComfyUI checkout. Runs the CLI against a mock backend that fakes generation. Use this for developing/testing the CLI, validation, and surrounding application code.
- **GPU mode** — real inference via a headless ComfyUI import on an actual RTX 5090. Use this for developing the model-loading/inference pipeline itself, or for real generation.

Both modes share the same CLI and code paths outside the backend itself.

---

## CPU mode (dev, no GPU)

```bash
make sync    # installs uv if needed, then `uv sync`
make mock    # runs generate.py --mock, writes a placeholder output.mp4
make test    # runs the test suite (installs ffmpeg if needed)
```

Or without `make`:

```bash
uv sync
uv run python generate.py \
  --mock \
  --model fake.safetensors \
  --image tests/assets/test.jpg \
  --prompt "Test prompt" \
  --output output.mp4
uv run pytest -q
```

`--mock` needs no CUDA, no model files, and no network access.

---

## GPU mode (real inference)

### Automated

```bash
make setup
```

Installs `uv` and `ffmpeg` if missing, clones the pinned ComfyUI checkout (`COMFYUI_REF` in the `Makefile` — the real backend imports it headlessly as a library, no server/UI, rather than vendoring MiniMax H3 code), runs `uv sync --group gpu`, then verifies the whole chain (ffmpeg binaries, CUDA visible, ComfyUI importable). Safe to re-run — every step is a no-op if already done.

Then fetch weights and generate:

```bash
make fetch-stock                     # fixed Qwen encoder + video/audio VAEs (~21GB)
make fetch-dit DIT_URL=<url>         # your chosen swappable DiT checkpoint
uv run python generate.py \
  --model /workspace/ComfyUI/models/diffusion_models/my_h3_model.safetensors \
  --image input.jpg \
  --prompt "The person slowly turns toward the camera." \
  --output output.mp4
```

A bare `--output` filename (no directory) lands in `outputs/` (auto-created, gitignored); a path with an explicit directory is respected as-is.

### Step by step (without `make setup`)

```bash
make uv          # install uv if missing
make ffmpeg      # install ffmpeg if missing
make comfyui     # clone the pinned ComfyUI checkout into COMFYUI_ROOT
make sync-gpu    # uv sync --group gpu (torch, torchaudio, comfy-kitchen, etc.)
make check       # verify ffmpeg, CUDA, and the ComfyUI import all actually work
```

Each target is independently a no-op if already satisfied, so you can re-run any of them freely. `make check` on its own re-verifies setup any time.

### Configuration

Two env vars, read once by `h3.py` at import time:

| Var | Default | Meaning |
|---|---|---|
| `COMFYUI_ROOT` | sibling `ComfyUI` dir next to this repo checkout | Where the real backend imports `comfy.sd` etc. from. |
| `MODELS_ROOT` | `$COMFYUI_ROOT/models` | Where the fixed Qwen/VAE weights and swappable DiT checkpoints live (ComfyUI's standard layout: `text_encoders/`, `vae/`, `diffusion_models/`). |

`COMFYUI_ROOT`'s default (in both the `Makefile` and `h3.py`) is computed from the repo checkout's own location, not a hardcoded path — it works unmodified whether the parent directory is `workspace` (RunPod) or `workspaces` (Codespaces). Override either var if your setup mounts things elsewhere.

Nothing else is configurable by design — see `objective/objective.md`'s "Do not expose" list.

### Hugging Face authentication (optional)

Needed only to avoid rate limits or access gated repos when fetching models:

- `export HF_TOKEN=<your token>`, or
- a `.env` file in the repo root with `HF_TOKEN=<your token>` (gitignored, loaded automatically by `fetch_models.sh`; an explicit shell `HF_TOKEN` wins over it).

The token is only ever sent to `huggingface.co`, never to a non-HF URL like CivitAI. (Alternative: `uvx --from huggingface_hub hf auth login` once, stores a token `huggingface_hub` reuses automatically.)

### Known tested checkpoints

One community int8/ConvRot checkpoint has been verified end-to-end — see `objective/status.md` for the verification trail and the other checkpoints still to be tested. It's a turbo/LoRA-merged variant needing only ~4-6 steps rather than the default 20 (`--steps 5`); that's checkpoint-specific, not a CLI default change.

### Measured performance

First successful real generation (RTX 5090): 1376×768, 124 frames, 5 steps (turbo checkpoint above) → 162.4s total (1.2s load + 161.2s generation). Full stage breakdown and peak VRAM/RAM aren't captured yet — see `objective/status.md`.

### Troubleshooting: CUDA OOM

`generate.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` itself (an explicit shell value always wins) — without it, denoising can OOM on a 32GB card from allocator fragmentation even when the model fits. If you still hit CUDA OOM:

- Check `nvidia-smi` — nothing else should be holding VRAM.
- Lower `--steps` on a base (non-turbo) checkpoint to rule it out as a variable.
- Don't work around it by changing resolution or frame count — those aren't exposed on the CLI by design; report the failure and investigate instead.

---

## Web UI

A very small Flask UI for queueing generations from a browser (including a phone, via RunPod's HTTP proxy) instead of running the CLI by hand. Same `inputs/`/`outputs/` folders and `h3.py` backend as the CLI.

```bash
make serve-mock   # no GPU/model files needed, for developing/testing the UI itself
make serve        # real backend; requires a DiT checkpoint already in diffusion_models/
```

Then open `http://localhost:8000` (override with `PORT=...`). On RunPod, expose that port as an HTTP service in the pod config and use the proxy URL it gives you instead.

The real backend keeps the fixed Qwen/VAE models and the auto-discovered DiT checkpoint loaded in memory across queued jobs instead of reloading them from disk every time — only the first generation after startup pays the full load cost. Up to 5 prompts can be queued; "Delete queue" clears pending jobs only, a job already generating always finishes. If more than one `.safetensors` file exists in `diffusion_models/`, set `H3_MODEL_PATH` to pick one explicitly.

## Checkpoint support

Only the MiniMax H3 FL2VA DiT is swappable, as a filesystem path to a `.safetensors` file (BF16, or supported INT8/INT8-ConvRot). `h3.validate_checkpoint()` inspects the header and rejects GGUF, malformed files, and anything without the real H3 tensor signature. **Known limitation:** FL2VA and Ref2VA checkpoints have identical tensor layouts — Ref2VA rejection is filename-based only, not a real content check.
