# videomodeltests

Minimal CLI + a lightweight web UI for MiniMax H3 (FL2VA) image-to-video + audio generation, targeting a single RTX 5090. See `objective/objective.md` for the CLI's full spec, `objective/ui.md` for the web UI's, and `objective/status.md` for detailed progress notes. This is inference experimentation tooling, not a product.

## Status

Phases 1-3 of the objective's plan are done: local skeleton, full real backend, and a first successful real generation on the target hardware (playable MP4, video + audio, within VRAM). Phase 4 (performance) and Phase 5 (verifying more checkpoints) are in progress — see `objective/status.md`. A web UI for queueing generations from a browser has also been added — see `objective/ui.md` — and since then has grown text-to-video support, a frame extractor (pull a frame out of a generated video and save it back as a new input image), and input/output thumbnails.

## Two modes

- **CPU mode** — no GPU, no model weights, no ComfyUI checkout, no network access. Runs the CLI or the web UI against a mock backend that fakes generation. Use this for developing/testing application code (CLI, validation, the web UI itself).
- **GPU mode** — real inference via a headless ComfyUI import on an actual RTX 5090. Use this for developing the model-loading/inference pipeline, or for real generation.

Both modes share the same CLI, web UI, and code paths outside the backend itself.

---

## CPU mode (dev, no GPU)

```bash
make sync    # installs uv if needed, then `uv sync`
```

### CLI

```bash
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

Other optional flags (real or mock): `--seed` (default 42), `--steps` (default 20), `--duration` — one of `5, 7.5, 10, 12.5, 15, 17.5, 20` seconds (default 5; each maps internally to a fixed frame count, see `h3.DURATION_PRESETS`). Omit `--image` entirely for text-to-video.

### Web UI

```bash
make serve-mock   # starts the UI at http://localhost:8000 (override with PORT=...) against the mock backend
```

Open the printed URL in a browser. Upload an image (or leave the image dropdown on "None (text-to-video)"), queue a prompt, and the mock backend fakes a full generation end-to-end (progress log, output dropdown, video player) — no GPU, model files, or network access involved. Use this to develop/test the UI itself before touching real hardware.

---

## GPU mode (real inference)

### Automated setup

```bash
make setup
```

Installs `uv` and `ffmpeg` if missing, clones the pinned ComfyUI checkout (`COMFYUI_REF` in the `Makefile` — the real backend imports it headlessly as a library, no server/UI, rather than vendoring MiniMax H3 code), runs `uv sync --group gpu`, then verifies the whole chain (ffmpeg binaries, CUDA visible, ComfyUI importable). Safe to re-run — every step is a no-op if already done.

### Fetch weights

```bash
make fetch-stock                     # fixed Qwen encoder + video/audio VAEs (~21GB)
make fetch-dit DIT_URL=<url>         # your chosen swappable DiT checkpoint
```

### CLI

```bash
uv run python generate.py \
  --model /workspace/ComfyUI/models/diffusion_models/my_h3_model.safetensors \
  --image input.jpg \
  --prompt "The person slowly turns toward the camera." \
  --output output.mp4
```

A bare `--output` filename (no directory) lands in `outputs/` (auto-created, gitignored); a path with an explicit directory is respected as-is. The same applies to `--image`, which lands in `inputs/`.

### Web UI

```bash
make serve        # requires a DiT checkpoint already in diffusion_models/ (see "Fetch weights" above)
```

Open `http://localhost:8000` (override with `PORT=...`). On RunPod, expose that port as an HTTP service in the pod config and use the proxy URL it gives you instead — the server binds `0.0.0.0` and has no RunPod-specific code, so this is purely a deploy-time choice.

The DiT checkpoint is auto-discovered from `diffusion_models/` at startup — if more than one `.safetensors` file is present there, set `H3_MODEL_PATH` to pick one explicitly. The real backend keeps the fixed Qwen/VAE models and the discovered DiT loaded in memory across queued jobs instead of reloading them from disk every time — only the first generation after startup pays the full load cost. Up to 5 prompts can be queued; "Clear queue" removes pending jobs only, a job already generating always finishes.

UI features beyond the basic upload/prompt/generate flow:

- **Text-to-video** — pick "None (text-to-video)" in the image dropdown instead of an uploaded image.
- **Frame extractor** — pick any video already in `outputs/`, scrub to a frame, and save it straight back into `inputs/` as a new starting image (handy for chaining generations). The preview pillarboxes non-16:9 videos to show their real aspect ratio rather than stretching them.
- **Thumbnails** — the input-image and output-video dropdowns show a preview thumbnail next to the current selection.

### Step by step (without `make setup`)

```bash
make uv          # install uv if missing
make ffmpeg      # install ffmpeg if missing
make comfyui     # clone the pinned ComfyUI checkout into COMFYUI_ROOT
make sync-gpu    # uv sync --group gpu (torch, torchaudio, comfy-kitchen, etc.)
make check       # verify ffmpeg, CUDA, and the ComfyUI import all actually work
```

Each target is independently a no-op if already satisfied, so you can re-run any of them freely. `make check` on its own re-verifies setup any time.

---

## Configuration

Env vars, read once at import time:

| Var | Default | Meaning |
|---|---|---|
| `COMFYUI_ROOT` | sibling `ComfyUI` dir next to this repo checkout | Where the real backend imports `comfy.sd` etc. from (`h3.py`). |
| `MODELS_ROOT` | `$COMFYUI_ROOT/models` | Where the fixed Qwen/VAE weights and swappable DiT checkpoints live (ComfyUI's standard layout: `text_encoders/`, `vae/`, `diffusion_models/`) (`h3.py`). |
| `H3_MODEL_PATH` | unset (auto-discover) | Web UI only: forces a specific DiT checkpoint instead of auto-discovering the one file in `diffusion_models/` — required if more than one is present (`web/server.py`). |
| `H3_MOCK` | unset (real backend) | Web UI only: set to run `web/server.py` against the mock backend instead of the real one (this is what `make serve-mock` sets). |
| `PORT` | `8000` | Web UI only: the port `web/server.py` listens on. |

`COMFYUI_ROOT`'s default (in both the `Makefile` and `h3.py`) is computed from the repo checkout's own location, not a hardcoded path — it works unmodified whether the parent directory is `workspace` (RunPod) or `workspaces` (Codespaces). Override any of these if your setup mounts things elsewhere.

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

`--duration` presets beyond 10s (12.5/15/17.5/20s) are new and unmeasured — the 10s preset alone already peaks at ~28GB VRAM on a 32GB card, so treat anything longer as untested on the 32GB target until it's been run and measured.

### Troubleshooting: CUDA OOM

`generate.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` itself (an explicit shell value always wins) — without it, denoising can OOM on a 32GB card from allocator fragmentation even when the model fits. If you still hit CUDA OOM:

- Check `nvidia-smi` — nothing else should be holding VRAM.
- Lower `--steps` on a base (non-turbo) checkpoint to rule it out as a variable.
- Don't work around it by changing resolution or frame count — those aren't exposed on the CLI by design; report the failure and investigate instead.

---

## Checkpoint support

Only the MiniMax H3 FL2VA DiT is swappable, as a filesystem path to a `.safetensors` file (BF16, or supported INT8/INT8-ConvRot). `h3.validate_checkpoint()` inspects the header and rejects GGUF, malformed files, and anything without the real H3 tensor signature. **Known limitation:** FL2VA and Ref2VA checkpoints have identical tensor layouts — Ref2VA rejection is filename-based only, not a real content check.
