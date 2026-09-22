# videomodeltests

Minimal CLI + a lightweight web UI for MiniMax H3 (FL2VA) image-to-video + audio generation, targeting a single RTX 5090. See `objective/objective.md` for the CLI's full spec, `objective/ui.md` for the web UI's, and `objective/status.md` for progress notes, known limitations, and measured performance. This is inference experimentation tooling, not a product.

## Two modes

- **CPU mode** — no GPU, no model weights, no ComfyUI checkout, no network access. Runs the CLI or the web UI against a mock backend that fakes generation. Use this for developing/testing application code (CLI, validation, the web UI itself).
- **GPU mode** — real inference via a headless ComfyUI import on an actual RTX 5090. Use this for real generation.

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

Other optional flags (real or mock): `--seed` (default 42), `--steps` (default 20), `--duration` — one of `5, 7.5, 10, 12.5, 15, 17.5, 20` seconds (default 5; each maps internally to a fixed frame count, see `h3.DURATION_PRESETS`), `--resolution` — `768` (default, native/recommended) or `512` (faster, lower quality — see `h3.RESOLUTION_PRESETS`). Omit `--image` entirely for text-to-video.

### Web UI

```bash
make serve-mock   # starts the UI at http://localhost:8000 (override with PORT=...) against the mock backend
```

Open the printed URL in a browser. Upload an image (or leave the image dropdown on "None (text-to-video)"), queue a prompt, and the mock backend fakes a full generation end-to-end (progress log, output dropdown, video player) — no GPU, model files, or network access involved.

---

## GPU mode (real inference)

### Automated setup

```bash
make setup
```

Installs `uv` and `ffmpeg` if missing, clones the pinned ComfyUI checkout (`COMFYUI_REF` in the `Makefile` — the real backend imports it headlessly as a library, no server/UI), runs `uv sync --group gpu`, then verifies the whole chain (ffmpeg binaries, CUDA visible, ComfyUI importable). Safe to re-run — every step is a no-op if already done.

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

Open `http://localhost:8000` (override with `PORT=...`). On RunPod, expose that port as an HTTP service in the pod config and use the proxy URL it gives you instead — the server binds `0.0.0.0` and has no RunPod-specific code.

The DiT checkpoint is auto-discovered from `diffusion_models/` at startup — if more than one `.safetensors` file is present there, set `H3_MODEL_PATH` to pick one explicitly. The real backend keeps the fixed Qwen/VAE models and the discovered DiT loaded in memory across queued jobs instead of reloading them from disk every time. Up to 5 prompts can be queued; "Clear queue" removes pending jobs only, a job already generating always finishes.

UI features beyond the basic upload/prompt/generate flow:

- **Text-to-video** — pick "None (text-to-video)" in the image dropdown instead of an uploaded image.
- **Resolution knob** — `768px (default)` or `512px (fast)`, same two presets as the CLI's `--resolution`.
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

### Fast downloads (hf-xet)

`fetch_models.sh` installs `hf` with the `hf_xet` extra for faster Hugging Face transfers. On a box with at least ~64GB RAM it also auto-sets `HF_XET_HIGH_PERFORMANCE=1` (saturates network + all CPU cores); override either direction with an explicit `HF_XET_HIGH_PERFORMANCE=0` or `=1`. Only affects `fetch_models.sh`'s HF downloads, not the curl fallback for non-HF sources like CivitAI.

### Docker (for a RunPod template)

The `Dockerfile` `git clone`s this repo itself during the build rather than copying local files, so it's a single self-contained file — build it from any directory, on any machine, with no local checkout needed:

```bash
curl -O https://raw.githubusercontent.com/ashwin2rai/videomodeltests/main/Dockerfile
docker build --platform linux/amd64 -t h3-test .
docker run --gpus all -p 8000:8000 -e DIT_URL=<url-to-your-DiT-checkpoint> h3-test
```

RunPod GPU pods are always `linux/amd64` — always pass `--platform linux/amd64` explicitly when building, especially from an Apple Silicon Mac or other ARM machine (which otherwise defaults to `arm64` and fails at container start with `exec format error`, since none of the image's binaries match the host's CPU architecture).

Build from a fork/branch with `--build-arg REPO_URL=... --build-arg REPO_REF=...`.

The image contains everything needed to serve the UI and run real generation — the pinned ComfyUI checkout, the `gpu` dependency group, ffmpeg — except model weights. `scripts/docker-entrypoint.sh` (the image's `ENTRYPOINT`) fetches the fixed stock models on every container start, and the DiT checkpoint too if `DIT_URL` is set (optionally with `DIT_NAME` to name the file); there's no persistent volume, so this re-downloads on every fresh container. Set `HF_TOKEN`/`HF_XET_HIGH_PERFORMANCE` as normal `docker run -e` env vars.

`CMD` defaults to `serve` (the real backend); other entrypoint modes: `serve-mock` (no GPU/weights needed), `check` (runs `scripts/check_env.py` and exits), `bash` (a shell, for debugging — e.g. `docker run --gpus all -it h3-test bash`). On RunPod, set the Container Disk size well above the image size plus ~90GB (stock models + a DiT checkpoint), expose the container's port (`PORT`, default 8000) as an HTTP service, and set `DIT_URL` as a template env var.

### Troubleshooting: CUDA OOM

`generate.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` itself (an explicit shell value always wins) — without it, denoising can OOM on a 32GB card from allocator fragmentation even when the model fits. If you still hit CUDA OOM:

- Check `nvidia-smi` — nothing else should be holding VRAM.
- Lower `--steps` on a base (non-turbo) checkpoint to rule it out as a variable.
- Don't work around it by changing resolution or frame count — those aren't exposed on the CLI by design; report the failure and investigate instead.

---

## Checkpoint support

Only the MiniMax H3 FL2VA DiT is swappable, as a filesystem path to a `.safetensors` file (BF16, or supported INT8/INT8-ConvRot). `h3.validate_checkpoint()` inspects the header and rejects GGUF, malformed files, and anything without the real H3 tensor signature. FL2VA and Ref2VA checkpoints share identical tensor layouts, so Ref2VA rejection here is filename-based only, not a content check.
