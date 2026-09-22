# videomodeltests

Minimal CLI + web UI for MiniMax H3 (FL2VA) image-to-video + audio generation, targeting a single RTX 5090. See `objective/objective.md` (CLI spec), `objective/ui.md` (web UI spec), `objective/status.md` (progress, rationale, known issues, measured performance).

## Two modes

- **CPU mode** — mock backend, no GPU/weights/network. For developing the app itself.
- **GPU mode** — real inference via a headless ComfyUI import.

Both share the same CLI, web UI, and code paths outside the backend itself.

---

## CPU mode

```bash
make sync         # installs uv if needed, then `uv sync`
make mock         # generate.py --mock -> a placeholder output.mp4
make test         # runs the test suite
make serve-mock   # web UI at http://localhost:8000 against the mock backend
```

Without `make`:

```bash
uv sync
uv run python generate.py \
  --mock --model fake.safetensors --image tests/assets/test.jpg \
  --prompt "Test prompt" --output output.mp4
uv run pytest -q
```

Flags (real or mock): `--seed` (default 42), `--steps` (default 20), `--duration` — one of `5, 7.5, 10, 12.5, 15, 17.5, 20` seconds (default 5), `--resolution` — `768` (default) or `512` (faster). Omit `--image` for text-to-video.

---

## GPU mode

```bash
make setup                    # uv, ffmpeg, ComfyUI checkout, gpu deps, env check
make fetch-stock               # fixed Qwen encoder + video/audio VAEs (~21GB)
make fetch-dit DIT_URL=<url>   # your DiT checkpoint
make serve                     # web UI at http://localhost:8000
```

CLI:

```bash
uv run python generate.py \
  --model <path-to-dit>.safetensors \
  --image input.jpg \
  --prompt "The person slowly turns toward the camera." \
  --output output.mp4
```

A bare `--output`/`--image` filename lands in `outputs/`/`inputs/` (auto-created, gitignored); a path with an explicit directory is respected as-is.

Web UI: the DiT is auto-discovered from `diffusion_models/`; set `H3_MODEL_PATH` if more than one is present. Up to 5 prompts queue at once. Text-to-video, the `512px (fast)` resolution preset, and a frame extractor (grab a frame from any output video back into `inputs/`) are in the UI alongside the basic upload/prompt/generate flow.

Step by step instead of `make setup`: `make uv`, `make ffmpeg`, `make comfyui`, `make sync-gpu`, `make check` — each is a no-op if already satisfied, and `make check` re-verifies the setup on its own at any time.

---

## Configuration

| Var | Default | Meaning |
|---|---|---|
| `COMFYUI_ROOT` | sibling `ComfyUI` dir | Where the real backend imports `comfy.sd` from. |
| `MODELS_ROOT` | `$COMFYUI_ROOT/models` | Qwen/VAE/DiT weights (ComfyUI's standard layout). |
| `H3_MODEL_PATH` | auto-discover | Web UI: force a specific DiT checkpoint. |
| `H3_MOCK` | unset | Web UI: run against the mock backend (`make serve-mock` sets this). |
| `PORT` | `8000` | Web UI listen port. |
| `HF_TOKEN` | unset | Hugging Face token (higher rate limits, gated repos) — or a `.env` file in the repo root, or `uvx --from huggingface_hub hf auth login` once. |
| `HF_XET_HIGH_PERFORMANCE` | auto (≥64GB RAM) | Force hf-xet's fast transfer mode on (`1`) or off (`0`). |

Nothing else is configurable by design — see `objective/objective.md`'s "Do not expose" list.

### Docker (RunPod template)

```bash
curl -O https://raw.githubusercontent.com/ashwin2rai/videomodeltests/main/Dockerfile
docker build --platform linux/amd64 -t h3-test .
docker run --gpus all -p 8000:8000 -e DIT_URL=<url-to-your-dit-checkpoint> h3-test
```

`--platform linux/amd64` is required when building from an ARM machine (e.g. Apple Silicon). Build from a fork/branch with `--build-arg REPO_URL=... --build-arg REPO_REF=...`. `CMD` defaults to `serve`; other modes: `serve-mock`, `check`, `bash`. On RunPod: size the Container Disk well above image size + ~90GB (no persistent volume — models re-download every start), expose `PORT` (default `8000`) as an HTTP service, and set `DIT_URL`/`HF_TOKEN` as template env vars.

The container updates its own checkout from git on every start, so a restart alone picks up app-code changes without a rebuild; set `SKIP_GIT_PULL=1` to pin it to exactly what was baked into the image instead. Rebuild the image whenever `pyproject.toml`/`uv.lock` change — the entrypoint warns if it detects that case. See `objective/status.md` for how this works.

### Troubleshooting: CUDA OOM

If generation OOMs despite the model fitting in VRAM: check `nvidia-smi` for other processes holding memory, and try a lower `--steps` on a non-turbo checkpoint to rule it out as a variable. Don't work around it by changing resolution/frame count — not exposed by design. See `objective/status.md` for the root cause already fixed here.

---

## Checkpoint support

Only the MiniMax H3 FL2VA DiT is swappable, as a `.safetensors` path (BF16, or supported INT8/INT8-ConvRot). See `objective/status.md` for validation details and the FL2VA/Ref2VA limitation.
