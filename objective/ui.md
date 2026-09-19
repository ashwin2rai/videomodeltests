# Objective: Lightweight Web UI

CLI generator work (`generate.py`/`h3.py`, see `objective/objective.md`) is paused as of Phase 3 completion to build a small web UI on top of it, so generation can be kicked off and monitored from a browser — including a phone, via RunPod's HTTP proxy — instead of SSH'ing in and running the CLI by hand.

This is a separate, additive layer. It does not replace or change the CLI's behavior; it reuses `h3.py` directly.

## UI elements (and nothing more)

- Upload button (adds an image to `inputs/`)
- Dropdown of images already in `inputs/`
- Text prompt input
- Knobs: duration, steps, seed
- Generate button (adds the current form to a queue)
- Short progress line (queue count + current stage, e.g. `generating: 3/5`)
- Video player + dropdown of videos in `outputs/`

Deferred to a future v2 (explicitly out of scope now): selecting a frame from a played video and saving it back into `inputs/`.

## Architecture decisions

- **Framework**: Flask. A deliberate, explicit exception to `objective.md`'s "no Flask" line — scoped entirely to the new `web/` layer; `h3.py`/`generate.py` stay framework-free. New base dependency (needed for mock-mode local dev too, not just GPU mode).
- **Progress**: plain polling (`GET /api/status` every ~1s), not SSE/WebSockets.
- **No model-selector UI element.** The server auto-discovers the one DiT checkpoint already downloaded into `{MODELS_ROOT}/diffusion_models/` at startup (`h3.discover_dit_checkpoint()`) and uses it for the server's whole life. An `H3_MODEL_PATH` env var overrides/disambiguates if more than one checkpoint is present. Switching checkpoints means downloading a different one and restarting the server.
- **Backend selection is explicit at server startup**, not auto-detected and not a UI toggle: `make serve` (real `H3Backend`) vs `make serve-mock` (`MockH3Backend`), mirroring the existing `sync`/`sync-gpu` convention. `backend.load()` runs eagerly at startup so a broken setup fails loudly in the terminal immediately, rather than silently degrading to mock or failing on first click.
- **Efficient queueing via a stateful backend.** `H3Backend` now caches loaded models on `self` across calls (was: reload everything from disk on every single `generate()` call):
  - Qwen + both VAEs (fixed paths): loaded once, ever.
  - The DiT: loaded once (cached by path) and reused unless the path changes.
  - The existing GPU offload lifecycle between encode/denoise/decode phases (`unload_all_models()`) is unchanged — caching only avoids re-reading from disk, not how much is ever resident on the GPU at once.
  - This is backward-compatible: `generate.py`'s CLI creates a fresh `H3Backend()` per process invocation, so caching never triggers there.
  - **Always cache, no RAM-fit fallback logic** (deliberate choice — not solving for a checkpoint size class not yet in use). Instead, `h3.warn_if_ram_tight()` logs a one-line warning (real file sizes vs. `psutil.virtual_memory().total`, plus a rough ~15GB overhead allowance) if caching looks tight — visibility only, so a future crash has a log line to explain it, not a gate.
- **Queue semantics**: max 5 pending jobs (`queue.Queue(maxsize=5)`). "Delete queue" clears pending jobs only; a job already generating always finishes (no mid-denoise cancellation).
- **Hosting**: RunPod's built-in HTTP proxy (expose the port in the pod config). No app-level auth. The server binds `0.0.0.0` on a configurable `PORT` env var with zero RunPod-specific code, so this is a pure deploy-time choice, not an architectural lock-in — the same code works behind any other cloud's firewall/port-forward later.

## File layout

```
web/
├── server.py        # Flask app: routes, worker thread, queue, persistent backend
└── static/
    └── index.html   # single-page UI, inline <style>/<script>, no build step, no framework
```

Plus: `h3.py` (`H3Backend.__init__`, `discover_dit_checkpoint()`, `warn_if_ram_tight()`), `pyproject.toml` (`flask` base dependency), `Makefile` (`serve`, `serve-mock` targets).

## Explicitly deferred / out of scope

- Frame-extraction-and-save-to-`inputs/` (v2).
- Switching the DiT checkpoint without restarting the server.
- Mid-generation cancellation.
- Any auth/access control (revisit only if it moves off RunPod's proxy).
- SSE/WebSockets, per-item queue deletion/reordering.
- A RAM-fit fallback (only the warning exists).

---

# Status (2026-09-19)

**Built and verified on the local CPU dev machine:**

- `h3.py`: `H3Backend` is now stateful (constructor + cache-on-`self` logic in `generate()`); `discover_dit_checkpoint()` and `warn_if_ram_tight()` added. Existing test suite still passes unchanged (46 passed, 1 skipped) — no regression to the CLI path.
- `discover_dit_checkpoint()` hand-verified for all four cases: zero candidates, one candidate, multiple candidates, and the `H3_MODEL_PATH` override.
- `web/server.py` + `web/static/index.html` built per the architecture above.
- `make serve-mock` hand-tested end-to-end: index page loads, image upload validates and lands in `inputs/`, 20+ concurrent job submissions all enqueue and drain correctly through the worker thread with distinct timestamped output files (no collisions), `/api/status` reports queue length + current job progress/message correctly, `/api/queue` DELETE drains cleanly.
- `ruff check .` clean.
- README updated with a "Web UI" section (`make serve` / `make serve-mock`, `H3_MODEL_PATH`, RunPod proxy note).

**Not yet verified — needs the real RunPod GPU box:**

- That the caching actually produces a measurable speedup on a second queued job (no Qwen/VAE/DiT reload from disk).
- `warn_if_ram_tight()` firing correctly against real file sizes and real `psutil.virtual_memory().total` on the actual instance (only exercised on CPU dev machine so far, where `psutil` isn't installed and the function no-ops via its `ImportError` guard).
- `discover_dit_checkpoint()` against a real `diffusion_models/` directory with a real downloaded checkpoint.
- The full real-generation flow triggered from the browser UI itself (queue → progress text advancing through real denoising steps → a real playable MP4 appearing in the output dropdown).
- Delete-queue's "pending only" semantic under real generation timing (only demonstrated against near-instant mock jobs so far; correct by construction from `queue.Queue`'s single-consumer semantics, but not observed under a multi-minute real job).

**Not started:** the v2 frame-extraction/save-to-`inputs/` feature.
