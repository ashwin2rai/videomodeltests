# Objective

Build an extremely small Python repository for testing MiniMax H3 image-to-video generation on a RunPod RTX 5090.

The repository should accept:

* a MiniMax H3 FL2VA DiT checkpoint in `.safetensors` format
* an input image
* a text prompt

and produce:

* a playable MP4
* containing both generated video and generated audio

The primary goal is **fast, minimal inference experimentation**, not a production application.

---

# Target Environment

Final inference environment:

* 1× NVIDIA RTX 5090
* 32 GB VRAM
* approximately 92 GB system RAM
* 16 vCPU
* CUDA-capable RunPod image
* approximately 100 GB total disk

Development will initially happen on a small computer that:

* may have no NVIDIA GPU
* has no model checkpoints
* should not download any model weights
* cannot perform real MiniMax inference

The repository must therefore be fully developable and testable locally without CUDA or model files.

---

# V1 Success Criterion

The project is successful when the following workflow works on RunPod:

```bash
python generate.py \
  --model /models/my_h3_model.safetensors \
  --image input.jpg \
  --prompt "The person slowly turns toward the camera." \
  --output output.mp4
```

The resulting `output.mp4` must:

* contain video
* contain generated audio
* be playable in normal desktop/mobile browsers and video players

The application should run within the RTX 5090's 32 GB VRAM limit.

Performance matters after basic correctness is established.

---

# Checkpoint Contract

V1 targets:

**ComfyUI-compatible MiniMax H3 FL2VA safetensors checkpoints.**

Expected checkpoint types include:

* BF16
* supported INT8
* supported INT8/ConvRot

Explicitly out of scope:

* GGUF
* arbitrary unknown safetensors layouts
* LoRAs
* model conversion utilities
* training
* Ref2VA

Only the MiniMax H3 DiT should be swappable by the user.

---

# Fixed Supporting Models

Use one known configuration for all supporting components.

These should not be exposed as normal CLI options.

Use fixed:

* MiniMax-compatible Qwen3-VL-32B compressed text/vision encoder
* MiniMax H3 video VAE
* MiniMax H3 audio VAE

Recommended Qwen encoder:

```text
qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
```

Keep the supporting model locations in a small constants/config section rather than scattering paths through the code.

Do not experiment with smaller Qwen models in V1.

---

# CLI

Keep the public interface intentionally tiny.

Required:

```text
--model
--image
--prompt
--output
```

Optional:

```text
--seed
--steps
```

Suggested defaults:

```text
seed: 42
steps: 20
frames: 124
fps: 24
```

Do not expose:

* negative prompt
* guidance scale
* arbitrary resolution
* scheduler selection
* quantization settings
* attention backend
* VAE paths
* Qwen path
* LoRA options
* frame count
* FPS

These can become development constants internally.

---

# Resolution

Do not expose free-form width and height.

V1 should use the native/recommended MiniMax H3 image-to-video resolution strategy.

Use:

```text
short edge = 768 px
```

Preserve the source image aspect ratio as closely as possible while producing dimensions valid for H3, including required dimension alignment.

For a typical 16:9 input this will be approximately:

```text
1344 × 768
```

Implement this in one small helper function.

Avoid complex resize, crop, padding, or preset systems.

Later performance experiments may evaluate a smaller canvas, but that is not part of the initial CLI.

---

# Minimal Architecture

Prefer something approximately this small:

```text
minimax-h3-test/
├── README.md
├── pyproject.toml
├── generate.py
├── h3.py
├── tests/
│   ├── test_cli.py
│   └── test_mock.py
└── .gitignore
```

Avoid adding files unless they have a clear purpose.

---

# h3.py

This contains the inference functionality.

Keep the API extremely small.

Conceptually:

```python
class H3Backend:
    def load(self):
        ...

    def generate(
        self,
        model_path,
        image_path,
        prompt,
        output_path,
        seed=42,
        steps=20,
        progress_callback=None,
    ):
        ...
```

Exact class names and structure are not important.

Simplicity is.

There should be one clean boundary between application code and actual GPU inference.

---

# Mock Backend

Provide a very small mock implementation for local development.

For example:

```python
class MockH3Backend:
    ...
```

It should require:

* no CUDA
* no Hugging Face access
* no model files
* no large dependencies

It should exercise the same basic application flow:

```text
load
→ validate image
→ report progress
→ pretend to generate
→ create or copy a tiny test output
```

The mock backend exists only so the CLI and surrounding application can be developed before RunPod is started.

Do not build an elaborate mocking framework.

A simple alternate class is enough.

Possible usage:

```bash
python generate.py \
  --mock \
  --model fake.safetensors \
  --image tests/assets/test.jpg \
  --prompt "Test prompt" \
  --output output.mp4
```

`--mock` is acceptable as a development-only option.

---

# Real Backend

The real backend should:

1. validate CUDA availability
2. load the fixed Qwen encoder
3. load the requested H3 FL2VA DiT
4. load the video/audio VAEs
5. preprocess the image
6. encode the image and prompt
7. run H3 inference
8. decode the generated video/audio
9. mux them into an MP4
10. report timing and memory statistics

Prefer existing maintained MiniMax H3 / ComfyUI model-loading code over writing custom checkpoint conversion logic.

The purpose of this repository is inference experimentation, not implementing MiniMax H3 from scratch.

---

# Important Loader Requirement

The loader must support the intended ComfyUI-style H3 safetensors layout.

Do not design the application specifically around Hugging Face repository IDs.

The primary model input is a filesystem path:

```text
/path/to/model.safetensors
```

This is important because the intended checkpoints may come from CivitAI, Hugging Face, or manual downloads.

Do not assume that every `.safetensors` file is compatible.

If practical, perform a small amount of checkpoint inspection before loading and produce a useful error for unsupported files.

Do not attempt to build a universal model-format detector.

---

# GPU Memory Strategy

The machine has 32 GB VRAM, so do not assume every component remains resident on the GPU simultaneously.

Prefer this conceptual lifecycle:

```text
Qwen
  ↓
encode image + prompt
  ↓
conditioning embeddings
  ↓
Qwen can be offloaded
  ↓
H3 DiT inference
  ↓
VAE decoding
```

Use existing model/offloading mechanisms where possible.

Avoid writing complicated manual memory-management machinery until measurements show it is necessary.

---

# Performance Priority

Performance is more important than maximizing compatibility across dozens of checkpoints.

Expected usage is approximately 2–3 selected H3 checkpoints.

After one model successfully generates a video, immediately record:

```text
model load time
conditioning/encoding time
denoising time
decode time
total generation time
peak VRAM
peak system RAM if easy to obtain
output resolution
steps
frames
```

Use these numbers before making optimizations.

Do not prematurely add large optimization frameworks.

---

# Progress Reporting

Even in V1, expose a tiny callback internally:

```python
progress_callback(progress, message)
```

Example stages:

```text
loading models
encoding prompt and image
generating video
decoding
writing output
done
```

If the underlying inference implementation exposes reliable per-step progress, use it.

Otherwise coarse stage progress is sufficient.

Do not build frontend or networking code around this yet.

---

# Local Tests

Keep tests deliberately simple.

They should run on the small development machine without CUDA or model downloads.

Test approximately:

### CLI parsing

Verify required arguments and defaults.

### Image validation

Reject nonexistent/unreadable images.

### Resolution helper

Verify common input ratios produce valid aligned dimensions.

### Mock generation

Verify the complete mock command creates an output file.

### Progress callback

Verify progress events can be received.

### Error handling

Verify useful errors for missing files and bad arguments.

Do not mock PyTorch internals.

Do not create elaborate integration test infrastructure.

---

# RunPod Smoke Test

The first real integration test happens on RunPod.

Procedure:

1. clone repository
2. install dependencies
3. copy/download fixed Qwen and VAE weights
4. obtain one intended H3 FL2VA checkpoint
5. run generation
6. verify MP4 video
7. verify audio
8. record performance measurements

The first objective is simply:

```text
one real checkpoint
+
one real image
+
one prompt
=
one valid MP4
```

Once this works, test the other one or two intended community checkpoints.

Do not spend time supporting models outside the intended set.

---

# Performance Work After First Successful Generation

Optimize only after collecting baseline measurements.

Suggested order:

1. eliminate obviously unnecessary CPU↔GPU transfers
2. ensure the Qwen encoder is not unnecessarily occupying VRAM during DiT inference
3. verify the selected quantized checkpoint is loaded natively rather than expanded into a much larger representation
4. profile attention/denoising performance
5. investigate RTX 5090-friendly attention implementations if beneficial
6. benchmark fewer inference steps if quality permits
7. optionally benchmark the supported smaller H3 resolution against the native 768-short-edge configuration

Change one variable at a time.

Keep a known-good baseline command for comparisons.

---

# Logging

Default console output should be concise.

Something approximately like:

```text
MiniMax H3

Model: my_model.safetensors
Input: photo.jpg
Canvas: 1344x768
Frames: 124
Steps: 20
Seed: 42

Loading models...
Encoding...
Generating: 12/20
Decoding...
Writing output.mp4...

Done.

Load:       18.2s
Generation: 94.1s
Total:      116.7s
Peak VRAM:  27.4 GB
```

Avoid verbose library/debug logs unless explicitly enabled.

---

# Error Handling

Give clear errors for:

* CUDA unavailable
* unsupported checkpoint
* checkpoint missing
* image missing
* unreadable image
* CUDA OOM
* insufficient disk space where detectable
* video encoding failure

For CUDA OOM, report the failure clearly.

Do not automatically change resolution, steps, or model configuration behind the user's back.

---

# Output Encoding

Produce one normal `.mp4`.

It should contain:

* H.264-compatible video where practical
* AAC-compatible audio where practical

Prefer an existing library/helper already used by the inference stack.

Using ffmpeg is acceptable if needed.

Do not implement custom media encoding.

---

# Dependency Philosophy

Use as few dependencies as reasonably possible.

Do not introduce:

* FastAPI
* Flask
* Gradio
* Streamlit
* React
* Redis
* Celery
* databases
* Docker orchestration
* configuration frameworks
* plugin frameworks

unless something becomes genuinely necessary.

After the first successful RunPod generation, pin the working versions of the important ML dependencies.

---

# README

Keep README operational and short.

Include:

* project purpose
* supported checkpoint type
* hardware target
* local mock-development setup
* RunPod setup
* model locations
* dependency installation
* example mock command
* example real command
* known tested checkpoint(s)
* measured performance
* troubleshooting for CUDA OOM

Avoid turning the README into a general MiniMax H3 tutorial.

---

# Explicitly Out of Scope for V1

Do not implement:

* LoRA loading
* web UI
* mobile UI
* HTTP API
* job queues
* WebSockets
* multiple-user support
* authentication
* checkpoint downloading
* checkpoint marketplace integration
* CivitAI APIs
* model conversion
* GGUF
* prompt history
* galleries
* persistent metadata
* arbitrary video duration
* arbitrary output resolution
* batch generation

Future LoRA and frontend ideas should not add abstractions or code to V1.

Simply avoid unnecessarily coupling the inference implementation to the CLI.

---

# Development Sequence

## Phase 1 — local skeleton

Implement:

* repository structure
* CLI
* input validation
* resolution calculation
* progress callback
* mock backend
* lightweight tests

Everything must work without GPU/model files.

## Phase 2 — real H3 integration

Implement the real ComfyUI-compatible H3 backend.

Do not rewrite the local application structure.

Replace only the inference boundary.

## Phase 3 — first RunPod generation

Get one chosen checkpoint producing a valid video with audio.

Fix correctness and memory issues only.

## Phase 4 — performance pass

Measure the pipeline and optimize the dominant bottlenecks.

## Phase 5 — checkpoint verification

Verify the other 1–2 intended community checkpoints.

Do not expand support beyond those unless required.

---

# Development Guidance

Favor direct code over abstractions.

A useful rule for this project:

> If a layer exists only because it might theoretically be useful later, delete it.

The exceptions are:

* the tiny mock/real backend boundary
* the progress callback

Both solve immediate requirements.

Avoid creating generic interfaces for hypothetical future models.

Keep model-loading behavior explicit and easy to inspect.

Prefer using upstream/community MiniMax H3 functionality directly rather than wrapping every operation.

When unsure between a clever implementation and a boring implementation, choose the boring one.

---

# Definition of Done

V1 is finished when:

```bash
python generate.py \
  --model /models/h3_fl2va_int8.safetensors \
  --image photo.jpg \
  --prompt "A gentle breeze moves through the scene while the camera slowly pushes forward." \
  --output result.mp4
```

successfully produces a playable video with generated audio on the RTX 5090 RunPod, while staying within available VRAM.

The same repository must also be locally testable on a small non-GPU development machine using its mock backend.

Anything beyond that belongs in a later version.
