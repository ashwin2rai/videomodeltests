import json
import logging
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


SHORT_EDGE = 768
ALIGNMENT = 32
FPS = 24
DEFAULT_SEED = 42
DEFAULT_STEPS = 20

# Duration presets, not free-form frame count/FPS (see objective.md's "Do not
# expose: ... frame count, FPS"). Each value is the nearest frame count on H3's
# 17k+5 grid to the target duration at FPS=24: 5s->124 (5.167s), 7.5s->175
# (7.292s), 10s->243 (10.125s), 12.5s->294 (12.25s), 15s->362 (15.083s),
# 17.5s->413 (17.208s), 20s->481 (20.042s). 10s peaks at ~28GB VRAM on a 32GB
# RTX 5090, measured hands-on (see objective/status.md); presets beyond 10s
# scale VRAM usage further and haven't been measured against that 32GB target —
# confirm headroom before running them on a 32GB card.
DURATION_PRESETS = {5.0: 124, 7.5: 175, 10.0: 243, 12.5: 294, 15.0: 362, 17.5: 413, 20.0: 481}
DEFAULT_DURATION = 5.0

# Sampling defaults, not exposed on the CLI (see objective.md). MiniMaxH3ImageToVideo
# only produces a positive conditioning (no negative branch), which matches cfg=1.0 —
# comfy.samplers skips evaluating the negative conditioning entirely at cfg==1.0, so
# its exact contents don't matter. sampler/scheduler are ComfyUI's general-purpose
# defaults for flow-matching models (H3 uses ModelSamplingAV + CONST); unverified
# against real output quality until the first real RunPod generation.
SAMPLER_NAME = "euler"
SCHEDULER = "simple"
CFG = 1.0

MAX_SAFETENSORS_HEADER_SIZE = 100_000_000
MIN_OUTPUT_FREE_BYTES = 500 * 1024 * 1024
# Rough, unmeasured allowance for OS/Python/CUDA-context overhead on top of raw model
# weights, used only for the RAM headroom warning below -- not a hard limit.
RAM_OVERHEAD_ALLOWANCE_BYTES = 15 * 1024**3

H3_BLOCK_KEY_RE = re.compile(r"^blocks\.\d+\.")
H3_DISTINCTIVE_KEYS = {"video_patch_proj.weight", "audio_patch_proj.weight", "token_refiner.final_norm.weight"}

# Real H3 backend needs a ComfyUI checkout (see `make comfyui`, README) to import
# comfy.sd etc. as a library (headless, no server/UI) rather than
# vendoring/reimplementing MiniMax H3. Not needed for --mock or tests.
#
# Both roots are env-overridable (same convention as fetch_models.sh's HF_TOKEN)
# for templates that mount ComfyUI/models somewhere other than the default;
# MODELS_ROOT defaults to living under COMFYUI_ROOT since that's where
# `make comfyui` + ComfyUI's own model manager expect it.
#
# COMFYUI_ROOT's default mirrors the Makefile's: a sibling "ComfyUI" dir next
# to this repo checkout, resolved from this file's own location (not CWD) so
# it's correct regardless of where generate.py is invoked from. This works
# whether the parent dir is named "workspace" (RunPod) or "workspaces"
# (Codespaces) — see `make comfyui` in the Makefile.
COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT", str(Path(__file__).resolve().parent.parent / "ComfyUI"))
MODELS_ROOT = os.environ.get("MODELS_ROOT", f"{COMFYUI_ROOT}/models")

# Fixed supporting models (not user-configurable). Filenames are confirmed
# against the real Comfy-Org/MiniMax-H3 Hugging Face repo.
QWEN_ENCODER_PATH = f"{MODELS_ROOT}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE_PATH = f"{MODELS_ROOT}/vae/minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_PATH = f"{MODELS_ROOT}/vae/minimax_h3_audio_vae_fp32.safetensors"


def compute_resolution(width, height, short_edge=SHORT_EDGE, alignment=ALIGNMENT):
    """Matches ComfyUI's adapt_canvas() (comfy_extras/nodes_minimax_h3.py): short-edge
    scaling with a short_edge**2 * (7/4) pixel-area cap, per-axis rounded to `alignment`.
    The cap matters for extreme aspect ratios, which would otherwise exceed the area H3
    was trained on even after short-edge scaling."""
    max_pixels = short_edge * short_edge * 7 // 4  # 768 * 1344 at the default 768 short edge
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = short_edge * ratio, float(short_edge)
    else:
        nom_w, nom_h = float(short_edge), short_edge / ratio
    if nom_w * nom_h > max_pixels:
        scale = math.sqrt(max_pixels / (nom_w * nom_h))
        nom_w, nom_h = nom_w * scale, nom_h * scale
    return (
        max(alignment, round(nom_w / alignment) * alignment),
        max(alignment, round(nom_h / alignment) * alignment),
    )


# Canvas used for text-to-video (no input image to derive an aspect ratio from).
# H3's own ComfyUI node defaults to the same 16:9 values for this case.
T2V_WIDTH, T2V_HEIGHT = compute_resolution(16, 9)


def _require_file(path, what):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def timestamped_output_path(path):
    # e.g. output.mp4 -> output_20260918_143022_123456.mp4 — microsecond precision
    # so two runs completing within the same second (e.g. --mock) still get distinct files.
    path = Path(path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


def validate_prompt(prompt):
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is required and cannot be empty")


def validate_image(path):
    path = _require_file(path, "Image")
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return img.size
    except UnidentifiedImageError as e:
        raise ValueError(f"Not a valid image: {path}") from e


def preprocess_image(path, short_edge=SHORT_EDGE, alignment=ALIGNMENT):
    path = _require_file(path, "Image")
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"Not a valid image: {path}") from e

    width, height = img.size
    canvas_w, canvas_h = compute_resolution(width, height, short_edge, alignment)
    logger.info("Preprocessing image %s: %dx%d -> %dx%d", path, width, height, canvas_w, canvas_h)
    return img.resize((canvas_w, canvas_h), Image.LANCZOS)


def validate_checkpoint(path):
    path = _require_file(path, "Checkpoint")
    logger.info("Validating checkpoint: %s", path)
    file_size = path.stat().st_size

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic == b"GGUF":
            raise ValueError(
                f"GGUF checkpoints are not supported: {path}. "
                "Provide a ComfyUI-style safetensors H3 checkpoint."
            )

        f.seek(0)
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(f"Not a valid safetensors file: {path}")

        header_len = struct.unpack("<Q", header_len_bytes)[0]
        if header_len <= 0 or header_len > MAX_SAFETENSORS_HEADER_SIZE or header_len > file_size:
            raise ValueError(f"Not a valid safetensors file: {path}")

        try:
            header = json.loads(f.read(header_len))
        except json.JSONDecodeError as e:
            raise ValueError(f"Not a valid safetensors file: {path}") from e

    metadata = header.pop("__metadata__", None)
    if not header:
        raise ValueError(f"Checkpoint contains no tensors: {path}")

    keys = header.keys()
    has_blocks = any(H3_BLOCK_KEY_RE.match(k) for k in keys)
    has_distinctive_key = not H3_DISTINCTIVE_KEYS.isdisjoint(keys)
    if not (has_blocks and has_distinctive_key):
        raise ValueError(
            f"Does not look like a MiniMax H3 FL2VA checkpoint: {path}. "
            "Expected transformer 'blocks.N.*' tensors and video/audio patch-projection tensors."
        )

    if "ref2va" in path.name.lower():
        raise ValueError(
            f"Filename suggests a Ref2VA checkpoint, which is out of scope for V1: {path}. "
            "(This is a filename heuristic, not a content check — Ref2VA and FL2VA checkpoints "
            "have identical tensor layouts and cannot be distinguished by header inspection alone. "
            "If this is actually an FL2VA checkpoint, rename it.)"
        )

    if metadata is None:
        logger.warning(
            "Checkpoint has no __metadata__ (common for quantized variants) — "
            "architecture was confirmed from tensor names only: %s",
            path,
        )

    dtypes = sorted({tensor["dtype"] for tensor in header.values()})
    logger.info("Checkpoint OK: %d tensors, dtypes=%s: %s", len(header), dtypes, path)
    return {"tensor_count": len(header), "dtypes": dtypes, "metadata": metadata}


def discover_dit_checkpoint(models_root=None):
    """Finds the one DiT checkpoint already downloaded into MODELS_ROOT/diffusion_models,
    for callers (the web server) that don't take --model as an explicit per-request argument.
    An H3_MODEL_PATH env var always wins, for disambiguation or a non-standard location."""
    override = os.environ.get("H3_MODEL_PATH")
    if override:
        path = str(_require_file(override, "DiT checkpoint (H3_MODEL_PATH)"))
        logger.info("Using DiT checkpoint from H3_MODEL_PATH: %s", path)
        return path

    models_root = models_root or MODELS_ROOT
    candidates = sorted(Path(models_root, "diffusion_models").glob("*.safetensors"))
    if len(candidates) == 1:
        logger.info("Auto-discovered DiT checkpoint: %s", candidates[0])
        return str(candidates[0])
    if not candidates:
        raise ValueError(
            f"No DiT checkpoint found in {models_root}/diffusion_models. "
            "Fetch one first (see `make fetch-dit`), or set H3_MODEL_PATH explicitly."
        )
    raise ValueError(
        f"Found {len(candidates)} DiT checkpoints in {models_root}/diffusion_models: "
        f"{[c.name for c in candidates]}. Set H3_MODEL_PATH to pick one."
    )


def warn_if_ram_tight(dit_path):
    """Visibility only -- logs a warning if caching this DiT alongside the fixed Qwen/VAE
    models is likely to approach total system RAM. Does not change any behavior or block
    loading. Assumes each checkpoint's resident RAM cost roughly matches its file size;
    see objective/status.md's open question about whether quantized checkpoints are
    expanded into a larger representation at load time, which this can't detect."""
    try:
        import psutil
    except ImportError:
        return  # psutil is a `gpu`-group dependency; nothing to check without it

    fixed_bytes = sum(
        Path(p).stat().st_size
        for p in (QWEN_ENCODER_PATH, VIDEO_VAE_PATH, AUDIO_VAE_PATH)
        if Path(p).is_file()
    )
    dit_bytes = Path(dit_path).stat().st_size
    estimated = fixed_bytes + dit_bytes + RAM_OVERHEAD_ALLOWANCE_BYTES
    total = psutil.virtual_memory().total

    if estimated > total:
        logger.warning(
            "Estimated resident RAM for cached models (~%.1fGB: %.1fGB fixed models + "
            "%.1fGB DiT + %.1fGB overhead allowance) is close to or exceeds total system "
            "RAM (~%.1fGB). If generation crashes or the machine becomes unresponsive, "
            "this is the likely cause.",
            estimated / 1e9, fixed_bytes / 1e9, dit_bytes / 1e9,
            RAM_OVERHEAD_ALLOWANCE_BYTES / 1e9, total / 1e9,
        )


def check_disk_space(output_path, required_bytes=MIN_OUTPUT_FREE_BYTES):
    directory = Path(output_path).resolve().parent
    free = shutil.disk_usage(directory).free
    if free < required_bytes:
        raise OSError(
            f"Insufficient disk space in {directory}: {free} bytes free, need at least {required_bytes}"
        )
    logger.info("Disk space OK: %.1fGB free in %s", free / 1e9, directory)


def mux_mp4(frames, width, height, fps, pcm_audio, sample_rate, output_path, ffmpeg_bin="ffmpeg", channels=1):
    logger.info("Muxing MP4: %dx%d @ %dfps -> %s", width, height, fps, output_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    try:
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm_audio)

        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-i", str(wav_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(output_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=b"".join(frames),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"{ffmpeg_bin} not found — install ffmpeg to encode output video") from e

        if proc.returncode != 0:
            stderr_tail = proc.stderr.decode(errors="replace")[-2000:]
            raise RuntimeError(f"ffmpeg failed to encode {output_path}: {stderr_tail}")
    finally:
        wav_path.unlink(missing_ok=True)


def _reporter(progress_callback):
    def report(progress, message):
        if progress_callback:
            progress_callback(progress, message)

    return report


class MockH3Backend:
    def load(self):
        pass

    def generate(
        self,
        model_path,
        prompt,
        output_path,
        image_path=None,
        seed=DEFAULT_SEED,
        steps=DEFAULT_STEPS,
        frames=DURATION_PRESETS[DEFAULT_DURATION],
        progress_callback=None,
    ):
        report = _reporter(progress_callback)
        validate_prompt(prompt)
        logger.info(
            "Mock generate: model=%s image=%s prompt=%r frames=%d steps=%d seed=%d",
            model_path, image_path or "(none — text-to-video)", prompt, frames, steps, seed,
        )

        report(0.0, "loading models")
        report(0.1, "encoding prompt and image" if image_path else "encoding prompt")
        for i in range(1, steps + 1):
            report(0.1 + 0.7 * (i / steps), f"generating: {i}/{steps}")
        report(0.9, "decoding")
        report(0.95, "writing output")

        Path(output_path).write_bytes(b"MOCK MP4 OUTPUT - not a real video\n")
        logger.info("Mock output written: %s", output_path)

        report(1.0, "done")


def _import_comfy():
    """Import ComfyUI as a headless library (no server/UI) from a git-cloned checkout.
    See README for setup. Raises ImportError with a clear message if COMFYUI_ROOT is missing."""
    if not Path(COMFYUI_ROOT).is_dir():
        raise ImportError(
            f"ComfyUI checkout not found at {COMFYUI_ROOT}. "
            "Clone it (see README) before using the real H3 backend."
        )
    logger.info("Importing ComfyUI headlessly from %s", COMFYUI_ROOT)
    if COMFYUI_ROOT not in sys.path:
        sys.path.insert(0, COMFYUI_ROOT)
    import comfy.model_management
    import comfy.sample
    import comfy.sd
    import comfy.utils
    import nodes
    from comfy_extras.nodes_audio import vae_decode_audio
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo

    return comfy.sd, comfy.sample, comfy.utils, comfy.model_management, nodes, MiniMaxH3ImageToVideo, vae_decode_audio


class H3Backend:
    """Stateful across calls: `generate()` caches loaded models on `self` instead of
    reloading them from disk every time, so a caller that keeps one instance alive across
    many generate() calls (the web server's queue) only pays full load cost once. A caller
    that creates a fresh instance per call (generate.py's CLI) sees no behavior change --
    caching simply never has a chance to trigger."""

    def __init__(self):
        self._qwen = None
        self._video_vae = None
        self._audio_vae = None
        self._dit_path = None
        self._dit_model = None

    def load(self):
        try:
            import torch
        except ImportError:
            torch = None

        if torch is None or not torch.cuda.is_available():
            logger.warning("CUDA is not available on this machine")
            raise RuntimeError("CUDA is not available. The real H3 backend requires an NVIDIA GPU.")
        logger.info("CUDA available: %s", torch.cuda.get_device_name(0))

    def generate(
        self,
        model_path,
        prompt,
        output_path,
        image_path=None,
        seed=DEFAULT_SEED,
        steps=DEFAULT_STEPS,
        frames=DURATION_PRESETS[DEFAULT_DURATION],
        progress_callback=None,
    ):
        report = _reporter(progress_callback)

        validate_prompt(prompt)
        validate_checkpoint(model_path)
        if image_path:
            image = preprocess_image(image_path)
            width, height = image.size
        else:
            image = None
            width, height = T2V_WIDTH, T2V_HEIGHT
            logger.info("No image provided — using text-to-video canvas %dx%d", width, height)
        for path, what in (
            (QWEN_ENCODER_PATH, "Fixed Qwen text/vision encoder"),
            (VIDEO_VAE_PATH, "Fixed video VAE"),
            (AUDIO_VAE_PATH, "Fixed audio VAE"),
        ):
            _require_file(path, what)

        logger.info(
            "Generating %s: model=%s canvas=%dx%d frames=%d steps=%d seed=%d sampler=%s/%s cfg=%s",
            "image-to-video" if image is not None else "text-to-video",
            model_path, width, height, frames, steps, seed, SAMPLER_NAME, SCHEDULER, CFG,
        )

        report(0.0, "loading models")
        sd, sample, utils, model_management, nodes, MiniMaxH3ImageToVideo, vae_decode_audio = _import_comfy()
        import numpy as np
        import torch
        torch.cuda.reset_peak_memory_stats()

        # ComfyUI's own execution engine (execution.py) wraps every node call in
        # torch.inference_mode(); calling node classes directly (headless, no
        # server/UI) bypasses that entirely. Without it, autograd retains the full
        # computation graph across every op — comfy.samplers happens to wrap its own
        # sampling loop in no_grad() internally so denoising is unaffected, but
        # nothing else is, and VAE decode's graph accumulates unboundedly across a
        # multi-tile decode until it OOMs on a card that would otherwise have room.
        with torch.inference_mode():
            # Load (or reuse already-cached) models. Cached on `self`, not reloaded from
            # disk per call, so a caller that keeps one H3Backend alive across many
            # generate() calls (the web server's queue) only pays full load cost once.
            if self._qwen is None:
                logger.info("Loading Qwen text/vision encoder (cached for the rest of this process)")
                self._qwen = sd.load_clip([QWEN_ENCODER_PATH], clip_type=sd.CLIPType.MINIMAX)
            if self._video_vae is None:
                logger.info("Loading video VAE (cached for the rest of this process)")
                self._video_vae = sd.VAE(sd=utils.load_torch_file(VIDEO_VAE_PATH))
            if self._audio_vae is None:
                logger.info("Loading audio VAE (cached for the rest of this process)")
                self._audio_vae = sd.VAE(sd=utils.load_torch_file(AUDIO_VAE_PATH))
            if self._dit_path != model_path:
                logger.info("Loading DiT checkpoint: %s", model_path)
                warn_if_ram_tight(model_path)
                self._dit_model = sd.load_diffusion_model(model_path)
                self._dit_path = model_path
            else:
                logger.info("Reusing cached DiT checkpoint: %s", model_path)

            clip, video_vae, audio_vae, model = self._qwen, self._video_vae, self._audio_vae, self._dit_model

            report(0.1, "encoding prompt and image" if image is not None else "encoding prompt")
            image_tensor = None
            if image is not None:
                image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None, ...]
            positive, latent = MiniMaxH3ImageToVideo.execute(
                clip=clip,
                vae=video_vae,
                prompt=prompt,
                width=width,
                height=height,
                length=frames,
                first_frame=image_tensor,
            )
            # No real negative conditioning: cfg=1.0 means comfy never evaluates it.
            negative = [[torch.zeros_like(positive[0][0]), {}]]

            # Qwen + video VAE (used above for conditioning) stay VRAM-resident otherwise,
            # leaving the ~20GB DiT no room to fully load — see objective.md's GPU memory
            # lifecycle (Qwen -> encode -> offload -> DiT inference). 32GB VRAM is tight
            # enough that this isn't optional: without it, denoising OOMs on the first step.
            logger.info("Offloading Qwen and video VAE to free VRAM for the DiT")
            model_management.unload_all_models()

            def sampler_callback(step, x0, x, total_steps):
                report(0.1 + 0.7 * ((step + 1) / total_steps), f"generating: {step + 1}/{total_steps}")

            report(0.1, "generating video")
            latent_samples = latent["samples"]
            noise = sample.prepare_noise(latent_samples, seed)
            denoised = sample.sample(
                model, noise, steps, CFG, SAMPLER_NAME, SCHEDULER,
                positive, negative, latent_samples,
                callback=sampler_callback, seed=seed,
            )

            report(0.9, "decoding")
            logger.info("Offloading the DiT to free VRAM for VAE decode")
            model_management.unload_all_models()  # DiT no longer needed; give the VAEs full room
            video_latent = denoised.unbind()[0]
            video_images = nodes.VAEDecode().decode(video_vae, {"samples": video_latent})[0]
            if video_images.ndim == 5:  # combine [B, T, H, W, C] batches, as VAEDecode itself does
                video_images = video_images.reshape(-1, *video_images.shape[-3:])
            pixel_frames = [
                frame.tobytes()
                for frame in video_images.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
            ]

            audio = vae_decode_audio(audio_vae, {"samples": denoised})
            waveform = audio["waveform"][0]  # [C, L] float roughly in [-1, 1]
            pcm = waveform.clamp(-1, 1).mul(32767).round().to(torch.int16).cpu().numpy()
            channels = pcm.shape[0]
            pcm_bytes = pcm.T.copy().tobytes()  # interleave channels for the WAV container

        logger.info("Peak VRAM: %.1fGB", torch.cuda.max_memory_allocated() / 1e9)
        report(0.95, "writing output")
        mux_mp4(pixel_frames, width, height, FPS, pcm_bytes, audio["sample_rate"], output_path, channels=channels)

        report(1.0, "done")
