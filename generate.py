import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Must be set before torch initializes its CUDA allocator. Without this, real
# generation reliably OOMs on a 32GB card from allocator fragmentation alone,
# even when enough memory is technically free (see objective/status.md).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import h3

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniMax H3 image-to-video generation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=h3.DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=h3.DEFAULT_STEPS)
    parser.add_argument(
        "--duration", type=float, default=h3.DEFAULT_DURATION,
        choices=sorted(h3.DURATION_PRESETS), metavar="{5,7.5,10}",
        help="Approximate video duration in seconds",
    )
    parser.add_argument("--mock", action="store_true", help="Use the mock backend (development only)")
    return parser.parse_args(argv)


def fail(message):
    print(f"Error: {message}", file=sys.stderr)
    return 1


def main(argv=None):
    args = parse_args(argv)

    image_arg = Path(args.image)
    if image_arg.parent == Path("."):  # bare filename: default into inputs/
        image_arg = Path("inputs") / image_arg
    args.image = str(image_arg)

    try:
        width, height = h3.validate_image(args.image)
    except (FileNotFoundError, ValueError) as e:
        return fail(e)

    if not args.mock and not Path(args.model).is_file():
        return fail(f"Checkpoint not found: {args.model}")

    output_arg = Path(args.output)
    if output_arg.parent == Path("."):  # bare filename: default into outputs/
        output_arg = Path("outputs") / output_arg
    output_arg.parent.mkdir(parents=True, exist_ok=True)
    output_path = h3.timestamped_output_path(output_arg)

    try:
        h3.check_disk_space(output_path)
    except OSError as e:
        return fail(e)

    canvas_w, canvas_h = h3.compute_resolution(width, height)
    frames = h3.DURATION_PRESETS[args.duration]

    print("MiniMax H3")
    print()
    print(f"Model: {args.model}")
    print(f"Input: {args.image}")
    print(f"Output: {output_path}")
    print(f"Canvas: {canvas_w}x{canvas_h}")
    print(f"Frames: {frames} (~{frames / h3.FPS:.1f}s)")
    print(f"Steps: {args.steps}")
    print(f"Seed: {args.seed}")
    print()

    backend = h3.MockH3Backend() if args.mock else h3.H3Backend()

    def progress_callback(progress, message):
        if message == "writing output":
            logger.info("Writing output to: %s", output_path)
        if message != "done":
            print(f"{message.capitalize()}...")

    try:
        t0 = time.perf_counter()
        backend.load()
        t_loaded = time.perf_counter()
        backend.generate(
            model_path=args.model,
            image_path=args.image,
            prompt=args.prompt,
            output_path=output_path,
            seed=args.seed,
            steps=args.steps,
            frames=frames,
            progress_callback=progress_callback,
        )
        t_done = time.perf_counter()
    except Exception as e:
        return fail(e)

    load_time = t_loaded - t0
    generation_time = t_done - t_loaded
    total_time = t_done - t0

    print()
    print("Done.")
    print()
    print(f"Load:       {load_time:.1f}s")
    print(f"Generation: {generation_time:.1f}s")
    print(f"Total:      {total_time:.1f}s")

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
