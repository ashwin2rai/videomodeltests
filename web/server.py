import logging
import os
import queue
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import h3  # noqa: E402

logger = logging.getLogger(__name__)

INPUTS_DIR = REPO_ROOT / "inputs"
OUTPUTS_DIR = REPO_ROOT / "outputs"
STATIC_DIR = Path(__file__).resolve().parent / "static"
INPUTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

MOCK = bool(os.environ.get("H3_MOCK"))
MAX_QUEUE = 5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

app = Flask(__name__)

job_queue = queue.Queue(maxsize=MAX_QUEUE)

# backend/model_path/backend_ready/backend_error are set once by _init_backend(), which
# always finishes (main(), below) before the worker thread starts or Flask begins serving
# requests -- so no lock is needed for them. current_job is different: it's mutated by the
# worker thread while request threads read it via /api/status at arbitrary times during a
# job's life, so it's the one piece of state that needs state_lock.
state_lock = threading.Lock()
backend = None
model_path = None
backend_ready = False
backend_error = None
current_job = None  # {"prompt", "progress", "status", "message"} or None


def _init_backend():
    global backend, model_path, backend_ready, backend_error
    backend_ready = False
    backend_error = None
    try:
        if MOCK:
            backend = h3.MockH3Backend()
            model_path = "mock.safetensors"
        else:
            backend = h3.H3Backend()
            model_path = h3.discover_dit_checkpoint()
        backend.load()
        backend_ready = True
        logger.info("Backend ready (mock=%s, model=%s)", MOCK, model_path)
    except Exception as e:
        backend_error = str(e)
        logger.error("Backend failed to load: %s", e)


def _worker_loop():
    global current_job
    job_id = 0
    while True:
        job = job_queue.get()
        job_id += 1
        logger.info("Starting job %d: %r (queue_length=%d)", job_id, job["prompt"], job_queue.qsize())
        with state_lock:
            current_job = {
                "id": job_id, "prompt": job["prompt"], "progress": 0.0,
                "message": "starting", "status": "running",
            }

        def progress_callback(progress, message):
            with state_lock:
                current_job["progress"] = progress
                current_job["message"] = message

        output_path = h3.timestamped_output_path(OUTPUTS_DIR / "result.mp4")
        try:
            backend.generate(
                model_path=model_path,
                image_path=str(INPUTS_DIR / job["image"]),
                prompt=job["prompt"],
                output_path=output_path,
                seed=job["seed"],
                steps=job["steps"],
                frames=h3.DURATION_PRESETS[job["duration"]],
                progress_callback=progress_callback,
            )
        except Exception as e:
            logger.exception("Generation failed")
            with state_lock:
                current_job["status"] = "error"
                current_job["message"] = str(e)
        else:
            logger.info("Job finished: %r -> %s", job["prompt"], output_path)
            with state_lock:
                current_job["status"] = "done"
        finally:
            job_queue.task_done()


def _list_dir(directory, extensions):
    files = [p for p in directory.iterdir() if p.suffix.lower() in extensions]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in files]


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/inputs")
def list_inputs():
    return jsonify(_list_dir(INPUTS_DIR, IMAGE_EXTENSIONS))


@app.get("/api/outputs")
def list_outputs():
    return jsonify(_list_dir(OUTPUTS_DIR, {".mp4"}))


@app.get("/inputs/<path:name>")
def serve_input(name):
    return send_from_directory(INPUTS_DIR, name)


@app.get("/outputs/<path:name>")
def serve_output(name):
    return send_from_directory(OUTPUTS_DIR, name)


@app.post("/api/upload")
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file provided"}), 400

    filename = secure_filename(file.filename) or "upload.jpg"
    dest = h3.timestamped_output_path(INPUTS_DIR / filename)
    file.save(dest)

    try:
        h3.validate_image(dest)
    except (FileNotFoundError, ValueError) as e:
        dest.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 400

    return jsonify({"filename": dest.name})


@app.post("/api/queue")
def enqueue():
    if not backend_ready:
        return jsonify({"error": backend_error or "Backend is not ready"}), 503

    data = request.get_json(force=True, silent=True) or {}
    image = data.get("image")
    prompt = (data.get("prompt") or "").strip()
    try:
        duration = float(data.get("duration", h3.DEFAULT_DURATION))
        steps = int(data.get("steps", h3.DEFAULT_STEPS))
        seed = int(data.get("seed", h3.DEFAULT_SEED))
    except (TypeError, ValueError):
        return jsonify({"error": "duration/steps/seed must be numbers"}), 400

    if not image or not (INPUTS_DIR / image).is_file():
        return jsonify({"error": f"Image not found: {image}"}), 400
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    if duration not in h3.DURATION_PRESETS:
        return jsonify({"error": f"Invalid duration: {duration}"}), 400

    job = {"image": image, "prompt": prompt, "duration": duration, "steps": steps, "seed": seed}
    try:
        job_queue.put_nowait(job)
    except queue.Full:
        return jsonify({"error": f"Queue is full ({MAX_QUEUE}/{MAX_QUEUE})"}), 409

    logger.info("Enqueued job: %r (queue_length=%d)", prompt, job_queue.qsize())
    return jsonify({"queue_length": job_queue.qsize()})


@app.delete("/api/queue")
def clear_queue():
    drained = 0
    while True:
        try:
            job_queue.get_nowait()
        except queue.Empty:
            break
        job_queue.task_done()
        drained += 1
    logger.info("Cleared %d pending job(s) from the queue", drained)
    return jsonify({"drained": drained, "queue_length": job_queue.qsize()})


@app.get("/api/status")
def status():
    with state_lock:
        job = dict(current_job) if current_job else None
    return jsonify({
        "queue_length": job_queue.qsize(),
        "current_job": job,
        "backend_ready": backend_ready,
        "backend_error": backend_error,
        # Cheap (single stat syscall) change signal so the client only needs to fetch the
        # full (listing + per-file stat) /api/outputs when something has actually changed,
        # instead of on every ~1s poll tick regardless.
        "outputs_mtime": OUTPUTS_DIR.stat().st_mtime,
    })


def main():
    h3.setup_logging()
    _init_backend()
    threading.Thread(target=_worker_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
