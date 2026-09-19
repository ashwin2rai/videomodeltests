import io
import os
import queue
import threading
import time

import pytest

os.environ["H3_MOCK"] = "1"  # must be set before importing web.server: MOCK is read at import time

from conftest import ASSET_IMAGE  # noqa: E402

import h3  # noqa: E402
from web import server as server_module  # noqa: E402


def wait_until(predicate, timeout=2.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition not met within timeout")


class _BlockingBackend:
    """Stand-in backend whose generate() hangs until released, so tests can control
    exactly when the shared worker thread is "busy" vs. between jobs -- needed because
    the real MockH3Backend finishes so fast that queue-state assertions would otherwise
    race the worker thread nondeterministically."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, **kwargs):
        self.started.set()
        self.release.wait()


@pytest.fixture(scope="module", autouse=True)
def worker_thread():
    threading.Thread(target=server_module._worker_loop, daemon=True).start()


@pytest.fixture(autouse=True)
def mock_backend():
    # Every test starts from a known-good mock backend, regardless of what a
    # previous test (e.g. the discovery-failure regression test) left behind.
    server_module.MOCK = True
    server_module._init_backend()


@pytest.fixture(autouse=True)
def clean_queue():
    def drain():
        while True:
            try:
                server_module.job_queue.get_nowait()
                server_module.job_queue.task_done()
            except queue.Empty:
                break

    drain()
    yield
    drain()


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    monkeypatch.setattr(server_module, "INPUTS_DIR", inputs)
    monkeypatch.setattr(server_module, "OUTPUTS_DIR", outputs)
    return inputs, outputs


@pytest.fixture
def client():
    return server_module.app.test_client()


def upload_asset(client):
    with open(ASSET_IMAGE, "rb") as f:
        resp = client.post("/api/upload", data={"file": (f, "test.jpg")})
    assert resp.status_code == 200
    return resp.get_json()["filename"]


def make_job(image, prompt):
    return {"image": image, "prompt": prompt, "duration": 5.0, "steps": 1, "seed": 1}


def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"MiniMax H3" in resp.data


def test_upload_valid_image(client, isolated_dirs):
    inputs, _ = isolated_dirs
    filename = upload_asset(client)
    assert (inputs / filename).is_file()
    assert filename in client.get("/api/inputs").get_json()


def test_upload_rejects_non_image(client, isolated_dirs):
    inputs, _ = isolated_dirs
    resp = client.post("/api/upload", data={"file": (io.BytesIO(b"not an image"), "bad.jpg")})
    assert resp.status_code == 400
    assert list(inputs.iterdir()) == []  # rejected upload must not be left behind


def test_upload_requires_file(client):
    resp = client.post("/api/upload", data={})
    assert resp.status_code == 400


def test_enqueue_runs_job_to_completion(client, isolated_dirs):
    _, outputs = isolated_dirs
    image = upload_asset(client)
    resp = client.post("/api/queue", json={"image": image, "prompt": "hello", "duration": 5, "steps": 3, "seed": 1})
    assert resp.status_code == 200

    def is_done():
        job = client.get("/api/status").get_json()["current_job"]
        return job is not None and job["status"] == "done"

    wait_until(is_done)
    status = client.get("/api/status").get_json()
    assert status["current_job"]["id"] >= 1
    assert list(outputs.iterdir())  # mock backend actually wrote an output file


def test_enqueue_missing_image(client):
    resp = client.post("/api/queue", json={"image": "nope.jpg", "prompt": "hello", "duration": 5})
    assert resp.status_code == 400


def test_enqueue_missing_prompt(client, isolated_dirs):
    image = upload_asset(client)
    resp = client.post("/api/queue", json={"image": image, "prompt": "  ", "duration": 5})
    assert resp.status_code == 400


def test_enqueue_invalid_duration(client, isolated_dirs):
    image = upload_asset(client)
    resp = client.post("/api/queue", json={"image": image, "prompt": "hello", "duration": 6})
    assert resp.status_code == 400


def test_enqueue_rejects_when_backend_not_ready(client, monkeypatch):
    monkeypatch.setattr(server_module, "backend_ready", False)
    monkeypatch.setattr(server_module, "backend_error", "boom")
    resp = client.post("/api/queue", json={"image": "x.jpg", "prompt": "hello", "duration": 5})
    assert resp.status_code == 503
    assert resp.get_json()["error"] == "boom"
    assert client.get("/api/status").get_json()["backend_error"] == "boom"


def test_queue_full_returns_409(client, isolated_dirs, monkeypatch):
    image = upload_asset(client)
    blocker = _BlockingBackend()
    monkeypatch.setattr(server_module, "backend", blocker)
    try:
        server_module.job_queue.put_nowait(make_job(image, "in flight"))
        wait_until(blocker.started.is_set)  # now dequeued and "running"; queue itself is empty again

        for i in range(server_module.MAX_QUEUE):
            server_module.job_queue.put_nowait(make_job(image, f"filler {i}"))

        resp = client.post("/api/queue", json={"image": image, "prompt": "one too many", "duration": 5})
        assert resp.status_code == 409
    finally:
        blocker.release.set()
        wait_until(lambda: server_module.job_queue.qsize() == 0)


def test_clear_queue_drains_pending_only(client, monkeypatch):
    blocker = _BlockingBackend()
    monkeypatch.setattr(server_module, "backend", blocker)
    try:
        server_module.job_queue.put_nowait(make_job("x.jpg", "running"))
        wait_until(blocker.started.is_set)  # this one is in flight, no longer sitting in the queue

        server_module.job_queue.put_nowait(make_job("x.jpg", "pending-a"))
        server_module.job_queue.put_nowait(make_job("x.jpg", "pending-b"))

        resp = client.delete("/api/queue")
        assert resp.status_code == 200
        assert resp.get_json()["drained"] == 2
        assert client.get("/api/status").get_json()["queue_length"] == 0
    finally:
        blocker.release.set()
        wait_until(lambda: server_module.job_queue.qsize() == 0)


def test_init_backend_survives_checkpoint_discovery_failure(monkeypatch):
    monkeypatch.setattr(server_module, "MOCK", False)
    monkeypatch.setattr(h3, "discover_dit_checkpoint", lambda: (_ for _ in ()).throw(ValueError("no checkpoint")))

    server_module._init_backend()  # must not raise -- regression test for a real crash-on-startup bug

    assert server_module.backend_ready is False
    assert "no checkpoint" in server_module.backend_error
