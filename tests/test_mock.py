import datetime as datetime_module

import pytest
from conftest import ASSET_IMAGE

import generate
import h3


def _fake_datetime(*fixed_values):
    values = iter(fixed_values)

    class FakeDatetime(datetime_module.datetime):
        @classmethod
        def now(cls, tz=None):
            return next(values)

    return FakeDatetime


@pytest.mark.parametrize(
    "width,height,expected",
    [
        (1920, 1080, (1344, 768)),  # 16:9
        (1080, 1920, (768, 1344)),  # 9:16
        (1000, 1000, (768, 768)),  # 1:1
        (1024, 768, (1024, 768)),  # 4:3
        (4000, 1000, (2016, 512)),  # extreme 4:1 panorama, exercises the pixel-area cap
        (1000, 4000, (512, 2016)),  # extreme 1:4, exercises the pixel-area cap
    ],
)
def test_compute_resolution(width, height, expected):
    assert h3.compute_resolution(width, height) == expected


def test_compute_resolution_never_exceeds_pixel_budget():
    max_pixels = h3.SHORT_EDGE * h3.SHORT_EDGE * 7 // 4
    for width, height in [(4000, 1000), (1000, 4000), (10000, 100), (100, 10000)]:
        w, h = h3.compute_resolution(width, height)
        assert w * h <= max_pixels * 1.01  # rounding to the alignment grid can push slightly over


def test_validate_image_accepts_real_image():
    width, height = h3.validate_image(ASSET_IMAGE)
    assert (width, height) == (64, 64)


def test_validate_image_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        h3.validate_image(tmp_path / "missing.jpg")


def test_validate_image_rejects_non_image(tmp_path):
    bad_file = tmp_path / "bad.jpg"
    bad_file.write_text("not an image")
    with pytest.raises(ValueError):
        h3.validate_image(bad_file)


def test_timestamped_output_path_format(monkeypatch):
    monkeypatch.setattr(
        h3, "datetime", _fake_datetime(datetime_module.datetime(2026, 9, 18, 14, 30, 22, 123456))
    )
    assert h3.timestamped_output_path("output.mp4").name == "output_20260918_143022_123456.mp4"


def test_timestamped_output_path_unique_per_call(monkeypatch):
    monkeypatch.setattr(
        h3,
        "datetime",
        _fake_datetime(
            datetime_module.datetime(2026, 9, 18, 14, 30, 22),
            datetime_module.datetime(2026, 9, 18, 14, 30, 23),
        ),
    )
    first = h3.timestamped_output_path("output.mp4")
    second = h3.timestamped_output_path("output.mp4")
    assert first != second


def test_repeated_generation_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(
        h3,
        "datetime",
        _fake_datetime(
            datetime_module.datetime(2026, 9, 18, 14, 30, 22),
            datetime_module.datetime(2026, 9, 18, 14, 30, 23),
        ),
    )
    output = tmp_path / "out.mp4"
    for _ in range(2):
        rc = generate.main(
            [
                "--mock",
                "--model", "fake.safetensors",
                "--image", ASSET_IMAGE,
                "--prompt", "p",
                "--output", str(output),
            ]
        )
        assert rc == 0

    assert len(list(tmp_path.glob("out_*.mp4"))) == 2


def test_mock_cli_creates_output_file(tmp_path):
    output = tmp_path / "out.mp4"
    rc = generate.main(
        [
            "--mock",
            "--model", "fake.safetensors",
            "--image", ASSET_IMAGE,
            "--prompt", "Test prompt",
            "--output", str(output),
        ]
    )
    assert rc == 0
    written = list(tmp_path.glob("out_*.mp4"))
    assert len(written) == 1
    assert written[0].stat().st_size > 0


def test_mock_backend_progress_callback_events(tmp_path):
    events = []
    backend = h3.MockH3Backend()
    backend.load()
    backend.generate(
        model_path="fake.safetensors",
        image_path=ASSET_IMAGE,
        prompt="p",
        output_path=tmp_path / "out.mp4",
        steps=3,
        progress_callback=lambda progress, message: events.append((progress, message)),
    )
    assert len(events) > 0
    progresses = [p for p, _ in events]
    assert progresses == sorted(progresses)
    assert events[0][1] == "loading models"
    assert events[-1] == (1.0, "done")


def test_duration_presets_land_on_valid_frame_grid():
    for duration, frames in h3.DURATION_PRESETS.items():
        assert (frames - 5) % 17 == 0, f"{duration}s -> {frames} frames is not on the 17k+5 grid"
