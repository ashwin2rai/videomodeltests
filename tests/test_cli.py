import pytest
from conftest import ASSET_IMAGE

import generate
import h3


def test_required_args_missing_raises_system_exit():
    with pytest.raises(SystemExit):
        generate.parse_args(["--image", ASSET_IMAGE, "--prompt", "p", "--output", "out.mp4"])


def test_defaults():
    args = generate.parse_args(
        ["--model", "m.safetensors", "--image", ASSET_IMAGE, "--prompt", "p", "--output", "out.mp4"]
    )
    assert args.seed == 42
    assert args.steps == 20
    assert args.duration == h3.DEFAULT_DURATION
    assert args.resolution == h3.DEFAULT_RESOLUTION
    assert args.mock is False


def test_image_is_optional():
    args = generate.parse_args(["--model", "m.safetensors", "--prompt", "p", "--output", "out.mp4"])
    assert args.image is None


def test_invalid_duration_rejected():
    with pytest.raises(SystemExit):
        generate.parse_args(
            ["--model", "m.safetensors", "--image", ASSET_IMAGE, "--prompt", "p",
             "--output", "out.mp4", "--duration", "6"]
        )


def test_invalid_resolution_rejected():
    with pytest.raises(SystemExit):
        generate.parse_args(
            ["--model", "m.safetensors", "--image", ASSET_IMAGE, "--prompt", "p",
             "--output", "out.mp4", "--resolution", "600"]
        )


def test_missing_image_file_errors(tmp_path, capsys):
    output = tmp_path / "out.mp4"
    rc = generate.main(
        [
            "--mock",
            "--model", "fake.safetensors",
            "--image", str(tmp_path / "missing.jpg"),
            "--prompt", "p",
            "--output", str(output),
        ]
    )
    assert rc != 0
    assert "Error" in capsys.readouterr().err


def test_missing_model_file_errors_without_mock(tmp_path, capsys):
    output = tmp_path / "out.mp4"
    rc = generate.main(
        [
            "--model", str(tmp_path / "missing.safetensors"),
            "--image", ASSET_IMAGE,
            "--prompt", "p",
            "--output", str(output),
        ]
    )
    assert rc != 0
    assert "Error" in capsys.readouterr().err


def test_mock_does_not_require_model_file(tmp_path):
    output = tmp_path / "out.mp4"
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
    assert list(tmp_path.glob("out_*.mp4"))


def test_mock_cli_output_reflects_custom_steps_and_seed(tmp_path, capsys):
    output = tmp_path / "out.mp4"
    rc = generate.main(
        [
            "--mock",
            "--model", "fake.safetensors",
            "--image", ASSET_IMAGE,
            "--prompt", "p",
            "--output", str(output),
            "--steps", "5",
            "--seed", "99",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Steps: 5" in out
    assert "Seed: 99" in out
    assert out.count("Generating: ") == 5


def test_mock_cli_output_reflects_duration_preset(tmp_path, capsys):
    output = tmp_path / "out.mp4"
    rc = generate.main(
        [
            "--mock",
            "--model", "fake.safetensors",
            "--image", ASSET_IMAGE,
            "--prompt", "p",
            "--output", str(output),
            "--duration", "7.5",
        ]
    )
    assert rc == 0
    assert f"Frames: {h3.DURATION_PRESETS[7.5]}" in capsys.readouterr().out


def test_mock_cli_output_reflects_resolution_preset(tmp_path, capsys):
    output = tmp_path / "out.mp4"
    rc = generate.main(
        [
            "--mock",
            "--model", "fake.safetensors",
            "--image", ASSET_IMAGE,
            "--prompt", "p",
            "--output", str(output),
            "--resolution", "512",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "512px short edge" in out
    expected_w, expected_h = h3.compute_resolution(64, 64, 512)
    assert f"Canvas: {expected_w}x{expected_h}" in out


def test_insufficient_disk_space_errors_through_cli(tmp_path, capsys, monkeypatch):
    class FakeUsage:
        free = 10

    monkeypatch.setattr(h3.shutil, "disk_usage", lambda path: FakeUsage())

    output = tmp_path / "out.mp4"
    rc = generate.main(
        [
            "--mock",
            "--model", "fake.safetensors",
            "--image", ASSET_IMAGE,
            "--prompt", "p",
            "--output", str(output),
        ]
    )
    assert rc != 0
    assert "disk space" in capsys.readouterr().err.lower()
