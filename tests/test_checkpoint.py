import pytest
from conftest import ASSET_IMAGE, h3_shaped_header, write_safetensors

import h3


def test_validate_checkpoint_accepts_h3_shaped_file(tmp_path):
    path = tmp_path / "model.safetensors"
    write_safetensors(path, h3_shaped_header(), data=b"\x00" * 4)
    info = h3.validate_checkpoint(path)
    assert info["tensor_count"] == 2
    assert info["dtypes"] == ["BF16"]
    assert info["metadata"] == {"format": "pt"}


def test_validate_checkpoint_rejects_non_h3_shaped_file(tmp_path):
    path = tmp_path / "model.safetensors"
    write_safetensors(
        path,
        {
            "weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            "bias": {"dtype": "F32", "shape": [1], "data_offsets": [2, 6]},
        },
        data=b"\x00" * 6,
    )
    with pytest.raises(ValueError, match="MiniMax H3"):
        h3.validate_checkpoint(path)


def test_validate_checkpoint_rejects_ref2va_filename(tmp_path):
    path = tmp_path / "minimax_h3_ref2va_bf16.safetensors"
    write_safetensors(path, h3_shaped_header(), data=b"\x00" * 4)
    with pytest.raises(ValueError, match="Ref2VA"):
        h3.validate_checkpoint(path)


def test_validate_checkpoint_rejects_gguf(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 100)
    with pytest.raises(ValueError, match="GGUF"):
        h3.validate_checkpoint(path)


def test_validate_checkpoint_rejects_garbage(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"not a real safetensors file at all")
    with pytest.raises(ValueError):
        h3.validate_checkpoint(path)


def test_validate_checkpoint_rejects_empty_tensors(tmp_path):
    path = tmp_path / "model.safetensors"
    write_safetensors(path, {"__metadata__": {"format": "pt"}})
    with pytest.raises(ValueError, match="no tensors"):
        h3.validate_checkpoint(path)


def test_validate_checkpoint_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        h3.validate_checkpoint(tmp_path / "missing.safetensors")


def test_h3_backend_generate_raises_checkpoint_error_before_not_implemented(tmp_path):
    bad_path = tmp_path / "bad.safetensors"
    bad_path.write_bytes(b"garbage")
    with pytest.raises(ValueError):
        h3.H3Backend().generate(
            model_path=bad_path,
            image_path=ASSET_IMAGE,
            prompt="p",
            output_path=tmp_path / "out.mp4",
        )


def test_h3_backend_generate_raises_for_missing_fixed_models(tmp_path, monkeypatch):
    # This dev/test machine doesn't have the fixed Qwen/VAE weights (only present on
    # RunPod after `make fetch-stock`) — generate() should fail clearly on that, not
    # with an opaque traceback from deep inside comfy.
    monkeypatch.setattr(h3, "MODELS_ROOT", str(tmp_path / "no_models_here"))
    monkeypatch.setattr(h3, "QWEN_ENCODER_PATH", f"{h3.MODELS_ROOT}/text_encoders/qwen.safetensors")
    monkeypatch.setattr(h3, "VIDEO_VAE_PATH", f"{h3.MODELS_ROOT}/vae/video.safetensors")
    monkeypatch.setattr(h3, "AUDIO_VAE_PATH", f"{h3.MODELS_ROOT}/vae/audio.safetensors")

    good_path = tmp_path / "good.safetensors"
    write_safetensors(good_path, h3_shaped_header(), data=b"\x00" * 4)
    with pytest.raises(FileNotFoundError, match="Qwen"):
        h3.H3Backend().generate(
            model_path=good_path,
            image_path=ASSET_IMAGE,
            prompt="p",
            output_path=tmp_path / "out.mp4",
        )


def test_h3_backend_generate_without_image_raises_for_missing_fixed_models(tmp_path, monkeypatch):
    # Same as above, but with no image_path at all -- text-to-video must reach the same
    # point in generate() (past checkpoint validation, past the image/no-image branch)
    # without ever touching the filesystem for an image.
    monkeypatch.setattr(h3, "MODELS_ROOT", str(tmp_path / "no_models_here"))
    monkeypatch.setattr(h3, "QWEN_ENCODER_PATH", f"{h3.MODELS_ROOT}/text_encoders/qwen.safetensors")
    monkeypatch.setattr(h3, "VIDEO_VAE_PATH", f"{h3.MODELS_ROOT}/vae/video.safetensors")
    monkeypatch.setattr(h3, "AUDIO_VAE_PATH", f"{h3.MODELS_ROOT}/vae/audio.safetensors")

    good_path = tmp_path / "good.safetensors"
    write_safetensors(good_path, h3_shaped_header(), data=b"\x00" * 4)
    with pytest.raises(FileNotFoundError, match="Qwen"):
        h3.H3Backend().generate(
            model_path=good_path,
            prompt="p",
            output_path=tmp_path / "out.mp4",
        )


def test_h3_backend_generate_rejects_empty_prompt(tmp_path):
    # Prompt is validated before the checkpoint even gets opened -- a nonexistent
    # checkpoint path proves this raises for the *prompt*, not something else.
    with pytest.raises(ValueError, match="Prompt"):
        h3.H3Backend().generate(
            model_path=tmp_path / "missing.safetensors",
            prompt="   ",
            output_path=tmp_path / "out.mp4",
        )
