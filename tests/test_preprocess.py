import pytest
from conftest import ASSET_IMAGE, h3_shaped_header, write_safetensors
from PIL import Image

import h3


def test_preprocess_square_image():
    img = h3.preprocess_image(ASSET_IMAGE)
    assert img.size == h3.compute_resolution(64, 64)
    assert img.mode == "RGB"


def test_preprocess_landscape_image(tmp_path):
    path = tmp_path / "landscape.jpg"
    Image.new("RGB", (1920, 1080), color=(10, 20, 30)).save(path)
    img = h3.preprocess_image(path)
    assert img.size == (1344, 768)


def test_preprocess_respects_exif_orientation(tmp_path):
    path = tmp_path / "rotated.jpg"
    # Stored as 100x50 (landscape) but EXIF says rotate 90 (orientation 6),
    # so the effective orientation is 50x100 (portrait).
    base = Image.new("RGB", (100, 50), color=(200, 50, 50))
    exif = base.getexif()
    exif[0x0112] = 6  # Orientation tag
    base.save(path, exif=exif)

    img = h3.preprocess_image(path)
    assert img.size == h3.compute_resolution(50, 100)


def test_preprocess_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        h3.preprocess_image(tmp_path / "missing.jpg")


def test_preprocess_non_image_raises(tmp_path):
    bad_file = tmp_path / "bad.jpg"
    bad_file.write_text("not an image")
    with pytest.raises(ValueError):
        h3.preprocess_image(bad_file)


def test_h3_backend_generate_raises_for_missing_image(tmp_path):
    good_checkpoint = tmp_path / "good.safetensors"
    write_safetensors(good_checkpoint, h3_shaped_header(), data=b"\x00" * 4)
    with pytest.raises(FileNotFoundError):
        h3.H3Backend().generate(
            model_path=good_checkpoint,
            image_path=tmp_path / "missing.jpg",
            prompt="p",
            output_path=tmp_path / "out.mp4",
        )
