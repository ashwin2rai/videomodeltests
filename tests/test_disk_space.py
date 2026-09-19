import pytest

import h3


def test_check_disk_space_passes_with_real_free_space(tmp_path):
    h3.check_disk_space(tmp_path / "out.mp4")


def test_check_disk_space_raises_when_insufficient(tmp_path, monkeypatch):
    class FakeUsage:
        free = 10

    monkeypatch.setattr(h3.shutil, "disk_usage", lambda path: FakeUsage())

    with pytest.raises(OSError, match="Insufficient disk space"):
        h3.check_disk_space(tmp_path / "out.mp4")
