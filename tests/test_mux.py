import json
import math
import shutil
import struct
import subprocess

import pytest

import h3

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

WIDTH, HEIGHT, FPS = 64, 64, 5
SAMPLE_RATE = 8000


def synthetic_frames(n):
    for i in range(n):
        color = bytes([i * 20 % 256, 100, 200])
        yield color * (WIDTH * HEIGHT)


def synthetic_tone(seconds):
    n_samples = int(SAMPLE_RATE * seconds)
    samples = [int(3000 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE)) for i in range(n_samples)]
    return struct.pack(f"<{n_samples}h", *samples)


@requires_ffmpeg
def test_mux_mp4_creates_playable_output(tmp_path):
    output = tmp_path / "out.mp4"
    h3.mux_mp4(
        frames=synthetic_frames(5),
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        pcm_audio=synthetic_tone(1.0),
        sample_rate=SAMPLE_RATE,
        output_path=output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type,codec_name",
            "-of", "json", str(output),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    streams = json.loads(probe.stdout)["streams"]

    video_streams = [s for s in streams if s["codec_type"] == "video"]
    audio_streams = [s for s in streams if s["codec_type"] == "audio"]
    assert len(video_streams) == 1
    assert video_streams[0]["codec_name"] == "h264"
    assert len(audio_streams) == 1
    assert audio_streams[0]["codec_name"] == "aac"


def test_mux_mp4_missing_ffmpeg_raises_clear_error(tmp_path):
    output = tmp_path / "out.mp4"
    with pytest.raises(RuntimeError, match="ffmpeg"):
        h3.mux_mp4(
            frames=synthetic_frames(2),
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
            pcm_audio=synthetic_tone(0.1),
            sample_rate=SAMPLE_RATE,
            output_path=output,
            ffmpeg_bin="definitely-not-a-real-binary",
        )
