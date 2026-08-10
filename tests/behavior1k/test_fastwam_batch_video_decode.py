from __future__ import annotations

from pathlib import Path

import torch

from fastwam.datasets.lerobot.lerobot.datasets import video_utils


def test_torchcodec_batch_predecode_coalesces_same_file(monkeypatch) -> None:
    calls = []

    def fake_decode(path, timestamps, tolerance_s, device="cpu", **_kwargs):
        calls.append((str(path), list(timestamps), tolerance_s, device))
        return torch.arange(len(timestamps), dtype=torch.float32)[:, None]

    monkeypatch.setattr(video_utils, "localize_video_path", lambda path: Path(path))
    monkeypatch.setattr(video_utils, "decode_video_frames_torchcodec", fake_decode)
    requests = [
        (Path("/data/camera.mp4"), [1.0, 2.0], 0.001, "torchcodec"),
        (Path("/data/camera.mp4"), [5.0, 6.0, 7.0], 0.001, "torchcodec"),
    ]

    with video_utils.predecode_video_frames(requests):
        first = video_utils.decode_video_frames(
            Path("/data/camera.mp4"), [1.0, 2.0], 0.001, "torchcodec"
        )
        second = video_utils.decode_video_frames(
            Path("/data/camera.mp4"), [5.0, 6.0, 7.0], 0.001, "torchcodec"
        )

    assert calls == [
        ("/data/camera.mp4", [1.0, 2.0, 5.0, 6.0, 7.0], 0.001, "cpu")
    ]
    torch.testing.assert_close(first, torch.tensor([[0.0], [1.0]]))
    torch.testing.assert_close(second, torch.tensor([[2.0], [3.0], [4.0]]))
