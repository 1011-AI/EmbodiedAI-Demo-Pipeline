from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from pipelines.lerobot.behavior1k.checkpoint import (
    DELTA_MANIFEST,
    DELTA_WEIGHTS,
    save_pi05_delta_checkpoint,
)


class _FakeTensor:
    def __init__(self, value: str) -> None:
        self.value = value

    def detach(self):
        return self


class _FakeConfig:
    type = "pi05"
    train_expert_only = True

    @staticmethod
    def save_pretrained(path: Path) -> None:
        (path / "config.json").write_text('{"type":"pi05"}\n')


class _FakeTrainConfig:
    def __init__(self, base_path: Path) -> None:
        self.policy = SimpleNamespace(pretrained_path=base_path)

    @staticmethod
    def save_pretrained(path: Path) -> None:
        (path / "train_config.json").write_text("{}\n")


class _FakePolicy:
    config = _FakeConfig()

    @staticmethod
    def named_parameters():
        return (
            ("model.expert", SimpleNamespace(requires_grad=True)),
            ("model.base", SimpleNamespace(requires_grad=False)),
        )

    @staticmethod
    def state_dict():
        return {
            "model.expert": _FakeTensor("updated"),
            "model.base": _FakeTensor("base"),
        }


def test_delta_checkpoint_saves_only_trainable_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    saved = {}

    def fake_save(state, path):
        saved.update(state)
        Path(path).write_bytes(b"delta")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(save=fake_save))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    base_path = tmp_path / "models/pi05/base"
    base_path.mkdir(parents=True)
    checkpoint_dir = tmp_path / "runs/000002"

    save_pi05_delta_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=2,
        cfg=_FakeTrainConfig(base_path),
        policy=_FakePolicy(),
        optimizer=object(),
    )

    pretrained_dir = checkpoint_dir / "pretrained_model"
    assert set(saved) == {"model.expert"}
    assert (pretrained_dir / DELTA_WEIGHTS).read_bytes() == b"delta"
    manifest = json.loads((pretrained_dir / DELTA_MANIFEST).read_text())
    assert manifest["format"] == "pi05_trainable_delta_v1"
    assert manifest["base_pretrained_path"] == "models/pi05/base"
    assert manifest["tensor_count"] == 1
    assert manifest["resume_supported"] is False
