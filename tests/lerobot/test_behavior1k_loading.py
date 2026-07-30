from __future__ import annotations

from types import SimpleNamespace

from pipelines.lerobot.behavior1k import loading


class _FakeDevice:
    def __init__(self, kind: str, index: int | None = None) -> None:
        if ":" in kind:
            kind, raw_index = kind.split(":", 1)
            index = int(raw_index)
        self.type = kind
        self.index = index
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *_args):
        self.entered = False

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def current_device() -> int:
        return 3

    @staticmethod
    def device(target):
        return target


class _FakeTorch:
    cuda = _FakeCuda()

    @staticmethod
    def device(kind, index=None):
        if isinstance(kind, _FakeDevice):
            return kind
        return _FakeDevice(kind, index)


def test_direct_cuda_load_uses_current_rank_device_and_restores_patch(
    monkeypatch,
) -> None:
    calls = []

    def original_load_file(filename, *args, **kwargs):
        calls.append((filename, args, kwargs))
        return {"weight": "tensor"}

    fake_safetensors = SimpleNamespace(load_file=original_load_file)
    monkeypatch.setenv(loading.DIRECT_CUDA_LOAD_ENV, "1")
    monkeypatch.setattr(
        loading,
        "_load_runtime_modules",
        lambda: (_FakeTorch, fake_safetensors),
    )
    policy_cfg = SimpleNamespace(
        type="pi05",
        device="cuda",
        pretrained_path="/models/pi05",
    )

    def fake_make_policy(**kwargs):
        assert kwargs["cfg"] is policy_cfg
        result = fake_safetensors.load_file(
            "model.safetensors",
            backend="pread",
        )
        fake_safetensors.load_file("explicit.safetensors", device="cpu")
        fake_safetensors.load_file("positional.safetensors", "cpu")
        print("✓ Loaded state dict from model.safetensors")
        print("All keys loaded successfully!")
        return result

    result = loading.make_policy_with_memory_strategy(
        fake_make_policy,
        cfg=policy_cfg,
        ds_meta=object(),
    )

    assert result == {"weight": "tensor"}
    assert calls == [
        (
            "model.safetensors",
            (),
            {"backend": "pread", "device": "cuda:3"},
        ),
        ("explicit.safetensors", (), {"device": "cpu"}),
        ("positional.safetensors", ("cpu",), {}),
    ]
    assert fake_safetensors.load_file is original_load_file


def test_default_mode_delegates_without_importing_cuda_runtime(monkeypatch) -> None:
    monkeypatch.delenv(loading.DIRECT_CUDA_LOAD_ENV, raising=False)
    policy_cfg = SimpleNamespace(
        type="pi05",
        device="cuda",
        pretrained_path="/models/pi05",
    )
    observed = {}

    def fake_make_policy(**kwargs):
        observed.update(kwargs)
        print("✓ Loaded state dict from model.safetensors")
        print("All keys loaded successfully!")
        return "policy"

    assert (
        loading.make_policy_with_memory_strategy(
            fake_make_policy,
            cfg=policy_cfg,
            ds_meta="metadata",
        )
        == "policy"
    )
    assert observed["cfg"] is policy_cfg
    assert observed["ds_meta"] == "metadata"


def test_direct_cuda_load_restores_patch_when_policy_creation_fails(
    monkeypatch,
) -> None:
    def original_load_file(filename, *args, **kwargs):
        return (filename, args, kwargs)

    fake_safetensors = SimpleNamespace(load_file=original_load_file)
    monkeypatch.setenv(loading.DIRECT_CUDA_LOAD_ENV, "1")
    monkeypatch.setattr(
        loading,
        "_load_runtime_modules",
        lambda: (_FakeTorch, fake_safetensors),
    )
    policy_cfg = SimpleNamespace(
        type="pi05",
        device="cuda",
        pretrained_path="/models/pi05",
    )

    def failing_make_policy(**_kwargs):
        raise RuntimeError("construction failed")

    try:
        loading.make_policy_with_memory_strategy(
            failing_make_policy,
            cfg=policy_cfg,
            ds_meta=object(),
        )
    except RuntimeError as exc:
        assert str(exc) == "construction failed"
    else:  # pragma: no cover - test assertion branch.
        raise AssertionError("expected construction failure")

    assert fake_safetensors.load_file is original_load_file


def test_explicit_cuda_device_must_match_accelerate_current_device(
    monkeypatch,
) -> None:
    fake_safetensors = SimpleNamespace(load_file=lambda *_args, **_kwargs: {})
    monkeypatch.setenv(loading.DIRECT_CUDA_LOAD_ENV, "1")
    monkeypatch.setattr(
        loading,
        "_load_runtime_modules",
        lambda: (_FakeTorch, fake_safetensors),
    )
    policy_cfg = SimpleNamespace(
        type="pi05",
        device="cuda:1",
        pretrained_path="/models/pi05",
    )

    try:
        loading.make_policy_with_memory_strategy(
            lambda **_kwargs: object(),
            cfg=policy_cfg,
            ds_meta=object(),
        )
    except Exception as exc:
        assert "current=cuda:3" in str(exc)
    else:  # pragma: no cover - test assertion branch.
        raise AssertionError("expected device mismatch")
