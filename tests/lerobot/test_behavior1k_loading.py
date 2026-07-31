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
    loaded = []

    @staticmethod
    def device(kind, index=None):
        if isinstance(kind, _FakeDevice):
            return kind
        return _FakeDevice(kind, index)

    @classmethod
    def load(cls, path, **kwargs):
        cls.loaded.append((path, kwargs))
        return {"model.expert": "updated"}


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


def test_distributed_direct_load_is_serialized_under_small_cgroup(
    monkeypatch,
) -> None:
    events = []

    class FakeDistributed:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def get_world_size() -> int:
            return 3

        @staticmethod
        def get_rank() -> int:
            return 1

        @staticmethod
        def barrier() -> None:
            events.append("barrier")

    class FakeCuda:
        @staticmethod
        def synchronize() -> None:
            events.append("synchronize")

        @staticmethod
        def empty_cache() -> None:
            events.append("empty_cache")

    fake_torch = SimpleNamespace(distributed=FakeDistributed(), cuda=FakeCuda())
    monkeypatch.delenv(loading.SERIALIZE_DISTRIBUTED_LOAD_ENV, raising=False)
    monkeypatch.setattr(loading, "_cgroup_memory_limit_bytes", lambda: 16 * 1024**3)

    result = loading._call_with_distributed_load_strategy(
        fake_torch,
        lambda: events.append("load") or "policy",
    )

    assert result == "policy"
    assert events == [
        "barrier",
        "load",
        "synchronize",
        "empty_cache",
        "barrier",
        "barrier",
    ]


def test_distributed_load_auto_serializes_at_eight_gib_per_rank(
    monkeypatch,
) -> None:
    fake_torch = object()
    monkeypatch.delenv(loading.SERIALIZE_DISTRIBUTED_LOAD_ENV, raising=False)
    monkeypatch.setattr(
        loading,
        "_distributed_load_context",
        lambda _torch: (object(), 0, 4),
    )
    monkeypatch.setattr(
        loading,
        "_cgroup_memory_limit_bytes",
        lambda: 32 * 1024**3,
    )

    assert loading._serialize_distributed_load_requested(fake_torch) is True


def test_distributed_direct_load_serialization_can_be_disabled(
    monkeypatch,
) -> None:
    class FakeDistributed:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def get_world_size() -> int:
            return 8

        @staticmethod
        def get_rank() -> int:
            return 0

        @staticmethod
        def barrier() -> None:  # pragma: no cover - must remain unused.
            raise AssertionError("barrier should not be called")

    fake_torch = SimpleNamespace(distributed=FakeDistributed())
    monkeypatch.setenv(loading.SERIALIZE_DISTRIBUTED_LOAD_ENV, "false")

    assert (
        loading._call_with_distributed_load_strategy(fake_torch, lambda: "policy")
        == "policy"
    )


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


def test_delta_checkpoint_loads_base_then_applies_trainable_state(
    tmp_path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text("[project]\nname='test'\n")
    base_dir = project_root / "models/pi05/base"
    base_dir.mkdir(parents=True)
    (base_dir / "model.safetensors").write_bytes(b"base")
    checkpoint_dir = project_root / "runs/checkpoint/pretrained_model"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "trainable_state.pt").write_bytes(b"delta")
    (checkpoint_dir / "behavior1k_delta_checkpoint.json").write_text(
        '{"format":"pi05_trainable_delta_v1",'
        '"weights":"trainable_state.pt",'
        '"base_pretrained_path":"models/pi05/base"}'
    )

    fake_safetensors = SimpleNamespace(
        load_file=lambda *_args, **_kwargs: {"base": "tensor"},
    )
    monkeypatch.chdir(project_root)
    monkeypatch.setenv(loading.DIRECT_CUDA_LOAD_ENV, "1")
    monkeypatch.setattr(
        loading,
        "_load_runtime_modules",
        lambda: (_FakeTorch, fake_safetensors),
    )
    _FakeTorch.loaded = []
    policy_cfg = SimpleNamespace(
        type="pi05",
        device="cuda",
        pretrained_path=checkpoint_dir,
    )

    class FakePolicy:
        def __init__(self):
            self.loaded = None

        @staticmethod
        def state_dict():
            return {"model.base": "base", "model.expert": "old"}

        def load_state_dict(self, state, strict):
            self.loaded = (state, strict)
            return SimpleNamespace(unexpected_keys=[])

    policy = FakePolicy()

    def fake_make_policy(**kwargs):
        assert kwargs["cfg"].pretrained_path == base_dir.resolve()
        fake_safetensors.load_file("model.safetensors")
        print("✓ Loaded state dict from model.safetensors")
        print("All keys loaded successfully!")
        return policy

    loaded = loading.make_policy_with_memory_strategy(
        fake_make_policy,
        cfg=policy_cfg,
        ds_meta=object(),
    )

    assert loaded is policy
    assert policy.loaded == ({"model.expert": "updated"}, False)
    assert policy_cfg.pretrained_path == checkpoint_dir
    assert _FakeTorch.loaded[0][0] == checkpoint_dir / "trainable_state.pt"
    assert str(_FakeTorch.loaded[0][1]["map_location"]) == "cuda:3"
