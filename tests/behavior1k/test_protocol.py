from __future__ import annotations

import importlib
import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("msgpack")

from embodied_demo.behavior1k.protocol import (  # noqa: E402
    ACTION_DIM,
    ActionChunkBuffer,
    BehaviorPolicySession,
    ProtocolContractError,
    packb,
    unpackb,
    validate_action,
    validate_action_chunk,
)


class ChunkPolicy:
    def __init__(self) -> None:
        self.predict_calls = 0
        self.reset_calls = 0

    def predict_action_chunk(self, observation):
        self.predict_calls += 1
        offset = float(observation["offset"])
        return np.stack(
            [
                np.full(ACTION_DIM, offset + index, dtype=np.float64)
                for index in range(3)
            ]
        )

    def reset(self) -> None:
        self.reset_calls += 1


def test_optional_dependencies_are_lazy_at_module_import(monkeypatch) -> None:
    module = importlib.import_module("embodied_demo.behavior1k.protocol")
    calls: list[str] = []
    real_import = module.importlib.import_module

    def recording_import(name: str):
        calls.append(name)
        return real_import(name)

    monkeypatch.setattr(module.importlib, "import_module", recording_import)
    assert module.ACTION_DIM == 23
    assert calls == []


def test_codec_round_trip_preserves_numpy_arrays_and_scalars() -> None:
    payload = {
        "image": np.arange(24, dtype=np.uint8).reshape(2, 3, 4),
        "state": np.linspace(0, 1, ACTION_DIM, dtype=np.float32),
        "index": np.int64(7),
    }

    decoded = unpackb(packb(payload))

    np.testing.assert_array_equal(decoded["image"], payload["image"])
    np.testing.assert_array_equal(decoded["state"], payload["state"])
    assert decoded["image"].dtype == np.uint8
    assert decoded["state"].dtype == np.float32
    assert isinstance(decoded["index"], np.int64)
    assert decoded["index"] == 7


@pytest.mark.parametrize(
    "value",
    [
        np.array([object()], dtype=object),
        np.array([1 + 2j], dtype=np.complex64),
    ],
)
def test_codec_rejects_unsafe_numpy_dtypes(value) -> None:
    with pytest.raises(ProtocolContractError, match="dtype"):
        packb({"value": value})


def test_codec_rejects_malformed_array_byte_length() -> None:
    import msgpack

    malformed = msgpack.packb(
        {
            "value": {
                b"__ndarray__": True,
                b"data": b"\x00",
                b"dtype": "<f4",
                b"shape": (2,),
            }
        }
    )
    with pytest.raises(ProtocolContractError, match="byte length"):
        unpackb(malformed)


def test_action_validation_returns_finite_float32_contract() -> None:
    action = validate_action([float(index) for index in range(ACTION_DIM)])
    assert action.shape == (ACTION_DIM,)
    assert action.dtype == np.float32
    assert bool(np.isfinite(action).all())

    chunk = validate_action_chunk(
        np.zeros((4, ACTION_DIM), dtype=np.float64)
    )
    assert chunk.shape == (4, ACTION_DIM)
    assert chunk.dtype == np.float32


@pytest.mark.parametrize(
    "value, match",
    [
        (np.zeros(ACTION_DIM - 1), "dimension"),
        (np.zeros((2, 3, ACTION_DIM)), "shape"),
        (np.empty((0, ACTION_DIM)), "at least one"),
        (np.full(ACTION_DIM, np.nan), "finite"),
        (np.full(ACTION_DIM, np.inf), "finite"),
        (np.full(ACTION_DIM, "x"), "numeric"),
    ],
)
def test_action_validation_rejects_invalid_payloads(value, match: str) -> None:
    with pytest.raises(ProtocolContractError, match=match):
        validate_action_chunk(value)


def test_chunk_buffer_reuses_chunk_until_exhausted() -> None:
    calls = 0

    def predict():
        nonlocal calls
        calls += 1
        return np.stack(
            [
                np.full(ACTION_DIM, calls * 10 + index, dtype=np.float32)
                for index in range(3)
            ]
        )

    buffer = ActionChunkBuffer()
    actions = [buffer.next_action(predict) for _ in range(4)]

    assert calls == 2
    assert [float(action[0]) for action in actions] == [10.0, 11.0, 12.0, 20.0]
    assert buffer.remaining == 2


def test_chunk_buffer_execution_horizon_forces_early_replan() -> None:
    calls = 0

    def predict():
        nonlocal calls
        calls += 1
        return np.full((5, ACTION_DIM), calls, dtype=np.float32)

    buffer = ActionChunkBuffer(execution_horizon=2)
    outputs = [buffer.next_action(predict) for _ in range(3)]

    assert calls == 2
    assert [float(action[0]) for action in outputs] == [1.0, 1.0, 2.0]


def test_session_emits_metadata_before_requests_and_only_once() -> None:
    session = BehaviorPolicySession(
        ChunkPolicy(),
        metadata={"policy": "pi05", "action_dim": ACTION_DIM},
    )

    with pytest.raises(ProtocolContractError, match="metadata"):
        session.handle({"offset": 0})

    metadata = unpackb(session.open_frame())
    assert metadata == {"policy": "pi05", "action_dim": ACTION_DIM}

    with pytest.raises(ProtocolContractError, match="already"):
        session.open()


def test_session_returns_one_float32_action_and_reuses_chunk() -> None:
    policy = ChunkPolicy()
    session = BehaviorPolicySession(policy, metadata={"policy": "test"})
    session.open()

    responses = [session.handle({"offset": 10}) for _ in range(4)]

    assert policy.predict_calls == 2
    for response in responses:
        assert response is not None
        assert response["action"].shape == (ACTION_DIM,)
        assert response["action"].dtype == np.float32
        assert response["server_timing"]["infer_ms"] >= 0
    assert [float(response["action"][0]) for response in responses] == [
        10.0,
        11.0,
        12.0,
        10.0,
    ]
    assert "prev_total_ms" not in responses[0]["server_timing"]
    assert responses[1]["server_timing"]["prev_total_ms"] >= 0


def test_reset_frame_has_no_ack_and_clears_policy_and_chunk_state() -> None:
    policy = ChunkPolicy()
    session = BehaviorPolicySession(policy, metadata={"policy": "test"})
    session.open_frame()

    first = unpackb(session.handle_frame(packb({"offset": 5})))
    assert float(first["action"][0]) == 5.0
    assert policy.predict_calls == 1
    assert session.action_buffer.remaining == 2

    reset_response = session.handle_frame(packb({"reset": True}))
    assert reset_response is None
    assert policy.reset_calls == 1
    assert session.action_buffer.remaining == 0

    after_reset = unpackb(session.handle_frame(packb({"offset": 20})))
    assert float(after_reset["action"][0]) == 20.0
    assert "prev_total_ms" not in after_reset["server_timing"]
    assert policy.predict_calls == 2


def test_session_rejects_non_mapping_request() -> None:
    session = BehaviorPolicySession(ChunkPolicy())
    session.open()
    with pytest.raises(ProtocolContractError, match="mapping"):
        session.handle([1, 2, 3])
