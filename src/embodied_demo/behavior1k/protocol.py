"""BEHAVIOR-1K policy transport contract.

The official evaluator talks to a policy server with MessagePack frames that
preserve NumPy arrays.  A connection receives policy metadata first, then sends
observation mappings and receives one 23-D action per observation.  A request
containing the ``reset`` key resets policy state and deliberately has no reply.

This module owns the transport-independent session state.  A WebSocket runtime
can delegate binary frames to :meth:`BehaviorPolicySession.handle_frame`
without duplicating reset, action validation, or action-chunk buffering logic.
NumPy and MessagePack are imported lazily so importing ``embodied_demo`` keeps
the lightweight core dependency boundary intact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import importlib
import math
import time
from typing import Any

ACTION_DIM = 23


class ProtocolContractError(ValueError):
    """Raised when a policy request or response violates the wire contract."""


class ProtocolDependencyError(RuntimeError):
    """Raised when an optional protocol dependency is unavailable."""


def _optional_import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - exercised in minimal runtime environments.
        raise ProtocolDependencyError(
            f"{name} is required for the BEHAVIOR-1K policy protocol. "
            "Install the Behavior integration dependencies before serving a policy."
        ) from exc


def _numpy() -> Any:
    return _optional_import("numpy")


def _msgpack() -> Any:
    return _optional_import("msgpack")


def _to_numpy(value: Any) -> Any:
    """Convert a model result to NumPy without importing a model framework."""

    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    return value


def _extract_action_payload(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    if "actions" in value:
        return value["actions"]
    if "action" in value:
        return value["action"]
    raise ProtocolContractError(
        "policy mappings must contain an 'actions' chunk or a single 'action'"
    )


def validate_action_chunk(value: Any, *, action_dim: int = ACTION_DIM) -> Any:
    """Return a finite, contiguous ``float32[T, action_dim]`` action chunk."""

    if action_dim <= 0:
        raise ProtocolContractError("action_dim must be positive")

    np = _numpy()
    value = _to_numpy(_extract_action_payload(value))
    raw = np.asarray(value)
    if raw.dtype.kind not in ("i", "u", "f"):
        raise ProtocolContractError(
            f"actions must have a real numeric dtype, received {raw.dtype}"
        )

    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.ndim != 2:
        raise ProtocolContractError(
            f"actions must have shape ({action_dim},) or (T, {action_dim}), "
            f"received {tuple(raw.shape)}"
        )
    if raw.shape[0] == 0:
        raise ProtocolContractError("an action chunk must contain at least one action")
    if raw.shape[1] != action_dim:
        raise ProtocolContractError(
            f"action dimension must be {action_dim}, received {raw.shape[1]}"
        )

    actions = np.ascontiguousarray(raw, dtype=np.float32)
    if not bool(np.isfinite(actions).all()):
        raise ProtocolContractError("actions must contain only finite values")
    return actions


def validate_action(value: Any, *, action_dim: int = ACTION_DIM) -> Any:
    """Return one finite, contiguous ``float32[action_dim]`` action."""

    chunk = validate_action_chunk(value, action_dim=action_dim)
    if chunk.shape[0] != 1:
        raise ProtocolContractError(
            f"a single action was required, received a chunk of {chunk.shape[0]}"
        )
    return chunk[0].copy()


def _pack_array(value: Any) -> Any:
    np = _numpy()
    value = _to_numpy(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            raise ProtocolContractError(
                f"cannot serialize NumPy dtype {value.dtype} in a policy frame"
            )
        array = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": array.tobytes(),
            b"dtype": array.dtype.str,
            b"shape": array.shape,
        }
    if isinstance(value, np.generic):
        if value.dtype.kind in ("V", "O", "c"):
            raise ProtocolContractError(
                f"cannot serialize NumPy dtype {value.dtype} in a policy frame"
            )
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"unsupported MessagePack value: {type(value).__name__}")


def _unpack_array(value: dict[Any, Any]) -> Any:
    np = _numpy()
    if b"__ndarray__" in value:
        try:
            dtype = np.dtype(value[b"dtype"])
            shape = tuple(int(item) for item in value[b"shape"])
            data = value[b"data"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolContractError("malformed NumPy array payload") from exc
        if dtype.kind in ("V", "O", "c") or any(item < 0 for item in shape):
            raise ProtocolContractError("unsupported NumPy array payload")
        expected_bytes = dtype.itemsize * math.prod(shape)
        if not isinstance(data, bytes) or len(data) != expected_bytes:
            raise ProtocolContractError(
                "NumPy array payload byte length does not match dtype and shape"
            )
        return np.frombuffer(data, dtype=dtype).reshape(shape)

    if b"__npgeneric__" in value:
        try:
            dtype = np.dtype(value[b"dtype"])
            data = value[b"data"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolContractError("malformed NumPy scalar payload") from exc
        if dtype.kind in ("V", "O", "c"):
            raise ProtocolContractError("unsupported NumPy scalar payload")
        return dtype.type(data)
    return value


def packb(payload: Any) -> bytes:
    """Pack a protocol payload using the evaluator-compatible NumPy extension."""

    msgpack = _msgpack()
    return msgpack.packb(payload, default=_pack_array)


def unpackb(frame: bytes | bytearray | memoryview) -> Any:
    """Unpack an evaluator protocol frame and restore NumPy arrays."""

    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise ProtocolContractError("policy frames must be binary MessagePack data")
    msgpack = _msgpack()
    try:
        return msgpack.unpackb(
            bytes(frame),
            object_hook=_unpack_array,
            strict_map_key=False,
        )
    except ProtocolContractError:
        raise
    except Exception as exc:
        raise ProtocolContractError(f"invalid MessagePack policy frame: {exc}") from exc


class ActionChunkBuffer:
    """Serve one action at a time while reusing a model-predicted action chunk."""

    def __init__(
        self,
        *,
        action_dim: int = ACTION_DIM,
        execution_horizon: int | None = None,
    ) -> None:
        if action_dim <= 0:
            raise ProtocolContractError("action_dim must be positive")
        if execution_horizon is not None and execution_horizon <= 0:
            raise ProtocolContractError("execution_horizon must be positive when set")
        self.action_dim = action_dim
        self.execution_horizon = execution_horizon
        self._chunk: Any | None = None
        self._cursor = 0

    @property
    def remaining(self) -> int:
        if self._chunk is None:
            return 0
        return int(self._chunk.shape[0] - self._cursor)

    def reset(self) -> None:
        self._chunk = None
        self._cursor = 0

    def _refill(self, value: Any) -> None:
        chunk = validate_action_chunk(value, action_dim=self.action_dim)
        if self.execution_horizon is not None:
            chunk = chunk[: self.execution_horizon]
        self._chunk = chunk.copy()
        self._cursor = 0

    def next_action(self, predict: Callable[[], Any]) -> Any:
        if self.remaining == 0:
            self._refill(predict())
        action = self._chunk[self._cursor].copy()
        self._cursor += 1
        return action


class BehaviorPolicySession:
    """Pure connection state for the BEHAVIOR-1K policy server protocol."""

    def __init__(
        self,
        policy: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
        action_dim: int = ACTION_DIM,
        execution_horizon: int | None = None,
    ) -> None:
        self.policy = policy
        self.metadata = dict(metadata or {})
        self.action_buffer = ActionChunkBuffer(
            action_dim=action_dim,
            execution_horizon=execution_horizon,
        )
        self._opened = False
        self._previous_total_ms: float | None = None

    @property
    def opened(self) -> bool:
        return self._opened

    def open(self) -> dict[str, Any]:
        """Return the mandatory metadata-first payload exactly once."""

        if self._opened:
            raise ProtocolContractError("session metadata has already been emitted")
        self._opened = True
        return deepcopy(self.metadata)

    def open_frame(self) -> bytes:
        return packb(self.open())

    def _predict(self, observation: Mapping[str, Any]) -> Any:
        if hasattr(self.policy, "predict_action_chunk"):
            return self.policy.predict_action_chunk(observation)
        if hasattr(self.policy, "act"):
            return self.policy.act(observation)
        if callable(self.policy):
            return self.policy(observation)
        raise ProtocolContractError(
            "policy must define predict_action_chunk(observation), act(observation), "
            "or be callable"
        )

    def _reset(self) -> None:
        self.action_buffer.reset()
        self._previous_total_ms = None
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        """Handle one decoded request.

        Returning ``None`` means that the transport must not send a frame.  This
        is the required behavior for reset requests.
        """

        if not self._opened:
            raise ProtocolContractError("session metadata must be emitted before requests")
        if not isinstance(request, Mapping):
            raise ProtocolContractError("policy requests must decode to a mapping")

        if "reset" in request:
            self._reset()
            return None

        started = time.monotonic()
        observation = deepcopy(dict(request))
        action = self.action_buffer.next_action(lambda: self._predict(observation))
        infer_ms = (time.monotonic() - started) * 1000.0

        timing: dict[str, float] = {"infer_ms": infer_ms}
        if self._previous_total_ms is not None:
            timing["prev_total_ms"] = self._previous_total_ms
        response = {
            "action": validate_action(action, action_dim=self.action_buffer.action_dim),
            "server_timing": timing,
        }
        self._previous_total_ms = (time.monotonic() - started) * 1000.0
        return response

    def handle_frame(self, request_frame: bytes) -> bytes | None:
        request = unpackb(request_frame)
        response = self.handle(request)
        return None if response is None else packb(response)
