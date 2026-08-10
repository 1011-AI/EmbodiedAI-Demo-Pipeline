from __future__ import annotations

import asyncio
import functools
import time

import pytest

np = pytest.importorskip("numpy")
msgpack = pytest.importorskip("msgpack")
pytest.importorskip("websockets")

from embodied_demo.behavior1k.protocol import ACTION_DIM  # noqa: E402
from embodied_demo.behavior1k.server import (  # noqa: E402
    BehaviorWebSocketPolicyServer,
)


class FakeTransportPolicy:
    """Deterministic policy used only to test transport state, never model quality."""

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


def _official_pack_data(value):
    """Independent copy of the BEHAVIOR v3.9.1 msgpack-numpy extension."""

    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _official_unpack_data(value):
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


_official_packb = functools.partial(msgpack.packb, default=_official_pack_data)
_official_unpackb = functools.partial(
    msgpack.unpackb,
    object_hook=_official_unpack_data,
    strict_map_key=False,
)


async def _healthz(port: int) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        b"GET /healthz HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Connection: close\r\n\r\n"
    )
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()
    return response


async def _exercise_transport() -> None:
    from websockets.asyncio.client import connect

    policy = FakeTransportPolicy()
    server = BehaviorWebSocketPolicyServer(
        policy,
        host="127.0.0.1",
        port=0,
        metadata={"policy_type": "transport-test", "action_dim": ACTION_DIM},
        execution_horizon=2,
    )
    async with server.open_server():
        assert server.bound_port is not None
        health = await _healthz(server.bound_port)
        assert health.startswith(b"HTTP/1.1 200")
        assert health.endswith(b"OK\n")

        async with connect(
            f"ws://127.0.0.1:{server.bound_port}",
            compression=None,
            max_size=None,
        ) as websocket:
            metadata = _official_unpackb(await websocket.recv())
            assert metadata == {
                "policy_type": "transport-test",
                "action_dim": ACTION_DIM,
            }

            await websocket.send(_official_packb({"offset": 10.0}))
            first = _official_unpackb(await websocket.recv())
            assert first["action"].dtype == np.float32
            assert first["action"].shape == (ACTION_DIM,)
            assert float(first["action"][0]) == 10.0

            await websocket.send(_official_packb({"offset": 999.0}))
            second = _official_unpackb(await websocket.recv())
            # The second request consumes the cached chunk, independent of its
            # new observation, because execution_horizon is two.
            assert float(second["action"][0]) == 11.0
            assert policy.predict_calls == 1

            # Official reset has no ACK. Sending an observation immediately
            # after it makes the next received frame unambiguously an action.
            await websocket.send(_official_packb({"reset": True}))
            await websocket.send(_official_packb({"offset": 50.0}))
            after_reset = _official_unpackb(await websocket.recv())
            assert float(after_reset["action"][0]) == 50.0
            assert "prev_total_ms" not in after_reset["server_timing"]
            assert policy.reset_calls == 1
            assert policy.predict_calls == 2


def test_real_websocket_transport_health_metadata_reset_and_action() -> None:
    asyncio.run(_exercise_transport())


class SlowTransportPolicy(FakeTransportPolicy):
    def predict_action_chunk(self, observation):
        time.sleep(0.3)
        return super().predict_action_chunk(observation)


async def _exercise_single_client_and_nonblocking_healthz() -> None:
    import websockets
    from websockets.asyncio.client import connect

    server = BehaviorWebSocketPolicyServer(
        SlowTransportPolicy(),
        host="127.0.0.1",
        port=0,
    )
    async with server.open_server():
        assert server.bound_port is not None
        uri = f"ws://127.0.0.1:{server.bound_port}"
        async with connect(uri, compression=None, max_size=None) as primary:
            await primary.recv()

            secondary = await connect(uri, compression=None, max_size=None)
            with pytest.raises(websockets.ConnectionClosed) as closed:
                await secondary.recv()
            assert closed.value.rcvd is not None
            assert closed.value.rcvd.code == 1013

            await primary.send(_official_packb({"offset": 1.0}))
            pending_action = asyncio.create_task(primary.recv())
            await asyncio.sleep(0.03)
            started = time.monotonic()
            health = await _healthz(server.bound_port)
            health_latency = time.monotonic() - started
            assert health.startswith(b"HTTP/1.1 200")
            assert health_latency < 0.2
            await pending_action


def test_server_rejects_second_client_and_keeps_healthz_responsive() -> None:
    asyncio.run(_exercise_single_client_and_nonblocking_healthz())
