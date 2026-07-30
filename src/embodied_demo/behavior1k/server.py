"""WebSocket transport for the BEHAVIOR-1K policy protocol.

The wire format is implemented by :mod:`embodied_demo.behavior1k.protocol`.
This module only owns network lifecycle and the official ``/healthz`` route.
Model loading stays in each pipeline so this transport can also serve future
FastWAM and other policy backends without importing them.
"""

from __future__ import annotations

import asyncio
import http
import logging
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Mapping

from .protocol import BehaviorPolicySession, ProtocolContractError

LOGGER = logging.getLogger(__name__)


class PolicyServerDependencyError(RuntimeError):
    """Raised when the optional WebSocket runtime is unavailable."""


class BehaviorWebSocketPolicyServer:
    """Serve one policy through the official BEHAVIOR/OpenPI wire protocol.

    A fresh :class:`BehaviorPolicySession` is created per connection, so action
    chunk cursors never leak between evaluator connections.  Model calls are
    serialized with a process-local lock because one GPU module and its reset
    state are shared by all sessions.
    """

    def __init__(
        self,
        policy: Any,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: Mapping[str, Any] | None = None,
        action_dim: int = 23,
        execution_horizon: int | None = None,
    ) -> None:
        if not host:
            raise ValueError("host must not be empty")
        if port < 0 or port > 65535:
            raise ValueError("port must be between 0 and 65535")
        self.policy = policy
        self.host = host
        self.port = port
        self.metadata = dict(metadata or {})
        self.action_dim = action_dim
        self.execution_horizon = execution_horizon
        self.bound_port: int | None = None
        self._policy_lock: asyncio.Lock | None = None
        self._client_lock: asyncio.Lock | None = None
        self._model_executor: ThreadPoolExecutor | None = None

    @staticmethod
    def _websockets() -> tuple[Any, Any]:
        try:
            import websockets
            import websockets.asyncio.server as websocket_server
        except ImportError as exc:  # pragma: no cover - minimal core environment.
            raise PolicyServerDependencyError(
                "websockets is required to serve a BEHAVIOR policy; "
                "install the project behavior1k extra"
            ) from exc
        return websockets, websocket_server

    @staticmethod
    def _health_check(connection: Any, request: Any) -> Any | None:
        """Handle the official evaluator's HTTP readiness probe."""

        if getattr(request, "path", None) != "/healthz":
            return None
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        # Compatibility with the legacy websockets process_request contract.
        return (
            http.HTTPStatus.OK,
            [("Content-Type", "text/plain; charset=utf-8")],
            b"OK\n",
        )

    async def _handler(self, websocket: Any) -> None:
        websockets, _ = self._websockets()
        if self._client_lock is None or self._model_executor is None:
            raise RuntimeError("policy server runtime was not initialized")
        client_lock = self._client_lock
        # A single policy object owns mutable model and reset state. Refuse a
        # second evaluator instead of allowing it to corrupt the active rollout.
        if client_lock.locked():
            await websocket.close(
                code=1013,
                reason="Only one active evaluator connection is supported",
            )
            return
        await client_lock.acquire()
        try:
            session = BehaviorPolicySession(
                self.policy,
                metadata=self.metadata,
                action_dim=self.action_dim,
                execution_horizon=self.execution_horizon,
            )
            await websocket.send(session.open_frame())
            LOGGER.info(
                "BEHAVIOR policy connection opened from %s",
                websocket.remote_address,
            )
            async for request_frame in websocket:
                if isinstance(request_frame, str):
                    raise ProtocolContractError(
                        "BEHAVIOR policy requests must be binary MessagePack frames"
                    )
                # Inference is synchronous and GPU-backed.  The lock prevents a
                # second client from resetting or entering the same policy
                # module while this request is active.
                if self._policy_lock is None:  # pragma: no cover - open_server owns it.
                    raise RuntimeError("policy server lock was not initialized")
                async with self._policy_lock:
                    loop = asyncio.get_running_loop()
                    response_frame = await loop.run_in_executor(
                        self._model_executor,
                        session.handle_frame,
                        request_frame,
                    )
                # The official client intentionally doesn't wait for reset ACK.
                if response_frame is not None:
                    await websocket.send(response_frame)
        except websockets.ConnectionClosed:
            pass
        except Exception:
            LOGGER.exception("BEHAVIOR policy connection failed")
            await websocket.close(code=1011, reason="Policy inference failed")
            raise
        finally:
            client_lock.release()
            LOGGER.info("BEHAVIOR policy connection closed")

    @asynccontextmanager
    async def open_server(self) -> AsyncIterator[Any]:
        """Open the listener and yield the underlying websockets server.

        ``port=0`` is supported for transport tests.  The selected port is then
        exposed through :attr:`bound_port`.
        """

        _, websocket_server = self._websockets()
        self._policy_lock = asyncio.Lock()
        self._client_lock = asyncio.Lock()
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="behavior-policy",
        )
        self._model_executor = executor
        try:
            async with websocket_server.serve(
                self._handler,
                self.host,
                self.port,
                compression=None,
                max_size=None,
                process_request=self._health_check,
            ) as server:
                sockets = tuple(server.sockets or ())
                if not sockets:
                    raise RuntimeError(
                        "WebSocket server opened without a listening socket"
                    )
                self.bound_port = int(sockets[0].getsockname()[1])
                LOGGER.info(
                    "BEHAVIOR_POLICY_SERVER_READY host=%s port=%d",
                    self.host,
                    self.bound_port,
                )
                yield server
        finally:
            self.bound_port = None
            self._policy_lock = None
            self._client_lock = None
            self._model_executor = None
            executor.shutdown(wait=True, cancel_futures=True)

    async def run(self) -> None:
        async with self.open_server() as server:
            await server.serve_forever()

    def serve_forever(self) -> None:
        asyncio.run(self.run())
