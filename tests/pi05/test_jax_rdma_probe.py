from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/pi05/jax_rdma_probe.py"
SPEC = importlib.util.spec_from_file_location("pi05_jax_rdma_probe", SCRIPT)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_nccl_parser_accepts_ib_and_records_gdr() -> None:
    parsed = probe.parse_nccl_transport(
        "Channel 00 : 0[0] -> 8[0] via NET/IB/0/GDRDMA"
    )
    assert parsed == {
        "transport": "ib",
        "net_ib_observed": True,
        "net_socket_observed": False,
        "gdr_observed": True,
    }


def test_nccl_parser_rejects_socket_or_mixed_transport() -> None:
    assert probe.parse_nccl_transport("Using network NET/Socket")["transport"] == "socket"
    assert (
        probe.parse_nccl_transport("NET/IB then NET/Socket")["transport"]
        == "mixed_ib_socket"
    )
