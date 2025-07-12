"""Simple CBOR-based edge protocol.

All messages are length-prefixed CBOR bytes.
Schema (dict):
    {
        "robot_id": str,
        "tick": int,
        "type": "LATENT" | "ACTION",
        "payload": {
            # For LATENT
            "z": float list,
            "token_type": int,  # 0 REAL, 1 IMAG
            "delta_tau": float,
        }
    }
"""
from __future__ import annotations

import cbor2
from dataclasses import dataclass
from typing import Any, Dict

__all__ = ["encode", "decode", "LatentMsg", "ActionMsg"]


@dataclass
class LatentMsg:
    robot_id: str
    tick: int
    z: list[float]
    token_type: int  # 0/1
    delta_tau: float


@dataclass
class ActionMsg:
    robot_id: str
    tick: int
    action: list[float]


def encode(obj: LatentMsg | ActionMsg) -> bytes:
    if isinstance(obj, LatentMsg):
        doc: Dict[str, Any] = {
            "robot_id": obj.robot_id,
            "tick": obj.tick,
            "type": "LATENT",
            "payload": {
                "z": obj.z,
                "token_type": obj.token_type,
                "delta_tau": obj.delta_tau,
            },
        }
    else:
        doc = {
            "robot_id": obj.robot_id,
            "tick": obj.tick,
            "type": "ACTION",
            "payload": {"action": obj.action},
        }
    return cbor2.dumps(doc)


def decode(data: bytes) -> LatentMsg | ActionMsg:
    doc = cbor2.loads(data)
    if doc["type"] == "LATENT":
        p = doc["payload"]
        return LatentMsg(
            robot_id=doc["robot_id"],
            tick=doc["tick"],
            z=p["z"],
            token_type=p["token_type"],
            delta_tau=p["delta_tau"],
        )
    else:
        p = doc["payload"]
        return ActionMsg(robot_id=doc["robot_id"], tick=doc["tick"], action=p["action"]) 