"""Reference Python client that runs on Pico/Raspberry side to send latent data."""
import socket
from typing import List

from .edge_protocol import LatentMsg, encode, decode, ActionMsg


class BridgeClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, robot_id: str = "bot1") -> None:
        self.addr = (host, port)
        self.robot_id = robot_id
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.addr)
        self.tick = 0

    def send_latent(self, z: List[float], token_type: int, delta_tau: float):
        msg = LatentMsg(robot_id=self.robot_id, tick=self.tick, z=z, token_type=token_type, delta_tau=delta_tau)
        data = encode(msg)
        length = len(data).to_bytes(4, "big")
        self.sock.sendall(length + data)
        self.tick += 1

    def recv_action(self) -> ActionMsg | None:
        # blocking read length prefix
        header = self.sock.recv(4, socket.MSG_WAITALL)
        if not header:
            return None
        length = int.from_bytes(header, "big")
        payload = self.sock.recv(length, socket.MSG_WAITALL)
        return decode(payload)  # type: ignore 