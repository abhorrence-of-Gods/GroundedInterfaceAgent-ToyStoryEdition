import threading
import time
import socket
import struct
import pathlib
import sys
import asyncio

import pytest
import torch

from communication.edge_protocol import LatentMsg, ActionMsg, encode, decode
from engine.streaming_runner import StreamingRunner
from models.slw_transformer import SLWTransformer

# Ensure project root is on PYTHONPATH for local test invocation
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- Dummy model to keep StreamingRunner lightweight ----------------
class _DummyActionTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4096, 3)

    def forward(self, x: torch.Tensor):
        return self.fc(x)

    # for Gia compatibility
    def encode_action(self, x):
        return self.forward(x)

    def get_action_dist(self, latent):
        class Dist:
            def __init__(self, mean):
                self.mean = mean
            def rsample(self):
                return self.mean
        return Dist(self.forward(latent))


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_tower = _DummyActionTower()
        # register one parameter so .parameters() non-empty
        self._dummy = torch.nn.Parameter(torch.zeros(1))


@pytest.mark.timeout(10)
def test_edge_pipeline_integration():
    host, port = "127.0.0.1", 9999
    model = _DummyModel()

    # Start server in another thread
    runner = StreamingRunner(model, host=host, port=port)
    server_thread = threading.Thread(target=runner.start, daemon=True)
    server_thread.start()
    time.sleep(0.5)  # allow server to bind

    try:
        # Connect as raw client
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))

        # send one latent message and expect one action reply immediately
        z_vec = [0.0] * 4096
        msg = LatentMsg(robot_id="bot1", tick=0, z=z_vec, token_type=0, delta_tau=0.0)
        data = encode(msg)
        sock.sendall(struct.pack(">I", len(data)) + data)

        # read response
        hdr = sock.recv(4, socket.MSG_WAITALL)
        length = struct.unpack(">I", hdr)[0]
        payload = sock.recv(length, socket.MSG_WAITALL)
        action_msg = decode(payload)

        assert isinstance(action_msg, ActionMsg)
        assert action_msg.robot_id == "bot1"
        assert len(action_msg.action) == 3

        sock.close()

    finally:
        # Proper shutdown
        if runner.loop.is_running():
            runner.loop.call_soon_threadsafe(
                lambda: asyncio.create_task(runner._shutdown())
            )
        server_thread.join(timeout=3) 