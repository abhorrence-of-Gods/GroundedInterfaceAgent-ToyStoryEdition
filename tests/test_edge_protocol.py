import pytest
import torch
import pathlib, sys
# Ensure project root is on PYTHONPATH for local test invocation
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from communication.edge_protocol import LatentMsg, ActionMsg, encode, decode


def test_encode_decode_latent():
    msg = LatentMsg(robot_id="bot1", tick=1, z=[0.1, 0.2], token_type=0, delta_tau=0.0)
    data = encode(msg)
    msg2 = decode(data)
    assert isinstance(msg2, LatentMsg)
    assert msg2.robot_id == "bot1"
    assert msg2.z == [0.1, 0.2]


def test_encode_decode_action():
    msg = ActionMsg(robot_id="bot1", tick=2, action=[0.0, 1.0, 2.0])
    data = encode(msg)
    msg2 = decode(data)
    assert isinstance(msg2, ActionMsg)
    assert msg2.action == [0.0, 1.0, 2.0] 