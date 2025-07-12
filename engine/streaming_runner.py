"""Streaming Runner for Toy-Story Central Brain.

TCP server handling multiple robot clients. Each client sends CBOR-encoded
Latent messages; server feeds them into SLW Transformer together with imagined
tokens, then returns Action messages.

⚠️  This is a minimal reference implementation intended for simulation/PoC. It
is *not* hardened for production.
"""
from __future__ import annotations

import asyncio
import struct
from collections import deque
from typing import Deque, Dict, List, Tuple

import torch

from communication.edge_protocol import decode, encode, LatentMsg, ActionMsg
if TYPE_CHECKING:
    from models.gia_agent import GiaAgent
from models.slw_transformer import SLWTransformer

_TOKEN_REAL = 0
_TOKEN_IMAG = 1


class RobotState:
    """Stores recent latent states for a single robot."""
    def __init__(self, buffer_size: int = 32):
        self.z_buffer: Deque[torch.Tensor] = deque(maxlen=buffer_size)
        self.type_buffer: Deque[int] = deque(maxlen=buffer_size)
        self.dt_buffer: Deque[float] = deque(maxlen=buffer_size)
        self.last_tick: int = -1


class StreamingRunner:
    def __init__(self, model: GiaAgent, host: str = "0.0.0.0", port: int = 8765):
        self.model = model.eval()
        self.host = host
        self.port = port
        self.slw = SLWTransformer().to(next(model.parameters()).device)
        self.robot_states: Dict[str, RobotState] = {}
        self.loop = asyncio.get_event_loop()
        self.server = None

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        print(f"[Runner] Connection from {peer}")
        robot_id = None
        client_task = None
        try:
            client_task = asyncio.create_task(self._client_loop(reader, writer))
            await client_task
        except asyncio.CancelledError:
            print("[Runner] Client loop cancelled.")
        except ConnectionResetError:
            print(f"[Runner] Connection reset by peer {peer}")
        finally:
            if client_task:
                client_task.cancel()
            if robot_id and robot_id in self.robot_states:
                del self.robot_states[robot_id]
                print(f"[Runner] Cleaned up state for disconnected robot: {robot_id}")
            writer.close()
            await writer.wait_closed()
            print(f"[Runner] Closed connection for {peer}")

    async def _client_loop(self, reader, writer):
        while True:
            hdr = await reader.readexactly(4)
            length = struct.unpack(">I", hdr)[0]
            payload = await reader.readexactly(length)
            msg = decode(payload)
            
            if msg.robot_id is None:
                msg.robot_id = robot_id
                self.robot_states[msg.robot_id] = RobotState()

            if isinstance(msg, LatentMsg):
                # Don't await here, run policy in background
                asyncio.create_task(self._process_latent(msg, writer))
            else:
                pass # Ignore other message types

    async def _process_latent(self, msg: LatentMsg, writer):
        state = self.robot_states.setdefault(msg.robot_id, RobotState())
        device = next(self.model.parameters()).device
        z_t = torch.tensor(msg.z, dtype=torch.float32, device=device)
        
        # Add new REAL token to buffer
        state.z_buffer.append(z_t)
        state.type_buffer.append(_TOKEN_REAL)
        state.dt_buffer.append(msg.delta_tau)

        # --- Immediate action from latest context ---
        await self._compute_and_send(state, msg.robot_id, writer, msg.tick)

    async def _compute_and_send(self, state: RobotState, robot_id: str, writer, current_tick: int):
        seq = torch.stack(list(state.z_buffer)).unsqueeze(0)
        token_type = torch.tensor(list(state.type_buffer), device=seq.device).unsqueeze(0)
        dt = torch.tensor(list(state.dt_buffer), device=seq.device).unsqueeze(0).unsqueeze(-1)

        # Get SLW context
        context = self.slw(seq, token_type, delta_tau=dt)
        fused = context[:, -1]

        # Use GIA to get next action
        # This is a simplification; a full implementation would use plan_in_dream
        action_dist = self.model.action_tower.get_action_dist(fused)
        action_vec = action_dist.rsample().squeeze(0).detach().cpu().tolist()

        act_msg = ActionMsg(robot_id=robot_id, tick=current_tick, action=action_vec)
        data = encode(act_msg)
        try:
            writer.write(struct.pack(">I", len(data)) + data)
            await writer.drain()
        except ConnectionError:
            print(f"[Runner] Failed to send action to {robot_id}, connection likely closed.")

    async def _shutdown(self):
        print("[Runner] shutting down server...")
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)
        self.loop.stop()

    def start(self):
        server_coro = asyncio.start_server(self._handle_client, self.host, self.port)
        self.server = self.loop.run_until_complete(server_coro)
        print(f"[Runner] Serving on {self.server.sockets[0].getsockname()}")
        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.loop.run_until_complete(self._shutdown())
            print("[Runner] Server has been shut down.") 