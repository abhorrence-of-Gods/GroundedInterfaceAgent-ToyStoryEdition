from __future__ import annotations

"""Non-stationary reward environment wrapper.

Periodically switches the underlying reward mapping every *period* steps to
simulate non-stationary human preferences or task objectives.

Usage
-----
>>> base_env = SomeGymEnv()
>>> env = NonStationaryRewardEnv(base_env, period=1000, reward_modes=[
...     lambda r, info: r,                       # mode 0: identity
...     lambda r, info: -r,                      # mode 1: negate
...     lambda r, info: info.get('success', 0),  # mode 2: shaped by success flag
... ])
"""

import random
from typing import Callable, List

import gymnasium as gym

RewardFn = Callable[[float, dict], float]

__all__ = ["NonStationaryRewardEnv"]


class NonStationaryRewardEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, period: int, reward_modes: List[RewardFn]):
        super().__init__(env)
        assert period > 0, "period must be positive"
        assert len(reward_modes) >= 2, "Need at least two reward modes"
        self.period = period
        self.reward_modes = reward_modes
        self._step_count = 0
        self._mode_idx = 0

    def reset(self, **kwargs):  # type: ignore[override]
        self._step_count = 0
        self._mode_idx = 0
        return self.env.reset(**kwargs)

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        # apply non-stationary reward mapping
        reward_fn = self.reward_modes[self._mode_idx]
        reward = float(reward_fn(reward, info))

        # update counters
        self._step_count += 1
        if self._step_count % self.period == 0:
            self._mode_idx = (self._mode_idx + 1) % len(self.reward_modes)
            info["reward_mode"] = self._mode_idx
        return obs, reward, terminated, truncated, info 