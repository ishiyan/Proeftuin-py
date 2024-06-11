from typing import Any

import numpy as np
import gymnasium as gym

from .observer import Observer
from .renderer import Renderer
from .trader import Trader

class Environment(gym.Env):

    def __init__(self, observer: Observer, trader: Trader, renderer: Renderer):

        self._observer = observer
        self._trader = trader
        self._renderer = renderer

        # spaces
        self.action_space = self._trader.action_space
        self.observation_space = self._observer.observation_space

        # episode
        self._info = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.array, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))

        self._observer.reset()
        observation_window, ohlcv_window, _ =  self._observer.observe(self)
        #print('reset (-2)', ohlcv_window[-2])
        #print('reset (-1)', ohlcv_window[-1])

        info = self._trader.reset(ohlcv_window)
        self._info = info

        self._renderer.reset()
        self._renderer.render_frame(info)

        return observation_window, info

    def step(self, action):
        observation_window, ohlcv_window, truncated =  self._observer.observe(self)
        #print('step (-2)', ohlcv_window[-2])
        #print('step (-1)', ohlcv_window[-1])

        step_reward, info = self._trader.act(action, ohlcv_window, truncated)
        self._info = info
        self._renderer.render_frame(info)
        return observation_window, step_reward, False, truncated, info

    def render(self, mode='human'):
        self._renderer.render_all(self._info)

    def render_all(self, title=None):
        self._renderer.render_all(self._info, title=title)

    def close(self):
        self._renderer.close()

    def save_rendering(self, filepath):
        self._renderer.save_rendering(filepath)

    def pause_rendering(self):
        self._renderer.pause_rendering()
