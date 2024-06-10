from typing import Any

import numpy as np
import gymnasium as gym

from .enums import Actions, Positions
from .observer import Observer
from .renderer import Renderer

class ZeroEnv(gym.Env):

    def __init__(self, window_size, observer: Observer=None, renderer: Renderer=None):

        self._observer = observer
        self._renderer = renderer
        self.window_size = window_size # window size for observation lookback
        self.prices, self.signal_features = self._observer.get_data() ################
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        self.observation_space = self._observer.observation_space

        # episode
        self._start_tick = self.window_size #######################
        self._end_tick = len(self.prices) - 1 #######################
        self._truncated = None #######################
        self._current_tick = None #######################
        self._last_trade_tick = None #######################
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self.history = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.array, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))
        self._observer.reset() ################
        self._renderer.reset() ################

        self._truncated = False #######################
        self._current_tick = self._start_tick #######################
        self._last_trade_tick = self._current_tick - 1 #######################
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self.history = {}

        observation = self._get_observation()
        info = self._get_info()

        self._renderer.render_frame(self.prices, self._position_history, self._position, self._start_tick, self._current_tick, self._total_profit, self._total_reward)
        return observation, info

    def step(self, action):
        self._truncated = False #######################
        self._current_tick += 1 #######################

        if self._current_tick == self._end_tick: #######################
            self._truncated = True #######################

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick #######################

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        self._renderer.render_frame(self.prices, self._position_history, self._position, self._start_tick, self._current_tick, self._total_profit, self._total_reward)
        return observation, step_reward, False, self._truncated, info

    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position
        )

    def _get_observation(self):
        feat, pric = self._observer.observe()
        return feat

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode='human'):
        self._renderer.render_frame(self.prices, self._position_history, self._position, self._start_tick, self._current_tick, self._total_profit, self._total_reward)

    def render_all(self, title=None):
        self._renderer.render_all(self.prices, self._position_history, self._total_profit, self._total_reward)

    def close(self):
        self._renderer.close()

    def save_rendering(self, filepath):
        self._renderer.save_rendering(filepath)

    def pause_rendering(self):
        self._renderer.pause_rendering()

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError
