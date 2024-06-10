from abc import ABC, abstractmethod
from time import time
from typing import Any
import numpy as np

import matplotlib.pyplot as plt

from .enums import Positions

class Renderer(ABC):
    @property
    @abstractmethod
    def render_frame(self, prices: np.array, position_history: Any, current_position: Any, start_idx: int, current_idx: int, total_profit: float, total_reward: float, mode='human'):
        raise NotImplementedError()

    @abstractmethod
    def render_all(self, prices: np.array, position_history: Any, total_profit: float, total_reward: float, title=None):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @abstractmethod
    def save_rendering(self, filepath):
        raise NotImplementedError()

    @abstractmethod
    def pause_rendering(self):
        raise NotImplementedError()

class Any1Renderer(Renderer):
    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self._first_rendering = True

    def reset(self):
        self._first_rendering = True
        plt.figure(figsize=(12, 6))

    def render_frame(self, prices: np.array, position_history: Any, current_position: Any, start_idx: int, current_idx: int, total_profit: float, total_reward: float, mode='human'):
        if self.render_mode != 'human':
            return
        def _plot_position(position, prices, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, prices[tick], color=color)

        start_time = time()

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(prices)
            _plot_position(position_history[start_idx], prices, start_idx)

        _plot_position(current_position, prices, current_idx)

        plt.suptitle(
            "Total Reward: %.6f" % total_reward + ' ~ ' +
            "Total Profit: %.6f" % total_profit
        )

        end_time = time()
        process_time = end_time - start_time

        pause_time = (1 / self.metadata['render_fps']) - process_time
        assert pause_time > 0., "High FPS! Try to reduce the 'render_fps' value."
        plt.pause(pause_time)

    def render_all(self, prices: np.array, position_history: Any, total_profit: float, total_reward: float, title=None):
        window_ticks = np.arange(len(position_history))
        plt.plot(prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, prices[short_ticks], 'ro')
        plt.plot(long_ticks, prices[long_ticks], 'go')

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % total_reward + ' ~ ' +
            "Total Profit: %.6f" % total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    