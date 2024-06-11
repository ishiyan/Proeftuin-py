from time import time
import numpy as np

import matplotlib.pyplot as plt

from .any1_trader import Positions
from .renderer import Renderer

class Any1Renderer(Renderer):
    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self._frame_price_history = None
        self._frame_position_history = None
        self._frame_index = None

    def reset(self):
        self._frame_price_history = []
        self._frame_position_history = []
        self._frame_index = -1
        plt.figure(figsize=(12, 6))

    def render_frame(self, info: dict, mode='human'):
        price = info['current_price']
        position = info['position']
        self._frame_price_history.append(price)
        self._frame_position_history.append(position)
        self._frame_index += 1

        if self.render_mode != 'human':
            return

        def _plot_position(index, position, price):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(index, price, color=color)

        start_time = time()

        if self._frame_index == 0:
            plt.cla()
        plt.plot(self._frame_price_history)
        _plot_position(self._frame_index, position, price)

        plt.suptitle(
            "Total Reward: %.6f" % info['total_reward'] + ' ~ ' +
            "Total Profit: %.6f" % info['total_profit']
        )

        end_time = time()
        process_time = end_time - start_time

        pause_time = (1 / self.metadata['render_fps']) - process_time
        assert pause_time > 0., "High FPS! Try to reduce the 'render_fps' value."
        plt.pause(pause_time)

    def render_all(self, info: dict, title=None):
        window_ticks = np.arange(len(self._frame_price_history))

        short_ticks = []
        long_ticks = []
        short_pos = []
        long_pos = []
        for i, tick in enumerate(window_ticks):
            if self._frame_position_history[i] == Positions.Short:
                short_ticks.append(tick)
                short_pos.append(self._frame_price_history[tick])
            elif self._frame_position_history[i] == Positions.Long:
                long_ticks.append(tick)
                long_pos.append(self._frame_price_history[tick])

        plt.plot(self._frame_price_history)
        plt.plot(short_ticks, short_pos, 'ro')
        plt.plot(long_ticks, long_pos, 'go')

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % info['total_reward'] + ' ~ ' +
            "Total Profit: %.6f" % info['total_profit']
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    