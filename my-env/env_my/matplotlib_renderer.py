import os
from time import time
import numpy as np

import matplotlib.pyplot as plt

from .any1_trader import Positions
from .renderer import Renderer

class MatplotlibRenderer(Renderer):
    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(self, display: bool = True, save_format: str = None,
        path: str = 'charts', filename_prefix: str = 'chart_') -> None:
        super().__init__()

        self._volume_chart_height = 0.33

        self._df = None
        self.fig = None
        self._price_ax = None
        self._volume_ax = None
        self.net_worth_ax = None
        self._show_chart = display

        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        if self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

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

    def _create_figure(self) -> None:
        self.fig = plt.figure()

        self.net_worth_ax = plt.subplot2grid((6, 1), (0, 0),
            rowspan=2, colspan=1)
        self.price_ax = plt.subplot2grid((6, 1), (2, 0), rowspan=8,
            colspan=1, sharex=self.net_worth_ax)
        self.volume_ax = self.price_ax.twinx()
        plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90,
            top=0.90, wspace=0.2, hspace=0)

    def _render_trades(self, step_range, trades) -> None:
        trades = [trade for sublist in trades.values() for trade in sublist]

        for trade in trades:
            if trade.step in range(sys.maxsize)[step_range]:
                date = self._df.index.values[trade.step]
                close = self._df['close'].values[trade.step]
                color = 'green'

                if trade.side is TradeSide.SELL:
                    color = 'red'

                self.price_ax.annotate(' ', (date, close),
                                       xytext=(date, close),
                                       size="large",
                                       arrowprops=dict(arrowstyle='simple', facecolor=color))

    def _render_volume(self, step_range, times) -> None:
        self.volume_ax.clear()

        volume = np.array(self._df['volume'].values[step_range])

        self.volume_ax.plot(times, volume,  color='blue')
        self.volume_ax.fill_between(times, volume, color='blue', alpha=0.5)

        self.volume_ax.set_ylim(0, max(volume) / self._volume_chart_height)
        self.volume_ax.yaxis.set_ticks([])

    def _render_price(self, step_range, times, current_step) -> None:
        self.price_ax.clear()

        self.price_ax.plot(times, self._df['close'].values[step_range], color="black")

        last_time = self._df.index.values[current_step]
        last_close = self._df['close'].values[current_step]
        last_high = self._df['high'].values[current_step]

        self.price_ax.annotate('{0:.2f}'.format(last_close), (last_time, last_close),
                               xytext=(last_time, last_high),
                               bbox=dict(boxstyle='round',
                                         fc='w', ec='k', lw=1),
                               color="black",
                               fontsize="small")

        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * self._volume_chart_height, ylim[1])

    def _render_net_worth(self, step_range, times, current_step, net_worths) -> None:
        self.net_worth_ax.clear()
        self.net_worth_ax.plot(times, net_worths[step_range], label='Net Worth', color="g")
        self.net_worth_ax.legend()

        legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_time = times[-1]
        last_net_worth = list(net_worths[step_range])[-1]

        self.net_worth_ax.annotate('{0:.2f}'.format(last_net_worth), (last_time, last_net_worth),
            xytext=(last_time, last_net_worth),
            bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
            color="black", fontsize="small")

        self.net_worth_ax.set_ylim(min(net_worths) / 1.25, max(net_worths) * 1.25)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:
        if price_history is None:
            raise ValueError("renderers() is missing required positional argument 'price_history'.")

        if net_worth is None:
            raise ValueError("renderers() is missing required positional argument 'net_worth'.")

        if performance is None:
            raise ValueError("renderers() is missing required positional argument 'performance'.")

        if trades is None:
            raise ValueError("renderers() is missing required positional argument 'trades'.")

        if not self.fig:
            self._create_figure()

        if self._show_chart:
            plt.show(block=False)

        current_step = step - 1

        self._df = price_history
        if max_steps:
            window_size = max_steps
        else:
            window_size = 20

        current_net_worth = round(net_worth[len(net_worth)-1], 1)
        initial_net_worth = round(net_worth[0], 1)
        profit_percent = round((current_net_worth - initial_net_worth) / initial_net_worth * 100, 2)

        self.fig.suptitle('Net worth: $' + str(current_net_worth) +
                          ' | Profit: ' + str(profit_percent) + '%')

        window_start = max(current_step - window_size, 0)
        step_range = slice(window_start, current_step)

        times = self._df.index.values[step_range]

        if len(times) > 0:
            # self._render_net_worth(step_range, times, current_step, net_worths, benchmarks)
            self._render_net_worth(step_range, times, current_step, net_worth)
            self._render_price(step_range, times, current_step)
            self._render_volume(step_range, times)
            self._render_trades(step_range, trades)

        self.price_ax.set_xticklabels(times, rotation=45, horizontalalignment='right')

        plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)
        plt.pause(0.001)
