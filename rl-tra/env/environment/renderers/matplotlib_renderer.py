import io
from numbers import Real
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from ..accounts.account import Account
from ..providers.provider import Provider
from ..aggregators.trade_aggregator import TradeAggregator
from ..frame import Frame 
from .renderer import Renderer

GREENS_DARK = {
    'up': 'limegreen', 'down': 'tab:blue', 'buy': 'tab:gray', 'sell': 'tab:gray',
    'line': 'tab:blue', 'fig': '#202020', 'ax': '#303030', 'text': '#666666',
    'title': '#d0d0d0', 'alert': 'tab:red'}

GREENS_LIGHT = {
    'up': 'limegreen', 'down': 'tab:blue', 'buy': 'tab:gray', 'sell': 'tab:gray',
    'line': 'tab:blue', 'fig': 'white', 'ax': 'white', 'text': 'black',
    'title': 'black', 'alert': 'tab:red'}

BROWNY_DARK = {
    'up': 'tab:olive', 'down': 'tab:brown', 'buy': 'tab:gray', 'sell': 'tab:gray',
    'line': 'tab:brown', 'fig': '#202020', 'ax': '#303030', 'text': '#666666',
    'title': '#d0d0d0', 'alert': 'tab:red'}

BROWNY_LIGHT = {
    'up': 'tab:olive', 'down': 'tab:brown', 'buy': 'tab:gray', 'sell': 'tab:gray',
    'line': 'tab:brown', 'fig': 'white', 'ax': 'white', 'text': 'black',
    'title': 'black', 'alert': 'tab:red'}

# Linear interpolation for the figure widh
MIN_WIDTH, MAX_WIDTH = 12.8, 19.2 #6.4, 12.8 + 3.2 + 3.2/2=1.6
MIN_STEPS, MAX_STEPS = 128, 312
STEP_DELTA = (MAX_WIDTH - MIN_WIDTH) / (MAX_STEPS - MIN_STEPS)
FIXED_HEIGHT = 6.4 #4.8

class MatplotlibRenderer(Renderer):

    def __init__(self,
        vec_env_index: Optional[int] = None,
        dpi=120, # 96, 120
        colors = GREENS_DARK):

        self.vec_env_index = vec_env_index
        self.dpi = dpi # DPI to render the figure
        self.colors = colors

        self.account: Account = None
        self.provider: Provider = None
        self.aggregator: TradeAggregator = None
        self.episode = None
        self.episode_max_steps = None
        self.episode_number = None
        self.current_step = None
        self.current_position = None
        self.current_twr = None
        self.current_balance = None
        self.initial_balance = None
        self.total_reward = None

        self.figure = None
        self.axes = None

        self.df_columns = ['start', 'open', 'high', 'low', 'close', 'buy', 'sell', 'ror', \
            'twr', 'sharpe', 'sortino', 'calmar', 'reward', 'total reward',]
        self.df = None

    def reset(self, episode_number: int, episode_max_steps: Optional[int],
            account: Account, provider: Provider, aggregator: TradeAggregator,
            frames: List[Frame]):
        self.account = account
        self.provider = provider
        self.aggregator = aggregator

        self.episode = f'{provider.name} {aggregator.name}'
        self.episode_max_steps = episode_max_steps
        self.episode_number = episode_number
        self.current_step = 0
        self.current_position = self.account.position.quantity_signed
        self.current_twr = 1.0
        self.current_balance = self.account.initial_balance
        self.initial_balance = self.account.initial_balance
        self.total_reward = 0.0

        if self.figure is not None:
            plt.close(self.figure)

        def determine_figsize(episode_max_steps):
            width = MIN_WIDTH + STEP_DELTA * (episode_max_steps - MIN_STEPS)
            # Constrain width to be within [MIN_WIDTH, MAX_WIDTH]
            width = max(MIN_WIDTH, min(width, MAX_WIDTH))
            return (width, FIXED_HEIGHT)
            
        fig = plt.figure(dpi = self.dpi, layout='constrained',
            figsize=determine_figsize(episode_max_steps))

        gs = fig.add_gridspec(5, 4)
        ax1 = fig.add_subplot(gs[:3, :])
        ax2_1 = fig.add_subplot(gs[3, 0])
        ax2_2 = fig.add_subplot(gs[3, 1])
        ax2_23 = fig.add_subplot(gs[3:, 2])
        ax2_4 = fig.add_subplot(gs[3, 3])
        ax3_1 = fig.add_subplot(gs[4, 0], sharex = ax2_1)
        ax3_2 = fig.add_subplot(gs[4, 1], sharex = ax2_2)
        ax3_4 = fig.add_subplot(gs[4, 3], sharex = ax2_4)
        self.figure = fig
        self.axes = [ax1, ax2_1, ax3_1, ax2_2, ax3_2, ax2_23, ax2_4, ax3_4]
        for ax in self.axes:
            ax.tick_params(labelsize='small')
        for ax in [ax2_1, ax2_2, ax2_4]:
            ax.tick_params(labelbottom=False)
        # Limit y-axis labels to 2 decimals
        formatter = ticker.FormatStrFormatter('%.2f')
        for ax in self.axes:
            ax.yaxis.set_major_formatter(formatter)

        self.figure.set_facecolor(self.colors['fig'])
        for ax in self.axes:
            ax.set_facecolor(self.colors['ax'])
            ax.tick_params(labelsize='small', colors=self.colors['text'])
            ax.grid(color=self.colors['text'])

        # Pre-populate the DataFrame with zeros
        self.df = pd.DataFrame({col: [0] * self.episode_max_steps for col in self.df_columns})
        #self.df = pd.DataFrame({col: [None]*self.episode_max_steps for col in self.df_columns})
        self._append_reset(frames)

    def step(self, frames: List[Frame], reward: Real):
        #self.total_reward += reward # Already done in _append_step
        self.current_step += 1
        self._append_step(frames[-1], reward)

    def render(self):
        self._plot_candlesticks(self.df, self.axes[0]) # ax1
        #self._plot_bars(self.df, self.axes[0])

        row = self.df.iloc[-1]
        reward = f'reward {row["reward"]:.2f}'
        total_reward = f'total reward {self.total_reward:.2f}'
        sharpe = f'sharpe ratio {row["sharpe"]:.2f}'
        sortino = f'sortino ratio {row["sortino"]:.2f}'
        twr = f'time weighted return {row["twr"]:.2f}'
        ror = f'rate of return {row["sharpe"]:.2f}'
        roi = f'retirn on investment {row["sortino"]:.2f}'

        self._plot_column_line(self.df, self.axes[1], 'reward', reward) # ax2_1
        self._plot_column_line(self.df, self.axes[2], 'total reward', total_reward) # ax3_1
        self._plot_column_line(self.df, self.axes[3], 'sharpe', sharpe) # ax2_2
        self._plot_column_line(self.df, self.axes[4], 'sortino', sortino) # ax3_2
        self._plot_column_line(self.df, self.axes[5], 'twr', twr) # ax2_23
        self._plot_column_line(self.df, self.axes[6], 'sharpe', ror) # ax2_4
        self._plot_column_line(self.df, self.axes[7], 'sortino', roi) # ax3_4

        return self._figure_to_rgb_array(self.figure)
        
    def close(self):
        if self.figure is not None:
            plt.close(self.figure)

    def _append_reset(self, frames: List[Frame]):
        row_template = {
            'start': False,
            'open': 0.0,
            'high': 0.0,
            'low': 0.0,
            'close': 0.0,
            'buy': False,
            'sell': False,
            'ror': 0.0,
            'twr': 0.0,
            'sharpe': 0.0,
            'sortino': 0.0,
            'calmar': 0.0,
            'reward': 0.0,
            'total reward': 0.0,
            }
        #assert set(row.keys()) == set(self.df_columns)
        #assert len(row) == len(self.df_columns)
        w = self.episode_max_steps
        for i in range(w):
            frame = frames[i-w]
            row = row_template.copy()
            row['open'] = frame.open
            row['high'] = frame.high
            row['low'] = frame.low
            row['close'] = frame.close
            if i == w-1:
                row['start'] = True
            # Update the row at index i directly.
            self.df.iloc[i] = row

    def _append_step(self, frame: Frame, reward: Real):
        self.total_reward += reward
        position = self.account.position.quantity_signed
        position_change = position - self.current_position
        self.current_position = position
        balance = self.account.balance
        balance_return = balance / self.current_balance - 1.0
        self.current_balance = balance
        twr = self.current_twr * (balance_return + 1.0)
        self.current_twr = twr
        row = {
            'start': False,
            'open': frame.open,
            'high': frame.high,
            'low': frame.low,
            'close': frame.close,
            'buy': True if position_change > 0 else False,
            'sell': True if position_change < 0 else False,
            'ror': self.account.report.ror,
            'twr': twr - 1,
            'sharpe': self.account.report.sharpe_ratio,
            'sortino': self.account.report.sortino_ratio,
            'calmar': self.account.report.calmar_ratio,
            'reward': reward,
            'total reward': self.total_reward}
        #assert set(row.keys()) == set(self.df_columns)
        #assert len(row) == len(self.df_columns)
        self.df = self.df.shift(-1)
        self.df.iloc[-1] = row

    def _plot_title(self, ax):
        row = self.df.iloc[-1]
        sharpe = row['sharpe']
        sortino = row['sortino']
        calmar = row['calmar']
        color = self.colors['title']
        title = f'episode {self.episode_number} step {self.current_step} {self.episode.lower()}'
        if self.vec_env_index is not None:
            title = f'env {self.vec_env_index} {title}'
        if self.account.is_halted:
            title += ' ACCOUNT HALTED'
            color = self.colors['alert']
        title += '\n'
        if sharpe is not None:
            title += f'sharpe {sharpe:.2f} '
        if sortino is not None:
            title += f'sortino {sortino:.2f} '
        if calmar is not None:
            title += f'calmar {calmar:.2f} '
        ax.set_title(title, ha='left', x=0, fontsize='small', color=color)

    def _plot_buy_sell_markers(self, df, ax, set_legend=False):
        buy_color=self.colors['buy']
        sell_color=self.colors['sell']
        el_width=.8
        marker_size = (self.figure.dpi * ax.get_position().width *self.figure.get_figwidth() * el_width / len(df))**2
        marker_size /= 2
        df_sell = df[df['sell']]
        ax.scatter(df_sell.index, df_sell['high'] + .5, s=marker_size, marker=7, color=sell_color) # 'v' 7
        df_buy = df[df['buy']]
        ax.scatter(df_buy.index, df_buy['low'] - .5, s=marker_size, marker=6, color=buy_color) # '^' 6
        if set_legend:
            buy_line = matplotlib.lines.Line2D([], [], color=buy_color, marker='^', markersize=5, linestyle='None', label='buy')
            sell_line = matplotlib.lines.Line2D([], [], color=sell_color, marker='v', markersize=5, linestyle='None', label='sell')
            ax.legend(handles=[buy_line, sell_line], fontsize='small', loc='best')

    def _plot_candlesticks(self, df, ax):
        wick_width=.2
        body_width=.8
        up_color=self.colors['up']
        down_color=self.colors['down']

        ax.cla()
        ax.yaxis.grid(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

        self._plot_title(ax)

        # Draw vertical lines where 'start' is True
        start_indices = self.df[self.df['start']].index
        for idx in start_indices:
            ax.axvline(x=idx, color='green', linewidth=1)

        # Up and down price movements
        up = df[df.close >= df.open]
        down = df[df.close < df.open]
        # Plot up candlesticks
        ax.bar(up.index, up.close - up.open, body_width, bottom=up.open, color=up_color)
        ax.bar(up.index, up.high - up.close, wick_width, bottom=up.close, color=up_color)
        ax.bar(up.index, up.low - up.open, wick_width, bottom=up.open, color=up_color)
        # Plot down candlesticks
        ax.bar(down.index, down.close - down.open, body_width, bottom=down.open, color=down_color)
        ax.bar(down.index, down.high - down.open, wick_width, bottom=down.open, color=down_color)
        ax.bar(down.index, down.low - down.close, wick_width, bottom=down.close, color=down_color)
        # Plot buy and sell markers
        self._plot_buy_sell_markers(df, ax)

    def _plot_bars(self, df, ax):
        up_color=self.colors['up']
        down_color=self.colors['down']

        ax.cla()
        ax.yaxis.grid(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        self._plot_title(ax)

        min_line_width = 0.4  # Minimum line width
        max_line_width = 2.0  # Maximum line width
        max_elements = 1000  # Maximum number of elements for max line width
        num_elements = len(df)
        #line_width = max(min_line_width, max_line_width - (num_elements / max_elements) * (max_line_width - min_line_width))
        line_width = min(max_line_width, min_line_width + (num_elements / max_elements) * (max_line_width - min_line_width))

        # Up and down price movements
        up = df[df.close >= df.open]
        down = df[df.close < df.open]
        # Plot up prices
        ax.hlines(up.open, up.index, up.index+0.4, color=up_color, linewidth=line_width)
        ax.hlines(up.close, up.index, up.index+0.4, color=up_color, linewidth=line_width)
        ax.vlines(up.index+0.4, up.low, up.high, color=up_color, linewidth=line_width)
        # Plot down prices
        ax.hlines(down.open, down.index, down.index+0.4, color=down_color, linewidth=line_width)
        ax.hlines(down.close, down.index, down.index+0.4, color=down_color, linewidth=line_width)
        ax.vlines(down.index+0.4, down.low, down.high, color=down_color, linewidth=line_width)
        # Plot buy and sell markers
        self._plot_buy_sell_markers(df, ax)

    def _plot_column_line(self, df, ax, column, title):
        ax.cla()
        ax.yaxis.grid(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize='small', loc='left', color=self.colors['title'])
        ax.plot(df.index, df[column], color=self.colors['line'])

    def _figure_to_rgb_array(self, fig):
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw', dpi=self.dpi)
        io_buf.seek(0)
        rgb_array = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()
        return rgb_array