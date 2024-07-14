from collections import OrderedDict
from typing import List, Optional
from numbers import Real
import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..accounts.account import Account
from ..providers.provider import Provider
from ..aggregators.trade_aggregator import TradeAggregator
from ..frame import Frame 
from .renderer import Renderer

# https://matplotlib.org/stable/gallery/color/named_colors.html
GREENS_DARK = {
    'up': 'limegreen', 'down': 'tab:blue',
    'buy': 'mediumslateblue', 'buy_many': 'mediumpurple',
    'sell': 'mediumslateblue', 'sell_many': 'mediumpurple',
    'line': 'tab:blue', 'fig': '#202020', 'ax': '#303030', 'text': '#666666',
    'title': '#d0d0d0', 'alert': 'tab:red', 'start': 'dimgrey'}
GREENS_LIGHT = {
    'up': 'limegreen', 'down': 'tab:blue',
    'buy': 'mediumpurple', 'buy_many': 'mediumorchid',
    'sell': 'mediumpurple', 'sell_many': 'mediumorchid',
    'line': 'tab:blue', 'fig': 'white', 'ax': 'white', 'text': 'black',
    'title': 'black', 'alert': 'tab:red', 'start': 'green'}
BROWNY_DARK = {
    'up': 'tab:olive', 'down': 'tab:brown',
    'buy': 'mediumpurple', 'buy_many': 'mediumorchid',
    'sell': 'mediumpurple', 'sell_many': 'mediumorchid',
    'line': 'tab:brown', 'fig': '#202020', 'ax': '#303030', 'text': '#666666',
    'title': '#d0d0d0', 'alert': 'tab:red', 'start': 'green'}
BROWNY_LIGHT = {
    'up': 'tab:olive', 'down': 'tab:brown',
    'buy': 'mediumpurple', 'buy_many': 'mediumorchid',
    'sell': 'mediumpurple', 'sell_many': 'mediumorchid',
    'line': 'tab:brown', 'fig': 'white', 'ax': 'white', 'text': 'black',
    'title': 'black', 'alert': 'tab:red', 'start': 'green'}

# Linear interpolation for the figure widh
MIN_WIDTH, MAX_WIDTH = 12.8, 19.2 #6.4, 12.8 + 3.2 + 3.2/2=1.6
MIN_STEPS, MAX_STEPS = 128, 312
STEP_DELTA = (MAX_WIDTH - MIN_WIDTH) / (MAX_STEPS - MIN_STEPS)
FIXED_HEIGHT = 6.4 #4.8
def determine_figsize(episode_max_steps):
    width = MIN_WIDTH + STEP_DELTA * (episode_max_steps - MIN_STEPS)
    # Constrain width to be within [MIN_WIDTH, MAX_WIDTH]
    width = max(MIN_WIDTH, min(width, MAX_WIDTH))
    return (width, FIXED_HEIGHT)

SCROLL = 0
EXPAND = 1
SHRINK = 2

class MatplotlibPriceRenderer(Renderer):
    """
    Renders the trading environment in 'rgb_array' render mode.
    """

    def __init__(self,
        vec_env_index: Optional[int] = None,
        dpi: int=120,
        price_candlesticks: bool=True,
        history_behavior: str='scroll',
        color_theme:str = 'GREENS_DARK'):
        """
        Args:
            vec_env_index (int, optional):
                Index of the vectorized environment.
                Default: None.
            dpi (int, optional):
                DPI to render the figure.
                Default: 120.
            price_candlesticks (bool, optional):
                True: draw price candlesticks.
                False: draw price bars.
                Default: True.
            history_behavior (str, optional):
                'scroll':
                    Historical white space is preserved on the left side.

                    On new step, the history scrolls to the left.

                    The x-axis labels are fixed.
                'expand':
                    Future white space is preserved on the right side.

                    On new step, the history expands to the right.

                    The x-axis labels are fixed.
                'shrink':
                    On new step, the history shrinks to fit the new step into the window.

                    The x-axis labels are changing every step.
                Default: 'scroll'.
            color_theme (str, optional):
                'GREENS_DARK':
                    Dark greens color scheme.
                'GREENS_LIGHT':
                    Light greens color scheme.
                'BROWNY_DARK':
                    Dark browny color scheme.
                'BROWNY_LIGHT':
                    Light browny color scheme.
                Default: 'GREENS_DARK'.
        """
        self.vec_env_index = vec_env_index
        self.dpi = dpi
        self.price_candlesticks = price_candlesticks

        if history_behavior == 'scroll':
            self.price_history_behavior = SCROLL
            self.line_history_behavior = SCROLL
        elif history_behavior == 'expand':
            self.price_history_behavior = EXPAND
            self.line_history_behavior = EXPAND
        elif history_behavior == 'shrink':
            self.price_history_behavior = SHRINK
            self.line_history_behavior = SHRINK
        else:
            raise ValueError(f'History behavior {history_behavior}' \
                "must be one of: {'scroll', 'expand', 'shrink'}") 

        if color_theme == 'GREENS_DARK':
             self.colors = GREENS_DARK
        elif color_theme == 'GREENS_LIGHT':
            self.colors = GREENS_LIGHT
        elif color_theme == 'BROWNY_DARK':
            self.colors = BROWNY_DARK
        elif color_theme == 'BROWNY_LIGHT':
            self.colors = BROWNY_LIGHT
        else:
            raise ValueError(f'Color theme {color_theme}' \
                "must be one of: {'GREENS_DARK', 'GREENS_LIGHT', 'BROWNY_DARK', 'BROWNY_LIGHT'}")

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
        self.risk_free_rate = None

        self.figure = None
        self.axes = None

        self.df_lines_columns = ['ror', 'twr', 'sharpe', 'sortino', 'calmar', 'reward', 'total reward', 'max dd pct', 'roi']
        self.df_price_columns = ['start', 'open', 'high', 'low', 'close', 'buy', 'sell']
        self.df_price = None
        self.df_lines = None
        self.price_window_delta = None
        if self.price_history_behavior in [SCROLL, EXPAND, SHRINK]:
            self.row_price_nan = {
            'start': False,
            'open': np.nan,
            'high': np.nan,
            'low': np.nan,
            'close': np.nan,
            'buy': 0,
            'sell': 0,
            }
            assert set(self.row_price_nan.keys()) == set(self.df_price_columns)
            assert len(self.row_price_nan) == len(self.df_price_columns)

        if self.line_history_behavior in [SCROLL, EXPAND]:
            self.row_lines_zero = {
                'ror': 0,
                'twr': 0,
                'sharpe': 0,
                'sortino': 0,
                'calmar': 0,
                'reward': 0,
                'total reward': 0,
                'max dd pct': 0,
                'roi': 0,
            }
            assert set(self.row_lines_zero.keys()) == set(self.df_lines_columns)
            assert len(self.row_lines_zero) == len(self.df_lines_columns)
        elif self.line_history_behavior == SHRINK:
            self.row_lines_nan = {
                'ror': np.nan,
                'twr': np.nan,
                'sharpe': np.nan,
                'sortino': np.nan,
                'calmar': np.nan,
                'reward': np.nan,
                'total reward': np.nan,
                'max dd pct': np.nan,
                'roi': np.nan,
            }
            assert set(self.row_lines_nan.keys()) == set(self.df_lines_columns)
            assert len(self.row_lines_nan) == len(self.df_lines_columns)

    def reset(self, episode_number: int, episode_max_steps: Optional[int],
            account: Account, provider: Provider, aggregator: TradeAggregator,
            frames: List[Frame], observation: OrderedDict):
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
        self.risk_free_rate = f'(r-f rate {self.account.report.risk_free_rate * 100:.0f}%)'

        if self.figure is not None:
            plt.close(self.figure)            
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

        self.figure.set_facecolor(self.colors['fig'])
        for ax in self.axes:
            ax.set_facecolor(self.colors['ax'])
            ax.tick_params(labelsize='small', colors=self.colors['text'])
            ax.grid(color=self.colors['text'])

        self._append_reset(frames)

    def step(self, frames: List[Frame], reward: Real, observation: OrderedDict):
        self.current_step += 1
        self._append_step(frames[-1], reward)

    def render(self):
        if self.price_candlesticks:
            self._plot_candlesticks(self.df_price, self.axes[0]) # ax1
        else:
            self._plot_bars(self.df_price, self.axes[0]) # ax1

        row = self.df_lines.iloc[-1]
        reward = f'reward {row["reward"]:.2f}' \
            if not row["reward"] is None else 'reward'
        total_reward = f'total reward {self.total_reward:.2f}' \
            if not self.total_reward is None else 'total reward'
        s = f'sharpe ratio {self.risk_free_rate}'
        sharpe = f'{s}  {row["sharpe"]:.2f}' \
            if not np.isnan(row["sharpe"]) else s
        s = f'sortino ratio {self.risk_free_rate}'
        sortino = f'{s}  {row["sortino"]:.2f}' \
            if not np.isnan(row["sortino"]) else s
        s = f'calmar ratio {self.risk_free_rate}'
        calmar = f'{s}  {row["calmar"]:.2f}' \
            if not np.isnan(row["calmar"]) else s
        twr = f'time weighted return {row["twr"]:.2f}%' \
            if not row["twr"] is None else 'twr ratio'
        ror = f'rate of return {row["ror"]:.2f}' \
            if not np.isnan(row["ror"]) else 'rate of return'
        roi = f'return on investment {row["roi"]:.2f}' \
            if not np.isnan(row["roi"]) else 'return on investment'
        maxdd = f'max drawdown {row["max dd pct"]:.2f}' \
            if not np.isnan(row["max dd pct"]) else 'max drawdown'

        self._plot_column_line(self.df_lines, self.axes[1], 'reward', reward) # ax2_1
        self._plot_column_line(self.df_lines, self.axes[2], 'total reward', total_reward) # ax3_1
        self._plot_column_line(self.df_lines, self.axes[3], 'sharpe', sharpe) # ax2_2
        self._plot_column_line(self.df_lines, self.axes[4], 'sortino', sortino) # ax3_2
        self._plot_column_line(self.df_lines, self.axes[5], 'twr', twr) # ax2_23
        self._plot_column_line(self.df_lines, self.axes[6], 'ror', ror) # ax2_4
        #self._plot_column_line(self.df_lines, self.axes[7], 'roi', roi) # ax3_4
        #self._plot_column_line(self.df_lines, self.axes[7], 'max dd pct', maxdd) # ax3_4
        self._plot_column_line(self.df_lines, self.axes[7], 'calmar', calmar) # ax3_4

        return self._figure_to_rgb_array(self.figure)
        
    def close(self):
        if self.figure is not None:
            plt.close(self.figure)

    def _append_reset(self, frames: List[Frame]):
        frame = frames[-1]
        # Uncoment one or another to test the behavior
        #frames = frames[64:-1]
        #self.episode_max_steps -= 64
        if self.price_history_behavior == SCROLL:
            self.df_price = pd.DataFrame({col: [np.nan] * self.episode_max_steps \
                                        for col in self.df_price_columns})
            wf = len(frames)
            if wf < self.episode_max_steps:
                w = wf
                wdelta = self.episode_max_steps - wf
                self.df_price.loc[:, ['start']] = False
                self.df_price.loc[:, ['buy', 'sell']] = 0
                row_price = self.row_price_nan.copy()
                row_price['open'] = frame.open
                row_price['high'] = frame.open
                row_price['low'] = frame.open
                row_price['close'] = frame.open
                self.df_price.iloc[0] = row_price
            else:
                w = self.episode_max_steps
                wdelta = 0
            self.price_window_delta = wdelta
            for i in range(w):
                frame = frames[i-w]
                row_price = self.row_price_nan.copy()
                row_price['open'] = frame.open
                row_price['high'] = frame.high
                row_price['low'] = frame.low
                row_price['close'] = frame.close
                if i == w-1:
                    row_price['start'] = True
                self.df_price.iloc[i + wdelta] = row_price
        elif self.price_history_behavior == EXPAND:
            self.df_price = pd.DataFrame({col: [np.nan] * self.episode_max_steps \
                                        for col in self.df_price_columns})
            self.df_price.loc[:, ['start']] = False
            self.df_price.loc[:, ['buy', 'sell']] = 0
            row_price = self.row_price_nan.copy()
            row_price['open'] = frame.open
            row_price['high'] = frame.high
            row_price['low'] = frame.low
            row_price['close'] = frame.close
            self.df_price.iloc[0] = row_price
            # Set the last row prices to the same (opening) price,
            # so it will be in exicting price range but will be invisible.
            # Otherwise mathplotlib will not render all right rows with nan prices.
            row_price['high'] = frame.open
            row_price['low'] = frame.open
            row_price['close'] = frame.open
            self.df_price.iloc[-1] = row_price
        elif self.price_history_behavior == SHRINK:
            self.df_price = pd.DataFrame(columns = self.df_price_columns)
            row_price = self.row_price_nan.copy()
            row_price['open'] = frame.open
            row_price['high'] = frame.high
            row_price['low'] = frame.low
            row_price['close'] = frame.close
            self.df_price.loc[len(self.df_price)] = row_price

        if self.line_history_behavior == SCROLL:
            self.df_lines = pd.DataFrame({col: [np.nan] * self.episode_max_steps \
                                        for col in self.df_lines_columns})
            # Set the first row values to zero, otherwise mathplotlib
            # will not render all left rows with nan values.
            self.df_lines.iloc[0] = self.row_lines_zero
        elif self.line_history_behavior == EXPAND:
            self.df_lines = pd.DataFrame({col: [np.nan] * self.episode_max_steps \
                                        for col in self.df_lines_columns})
            # Set the last row values to zero, otherwise mathplotlib
            # will not render all right rows with nan values.
            self.df_lines.iloc[-1] = self.row_lines_zero
        elif self.line_history_behavior ==  SHRINK:
            self.df_lines = pd.DataFrame(columns = self.df_lines_columns)
            self.df_lines.loc[len(self.df_lines)] = self.row_lines_nan

    def _append_step(self, frame: Frame, reward: Real):
        position = self.account.position.quantity_signed
        position_change = position - self.current_position
        self.current_position = position

        row_price = self.row_price_nan.copy()
        row_price['open'] = frame.open
        row_price['high'] = frame.high
        row_price['low'] = frame.low
        row_price['close'] = frame.close
        row_price['buy'] = position_change if position_change > 0 else 0
        row_price['sell'] = -position_change if position_change < 0 else 0

        if self.price_history_behavior == SCROLL:
            self.df_price = self.df_price.shift(-1)
            self.price_window_delta -= 1
            if self.price_window_delta > 0:
                row_pr = self.row_price_nan.copy()
                row_pr['open'] = frame.open
                row_pr['high'] = frame.open
                row_pr['low'] = frame.open
                row_pr['close'] = frame.open
                self.df_price.iloc[0] = row_pr
            self.df_price.iloc[-1] = row_price
        elif self.price_history_behavior == EXPAND:
            self.df_price.iloc[self.current_step] = row_price
        elif self.price_history_behavior == SHRINK:
            self.df_price.loc[len(self.df_price)] = row_price

        self.total_reward += reward
        balance = self.account.balance
        balance_return = balance / self.current_balance - 1.0
        self.current_balance = balance
        twr = self.current_twr * (balance_return + 1.0)
        self.current_twr = twr
        sr = self.account.report.sharpe_ratio
        if sr is None:
            sr = np.nan
        elif sr > 10 or sr < -10:
            sr = np.nan
        roi = self.account.report.returns_on_investments
        if roi is None or len(roi) == 0:
            roi = np.nan
        else:
            roi = roi[-1]
        perf = self.account.position.roundtrip_performance
        #rt_np_pnl_pct = perf.net_profit_pnl_percentage
        #rt_avg_net_wl_pct = perf.average_net_winning_loosing_percentage
        #roi2 = self.account.max_drawdown
        #roi2 = self.account.position.roi
        #roi2 = self.account.report.net_profit
        roi2 = self.account.report.roi_mean

        row_lines = {
            'ror': self.account.report.ror \
                if self.account.report.ror is not None else np.nan,
            'twr': (twr - 1) * 100,
            'sharpe': sr,
            'sortino': self.account.report.sortino_ratio \
                if self.account.report.sortino_ratio is not None else np.nan,
            'calmar': self.account.report.calmar_ratio \
                if self.account.report.calmar_ratio is not None else np.nan,
            'reward': reward,
            'total reward': self.total_reward \
                if self.total_reward is not None else np.nan,
            'max dd pct': self.account.report.max_drawdown_percent \
                if self.account.report.max_drawdown_percent is not None else np.nan,
            'roi': roi2,#rt_avg_net_wl_pct,#rt_np_pnl_pct,#roi,
            }
        assert set(row_lines.keys()) == set(self.df_lines_columns)
        assert len(row_lines) == len(self.df_lines_columns)

        if self.line_history_behavior == SCROLL:
            self.df_lines = self.df_lines.shift(-1)
            self.df_lines.iloc[-1] = row_lines
            # Set the first row values to zero, otherwise mathplotlib
            # will not render all left rows with nan values.
            self.df_lines.iloc[0] = self.row_lines_zero
        elif self.line_history_behavior == EXPAND:
            self.df_lines.iloc[self.current_step] = row_lines
        elif self.line_history_behavior == SHRINK:
            self.df_lines.loc[len(self.df_lines)] = row_lines

    def _plot_title(self, ax):
        row = self.df_lines.iloc[-1]
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
        buy_many_color=self.colors['buy_many']
        sell_many_color=self.colors['sell_many']
        el_width=.8
        marker_size = (self.figure.dpi * ax.get_position().width * self.figure.get_figwidth() * el_width / len(df))**2
        marker_size /= 2
        df_sell = df[df['sell'] > 0]
        sell_colors = np.where(df_sell['sell'] <= 1, sell_color, sell_many_color)
        ax.scatter(df_sell.index, df_sell['high'] + 1.5, s=marker_size, marker=7, # 'v' 7
            color=sell_colors)
        df_buy = df[df['buy'] > 0]
        buy_colors = np.where(df_buy['buy'] <= 1, buy_color, buy_many_color)
        ax.scatter(df_buy.index, df_buy['low'] - 1.5, s=marker_size, marker=6, # '^' 6
            color=buy_colors)
        if set_legend:
            buy_line = matplotlib.lines.Line2D([], [], color=buy_color,
                marker='^', markersize=5, linestyle='None', label='buy')
            buy_many_line = matplotlib.lines.Line2D([], [], color=buy_many_color,
                marker='^', markersize=5, linestyle='None', label='buy many')
            sell_line = matplotlib.lines.Line2D([], [], color=sell_color,
                marker='v', markersize=5, linestyle='None', label='sell')
            sell_many_line = matplotlib.lines.Line2D([], [], color=sell_many_color,
                marker='v', markersize=5, linestyle='None', label='sell many')
            ax.legend(handles=[buy_line, buy_many_line, sell_line, sell_many_line],
                fontsize='small', loc='best')

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
        start_indices = df[df['start']].index
        for idx in start_indices:
            ax.axvline(x=idx, color=self.colors['start'], linewidth=1)

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

        # Draw vertical lines where 'start' is True
        start_indices = df[df['start']].index
        for idx in start_indices:
            ax.axvline(x=idx, color=self.colors['start'], linewidth=1)

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