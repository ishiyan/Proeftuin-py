from collections import OrderedDict
from typing import List, Optional
from numbers import Real
import io

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

# Linear interpolation for the figure size
# 12.8/2 = 6.4/2 = 3.2/2 = 1.6/2 = 0.8/2 = 0.4/2 = 0.2
MIN_WIDTH, MAX_WIDTH = 6.4, 16.0
MIN_HEIGHT, MAX_HEIGHT = 3.2, 8.0
STEP_HEIGHT = 0.9142857
STEP_WIDTH = 0.1
def determine_figsize(observation_features: int, feature_length: int):
    height = STEP_HEIGHT * observation_features    
    #print(observation_features, '->', height)
    height = max(MIN_HEIGHT, min(height, MAX_HEIGHT))
    width = 2 * height # 1.61803398875
    #width = STEP_WIDTH * feature_length    
    #print(feature_length, '->', width)
    width = max(MIN_WIDTH, min(width, MAX_WIDTH))
    #print(width, 'x', height)
    return (width, height)

# https://matplotlib.org/stable/users/explain/colors/colormaps.html
CMAP = 'rainbow_r' # rainbow_r Spectral coolwarm_r bwr_r RdYlBu RdYlGn

class MatplotlibObservationRenderer(Renderer):
    """
    Renders step's observation in 'rgb_array' render mode.
    """

    def __init__(self,
        vec_env_index: Optional[int] = None,
        dpi: int=120,
        color_theme:str = 'GREENS_DARK'):
        """
        Args:
            vec_env_index (int, optional):
                Index of the vectorized environment.
                Default: None.
            dpi (int, optional):
                DPI to render the figure.
                Default: 120.
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

        self.figure = None
        self.axes = None
        self.ax_heatmap = None
        self.ax_heatmap_colorbar = None

        self.df_observation_corr = None
        self.df_observation = None
        self.observation_columns = None
        self.observation_rows = None

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

        self._fill_observation(observation)
        num_cols = len(self.df_observation.columns)
        num_rows = len(self.df_observation)
        self.observation_columns = num_cols
        self.observation_rows = num_rows

        if self.figure is not None:
            plt.close(self.figure)            
        fig = plt.figure(dpi = self.dpi, layout='constrained',
            figsize=determine_figsize(num_cols, num_rows))
        self.figure = fig
        self.ax_heatmap_colorbar = None

        gs = fig.add_gridspec(num_cols, 2)
        self.axes = []
        for i in range(num_cols):
            if num_cols != 1 and i == (num_cols - 1):
                ax = fig.add_subplot(gs[i, 0], sharex = self.axes[0])
            else:
                ax = fig.add_subplot(gs[i, 0])
                ax.tick_params(labelbottom=False)
            ax.tick_params(labelsize='small')
            self.axes.append(ax)

        self.ax_heatmap = fig.add_subplot(gs[:, 1])
        self.ax_heatmap.set_facecolor(self.colors['ax'])

        self.figure.set_facecolor(self.colors['fig'])
        for ax in self.axes:
            ax.set_facecolor(self.colors['ax'])
            ax.tick_params(colors=self.colors['text'])
            ax.grid(color=self.colors['text'])

    def step(self, frames: List[Frame], reward: Real, observation: OrderedDict):
        self.current_step += 1
        self._fill_observation(observation)
        assert len(self.df_observation.columns) == self.observation_columns
        assert len(self.df_observation) == self.observation_rows

    def render(self):
        self._plot_title(self.figure)
        for i, column in enumerate(self.df_observation.columns):            
            self._plot_column_line(self.df_observation, self.axes[i], column, column)
        self._plot_correllation_heatmap(self.df_observation_corr, self.ax_heatmap)
        return self._figure_to_rgb_array(self.figure)
        
    def close(self):
        if self.figure is not None:
            plt.close(self.figure)

    def _fill_observation(self, observation: OrderedDict):
        keys = list(observation.keys())
        #for key in keys:
        #    print(key, len(observation[key]))
        keys = [key for key in keys if key not in \
            ('open_180', 'close_180', 'high_180', 'low_180', 'volume_180',
            'zscore196_high_180', 'zscore180_low_196', 'zscore196_open_196',
            'yday_time_start_180', 'wday_time_start_180')]
        filtered_observation = {key: observation[key] for key in keys}
        self.df_observation = pd.DataFrame(filtered_observation)
        self.df_observation.columns = \
            [f'({i+1}) {col}' for i, col in enumerate(self.df_observation.columns)]
        self.df_observation_corr = self.df_observation.corr()

    def _plot_title(self, fig):
        color = self.colors['title']
        title = f'episode {self.episode_number} step {self.current_step} {self.episode.lower()}'
        if self.vec_env_index is not None:
            title = f'env {self.vec_env_index} {title}'
        if self.account.is_halted:
            title += ' ACCOUNT HALTED'
            color = self.colors['alert']        
        fig.suptitle(title, ha='left', x=0, fontsize='small', color=color)

    def _plot_column_line(self, df, ax: plt.Axes, column, title):
        ax.cla()
        ax.yaxis.grid(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize='small', loc='left', color=self.colors['title'])
        ax.plot(df.index, df[column], color=self.colors['line'])

    def _plot_correllation_heatmap(self, df_corr, ax: plt.Axes):
        ax.cla()
        cax = ax.imshow(df_corr, cmap=CMAP, interpolation='nearest', vmin=-1, vmax=1,
            aspect='auto')
        color_text = self.colors['text']
        color_title = self.colors['title']
        if self.ax_heatmap_colorbar is None:
            cb = plt.colorbar(cax, ax=ax)
            cb.set_ticks([-1, -0.5, 0, 0.5, 1])
            cb.ax.tick_params(labelsize='small', colors=color_title)
            self.ax_heatmap_colorbar = cb
        ax.set_title('Spearman correlation', fontsize='small', color=color_title)
        l = len(df_corr.columns)
        rl = range(l)
        ax.set_xticks(rl)
        ax.set_yticks(rl)
        ax.tick_params(axis='x', colors=color_text)
        ax.tick_params(axis='y', colors=color_text)
        r = range(1, l + 1)
        ax.set_xticklabels(r, fontsize='small', color=color_title)
        ax.set_yticklabels(r, fontsize='small', color=color_title)
        coeff_color = 'black'
        for i in rl:
            for j in rl:
                ax.text(j, i, f'{df_corr.iloc[i, j]:.2f}', ha='center',
                    va='center', fontsize='small', color=coeff_color)

    def _figure_to_rgb_array(self, fig):
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw', dpi=self.dpi)
        io_buf.seek(0)
        rgb_array = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()
        return rgb_array