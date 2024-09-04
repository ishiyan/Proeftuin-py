import os
from glob import glob
import json

import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers

# The following allows to save plots in SVG format.
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

name = 'ppo_01_xxx'
dir = f'./sb3/ppo_01/'
#dir = f'./sb3/{name}/'

ROLLING_WINDOW = 1000
CHART_PAGES = 1
Y_LIMIT = 1500
Y_LIMIT_MEAN = 30
Y_LIMIT_MEDIAN = 10

DPI = 120
WID = 12 # inches
HGT = 8 # inches
DARK_MODE = True

# https://matplotlib.org/stable/gallery/color/named_colors.html
COLOR_SCATTER = 'tab:blue'
COLOR_MEAN='orange'
COLOR_ZERO='red'
COLOR_FIG = 'white'
COLOR_AX = 'white'
COLOR_TXT = '#333333'
COLOR_TIT = 'black'
if DARK_MODE:
    COLOR_SCATTER = 'tab:blue'
    COLOR_MEAN='limegreen'
    COLOR_ZERO='red'
    COLOR_FIG = '#202020'
    COLOR_AX = '#303030'
    COLOR_TXT = '#a0a0a0' #'#666666'
    COLOR_TIT = '#d0d0d0'

TEXT_SCATTER = 'Raw rewards'
TEXT_MEAN = f'Rolling mean ({ROLLING_WINDOW} episodes)'
TEXT_MEDIAN = f'Rolling median ({ROLLING_WINDOW} episodes)'

def draw_reward_charts(dir: str, png: bool=True, svg: bool=True):
    glob_pattern = 'iter*.monitor.csv'
    files = glob(os.path.join(dir, glob_pattern))
    if len(files) == 0:
        raise ValueError(f'No files of the form "{glob_pattern}" found in {dir}')

    data_frames, headers = [], []
    for file_name in files:
        #print('reading:', file_name)
        with open(file_name) as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == '#'
            header = json.loads(first_line[1:])
            df = pd.read_csv(file_handler, index_col=None)
            headers.append(header)
            df['t'] += header['t_start']
        data_frames.append(df)
    df = pd.concat(data_frames)
    df.sort_values('t', inplace=True)
    df.reset_index(inplace=True)
    df['t'] -= min(header['t_start'] for header in headers)
    df['rolling_mean'] = df['r'].rolling(window=ROLLING_WINDOW).mean()
    df['rolling_median'] = df['r'].rolling(window=ROLLING_WINDOW).median()

    num_samples = len(df) // CHART_PAGES

    start = 0
    end = num_samples
    for page in range(1, CHART_PAGES+1):
        print(f'page {page}: {start} to {end}')
        df1 = df[start:end]
        fig = plt.figure(dpi = DPI, layout='constrained', figsize=(WID, HGT))
        gs = fig.add_gridspec(3, 1)
        axes = []
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.tick_params(labelbottom=False)
        axes.append(ax1)
        ax2 = fig.add_subplot(gs[1, 0], sharex = axes[0])
        axes.append(ax2)
        ax3 = fig.add_subplot(gs[2, 0], sharex = axes[0])
        axes.append(ax3)
        for ax in axes:
            ax.set_ylabel('Episode rewards', color=COLOR_TXT)
            ax.tick_params(labelsize='small', colors=COLOR_TXT)
        axes[-1].set_xlabel('Episodes', color=COLOR_TXT)
        axes[0].tick_params(labelbottom=False)
        axes[1].tick_params(labelbottom=False)

        fig.set_facecolor(COLOR_FIG)
        for i, ax in enumerate(axes):
            ax.set_facecolor(COLOR_AX)
            ax.scatter(df1.index, df1['r'], label=TEXT_SCATTER, s=0.01, color=COLOR_SCATTER)
            if i == 2:
                ax.set_title(TEXT_MEDIAN, fontsize='small', loc='left', color=COLOR_TIT)
                ax.plot(df1['rolling_median'] , label=TEXT_MEDIAN, color=COLOR_MEAN, linewidth=.5)
                ax.set_ylim(-Y_LIMIT_MEDIAN, Y_LIMIT_MEDIAN) 
            elif i == 1:
                ax.set_title(TEXT_MEAN, fontsize='small', loc='left', color=COLOR_TIT)
                ax.plot(df1['rolling_mean'], label=TEXT_MEAN, color=COLOR_MEAN, linewidth=.5)
                ax.set_ylim(-Y_LIMIT_MEAN, Y_LIMIT_MEAN) 
            elif i == 0:
                ax.set_title(name, fontsize='small', loc='left', color=COLOR_TIT)
                ax.set_ylim(-Y_LIMIT, Y_LIMIT) 
            ax.axhline(y=0.0, color=COLOR_ZERO, linewidth=.5)
        fname = dir+name+f'.rewards.{"dark" if DARK_MODE else "light"}'
        if CHART_PAGES > 1:
            fname = f'{fname}.page-{page}'
        print(f'filename {fname}, png={png}, svg={svg}')
        if png:
            plt.savefig(f'{fname}.png', format='png')
        if svg:
            plt.savefig(f'{fname}.svg', format='svg')
        plt.close(fig)
        start += num_samples
        end += num_samples

draw_reward_charts(dir)

