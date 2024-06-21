from collections import OrderedDict
from datetime import date, datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt

FIG_EXT = 'svg' # 'png' or 'svg'

if FIG_EXT == 'svg':
    # The following allows to save plots in SVG format.
    import matplotlib_inline
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

from env_my_intraday.actions.no_action import NoAction
from env_my_intraday.environment import Environment
from env_my_intraday.features import Copy
from env_my_intraday.features import Feature
from env_my_intraday.features import TimeEncoder
from env_my_intraday.features import PriceEncoder
from env_my_intraday.features import WindowScaler
from env_my_intraday.rewards.constant_reward import ConstantReward
from env_my_intraday.providers import Trade, Provider, BinanceMonthlyTradesProvider, BinanceMonthlyKlines1mToTradesProvider
from env_my_intraday import TradeAggregator, IntervalTradeAggregator
from env_my_intraday import Frame

# https://matplotlib.org/stable/gallery/color/named_colors.html
D_UP = 'limegreen'
D_DN = 'tab:blue'
D_LINE = 'tab:blue'
D_FIG = '#202020'
D_AX = '#303030'
D_TXT = '#666666'
D_TIT = '#d0d0d0'
KDE_LINE = 'limegreen'
HIST_BAR = 'tab:blue'

DO_COPY = False
DO_TIME_ENCODER = False
DO_RAW_PRICE = False
DO_PRICE_ENCODER = False
DO_WINDOW_SCALER_RAW = False
DO_WINDOW_SCALER_MINMAX = True

def plot_features(df, title, 
        panes: List[List[str]]=[],
        dark=True, show_legend=False, figsize=(8, 4)):
    fig = plt.figure(dpi=120, layout='constrained', figsize=figsize)
    if dark:
        fig.set_facecolor(D_FIG)
    ax_cnt = len(panes)
    ax_panes = []
    gs = fig.add_gridspec(3+ax_cnt, 1)
    ax = fig.add_subplot(gs[:3, 0])
    for i in range(ax_cnt):
        ax_panes.append(fig.add_subplot(gs[3+i, 0], sharex = ax))
    for a in [ax] + ax_panes:
        if dark:
            a.set_facecolor(D_AX)
            a.grid(color=D_TXT)
            a.tick_params(labelbottom=False, labelsize='small', colors=D_TXT)
        else:
            a.tick_params(labelbottom=False, labelsize='small')
        a.grid(False)
        a.tick_params(labelbottom=False)
    if ax_cnt > 0:
        ax_panes[-1].tick_params(labelbottom=True)
    else:
        ax.tick_params(labelbottom=True)
    if dark:
        ax.set_title(title, color=D_TIT)
    else:
        ax.set_title(title)

    wick_width=.2
    body_width=.8
    up_color=D_UP
    down_color=D_DN

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
    # Plot panes
    for i, pane in enumerate(panes):
        for column in pane:
            ax_panes[i].plot(df.index, df[column], label=column)#, color='tab:blue')
        if show_legend:
            legend = ax_panes[i].legend(loc='best', fontsize='small', )
            legend.get_frame().set_alpha(0.1)
            if dark:
                legend.get_frame().set_facecolor(D_FIG)
                for text in legend.get_texts():
                    text.set_color(color=D_TIT)
        else:
            ax_panes[i].legend().set_visible(False)
            tit = ' / '.join(pane)
            if dark:
                ax_panes[i].set_title(tit, fontsize='small', color=D_TIT)
            else:
                ax_panes[i].set_title(tit, fontsize='small')
    return fig

def plot_correllation_heatmap(df, title=None, cmap=None, coeff=False, coeff_color=None,
                                                        dark=True, feature_decimals=2):
    fig, ax = plt.subplots(dpi=120, layout='constrained')
    if dark:
        fig.set_facecolor(D_FIG)
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    if cmap is None:
        cmap = 'rainbow_r' # rainbow_r Spectral coolwarm_r bwr_r RdYlBu RdYlGn
    cax = ax.imshow(df, cmap=cmap, interpolation='nearest', vmin=-1, vmax=1)
    cb = fig.colorbar(cax)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    if dark:
        cb.set_label('Spearman correlation', fontsize='small', color=D_TIT)
        cb.ax.tick_params(labelsize='small', colors=D_TXT)
    else:
        cb.set_label('Spearman correlation', fontsize='small')
        cb.ax.tick_params(labelsize='small')
    if title is not None:
        if dark:
            ax.set_title(title, color=D_TIT)
        else:
            ax.set_title(title, color=D_TIT)
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.columns)))
    if dark:
        ax.tick_params(axis='x', colors=D_TXT)
        ax.set_xticklabels(df.columns, rotation=45, fontsize='small', color=D_TIT)
        ax.tick_params(axis='y', colors=D_TXT)
        ax.set_yticklabels(df.columns, fontsize='small', color=D_TIT)
    else:
        ax.set_xticklabels(df.columns, rotation=45, fontsize='small')
        ax.set_yticklabels(df.columns, fontsize='small')
    if coeff:
        if coeff_color is None:
            coeff_color = 'black'
        for i in range(len(df.columns)):
            for j in range(len(df.columns)):
                ax.text(j, i, f'{df.iloc[i, j]:.{feature_decimals}f}', ha='center', va='center',
                        fontsize='small', color=coeff_color)
    return fig

def plot_distribution_histogram(df, columns, bins='auto', dark=True,
                                show_legend=True, figsize=(4.8, 3.6)):
    """
        Don't plot multiple columns on the same histogram,
        because colors are ugly an plot is unreadable.
    """
    # bins: integer or 'auto', 'scott', 'rice', 'sturges', 'sqrt'    
    fig, ax = plt.subplots(dpi=120, layout='constrained', figsize=figsize)
    if dark:
        fig.set_facecolor(D_FIG)
        ax.set_facecolor(D_AX)
        ax.tick_params(labelbottom=True, labelsize='small', colors=D_TXT)
        ax.grid(color=D_TXT)
        ax.set_ylabel('probability density', fontsize='small', color=D_TXT)
    else:
        ax.tick_params(labelbottom=True, labelsize='small')
        ax.set_ylabel('probability density', fontsize='small')

    for column in columns:
        # Remove NaN values for kernel density estimate (KDE) calculation
        data = df[column].dropna()
        n, bins_, patches_ = ax.hist(data, bins=bins, density=True, color=HIST_BAR,
                edgecolor=D_AX if dark else 'white', label=column)
        # Calculate and plot KDE
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 1000)
        kde_values = kde(x_range)
        # Scale KDE to match histogram
        max_hist_y_value = max(n)
        max_kde_y_value = max(kde_values)
        scale_factor = max_hist_y_value / max_kde_y_value
        ax.plot(x_range, kde_values * scale_factor, color=KDE_LINE,
                label=column + ' KDE')
    ax.grid(False)
    if show_legend:
        legend = ax.legend(loc='best', fontsize='small')
        legend.get_frame().set_alpha(0.1)
        if dark:
            legend.get_frame().set_facecolor(D_AX)
            legend.get_frame().set_edgecolor(D_FIG)
            for text in legend.get_texts():
                text.set_color(color=D_TIT)
    else:
        ax.legend().set_visible(False)
        tit = 'probability density of ' + ' / '.join(columns) + ' (bins: ' + str(bins) + ')'
        if dark:
            ax.set_title(tit, fontsize='small', color=D_TIT)
        else:
            ax.set_title(tit, fontsize='small')
    return fig

def df_from_frames_and_observations(frames, observations):
    df = pd.DataFrame([f.__dict__ for f in frames])
    for key in observations[0].keys():
        df[key] = [o[key] for o in observations]
    #df.set_index('time_start', inplace=True)
    return df

def get_frames_and_observations(
        provider: Provider,
        aggregator:TradeAggregator,
        episode_steps: int,
        lookback_steps: int,
        features: List[Feature],
        datetime_cutoff: datetime
        ) -> Tuple[List[Frame], List[OrderedDict]]:
    frames: List[Frame] = []
    observations: List[OrderedDict] = []
    #lookback_steps = max(lookback_steps, max([f.period for f in features if f.period is not None]))
    lookback_steps = max(lookback_steps, max([f.period for f in features if f.period is not None], default=0))
    def process_frame(
            frame: Frame,
            frames: List[Frame], 
            observations: List[OrderedDict],
            features: List[Feature],
            lookback_steps: int):
        frames.append(frame)
        observation = OrderedDict()
        for feature in features:
            feature.process(frames, observation)
        observations.append(observation)
        if len(frames) > lookback_steps + 1:
            del frames[0]
            del observations[0]

    provider.reset(seek='first')
    aggregator.reset()
    frame_count = episode_steps + lookback_steps
    while True:
        if frame_count < 0:
            break
        try:
            trade = next(provider)
            if trade.datetime > datetime_cutoff:
                raise StopIteration
            frame = aggregator.aggregate([trade])
            if frame is not None:
                frame_count -= 1
                process_frame(frame, frames, observations,
                              features, lookback_steps)
        except StopIteration:
            frame = aggregator.finish()
            if frame is not None:
                process_frame(frame, frames, observations,
                              features, lookback_steps)
            break
    return frames, observations

def print_frames_and_observations(
        frames: List[Frame],
        observations = List[OrderedDict],
        filename: str = None,
        feature_decimals: int = 2):
    lines: List[str] = []
    header = '| time start | time end | open | high | low | close | volume'
    divider = '| --- | --- | --- | --- | --- | --- | ---'
    last_obs = observations[-1]
    last_keys = last_obs.keys()
    for key in last_keys:
        header += f' | {key}'
        divider += ' | ---'
    header += ' |'
    divider += ' |'
    lines.append(header)
    lines.append(divider)

    for i, frame in enumerate(frames):
        row = f"| {frame.time_start.strftime('%Y-%m-%d %H:%M:%S')} | " \
            f"{frame.time_end.strftime('%Y-%m-%d %H:%M:%S')} | " \
            f"{frame.open:.2f} | {frame.high:.2f} | {frame.low:.2f} | " \
            f"{frame.close:.2f} | {frame.volume:.2f}"
        obs = observations[i]
        for key in last_keys:
            v = obs[key]
            if isinstance(v, float):
                row += f" | {v:.{feature_decimals}f}"
            else:
                row += f" | {v}"
            #row += f" | {obs[key]:.{feature_decimals}f}"
        row += ' |'
        lines.append(row)
    if filename is not None:
        with open(filename, 'w') as f:
            for line in lines:
                f.write(line + '\n')
    else:
        for line in lines:
            print(line)

symbol = 'ETHUSDT'
dir = 'D:/data/binance_monthly_klines/'
provider = BinanceMonthlyKlines1mToTradesProvider(data_dir = dir, symbol = symbol,
            date_from = date(2024, 3, 1), date_to = date(2024, 4, 30), spread=0.5)
datetime_cutoff = datetime(2024, 4, 30, tzinfo=timezone.utc)

aggregator_1m = IntervalTradeAggregator(method='time',
                interval=1*60, duration=(1, 8*60*60))
aggregator_6h = IntervalTradeAggregator(method='time',
                interval=6*60*60, duration=(1, 800*60*60))

name_1m = f'{provider.name} {aggregator_1m.name}'
name_6h = f'{provider.name} {aggregator_6h.name}'

episode_steps = 16
lookback_steps = 196 # how many previous frames to consider

if DO_COPY:
    features: List[Feature] = [
        Copy(source=['volume'])
    ]
    frames_6h, states_6h = get_frames_and_observations(provider, aggregator_6h,
            episode_steps, lookback_steps, features, datetime_cutoff)
    print('copy', 'frames', len(frames_6h), 'states', len(states_6h))
    print_frames_and_observations(frames_6h, states_6h, filename=name_6h+' copy.txt')

if DO_TIME_ENCODER:
    features: List[Feature] = [
        TimeEncoder(source=['time_start','time_end'], yday=True, wday=True, tday=True, write_to='state'),
    ]
    frames_6h, states_6h = get_frames_and_observations(provider, aggregator_6h,
            episode_steps, lookback_steps, features, datetime_cutoff)
    print('time encoder', 'frames', len(frames_6h), 'states', len(states_6h))
    print_frames_and_observations(frames_6h, states_6h, filename=name_6h+' time encoder.txt')
    df = df_from_frames_and_observations(frames_6h, states_6h)
    #print(df.head())
    fig = plot_features(df, name_6h+' time encoder', show_legend=False, dark=True,
        panes=[['yday_time_start'],['wday_time_start'],['tday_time_start']])
    fig.savefig('_dark/'+name_6h+' time encoder.'+FIG_EXT)
    plt.show()
    plt.close(fig)
    fig = plot_features(df, name_6h+' time encoder', show_legend=False, dark=False,
        panes=[['yday_time_start'],['wday_time_start'],['tday_time_start']])
    fig.savefig('_light/'+name_6h+' time encoder.'+FIG_EXT)
    plt.show()
    plt.close(fig)
    corr = df[['yday_time_start', 'wday_time_start', 'tday_time_start']].corr()
    fig = plot_correllation_heatmap(corr, None, coeff=True, dark=True)
    fig.savefig('_dark/'+name_6h+' time encoder correlation.'+FIG_EXT)
    plt.show()
    plt.close(fig)
    fig = plot_correllation_heatmap(corr, None, coeff=True, dark=False)
    fig.savefig('_light/'+name_6h+' time encoder correlation.'+FIG_EXT)
    plt.show()
    plt.close(fig)

if DO_RAW_PRICE:
    features: List[Feature] = [
    ]
    frames_6h, states_6h = get_frames_and_observations(provider, aggregator_6h,
            episode_steps, lookback_steps, features, datetime_cutoff)
    print('raw price', 'frames', len(frames_6h), 'states', len(states_6h))
    print_frames_and_observations(frames_6h, states_6h, filename=name_6h+' raw price.txt')
    df = df_from_frames_and_observations(frames_6h, states_6h)
    #print(df.head())
    fig = plot_features(df, name_6h, show_legend=False, dark=True,
        panes=[])
    fig.savefig('_dark/'+name_6h+' raw price.'+FIG_EXT)
    plt.show()
    plt.close(fig)
    fig = plot_features(df, name_6h, show_legend=False, dark=False,
        panes=[])
    fig.savefig('_light/'+name_6h+' raw price.'+FIG_EXT)
    plt.show()
    plt.close(fig)
    corr = df[['open', 'high', 'low', 'close']].corr()
    fig = plot_correllation_heatmap(corr, None, coeff=True, dark=True)
    fig.savefig('_dark/'+name_6h+' raw price correlation.'+FIG_EXT)
    plt.show()
    plt.close(fig)
    fig = plot_correllation_heatmap(corr, None, coeff=True, dark=False)
    fig.savefig('_light/'+name_6h+' raw price correlation.'+FIG_EXT)
    plt.show()
    plt.close(fig)

if DO_PRICE_ENCODER:
    features: List[Feature] = [
        PriceEncoder(source=['open', 'high', 'low', 'close'], method='return', period=2, base='close', write_to='state'),
        PriceEncoder(source=['open', 'high', 'low', 'close'], method='logreturn', period=2, base='close', write_to='state'),
    ]
    frames_6h, states_6h = get_frames_and_observations(provider, aggregator_6h,
            episode_steps, lookback_steps, features, datetime_cutoff)
    print('price encoder', 'frames', len(frames_6h), 'states', len(states_6h))
    print_frames_and_observations(frames_6h, states_6h, filename=name_6h+' price encoder.txt', feature_decimals=6)
    df = df_from_frames_and_observations(frames_6h, states_6h)
    #print(df.head())
    fig = plot_features(df, name_6h+' price encoder', show_legend=False, dark=True,
        panes=[
            ['return_2_open_close', 'return_2_high_close', 'return_2_low_close', 'return_2_close_close'],
            ['logreturn_2_open_close', 'logreturn_2_high_close', 'logreturn_2_low_close', 'logreturn_2_close_close'],
        ])
    fig.savefig('_dark/'+name_6h+' price encoder all.'+FIG_EXT)
    plt.show()
    plt.close(fig)
    fig = plot_features(df, name_6h+' price encoder', show_legend=False, dark=False,
        panes=[
            ['return_2_open_close', 'return_2_high_close', 'return_2_low_close', 'return_2_close_close'],
            ['logreturn_2_open_close', 'logreturn_2_high_close', 'logreturn_2_low_close', 'logreturn_2_close_close'],
        ])
    fig.savefig('_light/'+name_6h+' price encoder all.'+FIG_EXT)
    plt.show()
    plt.close(fig)
    corr = df[['open', 'high', 'low', 'close',
        'return_2_open_close', 'return_2_high_close', 'return_2_low_close', 'return_2_close_close',
        'logreturn_2_open_close', 'logreturn_2_high_close', 'logreturn_2_low_close', 'logreturn_2_close_close'
        ]].corr()
    fig = plot_correllation_heatmap(corr, None, coeff=True, dark=True)
    fig.savefig('_dark/'+name_6h+' price encoder correlation all.'+FIG_EXT)
    plt.show()
    plt.close(fig)
    fig = plot_correllation_heatmap(corr, None, coeff=True, dark=False)
    fig.savefig('_light/'+name_6h+' price encoder correlation all.'+FIG_EXT)
    plt.show()
    plt.close(fig)
    # Without log returns
    fig = plot_features(df, name_6h+' price encoder', show_legend=False, dark=True,
        panes=[
            ['return_2_open_close'], ['return_2_high_close'], ['return_2_low_close'], ['return_2_close_close'],
        ])
    fig.savefig('_dark/'+name_6h+' price encoder ret.'+FIG_EXT)
    plt.show()
    plt.close(fig)
    fig = plot_features(df, name_6h+' price encoder', show_legend=False, dark=False,
        panes=[
            ['return_2_open_close'], ['return_2_high_close'], ['return_2_low_close'], ['return_2_close_close'],
        ])
    fig.savefig('_light/'+name_6h+' price encoder ret.'+FIG_EXT)
    plt.show()
    plt.close(fig)
    corr = df[['open', 'high', 'low', 'close',
        'return_2_open_close', 'return_2_high_close', 'return_2_low_close', 'return_2_close_close',
        ]].corr()
    fig = plot_correllation_heatmap(corr, None, coeff=True, dark=True)
    fig.savefig('_dark/'+name_6h+' price encoder correlation ret+p.'+FIG_EXT)
    plt.show()
    plt.close(fig)
    fig = plot_correllation_heatmap(corr, None, coeff=True, dark=False)
    fig.savefig('_light/'+name_6h+' price encoder correlation ret+p.'+FIG_EXT)
    plt.show()
    plt.close(fig)
    corr = df[[
        'return_2_open_close', 'return_2_high_close', 'return_2_low_close', 'return_2_close_close',
        ]].corr()
    fig = plot_correllation_heatmap(corr, None, coeff=True, dark=True)
    fig.savefig('_dark/'+name_6h+' price encoder correlation ret.'+FIG_EXT)
    plt.show()
    plt.close(fig)
    fig = plot_correllation_heatmap(corr, None, coeff=True, dark=False)
    fig.savefig('_light/'+name_6h+' price encoder correlation ret.'+FIG_EXT)
    plt.show()
    plt.close(fig)

if DO_WINDOW_SCALER_RAW:
    features: List[Feature] = [
        WindowScaler(source=['open', 'high', 'low', 'close'], method='raw',
                            copy_period=2, write_to='state'),
    ]
    frames_6h, states_6h = get_frames_and_observations(provider, aggregator_6h,
            episode_steps, lookback_steps, features, datetime_cutoff)
    print('window scaler raw', 'frames', len(frames_6h), 'states', len(states_6h))
    print_frames_and_observations(frames_6h, states_6h, filename=name_6h+' window scaler raw.txt', feature_decimals=6)
    df = df_from_frames_and_observations(frames_6h, states_6h)

if DO_WINDOW_SCALER_MINMAX:
    features: List[Feature] = [
        WindowScaler(source=['open', 'high', 'low', 'close'], method='minmax',
                            scale_period=64, copy_period=1, write_to='state'),
    ]
    frames_6h, states_6h = get_frames_and_observations(provider, aggregator_6h,
            episode_steps, lookback_steps, features, datetime_cutoff)
    print('window scaler minmax', 'frames', len(frames_6h), 'states', len(states_6h))
    print_frames_and_observations(frames_6h, states_6h, filename=name_6h+' minmax scaling.txt', feature_decimals=6)
    df = df_from_frames_and_observations(frames_6h, states_6h)
    df = df.rename(columns={
        'minmax(64)_1_open': 'minmax(64) open',
        'minmax(64)_1_high': 'minmax(64) high',
        'minmax(64)_1_low': 'minmax(64) low',
        'minmax(64)_1_close': 'minmax(64) close'})

    for dark in [True, False]:
        fig = plot_features(df, name_6h+' minmax scaling', show_legend=False, dark=dark,
            panes=[['minmax(64) open'], ['minmax(64) high'], ['minmax(64) low'], ['minmax(64) close'],
        ])
        fig.savefig('_'+('dark' if dark else 'light')+'/'+name_6h+' minmax scaling.'+FIG_EXT)
        #plt.show()
        plt.close(fig)

    corr = df[['open', 'high', 'low', 'close',
            'minmax(64) open', 'minmax(64) high', 'minmax(64) low', 'minmax(64) close',
        ]].corr(method='pearson') # Pearson correlation coefficient
    for dark in [True, False]:
        fig = plot_correllation_heatmap(corr, None, coeff=True, dark=dark)
        fig.savefig('_'+('dark' if dark else 'light')+'/'+name_6h+' minmax scaling correlation with price.'+FIG_EXT)
        #plt.show()
        plt.close(fig)

    corr = df[[
            'minmax(64) open', 'minmax(64) high', 'minmax(64) low', 'minmax(64) close',
        ]].corr(method='pearson') # Pearson correlation coefficient
    for dark in [True, False]:
        fig = plot_correllation_heatmap(corr, None, coeff=True, dark=dark)
        fig.savefig('_'+('dark' if dark else 'light')+'/'+name_6h+' minmax scaling correlation.'+FIG_EXT)
        #plt.show()
        plt.close(fig)

    for column in ['open', 'high', 'low', 'close', 'minmax(64) open',
                   'minmax(64) high', 'minmax(64) low', 'minmax(64) close']:
        for dark in [True, False]:
            fig = plot_distribution_histogram(df, [column], dark=dark)
            fig.savefig('_'+('dark' if dark else 'light')+'/'+name_6h+' distr '+column+'.'+FIG_EXT)
            #plt.show()
            plt.close(fig)
