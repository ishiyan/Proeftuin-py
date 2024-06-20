from collections import OrderedDict
from datetime import date, datetime, timezone
from typing import List, Tuple

import pandas as pd
from matplotlib import pyplot as plt

FIG_EXT = 'svg' # 'png' or 'svg'

if FIG_EXT == 'svg':
    # The following allows to save plots in SVG format.
    import matplotlib_inline
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


from env_my_intraday.actions.no_action import NoAction
from env_my_intraday.environment import Environment
from env_my_intraday.features.copy import Copy
from env_my_intraday.features.feature import Feature
from env_my_intraday.features.time_encoder import TimeEncoder
from env_my_intraday.rewards.constant_reward import ConstantReward
from env_my_intraday.providers import Trade, Provider, BinanceMonthlyTradesProvider, BinanceMonthlyKlines1mToTradesProvider
from env_my_intraday import TradeAggregator, IntervalTradeAggregator
from env_my_intraday import Frame

def plot_candlesticks(df, title, dark=True, figsize=(8, 4)):
    fig = plt.figure(dpi=120, layout='constrained', figsize=figsize)
    fig.set_facecolor('#202020' if dark else 'white')
    gs = fig.add_gridspec(4, 1)
    ax = fig.add_subplot(gs[:3, 0])
    ax2 = fig.add_subplot(gs[3, 0], sharex = ax)
    for a in (ax, ax2):
        a.set_facecolor('#303030' if dark else 'white')
        a.tick_params(labelsize='small', colors='#666666' if dark else 'black')
        a.grid(color='#666666' if dark else 'black')
        a.grid(False)
    ax.tick_params(labelbottom=False)
    ax.set_title(title, color='#d0d0d0' if dark else 'black')

    wick_width=.2
    body_width=.8
    up_color='limegreen'
    down_color='tab:blue'

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
    # Plot volumw
    ax2.vlines(df.index, [0], df.volume, color = 'tab:blue')
    return fig

def print_frames(frames: List[Frame], rows: int = 16, table: bool = True):
    print("| Time start | Time end | Ticks | Open | High | Low | Close | Volume | Money |")
    print("| ---------- | -------- | ----- | ---- | ---- | --- | ----- | ------ | ----- |")
    for i, frame in enumerate(frames):
        if i >= rows:
            break
        if table:
            print(f"| {frame.time_start.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"{frame.time_end.strftime('%Y-%m-%d %H:%M:%S')} | {frame.ticks} | "
                f"{frame.open:.2f} | {frame.high:.2f} | {frame.low:.2f} | "
                f"{frame.close:.2f} | {frame.volume:.2f} | {frame.money:.2f} |")
        else:
            print(f"Time: {frame.time_start.strftime('%Y-%m-%d %H:%M:%S')} - "
                f"{frame.time_end.strftime('%Y-%m-%d %H:%M:%S')}, Ticks: {frame.ticks}, "
                f"O: {frame.open:.2f}, H: {frame.high:.2f}, L: {frame.low:.2f}, "
                f"C: {frame.close:.2f}, V: {frame.volume:.2f}, M: {frame.money:.2f}, ")

def get_frames2(provider: Provider, aggregator:TradeAggregator,
        number_of_frames: int, datetime_cutoff: datetime) -> Tuple[pd.DataFrame, List[Frame]]:    
    frames: List[Frame] = []
    df_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(columns = df_columns)
    def add_frame(frame: Frame, df: pd.DataFrame, frames: List[Frame]):
        if frame is None:
            return
        frames.append(frame)
        row = {
            'datetime': frame.time_end,
            'open': frame.open,
            'high': frame.high,
            'low': frame.low,
            'close': frame.close,
            'volume': frame.volume}
        df.loc[len(df)] = row # Only use with a RangeIndex!

    provider.reset(seek='first')
    aggregator.reset()
    while True:
        if len(df) >= number_of_frames:
            break
        try:
            trade = next(provider)
            if trade.datetime > datetime_cutoff:
                raise StopIteration
            frame = aggregator.aggregate([trade])
            add_frame(frame, df, frames)
        except StopIteration:
            frame = aggregator.finish()
            add_frame(frame, df, frames)
            break
    return df, frames

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
        rows: int = 16):
    header = '| Time end | Open | High | Low | Close | Volume '
    divider = '| --- | --- | --- | --- | --- | --- '
    last_obs = observations[-1]
    last_keys = last_obs.keys()
    for key in last_keys:
        header += f' | {key}'
        divider += ' | ---'
    header += ' |'
    divider += ' |'
    print(header)
    print(divider)

    for i, frame in enumerate(frames):
        if i > rows:
            break
        row = f"| {frame.time_end.strftime('%Y-%m-%d %H:%M:%S')} | " \
            f"{frame.open:.2f} | {frame.high:.2f} | {frame.low:.2f} | " \
            f"{frame.close:.2f} | {frame.volume:.2f} "
        obs = observations[i]
        for key in last_keys:
            row += f" | {obs[key]:.2f}"
        row += ' |'
        print(row)

def run(provider: Provider, aggregator: TradeAggregator, number_of_frames: int,
    datetime_cutoff: datetime, 
    episode_steps: int = 128, features: List[Feature] = [],
    rows_to_print: int = 16, dark: bool = False):

    symbol = 'ETHUSDT'
    dir = 'D:/data/binance_monthly_trades/'
    provider = BinanceMonthlyTradesProvider(data_dir = dir, symbol = symbol,
                date_from = date(2024, 5, 1), date_to = date(2024, 5, 2))
    aggregator = IntervalTradeAggregator(method='time',
                interval=1*60, duration=(1, 8*60*60))

    name = f'{provider.name} {aggregator.name}'
    df, frames = get_frames2(provider, aggregator, number_of_frames, datetime_cutoff)
    print(name)
    print_frames(frames, rows_to_print)
    fig = plot_candlesticks(df, name, dark=dark)
    fig.savefig(f'{name}.{FIG_EXT}')
    #plt.show()
    plt.close(fig)

def run2(provider: Provider, aggregator: TradeAggregator, number_of_frames: int,
    datetime_cutoff: datetime, rows_to_print: int = 16, dark: bool = False):
    name = f'{provider.name} {aggregator.name}'
    df, frames = get_frames2(provider, aggregator, number_of_frames, datetime_cutoff)
    print(name)
    print_frames(frames, rows_to_print)
    fig = plot_candlesticks(df, name, dark=dark)
    fig.savefig(f'{name}.{FIG_EXT}')
    #plt.show()
    plt.close(fig)

symbol = 'ETHUSDT'
dir = 'D:/data/binance_monthly_trades/'
provider = BinanceMonthlyTradesProvider(data_dir = dir, symbol = symbol,
                date_from = date(2024, 5, 1), date_to = date(2024, 5, 2))
aggregator = IntervalTradeAggregator(method='time',
                interval=1*60, duration=(1, 8*60*60))
datetime_cutoff = datetime(2024, 5, 2, tzinfo=timezone.utc)
episode_steps = 128
lookback_steps = 16 # how many previous frames to consider

dir2 = 'D:/data/binance_monthly_klines/'
provider2 = BinanceMonthlyKlines1mToTradesProvider(data_dir = dir, symbol = symbol,
                date_from = date(2024, 3, 1), date_to = date(2024, 5, 31), spread=0.1)
aggregator2 = IntervalTradeAggregator(method='time',
                interval=4*60*60, duration=(1, 800*60*60))
datetime_cutoff2 = datetime(2024, 5, 31, tzinfo=timezone.utc)
lookback_steps2 = 64 # how many previous frames to consider

"""
features: List[Feature] = [
        Copy(source=['volume'])
]
frames, states = get_frames_and_observations(provider, aggregator,
            episode_steps, lookback_steps, features, datetime_cutoff)
print('frames', len(frames), 'states', len(states))
print_frames_and_observations(frames, states, rows=16)
"""

"""
features: List[Feature] = [
        TimeEncoder(source=['time_end'], yday=True, wday=True, write_to='state'),
]
frames, states = get_frames_and_observations(provider, aggregator,
            episode_steps, lookback_steps, features, datetime_cutoff)
print('frames', len(frames), 'states', len(states))
print_frames_and_observations(frames, states, rows=16)
"""

features: List[Feature] = [
        TimeEncoder(source=['time_end'], yday=True, wday=True, write_to='state'),
]
frames, states = get_frames_and_observations(provider2, aggregator2,
            episode_steps, lookback_steps2, features, datetime_cutoff2)
print('frames', len(frames), 'states', len(states))
print_frames_and_observations(frames, states, rows=lookback_steps2)
