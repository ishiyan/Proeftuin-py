from datetime import date, datetime, timezone
from typing import List, Tuple

import pandas as pd
from matplotlib import pyplot as plt

FIG_EXT = 'svg' # 'png' or 'svg'

if FIG_EXT == 'svg':
    # The following allows to save plots in SVG format.
    import matplotlib_inline
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

from environment import Trade, Provider, BinanceMonthlyTradesProvider
from environment import TradeAggregator, IntervalTradeAggregator
from environment import Frame

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

def get_frames(provider: Provider, aggregator:TradeAggregator,
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

def run(provider: Provider, aggregator: TradeAggregator, number_of_frames: int,
    datetime_cutoff: datetime, rows_to_print: int = 16, dark: bool = False):
    name = f'{provider.name} {aggregator.name}'
    df, frames = get_frames(provider, aggregator, number_of_frames, datetime_cutoff)
    print(name)
    print_frames(frames, rows_to_print)
    fig = plot_candlesticks(df, name, dark=dark)
    fig.savefig(f'{name}.{FIG_EXT}')
    #plt.show()
    plt.close(fig)

symbol = 'ETHUSDT'
dir = 'data/binance_monthly_trades/'
provider = BinanceMonthlyTradesProvider(data_dir = dir, symbol = symbol,
                date_from = date(2024, 5, 1), date_to = date(2024, 5, 2))

datetime_cutoff = datetime(2024, 5, 2, tzinfo=timezone.utc)
number_of_frames = 128

aggregator = IntervalTradeAggregator(method='time',
                interval=1*60, duration=(1, 8*60*60))
run(provider, aggregator, number_of_frames, datetime_cutoff)

aggregator = IntervalTradeAggregator(method='tick',
                interval=1*600, duration=(1, 8*60*60))
run(provider, aggregator, number_of_frames, datetime_cutoff)

aggregator = IntervalTradeAggregator(method='volume',
                interval=1*100, duration=(1, 8*60*60))
run(provider, aggregator, number_of_frames, datetime_cutoff)

aggregator = IntervalTradeAggregator(method='money',
                interval=1*6000*100, duration=(1, 8*60*60))
run(provider, aggregator, number_of_frames, datetime_cutoff)
