# Trading environment

This package implements [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
environment.

Its internal machinery is based on a stream of trades, which may be real ones
or simulated from aggregated price bars (*candles*, *klines*). This idea was
inspired by [Intraday](https://github.com/diovisgood/intraday) package.

## Converting price bars into the stream of trades

If our data source is a sequence of aggregated price bars (candlesticks, klines),
we have to convert them into a stream of simulated trades.

Since we don't have information about the movement of the price inside the pice bar,
we have to make a guess.

Per price bar, we randomly choose one of the two basic variants: `zig-zag 1` or `zig-zag 2`.

```text
zig-zag 1 (open -> high -> low -> close):

     high
     /\
open   \  close
        \/
        low

zig-zag 2 (open -> low -> high -> close):

          high
         /\
open \  /  \ close
      \/
      low
```

In the simplest case this will produce four trades, corresponding to the `open`,
`high`, `low` and `close` prices. We can generate more trades by linearly
interpolating these four price movement segments by introducing the `spread`.

Using the `spread`, we can calculate the price increment `step`.
The number of trades per segment will be determined by this price increment `step`.

```python
step = spread * (high + low) / 2
```

To create an *open -> high* sequence in *zig-zag 1*, we start from `open`
and end before `high`, incrementing by `step` each time.
For example, if `open` is 1, `high` is 5, and `step` is 1, this would
create the sequence *[1, 2, 3, 4]*.

We can experiment with different values of the `spread` to see how many trades we will generate.

```python
from environment import BarToTradeConverter

spread = 0.1
p, o = BarToTradeConverter._zigzag_open_high_low_close(
                    100.2, 100.4, 100.1, 100.3, spread)
for i in range(len(p)):
    print(i, p[i], o[i])
```

| spread | # trades |
| ------ | -------- |
| 0.4 | 4 |
| 0.3 | 5 |
| 0.2 | 6 |
| 0.1 | 11 |
| 0.05 | 18 |
| 0.01 | 73 |
| 0.005 | 144 |
| 0.001 | 703 |

Some type of price aggregation bars, for instance klines, have fields like
`volume`, `money`, `buy_volume`, `buy_money`. We can use them to
preserve the `VWAP` of the kline.

A volume weighted average price (`VWAP`) is defined for a sequence
of trades as follows.

```text
VWAP = (price1 * volume1 + price2 * volume2 + ... ) /
                            (volume1 + volume2 + ...)
```

Each trade has a buyer and a seller.
So called *buy trade* is a trade initiated by the buyer.

The extra fields of a kline allow us to compute separately
the `VWAP` of the *buy trades* and the `VWAP` of the *sell trades*.

```python
VWAP_buy = buy_money / buy_volume
VWAP_sell = (money - buy_money) / (volume - buy_volume)
```

Knowing them, we can adjust the volumes of the simulated
individual trades so the overall `VWAP` calculated on the
sequence of trades will match the `VWAP` of the original kline.

## Aggregating stream of trades into frames

Usually we aggregate by `time` interval, e.g. 5 minutes, which
produces *equi-time* frames.
This is not the only option. We can aggregate trades by:

- `volume`, producing *equi-volume* frames,
- `ticks`, producing *equi-tick* frames,
- `money`, producing *equi-money* frames.

In all these cases, the output frames will be slightly different.
You can explore the charts and tables below to see the difference.

The code used to produce these reseults is below.

```python
from datetime import date, datetime, timezone
from typing import List, Tuple

import pandas as pd
from matplotlib import pyplot as plt

FIG_EXT = 'svg' # 'png' or 'svg'

if FIG_EXT == 'svg':
    # The following allows to save plots in SVG format.
    import matplotlib_inline
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

from env_my_intraday import Trade, Provider, BinanceMonthlyTradesProvider
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
dir = 'D:/data/binance_monthly_trades/'
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
```

### Equi-time

![Equi-time chart](./readme/ETHUSDT@Binance(t)%20time@60.svg)

| Time start | Time end | Ticks | Open | High | Low | Close | Volume | Money |
| ---------- | -------- | ----- | ---- | ---- | --- | ----- | ------ | ----- |
| 2024-05-01 00:00:00 | 2024-05-01 00:01:00 | 804 | 3014.04 | 3016.83 | 3013.31 | 3016.82 | 361.80 | 1090668.35 |
| 2024-05-01 00:01:00 | 2024-05-01 00:02:00 | 539 | 3016.83 | 3019.69 | 3015.84 | 3017.72 | 128.95 | 389092.73 |
| 2024-05-01 00:02:00 | 2024-05-01 00:03:00 | 770 | 3017.72 | 3017.72 | 3012.19 | 3012.19 | 328.99 | 991904.22 |
| 2024-05-01 00:03:00 | 2024-05-01 00:04:00 | 472 | 3012.19 | 3014.00 | 3012.00 | 3013.99 | 81.52 | 245612.95 |
| 2024-05-01 00:04:00 | 2024-05-01 00:05:00 | 376 | 3013.99 | 3015.41 | 3013.24 | 3013.39 | 115.26 | 347408.82 |
| 2024-05-01 00:05:00 | 2024-05-01 00:06:00 | 521 | 3013.39 | 3016.83 | 3013.01 | 3016.41 | 196.93 | 593847.83 |
| 2024-05-01 00:06:00 | 2024-05-01 00:07:00 | 970 | 3016.41 | 3017.33 | 3014.65 | 3015.19 | 486.87 | 1468407.25 |

### Equi-tick

![Equi-tick chart](./readme/ETHUSDT@Binance(t)%20tick@600.svg)

| Time start | Time end | Ticks | Open | High | Low | Close | Volume | Money |
| ---------- | -------- | ----- | ---- | ---- | --- | ----- | ------ | ----- |
| 2024-05-01 00:00:00 | 2024-05-01 00:00:45 | 600 | 3014.04 | 3015.00 | 3013.31 | 3015.00 | 249.40 | 751687.13 |
| 2024-05-01 00:00:45 | 2024-05-01 00:01:35 | 600 | 3015.00 | 3019.69 | 3015.00 | 3017.71 | 195.96 | 591121.84 |
| 2024-05-01 00:01:35 | 2024-05-01 00:02:37 | 600 | 3017.70 | 3018.01 | 3013.63 | 3013.64 | 241.19 | 727576.01 |
| 2024-05-01 00:02:37 | 2024-05-01 00:03:19 | 600 | 3013.64 | 3013.80 | 3012.00 | 3013.05 | 190.63 | 574354.19 |
| 2024-05-01 00:03:19 | 2024-05-01 00:05:02 | 600 | 3013.10 | 3015.41 | 3013.00 | 3013.24 | 142.69 | 430064.09 |
| 2024-05-01 00:05:02 | 2024-05-01 00:06:06 | 600 | 3013.24 | 3016.83 | 3013.01 | 3015.55 | 257.79 | 777424.97 |
| 2024-05-01 00:06:06 | 2024-05-01 00:06:31 | 600 | 3015.50 | 3017.33 | 3014.65 | 3016.10 | 262.01 | 790198.70 |

### Equi-volume

![Equi-volume chart](./readme/ETHUSDT@Binance(t)%20volume@100.svg)

| Time start | Time end | Ticks | Open | High | Low | Close | Volume | Money |
| ---------- | -------- | ----- | ---- | ---- | --- | ----- | ------ | ----- |
| 2024-05-01 00:00:00 | 2024-05-01 00:00:21 | 301 | 3014.04 | 3014.79 | 3013.31 | 3014.22 | 101.25 | 305145.95 |
| 2024-05-01 00:00:21 | 2024-05-01 00:00:42 | 217 | 3014.21 | 3014.60 | 3013.58 | 3013.64 | 101.20 | 305008.34 |
| 2024-05-01 00:00:42 | 2024-05-01 00:00:51 | 164 | 3013.64 | 3015.99 | 3013.64 | 3015.51 | 100.76 | 303771.73 |
| 2024-05-01 00:00:51 | 2024-05-01 00:01:05 | 263 | 3015.51 | 3017.35 | 3015.51 | 3017.35 | 100.31 | 302542.98 |
| 2024-05-01 00:01:05 | 2024-05-01 00:02:02 | 454 | 3017.40 | 3019.69 | 3017.21 | 3017.21 | 100.72 | 303975.74 |
| 2024-05-01 00:02:02 | 2024-05-01 00:02:06 | 140 | 3017.21 | 3017.21 | 3016.11 | 3016.23 | 100.69 | 303748.05 |
| 2024-05-01 00:02:06 | 2024-05-01 00:02:40 | 309 | 3016.21 | 3016.68 | 3013.05 | 3013.06 | 104.58 | 315322.51 |

### Equi-money

![Equi-money chart](./readme/ETHUSDT@Binance(t)%20money@600000.svg)

| Time start | Time end | Ticks | Open | High | Low | Close | Volume | Money |
| ---------- | -------- | ----- | ---- | ---- | --- | ----- | ------ | ----- |
| 2024-05-01 00:00:00 | 2024-05-01 00:00:42 | 512 | 3014.04 | 3014.79 | 3013.31 | 3013.59 | 199.18 | 600321.69 |
| 2024-05-01 00:00:42 | 2024-05-01 00:01:04 | 386 | 3013.59 | 3016.83 | 3013.59 | 3016.18 | 199.01 | 600091.59 |
| 2024-05-01 00:01:04 | 2024-05-01 00:02:06 | 629 | 3016.18 | 3019.69 | 3016.11 | 3016.53 | 198.90 | 600145.57 |
| 2024-05-01 00:02:06 | 2024-05-01 00:02:45 | 480 | 3016.49 | 3016.68 | 3012.40 | 3012.40 | 199.07 | 600061.72 |
| 2024-05-01 00:02:45 | 2024-05-01 00:04:57 | 914 | 3012.40 | 3015.41 | 3012.00 | 3013.53 | 199.52 | 601233.10 |
| 2024-05-01 00:04:57 | 2024-05-01 00:05:52 | 493 | 3013.53 | 3016.71 | 3013.01 | 3016.71 | 200.80 | 605440.33 |
| 2024-05-01 00:05:52 | 2024-05-01 00:06:31 | 687 | 3016.75 | 3017.33 | 3014.65 | 3015.36 | 199.13 | 600630.51 |

## Feature engineering

Rinforcement Learning agents operate on the current `state` of environment, sometimes called `observation`.

We cannot use raw prices and volume in the `state` because different episodes may have different price and volume ranges.

Ideally, we want that `features` of the `state` are normalized to `[-1,1]` or `[0,1]` to speedup learning.

Below we inspect how we can encode, scale, and normalize our `features`.

## Price

Raw price values are not good for machine learning models, as they don't satisfy `i.i.d.` criteria.

We can encode all four `open`, `high`, `low` and `close` prices as returns from the previous `close` price.
![Encoding price as returns from the previous close](./readme/ETHUSDT@Binance(k)%20time@21600%20price%20encoder%20ret.svg)

The correlation with original prices is shown below.

![Correlation heatmap](./readme/ETHUSDT@Binance(k)%20time@21600%20price%20encoder%20correlation%20ret+p.svg)

![Correlation heatmap](./readme/ETHUSDT@Binance(k)%20time@21600%20price%20encoder%20correlation%20ret.svg)

### Time

Machine Learning models can't efficiently utilize raw timestamp values.

Given a timestamp they can hardly say:

- is it summer or winter?
- is it monday or friday?
- is it morning or noon?

We can convert timestamp into some floating numbers to make it easier
for models to use it.

- `day of year` in [0,1], where 0 is *January 1st*, 1 is *December 31st*
- `day of week` in [0,1], where 0 is *Monday*, 1 is *Sunday*
- `time of day` in [0,1], where 0 is *00:00:00*, 1 is *23:59:59*

Below is the chart showing encoded time.
![Time encoding chart](./readme/ETHUSDT@Binance(k)%20time@21600%20time%20encoder.svg)

You can explore the encoded values in the table below.

| time start | time end | yday time start | wday time start | tday time start | yday time end | wday time end | tday time end |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-03-05 00:00:00 | 2024-03-05 06:00:00 | 0.17 | 0.29 | 0.00 | 0.17 | 0.29 | 0.25 |
| 2024-03-05 06:00:00 | 2024-03-05 12:00:00 | 0.17 | 0.29 | 0.25 | 0.17 | 0.29 | 0.50 |
| 2024-03-05 12:00:00 | 2024-03-05 18:00:00 | 0.17 | 0.29 | 0.50 | 0.17 | 0.29 | 0.75 |
| 2024-03-05 18:00:00 | 2024-03-06 00:00:00 | 0.17 | 0.29 | 0.75 | 0.18 | 0.43 | 0.00 |
| 2024-03-06 00:00:00 | 2024-03-06 06:00:00 | 0.18 | 0.43 | 0.00 | 0.18 | 0.43 | 0.25 |
| 2024-03-06 06:00:00 | 2024-03-06 12:00:00 | 0.18 | 0.43 | 0.25 | 0.18 | 0.43 | 0.50 |
| 2024-03-06 12:00:00 | 2024-03-06 18:00:00 | 0.18 | 0.43 | 0.50 | 0.18 | 0.43 | 0.75 |
| 2024-03-06 18:00:00 | 2024-03-07 00:00:00 | 0.18 | 0.43 | 0.75 | 0.18 | 0.57 | 0.00 |
| 2024-03-07 00:00:00 | 2024-03-07 06:00:00 | 0.18 | 0.57 | 0.00 | 0.18 | 0.57 | 0.25 |
| 2024-03-07 06:00:00 | 2024-03-07 12:00:00 | 0.18 | 0.57 | 0.25 | 0.18 | 0.57 | 0.50 |
| 2024-03-07 12:00:00 | 2024-03-07 18:00:00 | 0.18 | 0.57 | 0.50 | 0.18 | 0.57 | 0.75 |
| 2024-03-07 18:00:00 | 2024-03-08 00:00:00 | 0.18 | 0.57 | 0.75 | 0.18 | 0.71 | 0.00 |
| 2024-03-08 00:00:00 | 2024-03-08 06:00:00 | 0.18 | 0.71 | 0.00 | 0.18 | 0.71 | 0.25 |
| 2024-03-08 06:00:00 | 2024-03-08 12:00:00 | 0.18 | 0.71 | 0.25 | 0.18 | 0.71 | 0.50 |
| 2024-03-08 12:00:00 | 2024-03-08 18:00:00 | 0.18 | 0.71 | 0.50 | 0.18 | 0.71 | 0.75 |
| 2024-03-08 18:00:00 | 2024-03-09 00:00:00 | 0.18 | 0.71 | 0.75 | 0.19 | 0.86 | 0.00 |
| 2024-03-09 00:00:00 | 2024-03-09 06:00:00 | 0.19 | 0.86 | 0.00 | 0.19 | 0.86 | 0.25 |
| 2024-03-09 06:00:00 | 2024-03-09 12:00:00 | 0.19 | 0.86 | 0.25 | 0.19 | 0.86 | 0.50 |
| 2024-03-09 12:00:00 | 2024-03-09 18:00:00 | 0.19 | 0.86 | 0.50 | 0.19 | 0.86 | 0.75 |
| 2024-03-09 18:00:00 | 2024-03-10 00:00:00 | 0.19 | 0.86 | 0.75 | 0.19 | 1.00 | 0.00 |
| 2024-03-10 00:00:00 | 2024-03-10 06:00:00 | 0.19 | 1.00 | 0.00 | 0.19 | 1.00 | 0.25 |
| 2024-03-10 06:00:00 | 2024-03-10 12:00:00 | 0.19 | 1.00 | 0.25 | 0.19 | 1.00 | 0.50 |
| 2024-03-10 12:00:00 | 2024-03-10 18:00:00 | 0.19 | 1.00 | 0.50 | 0.19 | 1.00 | 0.75 |
| 2024-03-10 18:00:00 | 2024-03-11 00:00:00 | 0.19 | 1.00 | 0.75 | 0.19 | 0.14 | 0.00 |
| 2024-03-11 00:00:00 | 2024-03-11 06:00:00 | 0.19 | 0.14 | 0.00 | 0.19 | 0.14 | 0.25 |
| 2024-03-11 06:00:00 | 2024-03-11 12:00:00 | 0.19 | 0.14 | 0.25 | 0.19 | 0.14 | 0.50 |
| 2024-03-11 12:00:00 | 2024-03-11 18:00:00 | 0.19 | 0.14 | 0.50 | 0.19 | 0.14 | 0.75 |
| 2024-03-11 18:00:00 | 2024-03-12 00:00:00 | 0.19 | 0.14 | 0.75 | 0.19 | 0.29 | 0.00 |
| 2024-03-12 00:00:00 | 2024-03-12 06:00:00 | 0.19 | 0.29 | 0.00 | 0.19 | 0.29 | 0.25 |
| 2024-03-12 06:00:00 | 2024-03-12 12:00:00 | 0.19 | 0.29 | 0.25 | 0.19 | 0.29 | 0.50 |
| 2024-03-12 12:00:00 | 2024-03-12 18:00:00 | 0.19 | 0.29 | 0.50 | 0.19 | 0.29 | 0.75 |

The correlation heatmap looks like this.
![Correlation heatmap](./readme/ETHUSDT@Binance(k)%20time@21600%20time%20encoder%20correlation.svg)

## Data sources

### Binance

The source for [Binance](https://www.binance.com/en) data is the [Binance Public Data](https://github.com/binance/binance-public-data/) project. The data can be downloaded
from the [Binance market data](https://data.binance.vision/) website.
The website has data for `spot`, `options` and `futures`, but only
`spot` symbols will be downloaded automatically.

| class | description |
| ----- | ----------- |
| BinanceMonthlyTradesProvider | Stream of trades from Binance monthly trade archives. |
| BinanceMonthlyKlines1mToTradesProvider | Simulates a stream of trades from the Binance 1-minute kline monthly archives. |

#### Trades per month

The `BinanceMonthlyTradesProvider` downloads monthly archives of spot trades from

`https://data.binance.vision/data/spot/monthly/trades/{symbol}/{symbol}-trades-{year:04}-{month:02}.zip`.

For instance, for symbol `ETHUSDT` and month `2024-05`, the download URL is

[https://data.binance.vision/data/spot/monthly/trades/ETHUSDT/ETHUSDT-trades-2024-05.zip](https://data.binance.vision/data/spot/monthly/trades/ETHUSDT/ETHUSDT-trades-2024-05.zip)

You can explore the full list of spot symbols [here](https://data.binance.vision/?prefix=data/spot/monthly/trades/). To specify this data provider, use the example code below.

```python
from datetime import date
from environment import BinanceMonthlyTradesProvider

provider = BinanceMonthlyTradesProvider(
    data_dir = 'data/binance_monthly_trades/',
    symbol = 'BTCUSDT',
    date_from = date(2024, 1, 1),
    date_to = date(2024, 5, 31))
```

The logic to get data per month is:

1. If required monthly `.feather` file (`BTCUSDT-trades-2024-01.feather`)
   is in the `data_dir`, load it. Do not go to the next steps.
2. If required monthly `.zip` file (`BTCUSDT-trades-2024-01.zip`) is in the
   `data_dir`, convert it to `.feather` file format for faster loading and
   delete the original `.zip` file. Go to step 1.
3. If required monthly `.zip` file is not in the `data_dir`,
   download it and go to step 2.

If needed, you can pre-download `.zip` files and convert them using the code below.
This code will not delete downloaded `.zip` files after conversion.

```python
from environment import BinanceMonthlyTradesProvider

def download_and_convert_monthly_trades(symbol, year, month):
    data_dir = 'data/binance_monthly_trades'
    file = f'{data_dir}/{symbol}-trades-{year:04}-{month:02}.zip'
    BinanceMonthlyTradesProvider.download_month_archive(symbol, year, month, file, 'spot')
    # comment the following line if you don't want to convert
    BinanceMonthlyTradesProvider.convert_month_archive(file)

symbol = 'BTCUSDT'

for year in range(2021, 2026):
    for month in range(1, 13):
        download_and_convert_monthly_trades(symbol, year, month)
```

#### Klines per month

The `BinanceMonthlyKlines1mToTradesProvider` downloads monthly archives of
1-minute spot klines from

`https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{year:04}-{month:02}.zip`.

For instance, for symbol `ETHUSDT` and month `2024-05`, the download URL is

[https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1m/ETHUSDT-1m-2024-05.zip](https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1m/ETHUSDT-1m-2024-05.zip)

You can explore the full list of spot symbols [here](https://data.binance.vision/?prefix=data/spot/monthly/klines/). To specify this data provider, use the example code below.

```python
from datetime import date
from environment import BinanceMonthlyKlines1mToTradesProvider

provider = BinanceMonthlyKlines1mToTradesProvider(
    data_dir = 'data/binance_monthly_klines/',
    symbol = 'BTCUSDT',
    date_from = date(2024, 1, 1),
    date_to = date(2024, 5, 31))
```

The logic to get data per month is:

1. If required monthly `.feather` file (`BTCUSDT-1m-2024-01.feather`)
   is in the `data_dir`, load it. Do not go to the next steps.
2. If required monthly `.zip` file (`BTCUSDT-1m-2024-01.zip`) is in the
   `data_dir`, convert it to `.feather` file format for faster loading and
   delete the original `.zip` file. Go to step 1.
3. If required monthly `.zip` file is not in the `data_dir`,
   download it and go to step 2.

If needed, you can pre-download `.zip` files and convert them using the code below.
This code will not delete downloaded `.zip` files after conversion.

```python
from environment import BinanceMonthlyKlines1mToTradesProvider

def download_and_convert_monthly_klines(symbol, year, month):
    data_dir = 'data/binance_monthly_klines'
    file = f'{data_dir}/{symbol}-1m-{year:04}-{month:02}.zip'
    BinanceMonthlyKlines1mToTradesProvider.download_month_archive(symbol, year, month, file, 'spot')
    # comment the following line if you don't want to convert
    BinanceMonthlyKlines1mToTradesProvider.convert_month_archive(file)

symbol = 'BTCUSDT'

for year in range(2021, 2026):
    for month in range(1, 13):
        download_and_convert_monthly_klines(symbol, year, month)
```

## CSV files

### Synthetic

| class | description |
| ----- | ----------- |
| SineTradesProvider | Generates a fake stream of trades to move price in a sinusoidal form. |

#### Sinusoid

Generates a fake stream of trades to move price in a sinusoidal form.
The following parameters define the sinusoid.

| class | description |
| ----- | ----------- |
| mean | The mean value of the sinusoid. |
| amplitude | The smplitude of the sinusoid. |
| SNRdb | The signal to noise ratio in decibels. The less this value - the more noise is added to the sinusoid. |
| period| The period of a sinusoid in seconds, or a range specified by two values, to take a random period at each episode. |
| frequency | The frequency of a sinusoid in Hertz, or a range specified by two values, to take a random frequency at each episode. |
| date_from | The starting date for simulated trades. |
| date_to | The ending date for simulated trades. |

You must specify either `frequency` or `period`.
