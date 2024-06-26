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

### Equi-tick

![Equi-tick chart](./readme/ETHUSDT@Binance(t)%20tick@600.svg)

### Equi-volume

![Equi-volume chart](./readme/ETHUSDT@Binance(t)%20volume@100.svg)

### Equi-money

![Equi-money chart](./readme/ETHUSDT@Binance(t)%20money@600000.svg)

## Feature engineering

Rinforcement Learning agents operate on the current `state` of environment, sometimes called `observation`.

We cannot use raw prices and volume in the `state` because different episodes may have different price and volume ranges.

Ideally, we want that `features` of the `state` are normalized to `[-1,1]` or `[0,1]` to speedup learning.

Below we inspect how we can encode, scale, and normalize our `features`.

## Price

Raw price values are not good for machine learning models, as they don't satisfy `i.i.d.` criteria.

For instance, let's look at ETHUSDT 6h klines.

![Price chart](./readme/raw-price/ETHUSDT@Binance(k)%20time@21600%20raw%20price.svg)

The prices are strongly correlated.

![Price correlation](./readme/raw-price/ETHUSDT@Binance(k)%20time@21600%20raw%20price%20corr.svg)

The price distributions are:

| open | high | low | close |
| --- | --- | --- | --- |
| ![open](./readme/raw-price/ETHUSDT@Binance(k)%20time@21600%20raw%20price%20distr%20open.svg) | ![high](./readme/raw-price/ETHUSDT@Binance(k)%20time@21600%20raw%20price%20distr%20high.svg) | ![low](./readme/raw-price/ETHUSDT@Binance(k)%20time@21600%20raw%20price%20distr%20low.svg) | ![close](./readme/raw-price/ETHUSDT@Binance(k)%20time@21600%20raw%20price%20distr%20close.svg) |

### Using returns instead if raw prices

We can encode all four `open`, `high`, `low` and `close` prices as returns or as log-returns from the previous `close` price.

![Encoding price as returns from the previous close](./readme/price-enc/ETHUSDT@Binance(k)%20time@21600%20price-enc%20ret.svg)

![Encoding price as log-returns from the previous close](./readme/price-enc/ETHUSDT@Binance(k)%20time@21600%20price-enc%20logret.svg)

It is funny to create candlestick charts based on these returns.

![Candlestick chart of returns](./readme/price-enc/ETHUSDT@Binance(k)%20time@21600%20price-enc%20ret%20candlesticks.svg)

![Candlestick chart of log-returns](./readme/price-enc/ETHUSDT@Binance(k)%20time@21600%20price-enc%20logret%20candlesticks.svg)

The correlation with original prices is shown below.

| with price | without price |
| --- | --- |
| ![return](./readme/price-enc/ETHUSDT@Binance(k)%20time@21600%20price-enc%20corr%20ret+logret+p.svg) | ![log-return](./readme/price-enc/ETHUSDT@Binance(k)%20time@21600%20price-enc%20corr%20ret+logret.svg) |

Return and log-return distributions:

| open | high | low | close |
| --- | --- | --- | --- |
| ![ret-open](./readme/price-enc/ETHUSDT@Binance(k)%20time@21600%20distr%20ret(c)%20open.svg) | ![ret-high](./readme/price-enc/ETHUSDT@Binance(k)%20time@21600%20distr%20ret(c)%20high.svg) | ![ret-low](./readme/price-enc/ETHUSDT@Binance(k)%20time@21600%20distr%20ret(c)%20low.svg) | ![ret-close](./readme/price-enc/ETHUSDT@Binance(k)%20time@21600%20distr%20ret(c)%20close.svg) |
| ![logret-open](./readme/price-enc/ETHUSDT@Binance(k)%20time@21600%20distr%20logret(c)%20open.svg) | ![logret-high](./readme/price-enc/ETHUSDT@Binance(k)%20time@21600%20distr%20logret(c)%20high.svg) | ![logret-low](./readme/price-enc/ETHUSDT@Binance(k)%20time@21600%20distr%20logret(c)%20low.svg) | ![logret-close](./readme/price-enc/ETHUSDT@Binance(k)%20time@21600%20distr%20logret(c)%20close.svg) |

### Min-max scaling

Also known as [*rescaling* or *min-max normalization*](https://en.wikipedia.org/wiki/Feature_scaling), this is the simplest method:
$$x'={\frac {x-{\text{min}}(x)}{{\text{max}}(x)-{\text{min}}(x)}}$$
where $x$ is an original value, $x'$ is the normalized value.

It rescales $x\prime$ to [0, 1].
We can generalize this metod to rescale to an arbitrary set of values [a, b]:
$$x'=a+{\frac {(x-{\text{min}}(x))(b-a)}{{\text{max}}(x)-{\text{min}}(x)}}$$

To rescale to [-1, 1] ($a=-1$, $b=1$), we can rewrite this equation as
$$x'=2{\frac {(x-{\text{min}}(x))}{{\text{max}}(x)-{\text{min}}(x)}}-1$$

This method is very sensitive to the presence of outliers.

In the following illustration, all four `open`, `high`, `low` and `close`
were scaled using `minmax` on the window of last 64 prices.

![Scaling price using minmax](./readme/scaler-minmax/ETHUSDT@Binance(k)%20time@21600%20scaler-minmax.svg)

It is interesting to create candlestick charts based on these scaled prices.

![Minmax candlesticks](./readme/scaler-minmax/ETHUSDT@Binance(k)%20time@21600%20scaler-minmax%20candlesticks.svg)

Minmax-scaled prices are very correlated with each other and with raw prices.

![Correlation heatmap](./readme/scaler-minmax/ETHUSDT@Binance(k)%20time@21600%20scaler-minmax%20corr+p.svg)

The distributions of scaled prices are:

| open | high | low | close |
| --- | --- | --- | --- |
| ![mmopen](./readme/scaler-minmax/ETHUSDT@Binance(k)%20time@21600%20distr%20minmax%20open.svg) | ![mmhigh](./readme/scaler-minmax/ETHUSDT@Binance(k)%20time@21600%20distr%20minmax%20high.svg) | ![mmlow](./readme/scaler-minmax/ETHUSDT@Binance(k)%20time@21600%20distr%20minmax%20low.svg) | ![mmclose](./readme/scaler-minmax/ETHUSDT@Binance(k)%20time@21600%20distr%20minmax%20close.svg) |

### Z-score scaling

Also known as [*standardization* or *Z-score normalization*](https://en.wikipedia.org/wiki/Feature_scaling), this method gets its name from the concept of the [*standard score*](https://en.wikipedia.org/wiki/Standard_score) used in statistics.

The *standard score* is the number of *standard deviations* by which the observed value is above or below the *mean* value:
$$x'={\frac {x-{\bar {x}}}{\sigma }}$$
where ${\bar{x}}$ is the mean of the $x$ vector, and $\sigma$ is its standard deviation.

This method cannot guarantee balanced feature scales in the presence of outliers, because they have an influence when computing the empirical mean and standard deviation.

Also, the rescaled data doesn't fit into [-1, 1] interval.

In the following illustration, all four `open`, `high`, `low` and `close`
were scaled using `Z-score` on the window of last 64 prices.

![Scaling price using zscore](./readme/scaler-zscore/ETHUSDT@Binance(k)%20time@21600%20scaler-zscore.svg)

It is interesting to create candlestick charts based on these scaled prices.

![Zscore candlesticks](./readme/scaler-zscore/ETHUSDT@Binance(k)%20time@21600%20scaler-zscore%20candlesticks.svg)

Z-score-scaled prices are very correlated with each other and with raw prices.

![Correlation heatmap](./readme/scaler-zscore/ETHUSDT@Binance(k)%20time@21600%20scaler-zscore%20corr+p.svg)

The distributions of scaled prices are:

| open | high | low | close |
| --- | --- | --- | --- |
| ![zopen](./readme/scaler-zscore/ETHUSDT@Binance(k)%20time@21600%20distr%20zscore%20open.svg) | ![zhigh](./readme/scaler-zscore/ETHUSDT@Binance(k)%20time@21600%20distr%20zscore%20high.svg) | ![zlow](./readme/scaler-zscore/ETHUSDT@Binance(k)%20time@21600%20distr%20zscore%20low.svg) | ![zclose](./readme/scaler-zscore/ETHUSDT@Binance(k)%20time@21600%20distr%20zscore%20close.svg) |

### Robust scaling

The [Robust](https://en.wikipedia.org/wiki/Robust_measures_of_scale)
scaling subtracts the *median* and scales the data according to the *IQR* (Interquartile Range).
The *IQR* is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).

Since this method is based on percentiles, it is not influenced by a small number of large outliers.

In the following illustration, all four `open`, `high`, `low` and `close`
were scaled using `robust` on the window of last 64 prices.

![Scaling price using robust](./readme/scaler-robust/ETHUSDT@Binance(k)%20time@21600%20scaler-robust.svg)

It is interesting to create candlestick charts based on these scaled prices.

![Robust candlesticks](./readme/scaler-robust/ETHUSDT@Binance(k)%20time@21600%20scaler-robust%20candlesticks.svg)

Robust-scaled prices are very correlated with each other and with raw prices.

![Correlation heatmap](./readme/scaler-robust/ETHUSDT@Binance(k)%20time@21600%20scaler-robust%20corr+p.svg)

The distributions of scaled prices are:

| open | high | low | close |
| --- | --- | --- | --- |
| ![ropen](./readme/scaler-robust/ETHUSDT@Binance(k)%20time@21600%20distr%20robust%20open.svg) | ![rhigh](./readme/scaler-robust/ETHUSDT@Binance(k)%20time@21600%20distr%20robust%20high.svg) | ![rlow](./readme/scaler-robust/ETHUSDT@Binance(k)%20time@21600%20distr%20robust%20low.svg) | ![rclose](./readme/scaler-robust/ETHUSDT@Binance(k)%20time@21600%20distr%20robust%20close.svg) |

### OHLC ratios

In the following illustration, we show these ratios created from `open`, `high`, `low` and `close` prices.

![Ohlc ratios](./readme/ohlc-ratios/ETHUSDT@Binance(k)%20time@21600%20ohlc-ratios.svg)

The ratios are not very correlated with raw prices.

![Correlation heatmap](./readme/ohlc-ratios/ETHUSDT@Binance(k)%20time@21600%20ohlc-ratios%20corr+p.svg)

The distributions of ratios are:

| o-l / h-l | c-l / h-l |
| --- | --- |
| ![olhl](./readme/ohlc-ratios/ETHUSDT@Binance(k)%20time@21600%20distr%20ol_hl.svg) | ![clhl](./readme/ohlc-ratios/ETHUSDT@Binance(k)%20time@21600%20distr%20cl_hl.svg) |

### Technical analysis indicators

Here we try several common indicators created from `open`, `high`, `low` and `close` prices.

They are `efficiency ratio`, `market dimension`, `fractal dimension`, `stochastic oscillator`
and `Chaikin money flow`.

![effrati-chart](./readme/ta-1/ETHUSDT@Binance(k)%20time@21600%20ta-1%20effrati.svg)

![markdim-chart](./readme/ta-1/ETHUSDT@Binance(k)%20time@21600%20ta-1%20markdim.svg)

![fracdim-chart](./readme/ta-1/ETHUSDT@Binance(k)%20time@21600%20ta-1%20fracdim.svg)

![stoch-chart](./readme/ta-1/ETHUSDT@Binance(k)%20time@21600%20ta-1%20stoch.svg)

![cmf-chart](./readme/ta-1/ETHUSDT@Binance(k)%20time@21600%20ta-1%20cmf.svg)

The correlatin heatmap:

![Correlation heatmap](./readme/ta-1/ETHUSDT@Binance(k)%20time@21600%20ta-1%20corr+p.svg)

The distributions are:

| effrati | fracdim | markdim | stoch | cmf |
| --- | --- | --- | --- | --- |
| ![effrati](./readme/ta-1/ETHUSDT@Binance(k)%20time@21600%20distr%20effrati.svg) | ![fracdim](./readme/ta-1/ETHUSDT@Binance(k)%20time@21600%20distr%20fracdim.svg) | ![markdim](./readme/ta-1/ETHUSDT@Binance(k)%20time@21600%20distr%20markdim.svg) | ![stoch](./readme/ta-1/ETHUSDT@Binance(k)%20time@21600%20distr%20stoch.svg) | ![cmf](./readme/ta-1/ETHUSDT@Binance(k)%20time@21600%20distr%20cmf.svg) |

## Time

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
![Time encoding chart](./readme/time-enc/ETHUSDT@Binance(k)%20time@21600%20time%20enc.svg)

The correlation heatmap looks like this.
![Correlation heatmap](./readme/time-enc/ETHUSDT@Binance(k)%20time@21600%20time%20enc%20corr+p.svg)

The distributions are:

| tday | wday | yday |
| --- | --- | --- |
| ![tday](./readme/time-enc/ETHUSDT@Binance(k)%20time@21600%20distr%20tday.svg) | ![wday](./readme/time-enc/ETHUSDT@Binance(k)%20time@21600%20distr%20wday.svg) | ![yday](./readme/time-enc/ETHUSDT@Binance(k)%20time@21600%20distr%20yday.svg) |

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
