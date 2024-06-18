# Trading environment

This package implements [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
environment.

Its internal machinery is based on a stream of trades, which may be real ones
or simulated from aggregated price bars (*candles*, *klines*). This idea was
inspired by [Intraday](https://github.com/diovisgood/intraday) package.

## Converting price bars into the stream of trades

If our data source is a sequence of aggregated price bars (candlesticks, klines),
we have to convert the into a stream of simulated trades.

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

Then, the number of price steps per price segment will be

```python
step = spread * (high + low) / 2
```

We can experiment with different valuse of the 'spread' to see how many trades we will generate.

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

Some type of price aggregation bars (e.g. klines) have fields like
`volume`, `money`, `buy_volume`, `buy_money`. We can use them to
preserve the `VWAP` of the kline.

A volume weighted average price (`VWAP`) is defined for a sequence
of trades as follows.

```text
VWAP = (price1 * volume1 + price2 * volume2 + ... ) / (volume1 + volume2 + ...)
```

Each trade has a buyer and a seller.
So called `Buy` trade is a trade initiated by the buyer.

The extra fields of a kline allow us to compute separately
the `VWAP` of the buy trades and the `VWAP` of the sell trades.

```python
VWAP_buy = buy_money / buy_volume
VWAP_sell = (money - buy_money) / (volume - buy_volume)
```

Knowing them, we can adjust the volumes of the simulated
individual trades so the overall `VWAP` calculated on the
sequence of trades will match the `VWAP` of the original kline.

## Data sources

### Binance

The source for [Binance](https://www.binance.com/en) data is the [Binance Public Data](https://github.com/binance/binance-public-data/) project. The data can be downloaded
from the [Binance market data](https://data.binance.vision/) website.
The website has data for `spot`, `options` and `futures`, but only
`spot` symbols can be downloaded automatically.

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

If needed, you can pre-download `.zip` files using the code below.
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

If needed, you can pre-download `.zip` files using the code below.
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
