from datetime import date, datetime, timedelta, timezone
from typing import List

import pandas as pd
from matplotlib import pyplot as plt
#!pip install mplfinance
import mplfinance as mpf

#from providers import BinanceMonthlyTradesProvider
#from aggregators import IntervalTradeAggregator
#from actions import BuySellCloseAction
#from rewards import ConstantReward
#from environment import Environment

from env_my_intraday import BinanceMonthlyTradesProvider
from env_my_intraday import IntervalTradeAggregator
from env_my_intraday import BuySellCloseAction
from env_my_intraday import ConstantReward
from env_my_intraday import EMA, Copy, PriceEncoder
from env_my_intraday import Environment
from env_my_intraday import Trade, Provider, TradeAggregator
from env_my_intraday import Frame

def get_frames(provider: Provider, aggregator:TradeAggregator,
        number_of_frames: int, datetime_cutoff: datetime) -> pd.DataFrame:

    df_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(columns = df_columns)
    def add_frame(frame: Frame,df: pd.DataFrame):
        if frame is None:
            return
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
            add_frame(frame, df)
        except StopIteration:
            frame = aggregator.finish()
            add_frame(frame, df)
            break
    #df.set_index('datetime', inplace=True)
    return df

symbol = 'ETHUSDT'
dir = 'D:/data/binance_monthly_trades/'
provider = BinanceMonthlyTradesProvider(data_dir = dir, symbol = symbol,
                date_from = date(2024, 5, 1), date_to = date(2024, 5, 2))
provider_name = provider.name

datetime_cutoff=datetime(2024, 5, 2, tzinfo=timezone.utc)

aggregator = IntervalTradeAggregator(method='time',
                interval=1*60, duration=(1, 8*60*60))
aggregator_name = aggregator.name
df = get_frames(provider, aggregator, 128, datetime_cutoff)
#print(len(df))
#print(df.head())

def plot_mplfinance_candlesticks(df, title, figsize=None):
    df = df.copy()
    df.index = pd.DatetimeIndex(df.datetime)

    mc = mpf.make_marketcolors(
    up='palegreen',down='c',
    edge='inherit',
    wick='inherit',
    volume='inherit',
    ohlc='inherit')
    style = mpf.make_mpf_style(base_mpl_style='default', marketcolors=mc,
                                mavcolors=['skyblue', 'midnightblue'])
    mpf.plot(df, type='candle', volume=True, ylabel_lower='', ylabel='',
             style=style, title=title, figratio=(10, 6), figscale=0.85)#, figsize=figsize)

def plot_candlesticks(df, title, figsize=(8, 4)):
    fig = plt.figure(dpi=120, layout='constrained', figsize=figsize)
    fig.set_facecolor('#202020')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('#303030')
    ax.tick_params(labelsize='small', colors='#666666')
    ax.grid(color='#666666')
    ax.set_title(title, color='#d0d0d0')

    wick_width=.2
    body_width=.8
    up_color='limegreen'
    down_color='tab:blue'

    df = df.copy()
    #df['datetime'] = df['datetime'].dt.tz_convert('UTC')
    #df.datetime = df.datetime.astype('datetime64[ns]')
    #df.datetime = pd.to_datetime(df.datetime)
    #df.datetime = df.datetime.dt.strftime('%Y-%m-%d %H:%M:%S')
    #df.datetime = pd.to_datetime(df.datetime)
    #df.set_index('datetime', inplace=True)
    #df.sort_index(inplace=True)
    #df.index = pd.DatetimeIndex(df.datetime)
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
    return fig

#plot_mplfinance_candlesticks(df, title=f'{provider_name} {aggregator_name}', figsize=(10, 5))
fig = plot_candlesticks(df, f'{provider_name} {aggregator_name}')
plt.show()
#plt.close(fig)
plt.close()