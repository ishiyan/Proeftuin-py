from datetime import date, timedelta
import math

from env import Environment
from env import BinanceMonthlyKlines1mToTradesProvider
from env import SineTradesProvider
from env import IntervalTradeAggregator
from env import BalanceReturnReward
from env import BuySellHoldCloseAction
from env import Environment
from env import OhlcRatios, TimeEncoder, Scale, CopyPeriod

SYMBOL = 'ETHUSDT'
TIME_FRAME='1m'
SCALE_METHOD = 'zscore'
SCALE_PERIOD = 196
COPY_PERIOD = 196

class Env01(Environment):
    def __init__(self,
                 symbol='ETHUSDT',
                 time_frame='1m',
                 scale_period = 196,
                 copy_period = 196,
                 episode_max_steps=128,
                 render_mode=None,
                ):
        """
        Args:
            symbol str: 'ETHUSDT' or 'sine'
            time_frame: str: '1m' or '1h'
            scale_period int: 196
            copy_period int: 196
            episode_max_steps int: 128, 196
            render_mode: None, 'ansi', 'rgb_array' 
        """
        # --------------------------------- data
        if symbol == 'sine':
            provider = SineTradesProvider(mean = 100, amplitude = 90, SNRdb = 15,
                period=(timedelta(minutes=10), timedelta(minutes=30)),
                date_from = date(2018, 1, 1), date_to = date(2023, 12, 31))
        else:
            dir = 'env/data/binance_monthly_klines/'
            provider =[]
            for i in range(1, 2):
                provider.append(BinanceMonthlyKlines1mToTradesProvider(data_dir = dir, symbol = symbol,
                date_from = date(2018, 1, 1), date_to = date(2023, 12, 31)))
        SYMBOL = symbol
        # --------------------------------- aggregator
        time_frame_secs = 1*60 if time_frame == '1m' else 60*60
        aggregator = IntervalTradeAggregator(method='time',
            interval=time_frame_secs, duration=(1, 8*60*60))
        TIME_FRAME=time_frame
        # --------------------------------- features
        SCALE_PERIOD = scale_period
        COPY_PERIOD = copy_period
        features_pipeline = [
            Scale(source=['open', 'high', 'low', 'close', 'volume'], method=SCALE_METHOD,
                period=SCALE_PERIOD, write_to='frame'),
            OhlcRatios(write_to='frame'),
            TimeEncoder(source=['time_start'], yday=True, wday=True, tday=True, write_to='frame'),
            CopyPeriod(source=[
                (f'{SCALE_METHOD}{SCALE_PERIOD}_open', -math.inf, math.inf),
                (f'{SCALE_METHOD}{SCALE_PERIOD}_high', -math.inf, math.inf),
                (f'{SCALE_METHOD}{SCALE_PERIOD}_low', -math.inf, math.inf),
                (f'{SCALE_METHOD}{SCALE_PERIOD}_close', -math.inf, math.inf),
                (f'{SCALE_METHOD}{SCALE_PERIOD}_volume', -math.inf, math.inf),
                ('ol_hl', 0.0, 1.0),
                ('cl_hl', 0.0, 1.0),
                ('yday_time_start', 0.0, 1.0),
                ('wday_time_start', 0.0, 1.0),
                ('tday_time_start', 0.0, 1.0),
            ], copy_period=COPY_PERIOD)
        ]
        super().__init__(
            provider=provider,
            aggregator=aggregator,
            features_pipeline=features_pipeline,
            action_scheme=BuySellHoldCloseAction(),
            reward_scheme=BalanceReturnReward(),
            warm_up_duration=0,
            episode_max_duration= \
                time_frame_secs*(episode_max_steps + max(SCALE_PERIOD, COPY_PERIOD)),
            render_mode=render_mode,
            initial_balance=10000,
            episode_max_steps=episode_max_steps,
        )
