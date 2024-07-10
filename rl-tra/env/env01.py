from typing import Optional, Sequence, Union
from datetime import date, timedelta
import math

from env import Environment
from env import BinanceMonthlyKlines1mToTradesProvider
from env import SineTradesProvider
from env import IntervalTradeAggregator
from env import BalanceReturnReward, RewardScheme
from env import BuySellHoldCloseAction, ActionScheme
from env import Environment
from env import OhlcRatios, TimeEncoder, Scale, CopyPeriod

class Env01(Environment):
    def __init__(self,
                 symbol: Union[str, Sequence[str]]='ETHUSDT',
                 time_frame='1m',
                 scale_method='zscore',
                 scale_period=196,
                 copy_period=196,
                 episode_max_steps=128,
                 initial_balance=10000,
                 action_scheme: Optional[ActionScheme]=None,
                 reward_scheme: Optional[RewardScheme]=None,
                 render_mode=None,
                 vec_env_index: Optional[int] = None,
                ):
        """
        Args:
            symbol str or sequence(str): 'ETHUSDT', 'BTCUSDT', 'BTCEUR' or 'SINE'
            time_frame str: '1m' or '1h'
            scale_method str: 'zscore', 'minmax', 'meanstd'
            scale_period int: 196
            copy_period int: 196
            episode_max_steps int: 128, 196
            initial_balance float: 10000
            action_scheme: None, BuySellHoldCloseAction()
            reward_scheme: None, BalanceReturnReward()
            render_mode: None, 'ansi', 'rgb_array'
            vec_env_index: None, int
        """
        # --------------------------------- data
        if isinstance(symbol, str):
            symbols = [symbol]
        elif isinstance(symbol, Sequence):
            symbols = symbol
        else:
            raise ValueError('symbol must be str or sequence(str)')
        providers = []
        for symbol in symbols:
            symbol = symbol.upper()
            if symbol == 'SINE':
                provider = SineTradesProvider(mean=100, amplitude=90, SNRdb=15,
                    period=(timedelta(minutes=10), timedelta(minutes=30)),
                    date_from=date(2018, 1, 1), date_to=date(2024, 7, 1))
            elif symbol in ['ETHUSDT', 'BTCUSDT', 'BTCEUR']:
                dir = './env/data/binance_monthly_klines/'
                if symbol == 'BTCEUR':
                    date_from = date(2020, 1, 1)
                elif symbol == 'BTCUSDT':
                    date_from = date(2018, 1, 1)
                elif symbol == 'ETHUSDT':
                    date_from = date(2018, 1, 1)
                else:
                    date_from = date(2020, 1, 1)
                provider = BinanceMonthlyKlines1mToTradesProvider(data_dir=dir, spread=0.01,
                    symbol=symbol, date_from=date_from, date_to=date(2024, 5, 31))
            else:
                raise ValueError(f"symbol must be 'ETHUSDT', 'BTCUSDT', 'BTCEUR', 'SINE'")
            providers.append(provider)
        # --------------------------------- aggregator
        time_frame_secs = 1*60 if time_frame == '1m' else 60*60
        aggregator = IntervalTradeAggregator(method='time',
            interval=time_frame_secs, duration=(1, 24*60*60))
        # --------------------------------- features
        features_pipeline = [
            Scale(source=['open', 'high', 'low', 'close', 'volume'], method=scale_method,
                period=scale_period, write_to='frame'),
            OhlcRatios(write_to='frame'),
            TimeEncoder(source=['time_start'], yday=True, wday=True, tday=True, write_to='frame'),
            CopyPeriod(source=[
                (f'{scale_method}{scale_period}_open', -math.inf, math.inf),
                (f'{scale_method}{scale_period}_high', -math.inf, math.inf),
                (f'{scale_method}{scale_period}_low', -math.inf, math.inf),
                (f'{scale_method}{scale_period}_close', -math.inf, math.inf),
                (f'{scale_method}{scale_period}_volume', -math.inf, math.inf),
                ('ol_hl', 0.0, 1.0),
                ('cl_hl', 0.0, 1.0),
                ('yday_time_start', 0.0, 1.0),
                ('wday_time_start', 0.0, 1.0),
                ('tday_time_start', 0.0, 1.0),
            ], copy_period=copy_period)
        ]
        super().__init__(
            provider=providers,
            aggregator=aggregator,
            features_pipeline=features_pipeline,
            action_scheme=BuySellHoldCloseAction() \
                if action_scheme is None else action_scheme,
            reward_scheme=BalanceReturnReward() \
                if reward_scheme is None else reward_scheme,
            warm_up_duration=0,
            episode_max_duration=time_frame_secs* \
                (episode_max_steps + max(scale_period, copy_period)),
            render_mode=render_mode,
            initial_balance=initial_balance,
            halt_account_if_negative_balance=False,
            episode_max_steps=episode_max_steps,
            vec_env_index=vec_env_index,
        )
