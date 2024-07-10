from datetime import date, timedelta
import math

from environment import BinanceMonthlyKlines1mToTradesProvider
from environment import BinanceMonthlyTradesProvider
from environment import SineTradesProvider
from environment import IntervalTradeAggregator
from environment import BalanceReturnReward
from environment import BuySellHoldCloseAction
from environment import Environment
from environment import OhlcRatios, TimeEncoder, Scale, CopyPeriod
from environment import AccountCalmar, AccountSharpe, AccountSortino, AccountROI, AccountROR
from wrappers import HumanRendering

# --------------------------------- data
#SYMBOL = 'BTCEUR'
#SYMBOL = 'BTCUSDT'
SYMBOL = 'ETHUSDT'

WHAT = 'binance-klines'
#WHAT = 'binance-trades'
#WHAT = 'sine'

if WHAT == 'binance-klines':
    dir = 'data/binance_monthly_klines/'
    provider = BinanceMonthlyKlines1mToTradesProvider(data_dir = dir, symbol = SYMBOL,
                        date_from = date(2024, 1, 1), date_to = date(2024, 5, 31))
elif WHAT == 'binance-trades':
    dir = 'data/binance_monthly_trades/'
    provider = BinanceMonthlyTradesProvider(data_dir = dir, symbol = SYMBOL,
                        #date_from = date(2024, 5, 1), date_to = date(2024, 5, 31))
                        date_from = date(2018, 5, 1), date_to=date(2018, 5, 31))
elif WHAT == 'sine':
    SYMBOL = 'sine'
    provider = SineTradesProvider(mean = 100, amplitude = 90, SNRdb = 15,
                    period=(timedelta(minutes=10), timedelta(minutes=30)),
                    date_from = date(2024, 5, 1), date_to = date(2024, 5, 31))
else:
    raise ValueError('Unknown WHAT')

# --------------------------------- aggregator
TIME_FRAME='1m'
#TIME_FRAME='1h'
TIME_FRAME_SECS = 1*60 if TIME_FRAME == '1m' else 60*60
aggregator = IntervalTradeAggregator(method='time',
                interval=TIME_FRAME_SECS, duration=(1, 8*60*60))

# --------------------------------- features
SCALE_METHOD = 'zscore'
SCALE_PERIOD = 196
COPY_PERIOD = 196
features_pipeline = [
    Scale(source=['open', 'high', 'low', 'close', 'volume'], method=SCALE_METHOD,
          period=SCALE_PERIOD, write_to='frame'),
    OhlcRatios(write_to='frame'),
    TimeEncoder(source=['time_start'], yday=True, wday=True, tday=True, write_to='frame'),
    #AccountCalmar(write_to='frame'),
    #AccountSharpe(write_to='frame'),
    #AccountSortino(write_to='frame'),
    #AccountROI(write_to='frame'),
    #AccountROR(write_to='frame'),
    CopyPeriod(source=[
        ('open', -math.inf, math.inf),
        ('high', -math.inf, math.inf),
        ('low', -math.inf, math.inf),
        ('close', -math.inf, math.inf),
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
        #('roi', -math.inf, math.inf),
        #('ror', -math.inf, math.inf),
        #('sharpe', -math.inf, math.inf),
        #('calmar', -math.inf, math.inf),
        #('sortino', -math.inf, math.inf),
        ], copy_period=COPY_PERIOD)
]

# --------------------------------- environment
env = Environment(
    provider=provider,
    aggregator=aggregator,
    features_pipeline=features_pipeline,
    action_scheme=BuySellHoldCloseAction(),
    reward_scheme=BalanceReturnReward(),
    warm_up_duration=None,
    episode_max_duration=None,
    render_mode='rgb_array',
    initial_balance=10000,
    halt_account_if_negative_balance=False,
    episode_max_steps=180#128#196
)
env = HumanRendering(env)
# --------------------------------- loop
for episode in range(2): #4
    state, frame = env.reset()
    while True:
        action = env.action_space.sample()
        state, reward, terminated, truncated, frame = env.step(action)
        if terminated or truncated:
            break
env.close()
