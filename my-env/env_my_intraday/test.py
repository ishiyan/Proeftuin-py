    
#from providers import BarToTradeConverter
#spread = 0.001
#p, o = BarToTradeConverter._zigzag_open_high_low_close(
#                    #0.2, 0.4, 0.1, 0.3, spread)
#                    100.2, 100.4, 100.1, 100.3, spread)
#for i in range(len(p)):
#    print(i, p[i], o[i])
# 0.4   4
# 0.3   5
# 0.2  6
# 0.1-> 11
# 0.05-> 18
# 0.01-> 73
# 0.005-> 144
# 0.001-> 703

from datetime import date, timedelta
from matplotlib import pyplot as plt

#from providers import BinanceMonthlyTradesProvider
#from aggregators import IntervalTradeAggregator
#from actions import BuySellCloseAction
#from rewards import ConstantReward
#from environment import Environment

from env_my_intraday import BinanceMonthlyTradesProvider
from env_my_intraday import IntervalTradeAggregator
from env_my_intraday import BuySellCloseAction
from env_my_intraday import ConstantReward
from env_my_intraday import Environment

symbol = 'ETHUSDT'
dir = 'D:/data/binance_monthly_trades/'
provider = BinanceMonthlyTradesProvider(data_dir = dir, symbol = symbol,
                date_from = date(2024, 5, 1), date_to = date(2024, 5, 31))

aggregator = IntervalTradeAggregator(method='time',
                interval=1*60, duration=(1, 8*60*60))

env = Environment(
    provider=provider,
    aggregator=aggregator,
    action_scheme=BuySellCloseAction(),
    reward_scheme=ConstantReward(),
    warm_up_duration=2*60,
    episode_max_duration=None,
    episode_max_steps=1
)

state, frame = env.reset()
print(state)
