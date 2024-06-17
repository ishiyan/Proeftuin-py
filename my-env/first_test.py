from datetime import date, timedelta
import imageio
from matplotlib import pyplot as plt

from env_my_intraday import BinanceMonthlyKlines1mToTradesProvider
from env_my_intraday import BinanceMonthlyTradesProvider
from env_my_intraday import IntervalProcessor
from env_my_intraday import EMA, Copy, PriceEncoder
from env_my_intraday import BuySellCloseAction
from env_my_intraday import BalanceReward
from env_my_intraday import Environment

#dir = 'D:/data/binance_monthly_trades/'
#symbol = 'BTCEUR'
#symbol = 'BTCUSDT'
#symbol = 'ETHUSDT'
dir = 'env_my_intraday/data/binance_monthly_klines/'
symbol = 'BTCUSDT'

provider = BinanceMonthlyKlines1mToTradesProvider(data_dir=dir, symbol=symbol,
                                    date_from=date(2024, 1, 1), date_to=date(2024, 5, 31))

#provider = BinanceMonthlyTradesProvider(data_dir=dir, symbol=symbol,
#                                  #date_from=date(2024, 5, 1), date_to=date(2024, 5, 31))
#                                  date_from=date(2018, 5, 1), date_to=date(2018, 5, 31))
processor = IntervalProcessor(method='time', interval=1*60, duration=(1, 8*60*60))
period = 100
atr_name = f'ema_{period}_true_range'
features_pipeline = [
    PriceEncoder(source='close', write_to='both'),
    EMA(period=period, source='true_range', write_to='frame'),
    Copy(source=['volume'])
]
action_scheme = BuySellCloseAction()
reward_scheme = BalanceReward(norm_factor=atr_name)
env = Environment(
    provider=provider,
    processor=processor,
    features_pipeline=features_pipeline,
    action_scheme=action_scheme,
    reward_scheme=reward_scheme,
    warm_up_duration=None,
    episode_max_duration=None,#2*60*60,
    #render_mode='rgb_array',
    initial_balance=10000, #1000000
    episode_max_steps=197
)

frames=[]
for i in range(1):
    state, frame = env.reset()
    #rgb_array = env.render(mode='rgb_array')
    #frames = [rgb_array]
    while True:
        #env.render('human')
        #rgb_array = env.render(mode='rgb_array')
        #frames.append(rgb_array)
        #print(state)
        action = env.action_space.sample()
        #action = action_scheme.get_random_action()
        state, reward, terminated, truncated, frame = env.step(action)
        if terminated or truncated:
            rgb_array = env.render(mode='rgb_array')
            frames.append(rgb_array)
            break
plt.show() 
env.close()
#imageio.mimwrite(uri='random_agent_steps.gif', ims=frames, fps=3) # os.path.join('./videos/', 'random_agent.gif')
plt.close()