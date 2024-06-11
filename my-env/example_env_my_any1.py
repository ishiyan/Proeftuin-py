import pandas as pd
import matplotlib.pyplot as plt

from env_my import Any1Env

def read_adjusted_history_ohlcv(path_csv):
    history = pd.read_csv(path_csv, parse_dates=["Date"], index_col='Date', \
        dtype={'Open': float, 'High': float, 'Low': float, 'Close': float,'Volume': float})
    history.drop('Adj Close', axis=1, inplace=True)
    history = history.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    history.sort_index(inplace= True)
    history.dropna(inplace= True)
    history.drop_duplicates(inplace=True)
    return history

price_history = read_adjusted_history_ohlcv('yfinance/googl.1d.adjusted.csv')
#price_history

price_history['median'] = (price_history['high'] + price_history['low']) / 2
price_history['typical'] = (price_history['high'] + price_history['low'] + price_history['close']) / 3

price_history["f_open"] = price_history["open"]/price_history["close"]
price_history["f_high"] = price_history["high"]/price_history["close"]
price_history["f_low"] = price_history["low"]/price_history["close"]
price_history["f_close"] = price_history["close"].pct_change()
price_history["f_volume"] = price_history["volume"] / price_history["volume"].rolling(252).max()

price_history.dropna(inplace= True)
#price_history

env = Any1Env(df=price_history, frame_bound=(50, 303), window_size=10)#, render_mode='human')
#env = gym.make('Any1Env-v0', df=price_history, frame_bound=(50, 303), window_size=10)

observation = env.reset(seed=None)
while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # env.render()
    if terminated or truncated:
        print("info:", info)
        break

plt.cla()
env.unwrapped.render_all()
plt.show()
