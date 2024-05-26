from .trading_env import TradingEnv, Actions, Positions
from .stocks_env import StocksEnv

from gymnasium.envs.registration import register
register(
    id='StocksEnvAny2-v0',
    entry_point='env_any_2.stocks_env:StocksEnv',
    disable_env_checker = True
)
