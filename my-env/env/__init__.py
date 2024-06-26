from gymnasium.envs.registration import register

register(
    id='TradingEnv-v0',
    entry_point='env.environments:TradingEnv',
    disable_env_checker = True
)
register(
    id='MultiDatasetTradingEnv-v0',
    entry_point='env.environments:MultiDatasetTradingEnv',
    disable_env_checker = True
)
 