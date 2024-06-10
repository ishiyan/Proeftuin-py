import numpy as np

from .enums import Actions, Positions
from .zero_env import ZeroEnv
from .observer import DataframeObserver
from .renderer import Any1Renderer

class MbraneEnv(ZeroEnv):
    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(self, df, window_size, frame_bound, render_mode=None):
        assert render_mode is None or render_mode in self.metadata['render_modes']

        self.observer = DataframeObserver(df, window_size, frame_bound)
        self.renderer = Any1Renderer(render_mode)

        super().__init__(window_size, self.observer, self.renderer)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    def _calculate_reward(self, action):
        step_reward = 0
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            current_price = self.prices[self._current_tick] #######################
            last_trade_price = self.prices[self._last_trade_tick] #######################
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                step_reward += price_diff

        return step_reward

    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._truncated: #######################
            current_price = self.prices[self._current_tick] #######################
            last_trade_price = self.prices[self._last_trade_tick] #######################

            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price
