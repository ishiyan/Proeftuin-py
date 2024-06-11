from enum import Enum
from typing import Tuple
import numpy as np
import gymnasium as gym

from .trader import Trader

class Actions(Enum):
    Sell = 0
    Buy = 1

class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long

class Any1Trader(Trader):
    def __init__(self, window_size):
        self._action_space = gym.spaces.Discrete(len(Actions))
        self.window_size = window_size

        self.trade_fee_bid_percent = 0.01 # unit
        self.trade_fee_ask_percent = 0.005 # unit

        self._current_price = None
        self._last_traded_price = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    def reset(self, initial_price_window: np.array) -> dict:
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self.history = {}
        self._total_reward = 0.
        self._total_profit = 1. # unit
        self._last_traded_price = initial_price_window[-2][3]
        self._current_price = initial_price_window[-1][3]
        return self.get_info()

    def act(self, action: Actions, new_price_window: np.array, finished: bool) -> Tuple[float, dict]:
        side_change = \
            (action == Actions.Buy.value and self._position == Positions.Short) or \
            (action == Actions.Sell.value and self._position == Positions.Long)

        # Calculate reward
        step_reward = 0
        if side_change and self._position == Positions.Long:
            step_reward += self._current_price - self._last_traded_price
        self._total_reward += step_reward

        # Update profit
        if (side_change or finished) and (self._position == Positions.Long):
            units = (self._total_profit * (1 - self.trade_fee_ask_percent)) / self._last_traded_price
            self._total_profit = (units * (1 - self.trade_fee_bid_percent)) * self._current_price

        # Take action
        if side_change:
            self._position = self._position.opposite()
            self._last_traded_price = self._current_price

        # Update state
        self._position_history.append(self._position)
        self._current_price = new_price_window[-1][3]
        info = self.get_info()
        self._update_history(info)

        return step_reward, info

    def get_info(self):
        return dict(total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position,
            last_traded_price=self._last_traded_price,
            current_price=self._current_price)

    def get_history(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position,
            last_traded_price=self._last_traded_price,
            current_price=self._current_price,)
    
    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}
        for key, value in info.items():
            self.history[key].append(value)
