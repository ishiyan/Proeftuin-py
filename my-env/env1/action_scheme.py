from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import gymnasium as gym
import numpy as np

class ActionScheme(ABC):
    """Determines the action to take at each step of an episode."""

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Space:
        """The action space of the environment. (`Space`, read-only)"""
        raise NotImplementedError()

    @abstractmethod
    def perform(self, action: Any, truncated: bool, price: float) -> None:
        """Performs an action on the environment."""
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        """Resets the action scheme."""
        raise NotImplementedError()

# --------------------------------------------------

class Actions(Enum):
    Sell = 0
    Buy = 1

class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long

class Any1ActionScheme(ActionScheme):

    def __init__(self) -> None:
        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit
        self.action_space = gym.spaces.Discrete(len(Actions))

        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._last_trade_price = None
        self._current_price = None

    def perform(self, action: Any, truncated: bool, price: float) -> None:
        trade = True if \
            ((action == Actions.Buy.value and self._position == Positions.Short) or \
            (action == Actions.Sell.value and self._position == Positions.Long)) \
            else False
        if trade or truncated:
            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / self.last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * price
        if trade:
            self._position = self._position.opposite()
            self._last_trade_price = price
        self._current_price = price
        self._position_history.append(self._position)

    def reset(self) -> None:
        #self.portfolio.reset()
        #self.broker.reset()
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
