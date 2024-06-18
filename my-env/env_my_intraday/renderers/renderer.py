from abc import ABC, abstractmethod
from typing import Optional
from numbers import Real

from ..accounts.account import Account
from ..providers.provider import Provider
from ..aggregators.trade_aggregator import TradeAggregator
from ..frame import Frame 

class Renderer(ABC):

    @abstractmethod
    def reset(self, episode_number: int, episode_max_steps: Optional[int],
            account: Account, provider: Provider, aggregator: TradeAggregator,
            frame: Frame):
        raise NotImplementedError()

    @abstractmethod
    def step(self, frame: Frame, reward: Real):
        raise NotImplementedError()

    @abstractmethod
    def render(self):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()
    