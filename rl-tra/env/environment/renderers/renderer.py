from abc import ABC, abstractmethod
from typing import List, Optional
from numbers import Real

from ..accounts.account import Account
from ..providers.provider import Provider
from ..aggregators.trade_aggregator import TradeAggregator
from ..frame import Frame 

class Renderer(ABC):

    @abstractmethod
    def reset(self, episode_number: int, episode_max_steps: Optional[int],
            account: Account, provider: Provider, aggregator: TradeAggregator,
            frames: List[Frame]):
        pass

    @abstractmethod
    def step(self, frames: List[Frame], reward: Real):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass
    