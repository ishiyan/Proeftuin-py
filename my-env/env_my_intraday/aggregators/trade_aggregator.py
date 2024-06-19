from abc import ABC, abstractmethod
from typing import Sequence, Optional

from ..providers import Trade
from ..frame import Frame

class TradeAggregator(ABC):
    """
    Base abstract class to aggregate a sequence of trades into a frame.
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def aggregate(self, trades: Sequence[Trade]) -> Optional[Frame]:
        pass
    
    @abstractmethod
    def finish(self) -> Optional[Frame]:
        pass
    
    @property
    @abstractmethod
    def name(self):
        pass
