from typing import Sequence, Literal
from abc import abstractmethod

from ..providers import Trade
from .feature import Feature

class TradesFeature(Feature):
    """
    Base class for all feature processors which extract values from stream of trades instead of frames
    
    Attributes
    ----------
    trades_period : int
        How many latest trades you need to compute you feature's values?
        Note: not all features require previous trades. Specify 1 or 0 or None in this case.
    """
    def __init__(self, write_to: Literal['state', 'frame', 'both'] = 'state', **kwargs):
        super().__init__(write_to=write_to, **kwargs)
        self.trades_period = kwargs['trades_period'] if ('trades_period' in kwargs) else None
    
    @abstractmethod
    def update(self, trades: Sequence[Trade]):
        """
        Compute your feature values when new trade arrives

        Notes
        -----
        Important: this method is invoked only ONCE for each new trade.

        Parameters
        ----------
        trades : Sequence[Trade]
            Contains a list of latest trades.
            trades[-1] is the newest trade.
            And trades[-2] is the trade before the newest one.
        """
        raise NotImplementedError()
