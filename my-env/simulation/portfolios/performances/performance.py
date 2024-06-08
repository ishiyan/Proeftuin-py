from datetime import datetime

from .roundtrips import Roundtrip,RoundtripPerformance
from .drawdowns import Drawdown
from .pnls import PnL

class Performance:
    """Tracks performance of a portfolio or an individual position."""

    def __init__(self):
        self._pnl: PnL = PnL()
        self._drd: Drawdown = Drawdown()
        self._rtp: RoundtripPerformance = RoundtripPerformance()

    def add_roundtrip(self, roundtrip: Roundtrip):
        """Adds a roundtrip."""
        return self._rtp.add(roundtrip)

    def add_PnL(self, time: datetime, initial_amount: float, amount: float,
            unrealized_amount: float, cash_flow: float):
        """Adds a profit and loss value at a specific time."""
        return self._pnl.add(time, initial_amount, amount, unrealized_amount, cash_flow)

    def add_drawdown(self, time: datetime, value: float):
        """Adds a drawdown value at a specific time."""
        return self._drd.add(time, value)

    def get_PnL(self) -> PnL:
        """Returns the profit and loss performance."""
        return self._pnl
    
    def get_drawdown(self) -> Drawdown:
        """Returns the drawdown performance."""
        return self._drd

    def get_roundtrip(self) -> RoundtripPerformance:
        """Returns the roundtrip performance."""
        return self.rtp
    