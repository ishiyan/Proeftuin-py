from datetime import datetime
from typing import Dict, List, Tuple

from ..currencies import Currency, CurrencyConverter
from ..utils import ScalarSeries
from ..instruments import Instrument
from ..orders import OrderExecution, OrderSide
from ..accounts import Account
from .performances import RoundtripMatching, Roundtrip, Performance
from .positions import PortfolioPosition, PositionSide

class Portfolio:
    """Portfolio position.
    """
    
    def __init__(self,
        holder: str,
        initial_cash: float,
        currency: Currency,
        currency_converter: CurrencyConverter,
        roundtrip_matching: RoundtripMatching):

        self._initial_cash: float = initial_cash
        self._currency: Currency = currency
        self._currency_converter: CurrencyConverter = currency_converter
        self._roundtrip_matching: RoundtripMatching = roundtrip_matching
        self._account: Account = Account(holder, currency, currency_converter)
        self._positions: Dict[Instrument, PortfolioPosition] = {}
        self._executions: List[OrderExecution] = []
        self._performance: Performance = Performance()

        # here is a problem with the date-time !!!
        self._account.add(datetime.now(), initial_cash, currency, 'Initial cash')
