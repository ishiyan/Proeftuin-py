from datetime import datetime

from ..instruments import Instrument
from .side import OrderSide
from .time_in_force import OrderTimeInForce
from .type import OrderType

class Order:
    """A request to place an order in a single instrument.

    Parameters
    ----------
    instrument : Instrument
        The instrument to be traded.
    type : OrderType
        The type of the order.
    side : OrderSide
        The side of the order.
    time_in_force : OrderTimeInForce
        The time in force condition of the order.
    quantity : float
        The total order quantity (in units) to execute.
    minimum_quantity : float, optional
        The minimum quantity (in units) to be filled.
    limit_price : float, optional
        The limit price in instrument's currency per unit of quantity.
        Zero if not set.

        Required for limit, stop-limit, and limit-if-touched orders.

        For FX orders, should be the "all-in" rate (spot rate adjusted for forward points).
    stop_price : float, optional
        The stop price in instrument's currency per unit of quantity.
        Zero if not set.

        Required for stop, stop-limit, and limit-if-touched orders.
    trailing_distance : float, optional
        The trailing distance in instrument's currency.
        Zero if not set.

        Required for trailing-stop orders.
    creation_time : datetime, optional
        The date and time when the order was created by a trader, trading system, or intermediary.
    expiration_time : datetime, optional
        The order expiration date and time for the good-till-date orders
    note : str, optional
        A free-format text with notes on the order.
    """
    def __init__(self,
                 instrument: Instrument,
                 type: OrderType = OrderType.MARKET,
                 side: OrderSide = OrderSide.BUY,
                 time_in_force: OrderTimeInForce = OrderTimeInForce.DAY,
                 quantity: float = 1,
                 minimum_quantity: float = None,
                 limit_price: float = None,
                 stop_price: float = None,
                 trailing_distance: float = None,
                 creation_time: datetime = None,
                 expiration_time: datetime = None,
                 note: str = '',
                 ):
        self.instrument = instrument
        self.type = type
        self.side = side
        self.time_in_force = time_in_force
        self.quantity = quantity
        self.minimum_quantity = minimum_quantity
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.trailing_distance = trailing_distance
        self.creation_time = creation_time
        self.expiration_time = expiration_time
        self.note = note

    def __str__(self):
        return f'{self.instrument.symbol} {self.side} {self.type} {self.time_in_force} {self.quantity} {self.instrument.symbol} {self.type}'

    def __repr__(self):
        attributes = ['instrument', 'type', 'side', 'time_in_force', 'quantity',
            'minimum_quantity', 'limit_price', 'stop_price', 'trailing_distance',
            'creation_time', 'expiration_time', 'note']
        attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes ]
        #attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes if getattr(self, attr) is not None]
        return 'OrderSingle(' + ', '.join(attr_strings) + ')'
