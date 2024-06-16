from numbers import Real
from typing import Any, Union
import datetime as dt

from .execution_side import ExecutionSide

class Execution:
    """Contains properties of a fill or a partial fill based on
    a filled or partially-filled execution report of an order.

    Parameters
    ----------
    datetime : Union[Real, dt.datetime, dt.date, Any]
        The date and time of an associated execution report
        as assigned by the broker.
    side : ExecutionSide
        The execution order side, which determines the sign of the quantity.
    quantity : float
        The unsigned execution quantity, the sign is determined by the side.
    quantity_sign : float
        The sign of the execution quantity, which is 1 for buy executions
        and -1 for sell executions.
    commission : float
        The execution commission amount.
    commission_per_unit : float
        The commission amount per unit.
    price : float
        The execution price.
    amount : float
        The unsigned execution value (price times quantity).
    margin : float
        The unsigned execution margin (instrument margin times quantity).
    debt : float
        The execution debt in (amount minus margin).
    pnl : float
        The execution Profit and Loss.
        This value typically should be adjusted when adding to a position.
    realized_pnl : float
        The realized execution Profit and Loss.
        This value typically should be adjusted when adding to a position.
    cash_flow : float
        The execution cash flow (price times negative signed quantity).
    unrealized_quantity : float
        The unsigned unrealized quantity in units.
    unrealized_price_high : float
        The highest price of the unrealized quantity.
    unrealized_price_low : float
        The lowest price of the unrealized quantity.
    """
    def __init__(self,
                 datetime: Union[Real, dt.datetime, dt.date, Any],
                 side: ExecutionSide,
                 quantity: float,
                 price: float,
                 commission: float,
                 margin_per_unit: float = None):
        qty_abs = abs(quantity)
        qty_sign = -1.0 if side.is_sell() else 1.0
        amount_abs = price * qty_abs
        margin_abs = qty_abs * margin_per_unit if margin_per_unit is not None else 0.0
        debt = 0.0 if amount_abs == 0 else amount_abs - margin_abs
        
        self.datetime = datetime
        self.side = side
        self.quantity = qty_abs
        self.quantity_sign = qty_sign
        self.commission = commission
        self.commission_per_unit = commission / qty_abs if qty_abs != 0 else 0.0
        self.price = price
        self.amount = amount_abs
        self.margin = margin_abs
        self.debt = debt
        self.pnl = -commission # Will be updated when adding to position.
        self.realized_pnl = 0 # Will be updated when adding to position.
        self.cash_flow = -qty_sign * amount_abs
        self.unrealized_quantity = qty_abs
        self.unrealized_price_high = price
        self.unrealized_price_low = price
