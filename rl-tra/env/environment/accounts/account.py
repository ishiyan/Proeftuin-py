from typing import Callable, Union, Any
from numbers import Real
import datetime as dt

from .execution_side import ExecutionSide
from .daycounting import DayCountConvention
from .performance import Performance
from .position import Position

class Account(object):
    def __init__(self,
                 initial_balance: Real = 10000,
                 annual_risk_free_rate: Real = 0.0,
                 annual_target_return: Real = 0.0,
                 day_count_convention: DayCountConvention = DayCountConvention.RAW):
        """
        Args:
            initial_balance real:
                Initial balance.
                Default: 10000
            annual_risk_free_rate Real:
                Annual rIsk-free rate (1% is 0.01).
                Default: 0.0
            annual_target_return Real:
                Annual rarget return (1% is 0.01).
                
                in context of Sortino ratio, it is the Minimum Acceptable
                Return (MAR, or Desired Target Return (DTR).
                Default: 0.0
            day_count_convention:
                Day count convention.
                Default: DayCountConvention.RAW
        """
        self.initial_balance = initial_balance
        self.annual_risk_free_rate = annual_risk_free_rate
        self.annual_target_return = annual_target_return
        self.day_count_convention = day_count_convention

        self.position = Position()
        self.performance = Performance(initial_balance=initial_balance,
            annual_risk_free_rate=annual_risk_free_rate,
            annual_target_return=annual_target_return,
            day_count_convention=day_count_convention)
        self.subscribers = {}
        self.cash = initial_balance
        self.balance = initial_balance
        self.is_halted = False
        
    def reset(self):
        self.position.reset()
        self.performance.reset()
        self.subscribers.clear()
        self.cash = self.initial_balance
        self.balance = self.initial_balance
        self.is_halted = False
        
    def subscribe(self, who: Any, callback: Callable[..., Any]):
        self.subscribers[who] = callback
        
    def unsubscribe(self, who: Any):
        del self.subscribers[who]
        
    def on_update(self, *args, **kwargs):
        for who, callback in self.subscribers.items():
            callback(*args, **kwargs)

    @property
    def has_no_position(self) -> bool:
        return self.position.quantity_signed == 0

    @property
    def has_position(self) -> bool:
        return self.position.quantity_signed != 0

    def update_performance(self, price_high: Real, price_low: Real,
        price_last: Real, datetime_last: dt.datetime,
        price_first: Real = None, datetime_first: dt.datetime = None):
        self.position.update_performance(
            price_high=price_high, price_low=price_low)
        self.balance = self.cash + self.position.value(price_last)
        self.performance.update(balance=self.balance, cash=self.cash,
            price_end_period=price_last, time_end_period=datetime_last,
            price_start_period=price_first, time_start_period=datetime_first)

    def update_balance(self, price: Real):
        self.position.update_performance(price_high=price, price_low=price)
        self.balance = self.cash + self.position.value(price)
                
    def close_position(self,
                       datetime: Union[Real, dt.datetime, dt.date, Any],
                       price: Real,
                       commission: Real = 0,
                       notes: str = None):
        rts, cash_flow = self.position.close(datetime = datetime, \
            price = price, commission = commission, notes = notes)
        self.cash += cash_flow
        self.update_balance(price)
        self.performance.add(rts)

    def execute(self,
               datetime: Union[Real, dt.datetime, dt.date, Any],
               operation: str,
               quantity: Real,
               price: Real,
               commission: Real = 0,
               notes: str = None):
        assert isinstance(operation, str) and (operation in 'BS'), \
            ValueError('Account:update: Invalid operation')
        assert isinstance(quantity, Real) and (quantity > 0), \
            ValueError(f'Invalid quantity {quantity}')
        assert isinstance(price, Real), \
            ValueError(f'Invalid price {price}')
        assert isinstance(commission, Real), \
            ValueError(f'Invalid commission {commission}')

        side = ExecutionSide.BUY if (operation == 'B') else ExecutionSide.SELL
        rts, cash_flow = self.position.execute(datetime = datetime, side = side, \
            quantity = quantity, price = price, commission = commission, notes = notes)
        
        self.cash += cash_flow
        self.update_balance(price)
        self.performance.add(rts)
        return self.balance
