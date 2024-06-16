from typing import Callable, Union, Any
from numbers import Real
import datetime as dt

from .execution_side import ExecutionSide
from .day_count_convention import DayCountConvention
from .performance_record import PerformanceRecord
from .performance import Performance
from .position import Position

class Account(object):
    def __init__(self,
                 initial_balance: Real = 10000,
                 risk_free_rate: Real = 0.0,
                 convention: DayCountConvention = DayCountConvention.RAW):
        self.initial_balance = initial_balance

        self.position = Position()
        self.cash = initial_balance
        self.balance = initial_balance
        self.max_balance = initial_balance
        self.min_balance = initial_balance
        self.max_drawdown = 0
        self.report = Performance(initial_balance=initial_balance, risk_free_rate=risk_free_rate, convention=convention)
        self.subscribers = {}
        self.is_halted = False
        
    def reset(self):
        self.position.reset()
        self.cash = self.initial_balance
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.min_balance = self.initial_balance
        self.max_drawdown = 0
        self.report.reset()
        self.subscribers.clear()
        
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
        
    def update_balance(self, price):
        self.balance = self.cash + self.position.value(price)

        if self.max_balance < self.balance:
            self.max_balance = self.balance
            self.min_balance = self.balance

        if self.min_balance > self.balance:
            self.min_balance = self.balance

            drawdown = (self.max_balance - self.min_balance)    
            if self.max_drawdown < drawdown:
                self.max_drawdown = drawdown
                
    def close_position(self,
                       datetime: Union[Real, dt.datetime, dt.date, Any],
                       price: Real,
                       commission: Real = 0,
                       notes: str = None):
        record, cash_flow = self.position.close(datetime = datetime, \
            price = price, commission = commission, notes = notes)
        if record is None:
            return
        self.cash += cash_flow
        self.update_balance(price)
        self.report.add(record)

    def update(self,
               datetime: Union[Real, dt.datetime, dt.date, Any],
               operation: str,
               quantity: Real,
               price: Real,
               commission: Real = 0,
               notes: str = None):
        assert isinstance(operation, str) and (operation in 'BS'), ValueError('Account:update: Invalid operation')
        assert isinstance(quantity, Real), ValueError('IVAN: Invalid quantity')
        assert isinstance(price, Real), ValueError(f'IVAN: Invalid price {price}')

        side = ExecutionSide.BUY if (operation == 'B') else ExecutionSide.SELL
        record, cash_flow = self.position.execute(datetime = datetime, side = side, \
            quantity = quantity, price = price, commission = commission, notes = notes)
        
        self.cash += cash_flow
        self.update_balance(price)
        if record is not None:
            self.report.add(record)
        return self.balance

    def __repr__(self):
        return f'{self.__class__.__name__}(initial_balance={self.initial_balance})'
    
    def __str__(self):
        return (
            f'{self.__class__.__name__}{{'
            f'cash={self.cash}, '
            f'balance={str(self.positibalanceon_datetime)}, '
            f'max_balance={self.max_balance}, '
            f'min_balance={self.min_balance}, '
            f'max_drawdown={self.max_drawdown}}}')
