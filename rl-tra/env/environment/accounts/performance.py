from typing import List
from numbers import Real
import datetime as dt

from .roundtrips import Roundtrip, RoundtripPerformance
from .daycounting import DayCountConvention
from .performances import Ratios, Periodicity

MAX_ELEMS = 500

class Performance(object):
    def __init__(self,
        initial_balance: Real = 100000,
        annual_risk_free_rate: Real = 0.,
        annual_target_return: Real = 0.,
        day_count_convention: DayCountConvention = DayCountConvention.RAW):
        """
        Args:
            initial_balance Real:
                Initial balance.
                Default: 10000
            annual_risk_free_rate Real:
                Annual rIsk-free rate (1% is 0.01).
                Default: 0.0
            annual_target_return Real:
                Annual raRget return (1% is 0.01).
                
                in context of Sortino ratio, it is the Minimum Acceptable
                Return (MAR, or Desired Target Return (DTR).
                Default: 0.0
            day_count_convention:
                Day count convention.
                Default: DayCountConvention.RAW
        """
        self.initial_balance = initial_balance
        
        self.roundtrips = RoundtripPerformance(
            initial_balance=initial_balance,
            annual_risk_free_rate=annual_risk_free_rate,
            annual_target_return=annual_target_return,
            day_count_convention=day_count_convention)

        self.daily = Ratios(
            periodicity=Periodicity.DAILY,
            annual_risk_free_rate=annual_risk_free_rate,
            annual_target_return=annual_target_return,
            day_count_convention=day_count_convention)

        self.times: List[dt.datetime] = []
        self.balances: List[float] = []
        self.cash: List[float] = []
        self._last_balance: float = None
        self._last_price: float = None
        self._last_time: dt.datetime = None

    def reset(self):
        self.roundtrips.reset()
        self.daily.reset()
        self.times.clear()
        self.balances.clear()
        self.cash.clear()
        self._last_balance = self.initial_balance
        self._last_price = None
        self._last_time = None
    
    def add(self, rts: List[Roundtrip]):
        for rt in rts:
            self.roundtrips.add_roundtrip(rt)
    
    def update(self, balance: float, cash: float,
        price_end_period: float, time_end_period: dt.datetime,
        price_start_period: float = None,
        time_start_period: dt.datetime = None):

        if time_start_period is None:
            time_start_period = self._last_time
        if price_start_period is None:
            price_start_period = self._last_price
        self._last_time = time_end_period
        self._last_price = price_end_period

        if time_start_period is None:
            # Called the very first time
            # without specifying the start time.
            return
        if price_start_period is None:
            price_start_period = price_end_period

        self.times.append(time_end_period)
        self.balances.append(balance)
        self.cash.append(cash)
        ret_pf = balance / self._last_balance  - 1
        ret_bm = price_end_period / price_start_period  - 1
        self._last_balance = balance

        self.daily.add_return(return_=ret_pf, return_benchmark=ret_bm,
            value=balance, time_start=time_start_period,
            time_end=time_end_period)
    
    @property
    def rate_of_return(self):
        """Rate of return"""
        if self.initial_balance == 0:
            return None
        return self._last_balance / self.initial_balance
    
    def __repr(self):
        return (
            f'{self.__class__.__name__}('
            f'initial_balance={self.initial_balance})'
        )
    
    def __str__(self):
        return '\n'.join([f'{str(k)}: {str(v)}' for (k, v) in self.__dict__.items()])
