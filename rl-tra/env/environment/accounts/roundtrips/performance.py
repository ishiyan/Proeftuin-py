from itertools import groupby
from numbers import Real
from typing import List
import numpy as np

from .side import RoundtripSide
from .roundtrip import Roundtrip
from ..daycounting import DayCountConvention, year_frac

class RoundtripPerformance:
    """Calculates the roundtrip performance statistics."""

    def __init__(self,
        initial_balance: Real = 100000.0,
        annual_risk_free_rate: Real = 0.0,
        annual_target_return: Real = 0.0,
        day_count_convention: DayCountConvention = DayCountConvention.RAW):
        """
        Args:
            initial_balance real:
                Initial balance.
                Default: 100000.0
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

        self.roundtrips: List[Roundtrip] = []
        self.returns_on_investments = []
        self.sortino_downside_returns = []
        self.returns_on_investments_annual = []
        self.sortino_downside_returns_annual = []

        self.first_time = None
        self.last_time = None
        self.max_net_pnl: float = 0
        self.max_drawdown: float = 0
        self.max_drawdown_percent: float = 0

        self.total_commission: float = 0
        self.gross_winning_commission: float = 0
        self.gross_loosing_commission: float = 0
        self.net_winning_commission: float = 0
        self.net_loosing_commission: float = 0
        self.gross_winning_long_commission: float = 0
        self.gross_loosing_long_commission: float = 0
        self.net_winning_long_commission: float = 0
        self.net_loosing_long_commission: float = 0
        self.gross_winning_short_commission: float = 0
        self.gross_loosing_short_commission: float = 0
        self.net_winning_short_commission: float = 0
        self.net_loosing_short_commission: float = 0

        self._net_pnl: float = 0
        self._gross_pnl: float = 0
        self._gross_winning_pnl: float = 0
        self._gross_loosing_pnl: float = 0
        self._net_winning_pnl: float = 0
        self._net_loosing_pnl: float = 0
        self._gross_long_pnl: float = 0
        self._gross_short_pnl: float = 0
        self._net_long_pnl: float = 0
        self._net_short_pnl: float = 0
        self._gross_long_winning_pnl: float = 0
        self._gross_long_loosing_pnl: float = 0
        self._net_long_winning_pnl: float = 0
        self._net_long_loosing_pnl: float = 0
        self._gross_short_winning_pnl: float = 0
        self._gross_short_loosing_pnl: float = 0
        self._net_short_winning_pnl: float = 0
        self._net_short_loosing_pnl: float = 0

        self._total_count: int = 0
        self._long_count: int = 0
        self._short_count: int = 0
        self._gross_winning_count: int = 0
        self._gross_loosing_count: int = 0
        self._net_winning_count: int = 0
        self._net_loosing_count: int = 0
        self._gross_long_winning_count: int = 0
        self._gross_long_loosing_count: int = 0
        self._net_long_winning_count: int = 0
        self._net_long_loosing_count: int = 0
        self._gross_short_winning_count: int = 0
        self._gross_short_loosing_count: int = 0
        self._net_short_winning_count: int = 0
        self._net_short_loosing_count: int = 0

        self._duration_sec: float = 0
        self._duration_sec_long: float = 0
        self._duration_sec_short: float = 0
        self._duration_sec_gross_winning: float = 0
        self._duration_sec_gross_loosing: float = 0
        self._duration_sec_net_winning: float = 0
        self._duration_sec_net_loosing: float = 0
        self._duration_sec_gross_long_winning: float = 0
        self._duration_sec_gross_long_loosing: float = 0
        self._duration_sec_net_long_winning: float = 0
        self._duration_sec_net_long_loosing: float = 0
        self._duration_sec_gross_short_winning: float = 0
        self._duration_sec_gross_short_loosing: float = 0
        self._duration_sec_net_short_winning: float = 0
        self._duration_sec_net_short_loosing: float = 0
        self._total_duration_annualized: float = 0

        self._total_mae: float = 0
        self._total_mfe: float = 0
        self._total_eff: float = 0  
        self._total_eff_entry: float = 0  
        self._total_eff_exit: float = 0  

        self._roi_mean: float = None
        self._roi_std: float = None
        self._roi_tdd: float = None
        self._roiann_mean: float = None
        self._roiann_std: float = None
        self._roiann_tdd: float = None

    def reset(self):
        self.roundtrips.clear()
        self.returns_on_investments.clear()
        self.sortino_downside_returns.clear()
        self.returns_on_investments_annual.clear()
        self.sortino_downside_returns_annual.clear()

        self.first_time = None
        self.last_time = None
        self.max_net_pnl = 0
        self.max_drawdown = 0
        self.max_drawdown_percent = 0

        self.total_commission = 0
        self.gross_winning_commission = 0
        self.gross_loosing_commission = 0
        self.net_winning_commission = 0
        self.net_loosing_commission = 0
        self.gross_winning_long_commission = 0
        self.gross_loosing_long_commission = 0
        self.net_winning_long_commission = 0
        self.net_loosing_long_commission = 0
        self.gross_winning_short_commission = 0
        self.gross_loosing_short_commission = 0
        self.net_winning_short_commission = 0
        self.net_loosing_short_commission = 0

        self._net_pnl = 0
        self._gross_pnl = 0
        self._gross_winning_pnl = 0
        self._gross_loosing_pnl = 0
        self._net_winning_pnl = 0
        self._net_loosing_pnl = 0
        self._gross_long_pnl = 0
        self._gross_short_pnl = 0
        self._net_long_pnl = 0
        self._net_short_pnl = 0
        self._gross_long_winning_pnl = 0
        self._gross_long_loosing_pnl = 0
        self._net_long_winning_pnl = 0
        self._net_long_loosing_pnl = 0
        self._gross_short_winning_pnl = 0
        self._gross_short_loosing_pnl = 0
        self._net_short_winning_pnl = 0
        self._net_short_loosing_pnl = 0

        self._total_count = 0
        self._long_count = 0
        self._short_count = 0
        self._gross_winning_count = 0
        self._gross_loosing_count = 0
        self._net_winning_count = 0
        self._net_loosing_count = 0
        self._gross_long_winning_count = 0
        self._gross_long_loosing_count = 0
        self._net_long_winning_count = 0
        self._net_long_loosing_count = 0
        self._gross_short_winning_count = 0
        self._gross_short_loosing_count = 0
        self._net_short_winning_count = 0
        self._net_short_loosing_count = 0

        self._duration_sec = 0
        self._duration_sec_long = 0
        self._duration_sec_short = 0
        self._duration_sec_gross_winning = 0
        self._duration_sec_gross_loosing = 0
        self._duration_sec_net_winning = 0
        self._duration_sec_net_loosing = 0
        self._duration_sec_gross_long_winning = 0
        self._duration_sec_gross_long_loosing = 0
        self._duration_sec_net_long_winning = 0
        self._duration_sec_net_long_loosing = 0
        self._duration_sec_gross_short_winning = 0
        self._duration_sec_gross_short_loosing = 0
        self._duration_sec_net_short_winning = 0
        self._duration_sec_net_short_loosing = 0
        self._total_duration_annualized = 0

        self._total_mae = 0
        self._total_mfe = 0
        self._total_eff = 0  
        self._total_eff_entry = 0  
        self._total_eff_exit = 0  

        self._roi_mean = None
        self._roi_std = None
        self._roi_tdd = None
        self._roiann_mean = None
        self._roiann_std = None
        self._roiann_tdd = None

    def add_roundtrip(self, roundtrip: Roundtrip):
        """Adds a roundtrip to the performance tracker."""
        self.roundtrips.append(roundtrip)
        self._total_count += 1
        comm = roundtrip.commission
        self.total_commission += comm
        secs = roundtrip.duration.total_seconds()
        self._duration_sec += secs
        self._total_mae += roundtrip.maximum_adverse_excursion
        self._total_mfe += roundtrip.maximum_favorable_excursion
        self._total_eff += roundtrip.total_efficiency
        self._total_eff_entry += roundtrip.entry_efficiency
        self._total_eff_exit += roundtrip.exit_efficiency

        net_pnl = roundtrip.net_pnl
        self._net_pnl += net_pnl
        if net_pnl > 0:
            self._net_winning_count += 1
            self._net_winning_pnl += net_pnl
            self.net_winning_commission += comm
            self._duration_sec_net_winning += secs
        elif net_pnl < 0:
            self._net_loosing_count += 1
            self._net_loosing_pnl += net_pnl
            self.net_loosing_commission += comm
            self._duration_sec_net_loosing += secs

        gross_pnl = roundtrip.gross_pnl
        self._gross_pnl += gross_pnl
        if gross_pnl > 0:
            self._gross_winning_count += 1
            self._gross_winning_pnl += gross_pnl
            self.gross_winning_commission += comm
            self._duration_sec_gross_winning += secs
        elif gross_pnl < 0:
            self._gross_loosing_count += 1
            self._gross_loosing_pnl += gross_pnl
            self.gross_loosing_commission += comm
            self._duration_sec_gross_loosing += secs

        if roundtrip.side == RoundtripSide.LONG:
            self._gross_long_pnl += gross_pnl
            self._net_long_pnl += net_pnl
            self._long_count += 1
            self._duration_sec_long += secs
            if gross_pnl > 0:
                self._gross_long_winning_count += 1
                self._gross_long_winning_pnl += gross_pnl
                self.gross_winning_long_commission += comm
                self._duration_sec_gross_long_winning += secs
            elif gross_pnl < 0:
                self._gross_long_loosing_count += 1
                self._gross_long_loosing_pnl += gross_pnl
                self.gross_loosing_long_commission += comm
                self._duration_sec_gross_long_loosing += secs
            if net_pnl > 0:
                self._net_long_winning_count += 1
                self._net_long_winning_pnl += gross_pnl
                self.net_winning_long_commission += comm
                self._duration_sec_net_long_winning += secs
            elif net_pnl < 0:
                self._net_long_loosing_count += 1
                self._net_long_loosing_pnl += gross_pnl
                self.net_loosing_long_commission += comm
                self._duration_sec_net_long_loosing += secs
        else:
            self._gross_short_pnl += gross_pnl
            self._net_short_pnl += net_pnl
            self._short_count += 1
            self._duration_sec_short += secs
            if gross_pnl > 0:
                self._gross_short_winning_count += 1
                self._gross_short_winning_pnl += gross_pnl
                self.gross_winning_short_commission += comm
                self._duration_sec_gross_short_winning += secs
            elif gross_pnl < 0:
                self._gross_short_loosing_count += 1
                self._gross_short_loosing_pnl += gross_pnl
                self.gross_loosing_short_commission += comm
                self._duration_sec_gross_short_loosing += secs
            if net_pnl > 0:
                self._net_short_winning_count += 1
                self._net_short_winning_pnl += gross_pnl
                self.net_winning_short_commission += comm
                self._duration_sec_net_short_winning += secs
            elif net_pnl < 0:
                self._net_short_loosing_count += 1
                self._net_short_loosing_pnl += gross_pnl
                self.net_loosing_short_commission += comm
                self._duration_sec_net_short_loosing += secs

        # Update first/last times and duration
        changed = False
        if (self.first_time is None) or (self.first_time > roundtrip.entry_time):
            self.first_time = roundtrip.entry_time
            changed = True
        if (self.last_time is None) or (self.last_time < roundtrip.exit_time):
            self.last_time = roundtrip.exit_time
            changed = True
        if changed:
            self._total_duration_annualized = \
                year_frac(self.first_time, self.last_time,
                          self.day_count_convention)

        roi = net_pnl / (roundtrip.quantity * roundtrip.entry_price)
        self.returns_on_investments.append(roi)
        self._roi_mean = np.mean(self.returns_on_investments)
        self._roi_std = np.std(self.returns_on_investments, ddof=0) # ddof=1
        downside = roi - self.annual_risk_free_rate
        if downside < 0:
            self.sortino_downside_returns.append(downside)
            self._roi_tdd = np.sqrt(np.mean(np.power(
                self.sortino_downside_returns, 2)))

        # Calculate annualized return-on-investment (roiann)
        # Duration is in fractional years
        d = year_frac(roundtrip.entry_time, roundtrip.exit_time,
                      self.day_count_convention)
        if d != 0:
            roiann = roi / d
            self.returns_on_investments_annual.append(roiann)
            self._roiann_mean = np.mean(self.returns_on_investments_annual)
            self._roiann_std = np.std(self.returns_on_investments_annual, ddof=0) # ddof=1
            downside = roiann - self.annual_risk_free_rate
            if downside < 0:
                self.sortino_downside_returns_annual.append(downside)
                self._roiann_tdd = np.sqrt(np.mean(np.power(
                    self.sortino_downside_returns_annual, 2)))

        # Calculate max drawdown
        if self.max_net_pnl < self._net_pnl:
            self.max_net_pnl = self._net_pnl
        dd = self.max_net_pnl - self._net_pnl
        if self.max_drawdown < dd:
            self.max_drawdown = dd
            self.max_drawdown_percent = self.max_drawdown / \
                (self.initial_balance + self.max_net_pnl)

    @property
    def roi_mean(self):
        """Mean value for returns on investments"""
        return self._roi_mean

    @property
    def roi_std(self):
        """Standard deviation over returns on investments"""
        return self._roi_std

    @property
    def roi_tdd(self):
        """Target downside deviation over returns on investments"""
        return self._roi_tdd

    @property
    def roiann_mean(self):
        """Mean value for annualized returns on investments"""
        return self._roiann_mean
    
    @property
    def roiann_std(self):
        """Standard deviation over annualized returns on investments"""
        return self._roiann_std
    
    @property
    def roiann_tdd(self):
        """Target downside deviation over annualized returns on investments"""
        return self._roiann_tdd
    
    @property
    def sharpe_ratio(self):
        """Sharpe ratio over returns on investments"""
        if (self._roi_mean is None) or (self._roi_std is None) or (self._roi_std == 0):
            return None
        return self._roi_mean / self._roi_std

    @property
    def sharpe_ratio_annual(self):
        """Sharpe ratio over annualized returns on investments"""
        if (self._roiann_mean is None) or (self._roiann_std is None) or (self._roiann_std == 0):
            return None
        return self._roiann_mean / self._roiann_std
    
    @property
    def sortino_ratio(self):
        """Sortino ratio over returns on investments"""
        if (self._roi_mean is None) or (self._roi_tdd is None) or (self._roi_tdd == 0):
            return None
        return (self._roi_mean - self.annual_risk_free_rate) / self._roi_tdd

    @property
    def sortino_ratio_annual(self):
        """Sortino ratio over annualized returns on investments"""
        if (self._roiann_mean is None) or (self._roiann_tdd is None) or (self._roiann_tdd == 0):
            return None
        return (self._roiann_mean - self.annual_risk_free_rate) / self._roiann_tdd
    
    @property
    def calmar_ratio(self):
        """Calmar ratio over returns on investments"""
        if (self._roi_mean is None) or (self.max_drawdown_percent == 0):
            return None
        return self._roi_mean / self.max_drawdown_percent

    @property
    def calmar_ratio_annual(self):
        """Calmar ratio over annualized returns on investments"""
        if (self._roiann_mean is None) or (self.max_drawdown_percent == 0):
            return None
        return self._roiann_mean / self.max_drawdown_percent
    
    @property
    def rate_of_return(self):
        """Rate of return"""
        if self.initial_balance == 0:
            return None
        return self._net_pnl / self.initial_balance
    
    @property
    def rate_of_return_annual(self):
        """Annualized rate of return"""
        if (self._total_duration_annualized == 0) or (self.initial_balance == 0):
            return None
        return (self._net_pnl / self.initial_balance) / self._total_duration_annualized
    
    @property
    def recovery_factor(self):
        rorann = self.rate_of_return_annual
        if rorann is None or self.max_drawdown_percent == 0:
            return None
        return rorann / self.max_drawdown_percent
    
    @property
    def gross_profit_ratio(self): # profit_factor
        """Returns the PnL ratio of the gross winning roundtrips over the gross loosing roundtrips."""
        return abs(self._gross_winning_pnl / self._gross_loosing_pnl) \
            if (self._gross_loosing_pnl != 0) else None
    
    @property
    def net_profit_ratio(self):
        """Returns the PnL ratio of the net winning roundtrips over the net loosing roundtrips."""
        return abs(self._net_winning_pnl / self._net_loosing_pnl) \
            if (self._net_loosing_pnl != 0) else None

    @property
    def gross_profit_long_ratio(self):
        """Returns the PnL ratio of the long gross winning roundtrips over the long gross loosing roundtrips."""
        return abs(self._gross_long_winning_pnl / self._gross_long_loosing_pnl) \
            if (self._gross_long_loosing_pnl != 0) else None
    
    @property
    def net_profit_long_ratio(self):
        """Returns the PnL ratio of the long net winning roundtrips over the long net loosing roundtrips."""
        return abs(self._net_long_winning_pnl / self._net_long_loosing_pnl) \
            if (self._net_long_loosing_pnl != 0) else None

    @property
    def gross_profit_short_ratio(self):
        """Returns the PnL ratio of the short gross winning roundtrips over the short gross loosing roundtrips."""
        return abs(self._gross_short_winning_pnl / self._gross_short_loosing_pnl) \
            if (self._gross_short_loosing_pnl != 0) else None
    
    @property
    def net_profit_short_ratio(self):
        """Returns the PnL ratio of the short net winning roundtrips over the short net loosing roundtrips."""
        return abs(self._net_short_winning_pnl / self._net_short_loosing_pnl) \
            if (self._net_short_loosing_pnl != 0) else None

    @property
    def total_count(self) -> int:
        """Returns the total number of roundtrips."""
        return self._total_count

    @property
    def long_count(self) -> int:
        """Returns the number of long roundtrips."""
        return self._long_count

    @property
    def short_count(self) -> int:
        """Returns the number of short roundtrips."""
        return self._short_count

    @property
    def gross_winning_count(self) -> int:
        """Returns the number of gross winning roundtrips."""
        return self._gross_winning_count

    @property
    def gross_loosing_count(self) -> int:
        """Returns the number of gross loosing roundtrips."""
        return self._gross_loosing_count

    @property
    def net_winning_count(self) -> int:
        """Returns the number of net winning roundtrips."""
        return self._net_winning_count

    @property
    def net_loosing_count(self) -> int:
        """Returns the number of net loosing roundtrips."""
        return self._net_loosing_count

    @property
    def gross_long_winning_count(self) -> int:
        """Returns the number of long gross winning roundtrips."""
        return self._gross_long_winning_count

    @property
    def gross_long_loosing_count(self) -> int:
        """Returns the number of long gross loosing roundtrips."""
        return self._gross_long_loosing_count

    @property
    def net_long_winning_count(self) -> int:
        """Returns the number of long net winning roundtrips."""
        return self._net_long_winning_count

    @property
    def net_long_loosing_count(self) -> int:
        """Returns the number of long net loosing roundtrips."""
        return self._net_long_loosing_count

    @property
    def gross_short_winning_count(self) -> int:
        """Returns the number of short gross winning roundtrips."""
        return self._gross_short_winning_count

    @property
    def gross_short_loosing_count(self) -> int:
        """Returns the number of short gross loosing roundtrips."""
        return self._gross_short_loosing_count

    @property
    def net_short_winning_count(self) -> int:
        """Returns the number of short net winning roundtrips."""
        return self._net_short_winning_count

    @property
    def net_short_loosing_count(self) -> int:
        """Returns the number of short net loosing roundtrips."""
        return self._net_short_loosing_count

    @property
    def gross_winning_ratio(self) -> float:
        """Returns the ratio of gross winning roundtrips."""
        return self._gross_winning_count / self._total_count \
            if self._total_count > 0 else 0.0

    @property
    def gross_loosing_ratio(self) -> float:
        """Returns the ratio of gross loosing roundtrips."""
        return self._gross_loosing_count / self._total_count \
            if self._total_count > 0 else 0.0
    
    @property
    def net_winning_ratio(self) -> float:
        """Returns the ratio of net winning roundtrips."""
        return self._net_winning_count / self._total_count \
            if self._total_count > 0 else 0.0

    @property
    def net_loosing_ratio(self) -> float:
        """Returns the ratio of net loosing roundtrips."""
        return self._net_loosing_count / self._total_count \
            if self._total_count > 0 else 0.0
    
    @property
    def gross_long_winning_ratio(self) -> float:
        """Returns the ratio of long gross winning roundtrips."""
        return self._gross_long_winning_count / self._long_count \
            if self._long_count > 0 else 0.0

    @property
    def gross_long_loosing_ratio(self) -> float:
        """Returns the ratio of long gross loosing roundtrips."""
        return self._gross_long_loosing_count / self._long_count \
            if self._long_count > 0 else 0.0
    
    @property
    def net_long_winning_ratio(self) -> float:
        """Returns the ratio of long net winning roundtrips."""
        return self._net_long_winning_count / self._long_count \
            if self._long_count > 0 else 0.0
    
    @property
    def net_long_loosing_ratio(self) -> float:
        """Returns the ratio of long net loosing roundtrips."""
        return self._net_long_loosing_count / self._long_count \
            if self._long_count > 0 else 0.0
    
    @property
    def gross_short_winning_ratio(self) -> float:
        """Returns the ratio of short gross winning roundtrips."""
        return self._gross_short_winning_count / self._short_count \
            if self._short_count > 0 else 0.0

    @property
    def gross_short_loosing_ratio(self) -> float:
        """Returns the ratio of short gross loosing roundtrips."""
        return self._gross_short_loosing_count / self._short_count \
            if self._short_count > 0 else 0.0
    
    @property
    def net_short_winning_ratio(self) -> float:
        """Returns the ratio of short net winning roundtrips."""
        return self._net_short_winning_count / self._short_count \
            if self._short_count > 0 else 0.0
    
    @property
    def net_short_loosing_ratio(self) -> float:
        """Returns the ratio of short net loosing roundtrips."""
        return self._net_short_loosing_count / self._short_count \
            if self._short_count > 0 else 0.0

    @property
    def total_gross_pnl(self) -> float:
        """Returns the total gross profit and loss of all roundtrips."""
        return self._gross_pnl

    @property
    def total_net_pnl(self) -> float:
        """Returns the total net profit and loss of all roundtrips (taking commission into account)."""
        return self._net_pnl

    @property
    def winning_gross_pnl(self) -> float:
        """Returns the total gross profit of all gross winning roundtrips."""
        return self._gross_winning_pnl
    
    @property
    def loosing_gross_pnl(self) -> float:
        """Returns the total gross loss of all gross loosing roundtrips."""
        return self._gross_loosing_pnl
    
    @property
    def winning_net_pnl(self) -> float:
        """Returns the total net profit of all net winning roundtrips (taking commission into account)."""
        return self._net_winning_pnl
    
    @property
    def loosing_net_pnl(self) -> float:
        """Returns the total net loss of all net loosing roundtrips (taking commission into account)."""
        return self._net_loosing_pnl

    @property
    def winning_gross_long_pnl(self) -> float:
        """Returns the PnL of all long gross winning roundtrips."""
        return self._gross_long_winning_pnl
    
    @property
    def loosing_gross_long_pnl(self) -> float:
        """Returns the PnL of all long gross loosing roundtrips."""
        return self._gross_long_loosing_pnl

    @property
    def winning_net_long_pnl(self) -> float:
        """Returns the PnL of all long net winning roundtrips."""
        return self._net_long_winning_pnl
    
    @property
    def loosing_net_long_pnl(self) -> float:
        """Returns the PnL of all long net loosing roundtrips."""
        return self._net_long_loosing_pnl

    @property
    def winning_gross_short_pnl(self) -> float:
        """Returns the PnL of all short gross winning roundtrips."""
        return self._gross_short_winning_pnl
    
    @property
    def loosing_gross_short_pnl(self) -> float:
        """Returns the PnL of all short gross loosing roundtrips."""
        return self._gross_short_loosing_pnl

    @property
    def winning_net_short_pnl(self) -> float:
        """Returns the PnL of all short net winning roundtrips."""
        return self._net_short_winning_pnl
    
    @property
    def loosing_net_short_pnl(self) -> float:
        """Returns the PnL of all short net loosing roundtrips."""
        return self._net_short_loosing_pnl
    
    @property
    def average_gross_pnl(self) -> float:
        """Returns the average gross profit and loss of a roundtrip."""
        return self._gross_pnl / self._total_count \
            if self._total_count > 0 else 0.0
    
    @property
    def average_net_pnl(self) -> float: # average_trade_result
        """Returns the average net profit and loss of a roundtrip (taking commission into account)."""
        return self._net_pnl / self._total_count \
            if self._total_count > 0 else 0.0
    
    @property
    def average_gross_long_pnl(self) -> float:
        """Returns the average gross profit and loss of a long roundtrip."""
        return self._gross_long_pnl / self._long_count \
            if self._long_count > 0 else 0.0
    
    @property
    def average_net_long_pnl(self) -> float: # average_long_trade_result
        """Returns the average net profit and loss of a long roundtrip (taking commission into account)."""
        return self._net_long_pnl / self._long_count \
            if self._long_count > 0 else 0.0
    
    @property
    def average_gross_short_pnl(self) -> float:
        """Returns the average gross profit and loss of a short roundtrip."""
        return self._gross_short_pnl / self._short_count \
            if self._short_count > 0 else 0.0
    
    @property
    def average_net_short_pnl(self) -> float: # average_short_trade_result
        """Returns the average net profit and loss of a short roundtrip (taking commission into account)."""
        return self._net_short_pnl / self._short_count \
            if self._short_count > 0 else 0.0
    
    @property
    def average_winning_gross_pnl(self) -> float:
        """Returns the average gross profit of a winning roundtrip."""
        return self._gross_winning_pnl / self._gross_winning_count \
            if self._gross_winning_count > 0 else 0.0
    
    @property
    def average_loosing_gross_pnl(self) -> float:
        """Returns the average gross loss of a loosing roundtrip."""
        return self._gross_loosing_pnl / self._gross_loosing_count \
            if self._gross_loosing_count > 0 else 0.0
    
    @property
    def average_winning_net_pnl(self) -> float:
        """Returns the average net profit of a winning roundtrip (taking commission into account)."""
        return self._net_winning_pnl / self._net_winning_count \
            if self._net_winning_count > 0 else 0.0
    
    @property
    def average_loosing_net_pnl(self) -> float:
        """Returns the average net loss of a loosing roundtrip (taking commission into account)."""
        return self._net_loosing_pnl / self._net_loosing_count \
            if self._net_loosing_count > 0 else 0.0
            
    @property
    def average_winning_gross_long_pnl(self) -> float:
        """Returns the average gross profit of a gross winning long roundtrip."""
        return self._gross_long_winning_pnl / self._gross_long_winning_count \
            if self._gross_long_winning_count > 0 else 0.0
    
    @property
    def average_loosing_gross_long_pnl(self) -> float:
        """Returns the average gross loss of a gross loosing long roundtrip."""
        return self._gross_long_loosing_pnl / self._gross_long_loosing_count \
            if self._gross_long_loosing_count > 0 else 0.0
    
    @property
    def average_winning_net_long_pnl(self) -> float:
        """Returns the average net profit of a net winning long roundtrip (taking commission into account)."""
        return self._net_long_winning_pnl / self._net_long_winning_count \
            if self._net_long_winning_count > 0 else 0.0
    
    @property
    def average_loosing_net_long_pnl(self) -> float:
        """Returns the average net loss of a net loosing long roundtrip (taking commission into account)."""
        return self._net_long_loosing_pnl / self._net_long_loosing_count \
            if self._net_long_loosing_count > 0 else 0.0
            
    @property
    def average_winning_gross_short_pnl(self) -> float:
        """Returns the average gross profit of a gross winning short roundtrip."""
        return self._gross_short_winning_pnl / self._gross_short_winning_count \
            if self._gross_short_winning_count > 0 else 0.0
    
    @property
    def average_loosing_gross_short_pnl(self) -> float:
        """Returns the average gross loss of a gross loosing short roundtrip."""
        return self._gross_short_loosing_pnl / self._gross_short_loosing_count \
            if self._gross_short_loosing_count > 0 else 0.0
    
    @property
    def average_winning_net_short_pnl(self) -> float:
        """Returns the average net profit of a net winning short roundtrip (taking commission into account)."""
        return self._net_short_winning_pnl / self._net_short_winning_count \
            if self._net_short_winning_count > 0 else 0.0
    
    @property
    def average_loosing_net_short_pnl(self) -> float:
        """Returns the average net loss of a net loosing short roundtrip (taking commission into account)."""
        return self._net_short_loosing_pnl / self._net_short_loosing_count \
            if self._net_short_loosing_count > 0 else 0.0

    @property
    def average_gross_winning_loosing_ratio(self) -> float:
        """Returns the average ratio of a gross winning or loosing roundtrip."""
        w = self.average_winning_gross_pnl
        l = self.average_loosing_gross_pnl
        return w / l if l != 0 else 0.0

    @property
    def average_net_winning_loosing_ratio(self) -> float:
        """Returns the average ratio of a net winning or loosing roundtrip (taking commission into account)."""
        w = self.average_winning_net_pnl
        l = self.average_loosing_net_pnl
        return w / l if l != 0 else 0.0

    @property
    def average_gross_winning_loosing_long_ratio(self) -> float:
        """Returns the average ratio of a gross winning or loosing long roundtrip."""
        w = self.average_winning_gross_long_pnl
        l = self.average_loosing_gross_long_pnl
        return w / l if l != 0 else 0.0

    @property
    def average_net_winning_loosing_long_ratio(self) -> float:
        """Returns the average ratio of a net winning or loosing long roundtrip (taking commission into account)."""
        w = self.average_winning_net_long_pnl
        l = self.average_loosing_net_long_pnl
        return w / l if l != 0 else 0.0

    @property
    def average_gross_winning_loosing_short_ratio(self) -> float:
        """Returns the average ratio of a gross winning or loosing short roundtrip."""
        w = self.average_winning_gross_short_pnl
        l = self.average_loosing_gross_short_pnl
        return w / l if l != 0 else 0.0

    @property
    def average_net_winning_loosing_short_ratio(self) -> float:
        """Returns the average ratio of a net winning or loosing short roundtrip (taking commission into account)."""
        w = self.average_winning_net_short_pnl
        l = self.average_loosing_net_short_pnl
        return w / l if l != 0 else 0.0
    
    @property
    def gross_profit_pnl_ratio(self) -> float:
        """Returns the PnL ratio of the gross winning roundtrips over the all roundtrips."""
        return self._gross_winning_pnl / self._gross_pnl \
            if self._gross_pnl != 0 else 0.0

    @property
    def net_profit_pnl_ratio(self) -> float:
        """Returns the PnL ratio of the net winning roundtrips over the all roundtrips."""
        return self._net_winning_pnl / self._net_pnl \
            if self._net_pnl != 0 else 0.0
    
    @property
    def gross_profit_pnl_long_ratio(self) -> float:
        """Returns the PnL ratio of the long gross winning roundtrips over the all long roundtrips."""
        return self._gross_long_winning_pnl / self._gross_long_pnl \
            if self._gross_long_pnl != 0 else 0.0

    @property
    def net_profit_pnl_long_ratio(self) -> float:
        """Returns the PnL ratio of the long net winning roundtrips over the all long roundtrips."""
        return self._net_long_winning_pnl / self._net_long_pnl \
            if self._net_long_pnl != 0 else 0.0
    
    @property
    def gross_profit_pnl_short_ratio(self) -> float:
        """Returns the PnL ratio of the short gross winning roundtrips over the all short roundtrips."""
        return self._gross_short_winning_pnl / self._gross_short_pnl \
            if self._gross_short_pnl != 0 else 0.0

    @property
    def net_profit_pnl_short_ratio(self) -> float:
        """Returns the PnL ratio of the short net winning roundtrips over the all short roundtrips."""
        return self._net_short_winning_pnl / self._net_short_pnl \
            if self._net_short_pnl != 0 else 0.0

    @property
    def average_duration_seconds(self) -> float:
        """Returns the average duration of a roundtrip in seconds."""
        return self._duration_sec / self._total_count \
            if self._total_count > 0 else 0.0
    
    @property
    def average_gross_winning_duration_seconds(self) -> float:
        """Returns the average duration of a gross winning roundtrip in seconds."""
        return self._duration_sec_gross_winning / self._gross_winning_count \
            if self._gross_winning_count > 0 else 0.0
    
    @property
    def average_gross_loosing_duration_seconds(self) -> float:
        """Returns the average duration of a gross loosing roundtrip in seconds."""
        return self._duration_sec_gross_loosing / self._gross_loosing_count \
            if self._gross_loosing_count > 0 else 0.0
    
    @property
    def average_net_winning_duration_seconds(self) -> float:
        """Returns the average duration of a net winning roundtrip in seconds."""
        return self._duration_sec_net_winning / self._net_winning_count \
            if self._net_winning_count > 0 else 0.0
    
    @property
    def average_net_loosing_duration_seconds(self) -> float:
        """Returns the average duration of a net loosing roundtrip in seconds."""
        return self._duration_sec_net_loosing / self._net_loosing_count \
            if self._net_loosing_count > 0 else 0.0
    
    @property
    def average_long_duration_seconds(self) -> float:
        """Returns the average duration of a long roundtrip in seconds."""
        return self._duration_sec_long / self._long_count \
            if self._long_count > 0 else 0.0
    
    @property
    def average_short_duration_seconds(self) -> float:
        """Returns the average duration of a short roundtrip in seconds."""
        return self._duration_sec_short / self._short_count \
            if self._short_count > 0 else 0.0
    
    @property
    def average_gross_winning_long_duration_seconds(self) -> float:
        """Returns the average duration of a long gross winning roundtrip in seconds."""
        return self._duration_sec_gross_long_winning / self._gross_long_winning_count \
            if self._gross_long_winning_count > 0 else 0.0
    
    @property
    def average_gross_loosing_long_duration_seconds(self) -> float:
        """Returns the average duration of a long gross loosing roundtrip in seconds."""
        return self._duration_sec_gross_long_loosing / self._gross_long_loosing_count \
            if self._gross_long_loosing_count > 0 else 0.0
    
    @property
    def average_net_winning_long_duration_seconds(self) -> float:
        """Returns the average duration of a long net winning roundtrip in seconds."""
        return self._duration_sec_net_long_winning / self._net_long_winning_count \
            if self._net_long_winning_count > 0 else 0.0
    
    @property
    def average_net_loosing_long_duration_seconds(self) -> float:
        """Returns the average duration of a long net loosing roundtrip in seconds."""
        return self._duration_sec_net_long_loosing / self._net_long_loosing_count \
            if self._net_long_loosing_count > 0 else 0.0
    
    @property
    def average_gross_winning_short_duration_seconds(self) -> float:
        """Returns the average duration of a short gross winning roundtrip in seconds."""
        return self._duration_sec_gross_short_winning / self._gross_short_winning_count \
            if self._gross_short_winning_count > 0 else 0.0
    
    @property
    def average_gross_loosing_short_duration_seconds(self) -> float:
        """Returns the average duration of a short gross loosing roundtrip in seconds."""
        return self._duration_sec_gross_short_loosing / self._gross_short_loosing_count \
            if self._gross_short_loosing_count > 0 else 0.0
    
    @property
    def average_net_winning_short_duration_seconds(self) -> float:
        """Returns the average duration of a short net winning roundtrip in seconds."""
        return self._duration_sec_net_short_winning / self._net_short_winning_count \
            if self._net_short_winning_count > 0 else 0.0
    
    @property
    def average_net_loosing_short_duration_seconds(self) -> float:
        """Returns the average duration of a short net loosing roundtrip in seconds."""
        return self._duration_sec_net_short_loosing / self._net_short_loosing_count \
            if self._net_short_loosing_count > 0 else 0.0
    
    @property
    def minimum_duration_seconds(self) -> float:
        """Returns the minimum duration of a roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips])
    
    @property
    def maximum_duration_seconds(self) -> float:
        """Returns the maximum duration of a roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips])
    
    @property
    def minimum_long_duration_seconds(self) -> float:
        """Returns the minimum duration of a long roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips if r.side == RoundtripSide.LONG])
    
    @property
    def maximum_long_duration_seconds(self) -> float:
        """Returns the maximum duration of a long roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips if r.side == RoundtripSide.LONG])
    
    @property
    def minimum_short_duration_seconds(self) -> float:
        """Returns the minimum duration of a short roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips if r.side == RoundtripSide.SHORT])
    
    @property
    def maximum_short_duration_seconds(self) -> float:
        """Returns the maximum duration of a short roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips if r.side == RoundtripSide.SHORT])
    
    @property
    def minimum_gross_winning_duration_seconds(self) -> float:
        """Returns the minimum duration of a gross winning roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips if r.gross_pnl > 0])
    
    @property
    def maximum_gross_winning_duration_seconds(self) -> float:
        """Returns the maximum duration of a gross winning roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips if r.gross_pnl > 0])
    
    @property
    def minimum_gross_loosing_duration_seconds(self) -> float:
        """Returns the minimum duration of a gross loosing roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips if r.gross_pnl < 0])
    
    @property
    def maximum_gross_loosing_duration_seconds(self) -> float:
        """Returns the maximum duration of a gross loosing roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips if r.gross_pnl < 0])
    
    @property
    def minimum_net_winning_duration_seconds(self) -> float:
        """Returns the minimum duration of a net winning roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips if r.net_pnl > 0])
    
    @property
    def maximum_net_winning_duration_seconds(self) -> float:
        """Returns the maximum duration of a net winning roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips if r.net_pnl > 0])
    
    @property
    def minimum_net_loosing_duration_seconds(self) -> float:
        """Returns the minimum duration of a net loosing roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips if r.net_pnl < 0])
    
    @property
    def maximum_net_loosing_duration_seconds(self) -> float:
        """Returns the maximum duration of a net loosing roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips if r.net_pnl < 0])
    
    @property
    def minimum_gross_winning_long_duration_seconds(self) -> float:
        """Returns the minimum duration of a long gross winning roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips \
                    if r.gross_pnl > 0 and r.side == RoundtripSide.LONG])
    
    @property
    def maximum_gross_winning_long_duration_seconds(self) -> float:
        """Returns the maximum duration of a long gross winning roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips \
                    if r.gross_pnl > 0 and r.side == RoundtripSide.LONG])
    
    @property
    def minimum_gross_loosing_long_duration_seconds(self) -> float:
        """Returns the minimum duration of a long gross loosing roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips \
                    if r.gross_pnl < 0 and r.side == RoundtripSide.LONG])
    
    @property
    def maximum_gross_loosing_long_duration_seconds(self) -> float:
        """Returns the maximum duration of a long gross loosing roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips \
                    if r.gross_pnl < 0 and r.side == RoundtripSide.LONG])
    
    @property
    def minimum_net_winning_long_duration_seconds(self) -> float:
        """Returns the minimum duration of a long net winning roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips \
                    if r.net_pnl > 0 and r.side == RoundtripSide.LONG])
    
    @property
    def maximum_net_winning_long_duration_seconds(self) -> float:
        """Returns the maximum duration of a long net winning roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips \
                    if r.net_pnl > 0 and r.side == RoundtripSide.LONG])
    
    @property
    def minimum_net_loosing_long_duration_seconds(self) -> float:
        """Returns the minimum duration of a long net loosing roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips \
                    if r.net_pnl < 0 and r.side == RoundtripSide.LONG])
    
    @property
    def maximum_net_loosing_long_duration_seconds(self) -> float:
        """Returns the maximum duration of a long net loosing roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips \
                    if r.net_pnl < 0 and r.side == RoundtripSide.LONG])
    
    @property
    def minimum_gross_winning_short_duration_seconds(self) -> float:
        """Returns the minimum duration of a short gross winning roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips \
                    if r.gross_pnl > 0 and r.side == RoundtripSide.SHORT])
    
    @property
    def maximum_gross_winning_short_duration_seconds(self) -> float:
        """Returns the maximum duration of a short gross winning roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips \
                    if r.gross_pnl > 0 and r.side == RoundtripSide.SHORT])
    
    @property
    def minimum_gross_loosing_short_duration_seconds(self) -> float:
        """Returns the minimum duration of a short gross loosing roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips \
                    if r.gross_pnl < 0 and r.side == RoundtripSide.SHORT])
    
    @property
    def maximum_gross_loosing_short_duration_seconds(self) -> float:
        """Returns the maximum duration of a short gross loosing roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips \
                    if r.gross_pnl < 0 and r.side == RoundtripSide.SHORT])
    
    @property
    def minimum_net_winning_short_duration_seconds(self) -> float:
        """Returns the minimum duration of a short net winning roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips \
                    if r.net_pnl > 0 and r.side == RoundtripSide.SHORT])
    
    @property
    def maximum_net_winning_short_duration_seconds(self) -> float:
        """Returns the maximum duration of a short net winning roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips \
                    if r.net_pnl > 0 and r.side == RoundtripSide.SHORT])
    
    @property
    def minimum_net_loosing_short_duration_seconds(self) -> float:
        """Returns the minimum duration of a short net loosing roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips \
                    if r.net_pnl < 0 and r.side == RoundtripSide.SHORT])
    
    @property
    def maximum_net_loosing_short_duration_seconds(self) -> float:
        """Returns the maximum duration of a short net loosing roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips \
                    if r.net_pnl < 0 and r.side == RoundtripSide.SHORT])

    @property
    def average_maximum_adverse_excursion(self) -> float:
        """Returns the average maximum adverse excursion of all roundtrips in percentage."""
        return self._total_mae / self._total_count if self._total_count > 0 else 0.0
    
    @property
    def average_maximum_favorable_excursion(self) -> float:
        """Returns the average maximum favorable excursion of all roundtrips in percentage."""
        return self._total_mfe / self._total_count if self._total_count > 0 else 0.0
    
    @property
    def average_entry_efficiency(self) -> float:
        """Returns the average entry efficiency of all roundtrips in percentage."""
        return self._total_eff_entry / self._total_count if self._total_count > 0 else 0.0
    
    @property
    def average_exit_efficiency(self) -> float:
        """Returns the average exit efficiency of all roundtrips in percentage."""
        return self._total_eff_exit / self._total_count if self._total_count > 0 else 0.0
    
    @property
    def average_total_efficiency(self) -> float:
        """Returns the average total efficiency of all roundtrips in percentage."""
        return self._total_eff / self._total_count if self._total_count > 0 else 0.0

    @property
    def average_maximum_adverse_excursion_gross_winning(self) -> float:
        """Returns the average maximum adverse excursion of all gross winning roundtrips in percentage."""
        return sum([r.maximum_adverse_excursion for r in self.roundtrips if r.gross_pnl > 0]) / \
            self._gross_winning_count if self._gross_winning_count > 0 else 0.0

    @property
    def average_maximum_adverse_excursion_gross_loosing(self) -> float:
        """Returns the average maximum adverse excursion of all gross loosing roundtrips in percentage."""
        return sum([r.maximum_adverse_excursion for r in self.roundtrips if r.gross_pnl < 0]) / \
            self._gross_loosing_count if self._gross_loosing_count > 0 else 0.0

    @property
    def average_maximum_adverse_excursion_net_winning(self) -> float:
        """Returns the average maximum adverse excursion of all net winning roundtrips in percentage."""
        return sum([r.maximum_adverse_excursion for r in self.roundtrips if r.net_pnl > 0]) / \
            self._net_winning_count if self._net_winning_count > 0 else 0.0

    @property
    def average_maximum_adverse_excursion_net_loosing(self) -> float:
        """Returns the average maximum adverse excursion of all net loosing roundtrips in percentage."""
        return sum([r.maximum_adverse_excursion for r in self.roundtrips if r.net_pnl < 0]) / \
            self._net_loosing_count if self._net_loosing_count > 0 else 0.0

    @property
    def average_maximum_favorable_excursion_gross_winning(self) -> float:
        """Returns the average maximum favorable excursion of all gross winning roundtrips in percentage."""
        return sum([r.maximum_favorable_excursion for r in self.roundtrips if r.gross_pnl > 0]) / \
            self._gross_winning_count if self._gross_winning_count > 0 else 0.0

    @property
    def average_maximum_favorable_excursion_gross_loosing(self) -> float:
        """Returns the average maximum favorable excursion of all gross loosing roundtrips in percentage."""
        return sum([r.maximum_favorable_excursion for r in self.roundtrips if r.gross_pnl < 0]) / \
            self._gross_loosing_count if self._gross_loosing_count > 0 else 0.0

    @property
    def average_maximum_favorable_excursion_net_winning(self) -> float:
        """Returns the average maximum favorable excursion of all net winning roundtrips in percentage."""
        return sum([r.maximum_favorable_excursion for r in self.roundtrips if r.net_pnl > 0]) / \
            self._net_winning_count if self._net_winning_count > 0 else 0.0

    @property
    def average_maximum_favorable_excursion_net_loosing(self) -> float:
        """Returns the average maximum favorable excursion of all net loosing roundtrips in percentage."""
        return sum([r.maximum_favorable_excursion for r in self.roundtrips if r.net_pnl < 0]) / \
            self._net_loosing_count if self._net_loosing_count > 0 else 0.0

    @property
    def average_entry_efficiency_gross_winning(self) -> float:
        """Returns the average entry efficiency of all gross winning roundtrips in percentage."""
        return sum([r.entry_efficiency for r in self.roundtrips if r.gross_pnl > 0]) / \
            self._gross_winning_count if self._gross_winning_count > 0 else 0.0

    @property
    def average_entry_efficiency_gross_loosing(self) -> float:
        """Returns the average entry efficiency of all gross loosing roundtrips in percentage."""
        return sum([r.entry_efficiency for r in self.roundtrips if r.gross_pnl < 0]) / \
            self._gross_loosing_count if self._gross_loosing_count > 0 else 0.0

    @property
    def average_entry_efficiency_net_winning(self) -> float:
        """Returns the average entry efficiency of all net winning roundtrips in percentage."""
        return sum([r.entry_efficiency for r in self.roundtrips if r.net_pnl > 0]) / \
            self._net_winning_count if self._net_winning_count > 0 else 0.0

    @property
    def average_entry_efficiency_net_loosing(self) -> float:
        """Returns the average entry efficiency of all net loosing roundtrips in percentage."""
        return sum([r.entry_efficiency for r in self.roundtrips if r.net_pnl < 0]) / \
            self._net_loosing_count if self._net_loosing_count > 0 else 0.0

    @property
    def average_exit_efficiency_gross_winning(self) -> float:
        """Returns the average exit efficiency of all gross winning roundtrips in percentage."""
        return sum([r.exit_efficiency for r in self.roundtrips if r.gross_pnl > 0]) / \
            self._gross_winning_count if self._gross_winning_count > 0 else 0.0

    @property
    def average_exit_efficiency_gross_loosing(self) -> float:
        """Returns the average exit efficiency of all gross loosing roundtrips in percentage."""
        return sum([r.exit_efficiency for r in self.roundtrips if r.gross_pnl < 0]) / \
            self._gross_loosing_count if self._gross_loosing_count > 0 else 0.0

    @property
    def average_exit_efficiency_net_winning(self) -> float:
        """Returns the average exit efficiency of all net winning roundtrips in percentage."""
        return sum([r.exit_efficiency for r in self.roundtrips if r.net_pnl > 0]) / \
            self._net_winning_count if self._net_winning_count > 0 else 0.0

    @property
    def average_exit_efficiency_net_loosing(self) -> float:
        """Returns the average exit efficiency of all net loosing roundtrips in percentage."""
        return sum([r.exit_efficiency for r in self.roundtrips if r.net_pnl < 0]) / \
            self._net_loosing_count if self._net_loosing_count > 0 else 0.0

    @property
    def average_total_efficiency_gross_winning(self) -> float:
        """Returns the average total efficiency of all gross winning roundtrips in percentage."""
        return sum([r.total_efficiency for r in self.roundtrips if r.gross_pnl > 0]) / \
            self._gross_winning_count if self._gross_winning_count > 0 else 0.0

    @property
    def average_total_efficiency_gross_loosing(self) -> float:
        """Returns the average total efficiency of all gross loosing roundtrips in percentage."""
        return sum([r.total_efficiency for r in self.roundtrips if r.gross_pnl < 0]) / \
            self._gross_loosing_count if self._gross_loosing_count > 0 else 0.0

    @property
    def average_total_efficiency_net_winning(self) -> float:
        """Returns the average total efficiency of all net winning roundtrips in percentage."""
        return sum([r.total_efficiency for r in self.roundtrips if r.net_pnl > 0]) / \
            self._net_winning_count if self._net_winning_count > 0 else 0.0

    @property
    def average_total_efficiency_net_loosing(self) -> float:
        """Returns the average total efficiency of all net loosing roundtrips in percentage."""
        return sum([r.total_efficiency for r in self.roundtrips if r.net_pnl < 0]) / \
            self._net_loosing_count if self._net_loosing_count > 0 else 0.0

    @property
    def max_consecutive_gross_winners(self) -> int:
        """Returns the maximum number of consecutive gross winners."""
        return max([len(list(g)) for k, g in \
            groupby([r.gross_pnl > 0 for r in self.roundtrips]) if k], default=0)
    
    @property
    def max_consecutive_gross_loosers(self) -> int:
        """Returns the maximum number of consecutive gross looser."""
        return max([len(list(g)) for k, g in \
            groupby([r.gross_pnl < 0 for r in self.roundtrips]) if k], default=0)
    
    @property
    def max_consecutive_net_winners(self) -> int:
        """Returns the maximum number of consecutive net winners."""
        return max([len(list(g)) for k, g in \
            groupby([r.net_pnl > 0 for r in self.roundtrips]) if k], default=0)
    
    @property
    def max_consecutive_net_loosers(self) -> int:
        """Returns the maximum number of consecutive net looser."""
        return max([len(list(g)) for k, g in \
            groupby([r.net_pnl < 0 for r in self.roundtrips]) if k], default=0)
