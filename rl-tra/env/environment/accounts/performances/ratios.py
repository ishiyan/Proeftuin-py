from datetime import datetime
from numbers import Real
import warnings
import numpy as np
from scipy.stats import kurtosis, skew

from ..daycounting import DayCountConvention, frac
from .periodicity import Periodicity

_SQRT2 = 1.4142135623730950488016887242097

class Ratios:
    """
    Various financial ratios to evaluate the performance of a strategy.
    """
    def __init__(self,
        periodicity: Periodicity = Periodicity.DAILY,
        annual_risk_free_rate: float = 0.,
        annual_target_return: float = 0., # TARGET RETURN !!!!
        day_count_convention: DayCountConvention = DayCountConvention.RAW):
        """
        Args:
            annual_risk_free_rate float:
                Annual risk-free rate.
                Default: 0.0
            annual_trading_days int:
                Annual trading days.
                Default: 252
            day_count_convention DayCountConvention:
                Day count convention.
                Default: DayCountConvention.RAW
        """
        self.periodicity = periodicity
        periods_per_annum = 252 if periodicity == Periodicity.DAILY \
            else 52 if periodicity == Periodicity.WEEKLY \
            else 12 if periodicity == Periodicity.MONTHLY \
            else 4 if periodicity == Periodicity.QUARTERLY \
            else 1
        self.periods_per_annum = periods_per_annum
        self.days_per_period = 1 if periodicity == Periodicity.DAILY \
            else 252 / 52 if periodicity == Periodicity.WEEKLY \
            else 252 / 12 if periodicity == Periodicity.MONTHLY \
            else 252 / 4 if periodicity == Periodicity.QUARTERLY \
            else 252

        self.risk_free_rate = annual_risk_free_rate \
            if annual_risk_free_rate == 0 or periods_per_annum == 1 \
            else ((1 + annual_risk_free_rate) ** (1/periods_per_annum) - 1)

        self.required_return = annual_target_return \
            if annual_target_return == 0 or periods_per_annum == 1 \
            else ((1 + annual_target_return) ** (1/periods_per_annum) - 1)
        self.day_count_convention = day_count_convention

        self.fractional_periods: np.ndarray = None
        self.returns: np.ndarray = None
        self._logret_sum: float = None
        self._drawdowns_cumulative: np.ndarray = None
        self._drawdowns_cumulative_min: float = None
        self._drawdowns_peaks: np.ndarray = None
        self._drawdowns_peaks_peak: int = None
        self._drawdown_continuous: np.ndarray = None
        self._drawdown_continuous_final: np.ndarray = None
        self._drawdown_continuous_finalized: bool = None
        self._drawdown_continuous_peak: int = None
        self._drawdown_continuous_inside: bool = None
        self._cumulative_return_plus_1: float = None
        self._cumulative_return_plus_1_100: float = None
        self._cumulative_return_geometric_mean: float = None
        self._cumulative_return_plus_1_max: float = None
        self._returns_mean: float = None
        self._returns_std: float = None
        self._returns_autocorr_penalty: float = None
        self._rf_mean: float = None
        self._rf_std: float = None
        self._excess_mean: float = None
        self._excess_std: float = None
        self._excess_autocorr_penalty: float = None
        self._required_mean: float = None
        self._required_lpm_1: float = None
        self._required_lpm_2: float = None
        self._required_lpm_3: float = None
        self._required_hpm_1: float = None
        self._required_hpm_2: float = None
        self._required_hpm_3: float = None
        self._required_autocorr_penalty: float = None
        self._avg_return: float = None
        self._avg_win: float = None
        self._avg_loss: float = None
        self._win_rate: float = None
        self._total_duration: float = None

    def reset(self):
        self.fractional_periods = np.array([])
        self.returns = np.array([])
        self._logret_sum = 0
        self._drawdowns_cumulative = np.array([])
        self._drawdowns_cumulative_min = np.inf
        self._drawdowns_peaks = np.array([])
        self._drawdowns_peaks_peak = 0
        self._drawdown_continuous = np.array([])
        self._drawdown_continuous_final = np.array([])
        self._drawdown_continuous_finalized = False
        self._drawdown_continuous_peak = 1
        self._drawdown_continuous_inside = False
        self._cumulative_return_plus_1 = 1
        self._cumulative_return_plus_1_100 = 1
        self._cumulative_return_plus_1_max = -np.inf
        self._total_duration = 0

    def add_return(self,
                   return_: float,
                   return_benchmark: float,
                   value: float,
                   time_start: datetime,
                   time_end: datetime):
        if self.periodicity == Periodicity.ANNUAL:
            fractional_period = frac(time_start, time_end,
                self.day_count_convention, False)
        else:
            fractional_period = frac(time_start, time_end,
                self.day_count_convention, True) / self.days_per_period
        #fractional_period = frac(time_start, time_end,
        #    self.day_count_convention, True) / self.days_per_period

        self.fractional_periods = np.append(self.fractional_periods, fractional_period)
        if fractional_period == 0:
            print('Zero fractional time period, perfomance not updated')
            return
        self._total_duration += fractional_period ### DO SMTH WITH IT

        # Normalized returns
        ret = return_ / fractional_period
        self.returns = np.append(self.returns, ret)
        l = len(self.returns)
        self._returns_mean = np.mean(self.returns)
        self._returns_std = \
            np.std(self.returns, ddof=1) if l > 1 else None
        self._returns_autocorr_penalty = self._autocorr_penalty(self.returns)

        tmp1 = self.returns[self.returns != 0]
        len1 = len(tmp1)
        self._avg_return = tmp1.mean() if len1 > 0 else None
        tmp2 = self.returns[self.returns > 0]
        len2 = len(tmp2)
        self._win_rate = len2 / len1 if len1 > 0 else None
        self._avg_win = tmp2.mean() if len2 > 0 else None
        tmp2 = self.returns[self.returns < 0]
        len2 = len(tmp2)
        self._avg_loss = tmp2.mean() if len2 > 0 else None

        # Excess returns (returns less risk-free rate)
        if self.risk_free_rate == 0:
            self._excess_mean = self._returns_mean
            self._excess_std = self._returns_std
            self._excess_autocorr_penalty = self._returns_autocorr_penalty
        else:
            tmp2 = self.returns - self.risk_free_rate
            self._excess_mean = np.mean(tmp2)
            self._excess_std = np.std(tmp2, ddof=1) if l > 1 else None
            self._excess_autocorr_penalty = self._autocorr_penalty(tmp2)

        # Lower partial moments for the raw returns (less required return)
        if self.required_return == 0:
            tmp2 = -self.returns
        else:
            tmp2 = self.required_return - self.returns
        # Set the minimum of each to 0
        tmp2 = tmp2.clip(min=0)
        # Calculate the sum of the excess returns to the power of order
        self._required_lpm_1 = np.sum(tmp2) / l
        self._required_lpm_2 = np.sum(tmp2 ** 2) / l
        self._required_lpm_3 = np.sum(tmp2 ** 3) / l

        # Higher partial moments for the raw returns (less required return)
        if self.required_return == 0:
            tmp2 = self.returns
            self._required_mean = self._returns_mean
            self._required_autocorr_penalty = self._returns_autocorr_penalty
        else:
            # Calculate the difference between the returns and the threshold
            tmp2 = self.returns - self.required_return
            self._required_mean = np.mean(tmp2)
            self._required_autocorr_penalty = self._autocorr_penalty(tmp2)
        # Set the minimum of each to 0
        tmp2 = tmp2.clip(min=0)
        # Calculate the sum of the excess returns to the power of order
        self._required_hpm_1 = np.sum(tmp2) / l
        self._required_hpm_2 = np.sum(tmp2 ** 2) / l
        self._required_hpm_3 = np.sum(tmp2 ** 3) / l

        # Cumulative returns
        retlog = np.log(return_ + 1) / fractional_period
        self._logret_sum += retlog
        ret1 = ret + 1
        if l == 1:
            cmr = ret1
            self._cumulative_return_plus_1 = ret1
            self._cumulative_return_geometric_mean = ret
        else:
            prev = (self._cumulative_return_geometric_mean + 1) ** (l - 1)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    cmr = prev * ret1
            except RuntimeWarning as e:
                print('RuntimeWarning mult cum ret: prev', prev, '* ret1', ret1, '->', cmr, 'steps', l, 'periodicity', self.periodicity)
                print('RuntimeWarning mult cum ret message:', e)
            cmr = np.exp(self._logret_sum)
            self._cumulative_return_plus_1 = cmr
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    self._cumulative_return_geometric_mean = pow(cmr, 1 / l) - 1
            except RuntimeWarning as e:
                print('RuntimeWarning pow cum ret: cmr', cmr, '** 1/l', 1/l, 'gm', self._cumulative_return_geometric_mean, 'steps', l, 'periodicity', self.periodicity)
                print('RuntimeWarning pow cum ret message:', e)
        if self._cumulative_return_plus_1_max < cmr:
            self._cumulative_return_plus_1_max = cmr

        # Drawdowns from peaks to valleys,
        # operates on cumulative returns.
        dd = cmr / self._cumulative_return_plus_1_max - 1
        if self._drawdowns_cumulative_min > dd:
            self._drawdowns_cumulative_min = dd
        self._drawdowns_cumulative = np.append(self._drawdowns_cumulative, dd)
        # Different drawdown calculation used in pain index, ulcer index
        dd = 1
        for j in range(self._drawdowns_peaks_peak + 1, l):
            dd *= (1 + self.returns[j] * 0.01)
        if dd > 1:
            self._drawdowns_peaks_peak = l - 1
            self._drawdowns_peaks = np.append(self._drawdowns_peaks, 0)
        else:
            self._drawdowns_peaks = np.append(self._drawdowns_peaks, (dd - 1) * 100)
        # Drawdown calculation used in Burke
        if l > 1:
            self._drawdown_continuous_finalized = False
            if ret < 0:
                if not self._drawdown_continuous_inside:
                    self._drawdown_continuous_inside = True
                    self._drawdown_continuous_peak = l - 2
                self._drawdown_continuous = np.append(self._drawdown_continuous, 0)
            else:
                if self._drawdown_continuous_inside:
                    dd = 1
                    j1 = self._drawdown_continuous_peak + 1
                    for j in range(j1, l-1):
                        dd = dd * (1 + self.returns[j] * 0.01)
                    self._drawdown_continuous = np.append(self._drawdown_continuous, (dd - 1) * 100)
                    self._drawdown_continuous_inside = False
                else:
                    self._drawdown_continuous = np.append(self._drawdown_continuous, 0)

    def _autocorr_penalty(self, returns) -> float:
        """Metric to account for auto correlation"""
        #num = len(returns)
        #if num < 3:
        #    return 1
        #try:
        #    with warnings.catch_warnings():
        #        warnings.simplefilter("error", RuntimeWarning)
        #        coef = np.abs(np.corrcoef(returns[:-1], returns[1:])[0, 1])
        #except RuntimeWarning as e:
        #    #print('RuntimeWarning autocorr_penalty returns:', returns)
        #    #print('RuntimeWarning autocorr_penalty message:', e)
        #    return 1
        #corr = [((num - x) / num) * coef**x for x in range(1, num)]
        #return np.sqrt(1 + 2 * np.sum(corr))
        return 1
    
    @property
    def cumulative_return(self):
        """Cumulative geometric returns"""
        return self._cumulative_return_plus_1 - 1
    
    @property
    def drawdowns_cumulative(self):
        """
        Drawdowns from peaks to valleys on cumulative geometric returns.
        """
        return self._drawdowns_cumulative
    
    @property
    def min_drawdowns_cumulative(self):
        """
        The minimum value of the drawdowns from peaks to valleys
        on cumulative geometric returns.
        """
        return self._drawdowns_cumulative_min
    
    @property
    def worst_drawdowns_cumulative(self):
        """
        The absolute value of the minimum value
        of the drawdowns from peaks to valleys
        on cumulative geometric returns.
        """
        return abs(self._drawdowns_cumulative_min)
    
    def drawdowns_peaks(self):
        """
        Drawdowns from peaks to valleys.
        """
        return self._drawdowns_peaks

    def drawdowns_continuous(self, peaks_only: bool = False, max_peaks: int = None):
        """
        Drawdowns on any continuous, uninterrupted losing return region.

        On every end of such uninterrupted negative return region,
        drawdown is the cumulative return ovr the region.

        Otherwise, drawdown is zero.
        
        Used in Burke ratio.

        Args:
            peaks_only bool:
                Return only the peaks, not zeroes.
                Default: False
            max_peaks int:
                Maximum number of peaks to return.
                Has no effect if `peaks_only` is False.
                Default: None
        """
        def finalize_calculation():
            if not self._drawdown_continuous_finalized:
                if self._drawdown_continuous_inside:
                    dd = 1
                    j1 = self._drawdown_continuous_peak + 1
                    for j in range(j1, len(self.returns)):
                        dd = dd * (1 + self.returns[j] * 0.01)
                    self._drawdown_continuous_final = np.append(self._drawdown_continuous,
                        (dd - 1) * 100)
                else:
                    self._drawdown_continuous_final = np.append(self._drawdown_continuous, 0)
                self._drawdown_continuous_finalized = True
        finalize_calculation()
        if not peaks_only:
            return self._drawdown_continuous_final
        drawdowns = self._drawdown_continuous_final[self._drawdown_continuous_final != 0]
        if max_peaks is not None:
            if len(drawdowns) > 0:
                drawdowns = np.sort(drawdowns)
                drawdowns = drawdowns[:max_peaks]
        return drawdowns

    @property
    def skew(self):
        """
        Calculates returns' skewness
        (the degree of asymmetry of a distribution around its mean)
        """
        return skew(self.returns) if len(self.returns) > 1 else None

    @property
    def kurtosis(self):
        """
        Calculates returns' kurtosis
         (the degree to which a distribution peak compared to a normal distribution)
        """
        return kurtosis(self.returns) if len(self.returns) > 1 else None

    # https://www.alternativesoft.com/the-difference-between-the-Sharpe-ratio-and-the-Smart-Sharpe-Ratio.html
    def sharpe_ratio(self,
        ignore_risk_free_rate: bool = False,
        autocorrelation_penalty: bool = False) -> float:
        """
        Ex post Sharpe ratio over excess or raw returns,
        with or without autocorrelation penalty.
        
        Args:
            ignore_risk_free_rate bool:
                Ignore the risk-free rate.
                
                If True, the ratio is calculated over raw returns.
                Sometimes this is called the "risk-return ratio".

                If False, the ratio is calculated over excess returns.
                Default: False
            autocorrelation_penalty bool:
                Apply autocorrelation penalty.
                Default: False
        """
        if ignore_risk_free_rate:
            if (self._returns_mean is None) or \
                (self._returns_std is None) or (self._returns_std == 0):
                return None
            denominator = self._returns_std
            if autocorrelation_penalty:
                denominator *= self._returns_autocorr_penalty
            return self._returns_mean / denominator
        else:
            if (self._excess_mean is None) or \
                (self._excess_std is None) or (self._excess_std == 0):
                return None
            denominator = self._excess_std
            if autocorrelation_penalty:
                denominator *= self._excess_autocorr_penalty
            return self._excess_mean / denominator

    def sortino_ratio(self,
        autocorrelation_penalty: bool = False,
        divide_by_sqrt2: bool = False) -> float:
        """
        Sortino ratio over excess returns,
        with or without autocorrelation penalty.

        Excess returns are returns less the required return.

        Note that the Sortino ratio doesn't use the risk-free rate.
        
        Args:
            autocorrelation_penalty bool:
                Apply autocorrelation penalty.
                Default: False
            divide_by_sqrt2 bool:
                Divide by the square root of 2.

                This Jack Schwager's version of the Sortino ratio
                allows for direct comparisons to the Sharpe.
                
                See here for more info: https://archive.is/wip/2rwFW                
                Default: False
        """
        if (self._required_mean is None) or \
            (self._required_lpm_2 is None) or (self._required_lpm_2 == 0):
            return None
        denominator = np.sqrt(self._required_lpm_2)
        if autocorrelation_penalty:
            denominator *= self._required_autocorr_penalty
        if divide_by_sqrt2:
            denominator *= _SQRT2
        return self._required_mean / denominator

    def omega_ratio(self):
        """
        Omega ratio over normalized returns
        """
        #if (self._required_hpm_1 is None) or \
        if (self._required_mean is None) or \
            (self._required_lpm_1 is None) or (self._required_lpm_1 == 0):
            return None

        #return self._required_hpm_1 / self._required_lpm_1
        return self._required_mean / self._required_lpm_1 + 1

    def kappa_ratio(self, order: int = 3):
        """
        Kappa ratio over normalized returns
        """
        if (self._required_mean is None):
            return None
        if order == 1:
            if (self._required_lpm_1 is None) or (self._required_lpm_1 == 0):
                return None
            return self._required_mean / self._required_lpm_1
        elif order == 2:
            if (self._required_lpm_2 is None) or (self._required_lpm_2 == 0):
                return None
            return self._required_mean / np.sqrt(self._required_lpm_2)
        elif order == 3:
            if (self._required_lpm_3 is None) or (self._required_lpm_3 == 0):
                return None
            return self._required_mean / (self._required_lpm_3 ** (1/3))
        else:
            if self.required_return == 0:
                tmp = -self.returns
            else:
                tmp = self.required_return - self.returns
            tmp = tmp.clip(min=0)
            lpm = np.sum(tmp ** order) / len(self.returns)
            if (lpm is None) or (lpm == 0):
                return None
            return self._required_mean / (lpm ** (1/order))

    def kappa3_ratio(self, order: int = 3):
        """
        Kappa order 3 ratio over normalized returns
        """
        if (self._required_mean is None) or \
            (self._required_lpm_3 is None) or (self._required_lpm_3 == 0):
            return None
        return self._required_mean / (self._required_lpm_3 ** (1/3))

    def bernardo_ledoit_ratio(self):
        """
        Bernardo and Ledoit ratio over normalized returns
        """
        l = len(self.returns)
        if l < 1:
            return None
        tmp = -self.returns
        tmp = tmp.clip(min=0)
        lpm_1 = np.sum(tmp) / l
        if lpm_1 is None or lpm_1 == 0:
            return None 
        tmp = self.returns.clip(min=0)
        hpm_1 = np.sum(tmp) / l
        return hpm_1 / lpm_1

    def upside_potential_ratio(self, full : bool = True):
        """
        The upside-potential ratio over normalized returns
        """
        if full:
            if (self._required_hpm_1 is None) or \
                (self._required_lpm_2 is None) or (self._required_lpm_2 == 0):
                return None
            return self._required_hpm_1 / np.sqrt(self._required_lpm_2)
        else:
            tmp = self.returns[self.returns < self.required_return]
            l = len(tmp)
            if l < 1:
                return None
            tmp = tmp - self.required_return
            lpm_2 = np.sum(tmp ** 2) / l
            if lpm_2 is None or lpm_2 == 0:
                return None
            tmp = self.returns[self.returns > self.required_return]
            if len(tmp) == 0:
                return None
            tmp = tmp - self.required_return
            #hpm_1 = np.sum(tmp) / l if l > 0 else None
            hpm_1 = np.mean(tmp)
            return hpm_1 / np.sqrt(lpm_2)

    def compound_growth_rate(self):
        """
        Compound (annual) growth rate (CAGR), or the geometric mean of the returns.
        """
        return self._cumulative_return_geometric_mean
    
    def calmar_ratio(self):
        """
        Calmar ratio over normalized returns
        """
        wdd = self.worst_drawdowns_cumulative
        if wdd == 0:
            return None
        cagr = self._cumulative_return_geometric_mean
        if cagr is None:
            return None
        return cagr / wdd
    
    def sterling_ratio(self, annual_excess_rate: float = 0):
        """
        Steling ratio over normalized returns

        Args:
            annual_excess_rate float:
                Annual excess rate to add to maximum drawdown.
                Default: 0.1 (10%)
        """
        #excess_rate = annual_excess_rate if self.is_annual \
        #    else ((1 + annual_excess_rate) ** (1/252) - 1)
        excess_rate = annual_excess_rate \
            if annual_excess_rate == 0 or self.periods_per_annum == 1 \
            else ((1 + annual_excess_rate) ** (1/self.periods_per_annum) - 1)

        wdd = self.worst_drawdowns_cumulative + excess_rate
        if wdd == 0:
            return None
        cagr = self._cumulative_return_geometric_mean
        if cagr is None:
            return None
        return cagr / wdd

    def burke_ratio(self, modified: bool = False):
        """
        Burke ratio of the return distribution.

        Args:
            modified bool:
                Which ratio to calculate, Burke ratio or modified Burke ratio.
                Default: False
        """
        rate = self._cumulative_return_geometric_mean - self.risk_free_rate
        if rate is None:
            return None
        drawdowns = self.drawdowns_continuous(peaks_only=True)
        if len(drawdowns) < 1:
            return None
        sqrt_sum_drawdowns_squared = np.sqrt(np.sum(np.square(drawdowns)))
        if sqrt_sum_drawdowns_squared == 0:
            return None
        burke = rate / sqrt_sum_drawdowns_squared
        if modified:
            burke *= np.sqrt(len(self.returns))
        return burke
  
    def pain_index(self):
        """
        Pain index over normalized returns
        """
        l = len(self._drawdowns_peaks)
        if l < 1:
            return None
        # By calculation, all values are <= 0, so we don't need abs()
        return -np.sum(self._drawdowns_peaks) / l

    def pain_ratio(self):
        """
        Pain ratio over normalized returns
        """
        rate = self._cumulative_return_geometric_mean - self.risk_free_rate
        if rate is None:
            return None
        l = len(self._drawdowns_peaks)
        if l < 1:
            return None
        # By calculation, all values are <= 0, so we don't need abs()
        pain_index = -np.sum(self._drawdowns_peaks) / l
        return (rate / pain_index) if pain_index != 0 else None

    def ulcer_index(self):
        """
        Ulcer index over normalized returns
        """
        l = len(self._drawdowns_peaks)
        if l < 1:
            return None
        ulcer_index = np.sqrt(np.sum(np.square(self._drawdowns_peaks)) / l)
        return ulcer_index
    
    def martin_ratio(self):
        """
        Ulcer ratio over normalized returns
        """
        rate = self._cumulative_return_geometric_mean - self.risk_free_rate
        if rate is None:
            return None
        l = len(self._drawdowns_peaks)
        if l < 1:
            return None
        ulcer_index = np.sqrt(np.sum(np.square(self._drawdowns_peaks)) / l)
        return (rate / ulcer_index) if ulcer_index != 0 else None

    ########################################

    @property
    def gain_to_pain_ratio(self):
        """
        Jack Schwager's GPR. See here for more info:
        https://archive.is/wip/2rwFW
        """
        # Note LPM is always positive, otherwise we have to do abs() around it
        #downside = self._required_lpm_1 * len(self.returns)
        #return (self.returns.sum() / downside) if downside != 0 else None
        return (self._returns_mean / self._required_lpm_1) \
            if self._required_lpm_1 != 0 else None

    @property
    def risk_of_ruin(self):
        """
        Calculates the risk of ruin
        (the likelihood of losing all one's investment capital)
        """
        wr = self._win_rate
        return ((1 - wr) / (1 + wr)) ** len(self.returns)
    

    @property
    def risk_return_ratio(self): # !!! DELETE
        """
        Calculates the return / risk ratio
        (Sharpe ratio without factoring in the risk-free rate)
        """
        if (self._returns_mean is None) or \
            (self._returns_std is None) or (self._returns_std == 0):
            return None
        return self._returns_mean / self._returns_std


# https://github.com/webclinic017/cryptoTrendFollowing/blob/main/main.py
#def annualising_factor(series):
#    return round(365 / ((np.diff(df.index.values).mean().astype(float) * 1e-9)/(24*3600)))

# https://github.com/sprksh/finance-calculator/blob/master/src/finance_calculator/calculators/ratio_calculator.py

# https://github.com/laholmes/risk-adjusted-return/blob/e0865a586d466a52ab1e549d46778e2aac0f0b0f/app.py#L336
# this one is good without pandas
# https://www.turingfinance.com/computational-investing-with-python-week-one/

# https://github.com/midas-research/profit-naacl

# https://github.com/AI-ApeX-DeV/TradeBot-USDT_BTC/blob/main/Metrics.txt


# https://www.pm-research.com/content/iijinvest/3/3/59
# Sortino, F.A.; Price, L.N. (1994). "Performance measurement in a downside risk framework". Journal of Investing. 3 (3): 50–8. doi:10.3905/joi.3.3.59
#
# "Sortino: A 'Sharper' Ratio" (PDF). Red Rock Capital. Retrieved February 16, 2014.
# http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
#
# https://en.wikipedia.org/wiki/Rate_of_return
# According to the CFA Institute's Global Investment Performance Standards (GIPS),[3]
#"Returns for periods of less than one year must not be annualized."
# https://www.cfainstitute.org/en/membership/professional-development/refresher-readings/gips-overview
# Overview of the Global Investment Performance Standards "GIPS Standards"
#
# https://github.com/Peter-Staadecker/Lions-and-Tigers-and-Sortinos-Oh-My/blob/main/tiingo%20analysis%20multi-yr%20monthly%20v4%20%20-%20API%20key%20blank.py
# min_acceptable_return_in_period
# ------------------ calculate Sortino ratio =  (stock return - minimum acceptable return)/ downside std dev
# note 1. returns are % growth, not growth ratio
# note 2. downside std dev counts all periods both above and below target in the std dev denominator,
#   i.e. the zero values in the downside are not thrown away. For emphasis on this point see
#   e.g. http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
# note 3. I use a geometric average for average growth in calculating the numerator of the Sortino ratio.
#   Some may prefer an arithmetic ratio. Based on some quick tests the difference is likely minor.
# note 4. I use std. deviation for the population, not for a sample. Again the differences are likely minor.
# note 5. There are however, major differences in ratios depending on whether the data frequency is daily, weekly
#   monthly or yearly. As a result, it seems that the ratios (up/down market capture and Sortino) are more
#   useful for comparison between stocks when measured with the same data frequency, rather than as an absolute measure.
#   The beta ratio seems least influenced by frequency.
#
# downsideStdDevAnnlzd: float = downsideStdDevPeriod * (PeriodsInYr ** 0.5)  # <----////////adjust if needed
#
#if dataFreqStr == "weekly":
#    PeriodsInYr = 52
#elif dataFreqStr == "monthly":
#    PeriodsInYr = 12
#elif dataFreqStr == "annually":
#    PeriodsInYr = 1
#elif dataFreqStr == "daily":
#    PeriodsInYr = 253  # using an average
#
# https://github.com/kunnn1/Quant-Calc/blob/main/quant_calc.py
# def calculate_sharpe_ratio(returns, risk_free_rate, trading_days=252):
#   excess_returns = returns - risk_free_rate / trading_days
#    sharpe_ratio = np.sqrt(trading_days) * excess_returns.mean() / excess_returns.std()
#    return sharpe_ratio
# def calculate_sortino_ratio(returns, risk_free_rate, trading_days=252):
#    downside_returns = returns[returns < 0]
#    excess_returns = returns - risk_free_rate / trading_days
#    sortino_ratio = np.sqrt(trading_days) * excess_returns.mean() / downside_returns.std()
#    return sortino_ratio
# def calculate_max_drawdown(returns):
#    cumulative_returns = (1 + returns).cumprod()
#    peak = cumulative_returns.cummax()
#    drawdown = (cumulative_returns - peak) / peak
#    max_drawdown = drawdown.min()
#    return max_drawdown
#
#
