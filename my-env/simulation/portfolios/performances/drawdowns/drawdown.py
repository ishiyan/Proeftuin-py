from datetime import datetime
import math
from typing import List, Tuple

from ....utils import ScalarSeries

class Drawdown:
    """Contains drawdown amount, percentage and their maximal values."""

    def __init__(self):
        self._watermark: ScalarSeries = ScalarSeries()
        self._amount: ScalarSeries = ScalarSeries()
        self._percentage: ScalarSeries = ScalarSeries()
        self._amount_max: ScalarSeries = ScalarSeries()
        self._percentage_max: ScalarSeries = ScalarSeries()

    def watermark(self) -> float:
        """
        Returns the current high watermark amount in positive values.
        """
        return self._watermark.current_value()

    def watermark_history(self) -> List[Tuple[datetime, float]]:
        """
        Returns the high watermark amount history as a time series.
        """
        return self._watermark.history()

    def amount(self) -> float:
        """
        Returns the current drawdown amount (in negative values).
        """
        return self._amount.current_value()

    def amount_history(self) -> List[Tuple[datetime, float]]:
        """
        Returns the drawdown amount (in negative values)
        history as a time series.
        """
        return self._amount.history()

    def percentage(self) -> float:
        """
        Returns the current drawdown percentage (in range [-100, 0]).
        """
        return self._percentage.current_value()

    def percentage_history(self) -> List[Tuple[datetime, float]]:
        """
        Returns the drawdown percentage (in range [-100, 0])
        history as a time series.
        """
        return self._percentage.history()

    def max_amount(self) -> float:
        """
        Returns the current maximal drawdown amount
        (the minimal negative historical value until now).
        """
        return self._amount_max.current_value()

    def max_amount_history(self) -> List[Tuple[datetime, float]]:
        """
        Returns the maximal drawdown amount
        (the minimal negative historical value before every sample)
        history as a time series.
        """
        return self._amount_max.history()

    def max_percentage(self) -> float:
        """
        Returns the current maximal drawdown percentage
        (the minimal negative historical value until now)
        in range [-100, 0].
        """
        return self._percentage_max.current_value()

    def max_percentage_history(self) -> List[Tuple[datetime, float]]:
        """
        Returns the maximal drawdown percentage
        (the minimal negative historical value before every sample, in range [-100, 0])
        history as a time series.
        """
        return self._percentage_max.history_value()

    def add(self, time: datetime, value: float):
        """
        Adds a drawdown record if the time is later the time of the last added record.

        Otherwise (if the time is earlier), the record will not be updated.
        """
        c = self._watermark.current()
        if c is None:
            self._watermark.add(time, value)
            return
        elif c[0] >= time:
            return
        elif c[1] < value:
            self._watermark.add(time, value)

        v = self._watermark.current_value()
        if v != 0.0:
            a = min(value - v, 0)
            p = min(a / v, 0)
        else:
            a = 0.0
            p = 0.0

        c = self._amount.current()
        if c is None:
            self._amount.add(time, a)
            self._percentage.add(time, p)
            self._amount_max.add(time, a)
            self._percentage_max.add(time, p)
        elif c[0] < time:
            self._amount.add(time, a)
            self._percentage.add(time, p)
            self._amount_max.add(time, min(a,
                self._amount_max.current_value()))
            self._percentage_max.add(time, min(a,
                self._percentage_max.current_value()))
