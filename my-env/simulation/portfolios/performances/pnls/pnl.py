from datetime import datetime
from typing import List, Tuple

from ....utils import ScalarSeries

class PnL:
    """Contains Profit and Loss amount, unrealized amount, and percentage."""

    def __init__(self):
        self._amount: ScalarSeries = ScalarSeries()
        self._amount_unrealized: ScalarSeries = ScalarSeries()
        self._percentage: ScalarSeries = ScalarSeries()

    def amount(self) -> float:
        """
        Returns the current Profit and Loss amount
        (the value plus the cash flow).
        """
        return self._amount.current_value()

    def amount_history(self) -> List[Tuple[datetime, float]]:
        """
        Returns the Profit and Loss amount (the value plus the cash flow)
        history as a time series.
        """
        return self._amount.history()

    def unrealized_amount(self) -> float:
        """
        Returns the current unrealized Profit and Loss amount
        (the theoretical marked-to-market gain or loss on the
        open position(s) valued at current market price)
        """
        return self._amount_unrealized.current_value()

    def unrealized_amount_history(self) -> List[Tuple[datetime, float]]:
        """
        Returns the unrealized  Profit and Loss amount
        (the theoretical marked-to-market gain or loss on the
        open position(s) valued at current market price)
        history as a time series.
        """
        return self._amount_unrealized.history()

    def percentage(self) -> float:
        """
        Returns the current Profit and Loss percentage
        (the PnL amount divided by the initial amount, expressed in %).
        """
        return self._percentage.current_value()

    def percentage_history(self) -> List[Tuple[datetime, float]]:
        """
        Returns the Profit and Loss percentage
        (the PnL amount divided by the initial amount, expressed in %)
        history as a time series.
        """
        return self._percentage.history()

    def add(self, time: datetime, initial_amount: float, amount: float,
            unrealized_amount: float, cash_flow: float):
        """
        Adds a PnL record if the time is later the time of the last added record.

        Otherwise (if the time is earlier), the record will not be added.
        """
        pct = (amount + cash_flow) / initial_amount * 100.0 if initial_amount != 0.0 else 0.0

        self._amount.add(time, amount + cash_flow)
        self._amount_unrealized.add(time, unrealized_amount)
        self._percentage.add(time, pct)
