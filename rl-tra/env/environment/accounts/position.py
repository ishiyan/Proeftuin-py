from enum import Enum
from typing import List, Tuple, Union, Any
from numbers import Real
import datetime as dt

from .performance_record import PerformanceRecord
from .execution_side import ExecutionSide
from .execution import Execution
from .roundtrips.matching import RoundtripMatching
from .roundtrips.roundtrip import Roundtrip
from .roundtrips.performance import RoundtripPerformance

class PositionSide(Enum):
    """Enumerates the sides of a position."""

    LONG = 'long'
    """The long position."""

    SHORT = 'short'
    """The short position."""

class Position(object):
    """
    Portfoloo position
    
    Parameters
    ----------
    margin_per_unit : Real
        The margin per unit of the position instrument.
        commission : Union[Real, Callable[[str, Real, Real], Real]]
        The commission per unit of the instrument.
    """
    def __init__(self,
                 margin_per_unit: Real = 0,
                 roundtrip_matching: RoundtripMatching = RoundtripMatching.FIFO):
        self.margin_per_unit = margin_per_unit
        self.roundtrip_matching = roundtrip_matching

        self.datetime = None
        self.side = PositionSide.LONG
        self.quantity_signed = 0
        self.average_price = None
        self.commission = 0
        self.roi = 0
        #######
        self.cash_flow = 0
        self.debt = 0
        self.margin = 0
        self.executions: List[Execution] = []
        self.roundtrip_performance = RoundtripPerformance()
        self.entry_amount = 0
        
    def reset(self):
        self.datetime = None
        self.side = PositionSide.LONG
        self.quantity_signed = 0
        self.average_price = None
        self.commission = 0
        self.roi = 0
        #######
        self.cash_flow = 0
        self.debt = 0
        self.margin = 0
        self.executions = []
        self.roundtrip_performance = RoundtripPerformance()
        self._entry_amount = 0
        
    def value(self, price):
        if (self.quantity_signed != 0) and (self.average_price != 0):
            self.roi = self.quantity_signed * (price - self.average_price) / abs(self.quantity_signed * self.average_price)
        else:
            self.roi = 0
        return self.quantity_signed * price
                
    def close(self,
                datetime: Union[Real, dt.datetime, dt.date, Any],
                price: Real,
                commission: Real = 0,
                notes: str = None) -> Tuple[PerformanceRecord, float]:
        if self.quantity_signed == 0:
            return None, 0

        self.commission += commission
        record = PerformanceRecord(
            operation = 1 if (self.quantity_signed > 0) else -1,
            amount = abs(self.quantity_signed),
            enter_date = self.datetime,
            enter_price = self.average_price,
            exit_date = datetime,
            exit_price = price,
            result = self.quantity_signed * (price - self.average_price) - self.commission,
            commission = self.commission,
            notes = notes
        )        
        side = ExecutionSide.SELL if (self.quantity_signed > 0) else ExecutionSide.BUY
        ex = Execution(datetime=datetime, side=side, quantity=abs(self.quantity_signed),
            price=price, commission=commission, margin_per_unit=self.margin_per_unit)
        rts = self._update_execution_pnl_and_match_roundtrips(ex, self.quantity_signed)
        for r in rts:
            self.roundtrip_performance.add_roundtrip(r)
        self.executions.append(ex)
        #self.quantity_signed = 0
        #_ = self._update_margin_and_debt(ex)

        cash_flow = self.quantity_signed * price - commission
        self.datetime = None
        self.quantity_signed = 0
        self.average_price = None
        self.commission = 0
        self.roi = 0
        self.margin = 0
        self.debt = 0
        return record, cash_flow # self.cash = self.cash + cash_flow

    def execute(self,
               datetime: Union[Real, dt.datetime, dt.date, Any],
               side: ExecutionSide,
               quantity: Real,
               price: Real,
               commission: Real = 0,
               notes: str = None) -> Tuple[PerformanceRecord, float]:
        """
        Update position by new operation.
        Each execution specifies datetime, side, quantity, price and commission.
        The position is updated according to the following scheme:
        ```
                            Execution quantity:
        Current quantity:  +2   +1   -1   -2
                      +2    I    I    D    C
                      +1    I    I    C    R
                       0    S    S    S    S
                      -1    R    C    I    I
                      -2    C    D    I    I

        S - new position
        I - increase current quantity
        D - decrease current quantity
        C - close current quantity
        R - revert current quantity (changing sign)
        ```
        Parameters
        ----------
        datetime : Union[Real, dt.datetime, dt.date, Any]
            The date and time of an associated execution report
            as assigned by the broker.

        :param operation: string 'B' - buy, or 'S' - sell
        :param quantity: positive number, > 0, may have floating point
        :param price: number, may have floating point
        :param commission: number, may have floating point
        :return: updated balance of account
        """
        assert isinstance(quantity, Real) and (quantity > 0), ValueError('Position:update: Invalid quantity')
        assert isinstance(price, Real), ValueError('Account:update: Invalid price')
        assert isinstance(commission, Real), ValueError('Account:update: Invalid commission')
        
        quantity_signed = quantity if (side == ExecutionSide.BUY) else -quantity
        new_quantity = self.quantity_signed + quantity_signed
        
        ex = Execution(datetime=datetime, side=side, quantity=quantity,
            price=price, commission=commission, margin_per_unit=self.margin_per_unit)

        if self.quantity_signed == 0: # NEW
            self.datetime = datetime
            #self.side = PositionSide.SHORT if quantity_signed < 0 else PositionSide.LONG
            self.quantity_signed = quantity_signed
            self.average_price = price
            self.commission += commission
            self.margin = ex.margin
            self.debt = ex.debt
            # ex.cash_flow = - quantity_signed * price
            self.cash_flow = ex.cash_flow - ex.commission
            self.entry_amount = ex.amount
            for e in self.executions:
                e.unrealized_quantity = 0
            self.executions.append(ex)
            return None, - quantity_signed * price - commission # ex.cash_flow - ex.commission

        elif new_quantity == 0: # CLOSE
            return self.close(datetime, price, commission, notes)
            
        elif self.quantity_signed * quantity_signed > 0: # INCREASE
            rts = self._update_execution_pnl_and_match_roundtrips(ex, self.quantity_signed)
            for r in rts:
                self.roundtrip_performance.add_roundtrip(r)
            self.quantity_signed = new_quantity
            _ = self._update_margin_and_debt(ex)
            self.executions.append(ex)
            self.cash_flow += ex.cash_flow - ex.commission
            self.average_price = (self.quantity_signed * self.average_price + quantity_signed * price) / new_quantity
            self.commission += commission
            return None, - quantity_signed * price - commission

        elif self.quantity_signed * new_quantity > 0: # DECREASE
            sign = 1 if (self.quantity_signed > 0) else -1
            record = PerformanceRecord(
                operation = sign,
                amount = quantity_signed,
                enter_date = self.datetime,
                enter_price = self.average_price,
                exit_date = datetime,
                exit_price = price,
                result = sign * quantity * (price - self.average_price) - commission,
                commission = commission,
                notes = notes
            )
            rts = self._update_execution_pnl_and_match_roundtrips(ex, self.quantity_signed)
            for r in rts:
                self.roundtrip_performance.add_roundtrip(r)
            self.quantity_signed = new_quantity
            _ = self._update_margin_and_debt(ex)
            self.executions.append(ex)
            # Note: we do not increase commission in this case!
            return record, - quantity_signed * price - commission

        elif self.quantity_signed * new_quantity < 0: # REVERTED
            report, cf = self.close(datetime, price, 0, notes) # ??? commission zero ???
            rts = self._update_execution_pnl_and_match_roundtrips(ex, self.quantity_signed)
            for r in rts:
                self.roundtrip_performance.add_roundtrip(r)
            self.quantity_signed = new_quantity
            _ = self._update_margin_and_debt(ex)
            self.executions.append(ex)
            self.datetime = datetime
            self.commission += commission
            self.average_price = price
            return report, cf - new_quantity * price - commission

        return None, 0

    def _update_execution_pnl_and_match_roundtrips(self,
        ex: Execution, qty_signed: float) -> List[Roundtrip]:
        """
        Updates the execution ex.pnl and ex.realized_pnl, and matches roundtrips
        for the given execution assuming the execution has not been appended
        to the history yet.
        """
        rts: List[Roundtrip] = []
        ex_sign = ex.quantity_sign
        ex_qty_left = ex.quantity
        commission_matched = 0.0
        amount_matched = 0.0

        if (qty_signed >= 0 and ex_sign < 0) or (qty_signed < 0 and ex_sign >= 0):
            # Execution and previous position have opposite sides.
            # Long position and sell execution or short position and buy execution.
            if self.roundtrip_matching == RoundtripMatching.FIFO:
                for e in self.executions:
                    if ex_qty_left <= 0:
                        break
                    # Skip if the full quantity has already been matched
                    # or execution sides are the same.
                    if (e.unrealized_quantity > 0) and (e.quantity_sign != ex_sign):
                        # Execution sides are opposite and
                        # there is an unmatched quantity.
                        min_qty = min(ex_qty_left, e.unrealized_quantity)
                        commission_matched += min_qty * \
                            (ex.commission_per_unit + e.commission_per_unit)
                        amount_matched += -ex_sign * min_qty * (ex.price - e.price)
                        ex_qty_left -= min_qty
                        rts.append(self._new_roundtrip(e, ex, min_qty))
            elif self.roundtrip_matching == RoundtripMatching.LIFO:
                for e in reversed(self.executions):
                    if ex_qty_left <= 0:
                        break
                    # Skip if the full quantity has already been matched
                    # or execution sides are the same.
                    if (e.unrealized_quantity > 0) and (e.quantity_sign != ex_sign):
                        # Execution sides are opposite and
                        # there is an unmatched quantity.
                        min_qty = min(ex_qty_left, e.unrealized_quantity)
                        commission_matched += min_qty * \
                            (ex.commission_per_unit + e.commission_per_unit)
                        amount_matched += -ex_sign * min_qty * (ex.price - e.price)
                        ex_qty_left -= min_qty
                        rts.append(self._new_roundtrip(e, ex, min_qty))
        ex.pnl += amount_matched
        ex.realized_pnl = amount_matched - commission_matched
        return rts

    def _new_roundtrip(self, entry: Execution, exit: Execution, qty: Real) -> Roundtrip:
        """Creates a new roundtrip for the given entry and exit executions."""
        entry.unrealized_quantity -= qty
        exit.unrealized_quantity -= qty
        return Roundtrip(entry, exit, qty)

    def _update_margin_and_debt(self, ex: Execution) -> float:
        """
        Updates the margin and debt and returns a signed increment
        in debt caused by this execution.

        Uses quantity_signed which should be updated for this execution.
        """
        if ex.margin == 0:
            return 0
        is_long = self.quantity_signed > 0
        is_short = self.quantity_signed < 0
        is_buy = ex.side.is_buy()
        is_sell = ex.side.is_sell()

        if (is_long and is_buy) or (is_short and is_sell):
            # Execution and updated position have the same directions.
            # Long position and buy execution or
            # short position and sell execution.
            self.margin += ex.margin
            self.debt += ex.debt
            return ex.debt
        elif (is_long and is_sell) or (is_short and is_buy):
            # Execution and updated position have opposite directions.
            # Long position and sell execution or
            # short position and buy execution.
            qty_diff = abs(self.quantity_signed) - ex.quantity
            if qty_diff > 0:
                # Executed less than updated position quantity.
                self.margin -= ex.margin
                delta = -self.debt * ex.quantity / abs(self.quantity_signed)
                self.debt += delta
                return delta
            elif qty_diff < 0:
                # Executed more than updated position quantity.
                self.margin = -qty_diff * self.margin_per_unit
                amt_diff = -qty_diff * ex.price - self.margin
                delta = amt_diff - self._debt
                self.debt = amt_diff
                return delta
            else:
                # Executed exactly the updated position quantity.
                self.margin = 0
                self.debt = 0
                return -ex.debt
        else:
            # Either order side or position side are unknown.
            return 0

    def __repr__(self):
        return (f'{self.__class__.__name__}('
            f'margin_per_unit={self.margin_per_unit}, '
            f'roundtrip_matching={self.roundtrip_matching})')

    def __str__(self):
        return (
            f'{self.__class__.__name__}{{'
            f'datetime={str(self.datetime)}, '
            f'quantity_signed={self.quantity_signed}, '
            f'average_price={self.average_price}, '
            f'commission={self.commission}, '
            f'roi={self.roi}, '
            f'cash_flow={self.cash_flow}, '
            f'debt={self.debt}, '
            f'margin={self.margin}}}')
