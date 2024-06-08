from datetime import datetime
from typing import List, Tuple

from ...currencies import Currency
from ...utils import ScalarSeries
from ...instruments import Instrument
from ...orders import OrderExecution, OrderSide
from ...accounts import Account
from ..performances import RoundtripMatching, Roundtrip, Performance
from .side import PositionSide

class PortfolioPosition:
    """Portfolio position.
    """
    
    def __init__(self, instrument: Instrument,
        execution: OrderExecution,
        account: Account,
        roundtrip_matching: RoundtripMatching):

        self._instrument = instrument
        self._price_factor = instrument.price_factor if instrument.price_factor else 1
        self._roundtrip_matching = roundtrip_matching
        self._executions: List[OrderExecution] = []
        self._amounts: ScalarSeries = ScalarSeries()
        self._performance: Performance = Performance()
        self._quantity_bought: float = 0
        self._quantity_sold: float = 0
        self._quantity_sold_short: float = 0
        self._quantity_signed: float = 0
        self._quantity: float = 0
        self._side: PositionSide = PositionSide.LONG
        self._cash_flow: float = 0
        self._entry_amount: float = 0
        self._debt: float = 0
        self._margin: float = 0
        self._price: float = 0

        self._initialize(execution, account)

    def _update_side_and_quantities(self, ex: OrderExecution, qty_signed: float):
        """
        Updates _side, _quantity, _quantity_signed, _quantity_bought,
        _quantity_sold and _quantity_sold_short for the given execution
        and the given current signed quantity.
        """
        if ex.side.is_buy():
            self._quantity_bought += ex.quantity
        elif ex.side == OrderSide.SELL or \
            ex.side == OrderSide.SELL_PLUS:
            self._quantity_sold += ex.quantity
        elif ex.side == OrderSide.SELL_SHORT or \
            ex.side == OrderSide.SELL_SHORT_EXEMPT:
            self._quantity_sold_short += ex.quantity

        qty_signed += ex.quantity_sign * ex.quantity
        self._quantity_signed = qty_signed
        self._quantity = abs(qty_signed)
        self._side = PositionSide.SHORT if qty_signed < 0 else PositionSide.LONG

    def _initialize(self, ex: OrderExecution, account: Account):
        """Re-initializes the closed position."""
        self._margin = ex.margin
        self._debt = ex.debt
        self._price = ex.price
        self._entry_amount = ex.amount
        self._quantity = ex.quantity

        for e in self._executions:
            e.unrealized_quantity = 0
        self._update_side_and_quantities(ex, 0)
        self._executions.append(ex)
        cf = ex.cash_flow - ex.commission_converted
        self._cash_flow = cf
        t = ex.report_time
        a = ex.amount
        self._amounts.add(t, a)
        self._performance.add_PnL(t, a, a, a, cf)
        self._performance.add_drawdown(t, a + cf)
        account.add_execution(ex)

    def _update_margin_and_debt(self, ex: OrderExecution) -> float:
        """
        Updates the _margin and _debt and returns a signed increment
        in debt caused by this execution.

        Uses _side and _quantity updated by _update_side_and_quantities().
        """
        if ex.margin == 0:
            return 0
        is_long = self._side == PositionSide.LONG
        is_short = self._side == PositionSide.SHORT
        is_buy = ex.side.is_buy()
        is_sell = ex.side.is_sell()

        if (is_long and is_buy) or (is_short and is_sell):
            # Execution and updated position have the same directions.
            # Long position and buy execution or
            # short position and sell execution.
            self._margin += ex.margin
            self._debt += ex.debt
            return ex.debt
        elif (is_long and is_sell) or (is_short and is_buy):
            # Execution and updated position have opposite directions.
            # Long position and sell execution or
            # short position and buy execution.
            qty_diff = self._quantity - ex.quantity
            if qty_diff > 0:
                # Executed less than updated position quantity.
                self._margin -= ex.margin
                delta = -self._debt * ex.quantity / self._quantity
                self._debt += delta
                return delta
            elif qty_diff < 0:
                # Executed more than updated position quantity.
                self._margin = -qty_diff * self._instrument.initial_margin
                amt_diff = -qty_diff * ex.price * self._price_factor - self._margin
                delta = amt_diff - self._debt
                self._debt = amt_diff
                return delta
            else:
                # Executed exactly the updated position quantity.
                self._margin = 0
                self._debt = 0
                return -ex.debt
        else:
            # Either order side or position side are unknown.
            return 0

    def _new_roundtrip(self, entry: OrderExecution, exit: OrderExecution, qty: float) -> Roundtrip:
        """Creates a new roundtrip for the given entry and exit executions."""
        entry.unrealized_quantity -= qty
        exit.unrealized_quantity -= qty
        return Roundtrip(self._instrument, entry, exit, qty)

    def _update_execution_pnl_and_match_roundtrips(self,
        ex: OrderExecution, qty_signed: float) -> List[Roundtrip]:
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
            if self._roundtrip_matching == RoundtripMatching.FIFO:
                for e in self._executions:
                    if ex_qty_left <= 0:
                        break
                    # Skip if the full quantity has already been matched
                    # or execution sides are the same.
                    if (e.unrealized_quantity > 0) and (e.quantity_sign != ex_sign):
                        # Execution sides are opposite and
                        # there is an unmatched quantity.
                        min_qty = min(ex_qty_left, e.unrealized_quantity)
                        commission_matched += min_qty * \
                            (ex.commission_converted_per_unit + e.commission_converted_per_unit)
                        amount_matched += -ex_sign * min_qty * (ex.price - e.price)
                        ex_qty_left -= min_qty
                        e.unrealized_quantity -= ex_qty_left
                        rts.append(self._new_roundtrip(e, ex, min_qty))
            elif self._roundtrip_matching == RoundtripMatching.LIFO:
                for e in reversed(self._executions):
                    if ex_qty_left <= 0:
                        break
                    # Skip if the full quantity has already been matched
                    # or execution sides are the same.
                    if (e.unrealized_quantity > 0) and (e.quantity_sign != ex_sign):
                        # Execution sides are opposite and
                        # there is an unmatched quantity.
                        min_qty = min(ex_qty_left, e.unrealized_quantity)
                        commission_matched += min_qty * \
                            (ex.commission_converted_per_unit + e.commission_converted_per_unit)
                        amount_matched += -ex_sign * min_qty * (ex.price - e.price)
                        ex_qty_left -= min_qty
                        e.unrealized_quantity -= ex_qty_left
                        rts.append(self._new_roundtrip(e, ex, min_qty))
        amount_matched *= self._price_factor
        ex.pnl += amount_matched
        ex.realized_pnl = amount_matched - commission_matched
        return rts

    def _update_price(self, t: datetime, price: float):
        """
        Updates _price, _amounts and _performance based on new price
        assuming _update_execution_pnl_and_match_roundtrips() has been
        called and the execution has been appended.
        """
        unrealized_amt = 0.0
        for e in self._executions:
            qty = e.unrealized_quantity
            if qty > 0:
                # Delta increments for the changed price.
                unrealized_amt += (price - e.price) * qty * e.quantity_sign
                e.unrealized_price_high = max(e.unrealized_price_high, price)
                e.unrealized_price_low = min(e.unrealized_price_low, price)
        self._price = price
        amt = price * self._price_factor * self._quantity_signed
        self._amounts.add(t, amt)
        self._performance.add_PnL(t, self._entry_amount, amt,
            unrealized_amt * self._price_factor, self._cash_flow)
        self._performance.add_drawdown(t, amt + self._cash_flow)


    def execute(self, ex: OrderExecution, account: Account) -> List[Roundtrip]:
        """
        Adds an execution to the existing position.
        The instrument of the execution should match the position instrument.
        """
        if self._quantity == 0:
            self._initialize(ex, account)
            return []
        qty_signed = self._quantity_signed
        rts = self._update_execution_pnl_and_match_roundtrips(ex, qty_signed)
        self._update_side_and_quantities(ex, qty_signed)
        _ =self._update_margin_and_debt(ex)
        self._executions.append(ex)
        self._cash_flow += ex.cash_flow - ex.commission_converted
        for r in rts:
            self._performance.add_roundtrip(r)
        account.execute(ex)
        self._update_price(ex.report_time, ex.price)
        return rts

    def instrument(self) -> Instrument:
        """Returns the instrument of the position."""
        return self._instrument
    
    def currency(self) -> Currency:
        """Returns the currency of the position."""
        return self._instrument.currency
    
    def execution_history(self) -> List[OrderExecution]:
        """Returns the list of executions."""
        return self._executions.copy()
    
    def debt(self) -> float:
        """Returns the debt in the instrument currency."""
        return self._debt
    
    def margin(self) -> float:
        """Returns the margin in the instrument currency."""
        return self._margin
    
    def leverage(self) -> float:
        """Returns the current position leverage ratio."""
        return self._amounts.current_value() / self._margin \
            if self._margin != 0 else 0
    
    def price(self) -> float:
        """Returns the price in the instrument currency."""
        return self._price
    
    def quantity_bought(self) -> float:
        """
        Returns the total unsigned quantity bought in this position.
        """
        return self._quantity_bought
    
    def quantity_sold(self) -> float:
        """
        Returns the total unsigned quantity sold in this position.
        """
        return self._quantity_sold
    
    def quantity_sold_short(self) -> float:
        """
        Returns the total unsigned quantity sold short in this position.
        """
        return self._quantity_sold_short

    def side(self) -> PositionSide:
        """Returns the position side (long or short)."""
        return self._side

    def quantity(self) -> float:
        """
        Returns the unsigned position quantity
        (bought minus sold minus sold short).
        """
        return self._quantity

    def cash_flow(self) -> float:
        """
        Returns the current cash flow (the sum of cash flows
        of all order executions) in instrument currency.
        """
        return self._cash_flow
    
    def amount(self) -> float:
        """
        Returns the current position amount (factored price
        times signed quantity) in instrument currency.
        """
        return self._amounts.current_value()
    
    def amount_history(self) -> List[Tuple[datetime, float]]:
        """Returns the history of position amounts."""
        return self._amounts.history()
    
    def performance(self) -> Performance:    
        """Returns the performance of the position."""
        return self._performance