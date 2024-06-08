from ....instruments.instrument import Instrument
from ....orders.execution import OrderExecution
from .side import RoundtripSide

class Roundtrip:
    """The immutable position round-trip.

    Parameters
    ----------
    instrument : Instrument
        The traded instrument.
    entry : OrderExecution
        The entry order execution.
    exit : OrderExecution
        The exit order execution.
    quantity : float
        The total unsigned quantity of the position.
    side : RoundtripSide
        The side of the round-trip.
    entry_time : datetime
        The date and time the position was opened.
    entry_price : float
        The (average) price at which the position was opened.
    exit_time : datetime
        The date and time the position was closed.
    exit_price : float
        The (average) price at which the position was closed.
    duration : timedelta
        The duration of the round-trip.
    highest_price : float
        The highest price of the instrument during the round-trip.
    lowest_price : float
        The lowest price of the instrument during the round-trip.
    commission : float
        The total commission paid for the round-trip.
    gross_pnl : float
        The gross Profit and Loss of the round-trip.
    net_pnl : float
        The net Profit and Loss of the round-trip.
    maximum_adverse_price : float
        The maximum adverse price during the round-trip.
    maximum_favorable_price : float
        The maximum favorable price during the round-trip.
    maximum_adverse_excursion : float
        The percentage of the Maximum Adverse Excursion (MAE)
        which measures the maximum potential loss per unit of
        quantity taken during the round-trip period.

        This statistical concept originally created by John Sweeney
        to measure the distinctive characteristics of profitable trades,
        can be used as part of an analytical process to distinguish
        between average trades and those that offer substantially greater
        profit potential.

        0% is the perfect MAE, 100% and higher is the worst possible MAE.
    maximum_favorable_excursion : float
        The percentage of the Maximum Favorable Excursion (MFE)
        which measures the peak potential profit per unit of
        quantity taken during the round-trip period.

        This statistical concept originally created by John Sweeney
        to measure the distinctive characteristics of profitable trades,
        can be used as part of an analytical process to distinguish
        between average trades and those that offer substantially greater
        profit potential.

        0% is the perfect MFE, 100% and higher is the worst possible MFE.
    entry_efficiency : float
        The Entry Efficiency which measures the percentage in range [0. 100]
        of the total round-trip potential taken by a round-trip given its
        entry and assuming the best possible exit during the round-trip period.

        It shows how close the entry price was to the best possible entry
        price during the round-trip.

        100% is the perfect efficiency, 0% is the worst possible efficiency.
    exit_efficiency : float
        The Exit Efficiency which measures the percentage in range [0. 100]
        the total round-trip potential taken by a round-trip given its exit
        and assuming the best possible entry during the round-trip period.

        It shows how close the exit price was to the best possible exit price
        during the round-trip.

        100% is the perfect efficiency, 0% is the worst possible efficiency.
    total_efficiency : float
        The Total Efficiency which measures the percentage in range [0. 100]
        of the total round-trip potential taken by a round-trip during the
        round-trip period.

        It shows how close the entry and exit prices were to the best possible
        entry and exit prices during the round-trip, or the ability to capture
        the maximum profit potential during the round-trip period.

        100% is the perfect efficiency, 0% is the worst possible efficiency.
    """
    def __init__(self,
                 instrument: Instrument,
                 entry: OrderExecution,
                 exit: OrderExecution,
                 quantity: float,
                 ):
        price_factor = instrument.price_factor if instrument.price_factor else 1.0
        side = RoundtripSide.SHORT if entry.side.is_sell() else RoundtripSide.LONG
        entry_p = entry.price
        exit_p = exit.price

        pnl = quantity * (entry_p - exit_p if side == RoundtripSide.SHORT
            else exit_p - entry_p) * price_factor

        commission = (entry.commission_converted / entry.quantity +
                      exit.commission_converted / exit.quantity) * quantity

        highest_p = max(entry.unrealized_price_high, exit.unrealized_price_high)
        lowest_p = min(entry.unrealized_price_low, exit.unrealized_price_low)
        delta = highest_p - lowest_p
        entry_efficiency = 0.0
        exit_efficiency = 0.0
        total_efficiency = 0.0

        if side == RoundtripSide.LONG:
            maximum_adverse_price = lowest_p
            maximum_favorable_price = highest_p
            maximum_adverse_excursion = 100.0 * (1.0 - lowest_p / entry_p)
            maximum_favorable_excursion = 100.0 * (highest_p / exit_p - 1.0)
            if delta != 0.0:
                entry_efficiency = 100.0 * (highest_p - entry_p) / delta
                exit_efficiency = 100.0 * (exit_p - lowest_p) / delta
                total_efficiency = 100.0 * (exit_p - entry_p) / delta
        else:
            maximum_adverse_price = highest_p
            maximum_favorable_price = lowest_p
            maximum_adverse_excursion = 100 * (highest_p / entry_p - 1.0)
            maximum_favorable_excursion = 100.0 * (1.0 - lowest_p / exit_p)
            if delta != 0.0:
                entry_efficiency = 100.0 * (entry_p - lowest_p) / delta
                exit_efficiency = 100.0 * (highest_p - exit_p) / delta
                total_efficiency = 100.0 * (entry_p - exit_p) / delta
        
        duration = exit.report_time - entry.report_time # timedelta from datetime ??? looa at usages

        self.instrument = instrument
        self.side = side
        self.quantity = quantity
        self.entry_time = entry.report_time
        self.entry_price = entry_p
        self.exit_time = exit.report_time
        self.exit_price = exit_p
        self.duration = duration
        self.highest_price = highest_p
        self.lowest_price = lowest_p
        self.commission = commission
        self.gross_pnl = pnl
        self.net_pnl = pnl - commission
        self.maximum_adverse_price = maximum_adverse_price
        self.maximum_favorable_price = maximum_favorable_price
        self.maximum_adverse_excursion = maximum_adverse_excursion
        self.maximum_favorable_excursion = maximum_favorable_excursion
        self.entry_efficiency = entry_efficiency
        self.exit_efficiency = exit_efficiency
        self.total_efficiency = total_efficiency
        super().__setattr__('_is_frozen', True)
    
    def __setattr__(self, name, value):
        if getattr(self, '_is_frozen', False):
            raise TypeError(f"Can't modify immutable instance")
        super().__setattr__(name, value)
