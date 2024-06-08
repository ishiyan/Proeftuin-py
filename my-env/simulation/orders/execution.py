from ..currencies import CurrencyConverter
from .execution_report import OrderExecutionReport

class OrderExecution:
    """Contains properties of a fill or a partial fill based on
    a Filled or Partially-filled execution report of an order.

    Parameters
    ----------
    report_ID : str
        The unique transaction identifier of an associated
        execution report as assigned by the sell-side.
    report_time : datetime.datetime
        The date and time of an associated execution report
        as assigned by the broker.
    side : OrderSide
        The execution order side, which determines the sign of the quantity.
    quantity : float
        The unsigned execution quantity, the sign is determined by the side.
    quantity_sign : float
        The sign of the execution quantity, which is 1 for buy orders
        and -1 for sell orders.
    currency : Currency
        The currency of the instrument.
    commission : float
        The execution commission amount in the commission currency.
    commission_currency : Currency
        The currency of the commission.
    commission_conversion_rate : float
        The conversion rate of the commission currency to the instrument currency.
    commission_converted : float
        The commission amount converted to the instrument currency.
    commission_converted_per_unit : float
        The commission amount converted to the instrument currency per unit.
    price : float
        The execution price in the instrument currency.
    amount : float
        The unsigned execution value in the instrument currency (factored price
        times quantity).
    margin : float
        The unsigned execution margin in instrument currency (instrument margin
        times quantity).
    debt : float
        The execution debt in instrument currency (amount minus margin).
    pnl : float
        The execution Profit and Loss in the instrument currency.
        This value typically should be adjusted when adding to a position.
    realized_pnl : float
        The realized execution Profit and Loss in the instrument currency.
        This value typically should be adjusted when adding to a position.
    cash_flow : float
        The execution cash flow in the instrument currency (factored price
        times negative signed quantity).
    unrealized_quantity : float
        The unsigned unrealized quantity in units.
    unrealized_price_high : float
        The highest price of the unrealized quantity.
    unrealized_price_low : float
        The lowest price of the unrealized quantity.
    """
    def __init__(self,
                 report: OrderExecutionReport,
                 currency_converter: CurrencyConverter,
                 ):
        qty_abs = abs(report.last_filled_quantity)
        side = report.order.side
        qty_sign = -1.0 if side.is_sell() else 1.0
        instrument = report.order.instrument
        price = report.last_filled_price
        price_factored = price
        price_factored = price * instrument.price_factor if instrument.price_factor else price
        amount_abs = price_factored * qty_abs
        margin_abs = qty_abs * instrument.initial_margin if instrument.initial_margin is not None else 0.0
        debt = 0.0 if amount_abs == 0 else amount_abs - margin_abs
        conv, rate = (currency_converter.convert(report.last_fill_commission,
            report.commission_currency, instrument.currency) 
            if report.commission_currency != instrument.currency
            else (report.last_fill_commission, 1.0))
        
        self.report_ID = report.ID
        self.report_time = report.transaction_time
        self.side = side
        self.quantity = qty_abs
        self.quantity_sign = qty_sign
        self.currency = instrument.currency
        self.commission = report.last_fill_commission
        self.commission_currency = report.commission_currency
        self.commission_conversion_rate = rate
        self.commission_converted = conv
        self.commission_converted_per_unit = conv / qty_abs if qty_abs != 0 else 0.0
        self.price = price
        self.amount = amount_abs
        self.margin = margin_abs
        self.debt = debt
        self.pnl = -conv, # Will be updated when adding to position.
        self.realized_pnl = 0, # Will be updated when adding to position.
        self.cash_flow = -qty_sign * amount_abs
        self.unrealized_quantity = qty_abs
        self.unrealized_price_high = price
        self.unrealized_price_low = price
