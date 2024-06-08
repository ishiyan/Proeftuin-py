from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import List

from .currencies import Currency, Currencies
from .instruments import Instrument, InstrumentType, InstrumentStatus

class OrderType(Enum):
    """Enumerates the types of an order."""

    MARKET = 'market'
    """
    Market order is an order to buy (sell) an asset at the
    ask (bid) price currently available in the marketplace.

    This order may increase the likelihood of a fill and the
    speed of execution, but provides no price protection and
    may fill at a price far lower (higher) than the current
    bid (ask).
    """

    MARKET_IF_TOUCHED = 'market_if_touched'
    """
    Market-if-touched order is an order to
    buy (sell) a stated amount of an asset as soon as the market
    goes below (above) a preset price, at which point it becomes
    a market order.

    This order is similar to a stop order, except that a
    market-if-touched sell order is placed above the current
    market price, and a stop sell order is placed below.
    """

    LIMIT = 'limit'
    """
    Limit order is an order to buy (sell) only at a specified
    limit price or better, above (below) the limit price.

    A limit order may not get filled if the price never
    reaches the specified limit price.
    """

    STOP = 'stop'
    """
    Stop order is a buy (sell) stop order which becomes
    a market order when the last traded price is
    greater (less) -than-or-equal to the stop price.

    A buy (sell) stop price is always above (below) the
    current market price.

    A stop order may not get filled if the price never
    reaches the specified stop price.
    """

    STOP_LIMIT = 'stop_limit'
    """
    Stop-limit order is a buy (sell) order
    which becomes a limit order when the last traded price
    is greater (less) -than-or-equal to the stop price.

    A buy (sell) stop price is always above (below)
    the current market price.

    A stop-limit order may not get filled if the price
    never reaches the specified stop price.
    """

    TRAILING_STOP = 'trailing_stop'
    """
    Trailing-stop order is a buy (sell) order
    entered with a stop price at a fixed amount above (below)
    the market price that creates a moving or trailing
    activation price, hence the name.

    If the market price falls (rises), the stop loss price
    rises (falls) by the increased amount, but if the stock
    price falls, the stop loss price remains the same.

    The reverse is true for a buy trailing stop order.
    """

    MARKET_ON_CLOSE = 'market_on_close'
    """
    Market on close order will execute as a
    market order as close to the closing price as possible.
    """

    MARKET_TO_LIMIT = 'market_to_limit'
    """
    Market-to-limit order is a market order
    to execute at the current best price.

    If the entire order does not immediately execute at the
    market price, the remainder of the order is re-submitted
    as a limit order with the limit price set to the price at
    which the market order portion of the order executed.
    """

    LIMIT_IF_TOUCHED = 'limit_if_touched'
    """
    Limit-if-touched order is designed to buy (or sell) a contract
    below (or above) the market, at the limit price or better.
    """

    LIMIT_ON_CLOSE = 'limit_on_close'
    """
    Limit-on-close order will fill at the closing price if that
    price is at or better than the submitted limit price.

    Otherwise, the order will be canceled.
    """

class OrderTimeInForce(Enum):
    """Enumerates time conditions an order is to be traded."""

    DAY = 'day'
    """
    Day order requires an order to be executed within the trading
    day on which it was entered.
    """

    IMMEDIATE_OR_CANCEL = 'immediate_or_cancel'
    """
    Immediate-or-cancel requires an order to be executed immediately
    in whole or partially.

    Any portion not so executed is to be canceled.
    Not to be confused with Fill-or-kill.
    """

    FILL_OR_KILL = 'fill_or_kill'
    """
    Fill-or-kill requires an order to be executed immediately in its entirety.

    If not so executed, the order is to be canceled.
    Not to be confused with Immediate-or-cancel.
    """

    GOOD_TILL_CANCELED = 'good_till_canceled'
    """
    Good-till-canceled requires an order to remain in effect until it is
    either executed or canceled.

    Typically, GTC orders will be automatically be cancelled if a corporate
    action on a security results in a stock split (forward or reverse),
    exchange for shares, or distribution of shares.
    """

    GOOD_TILL_DATE = 'good_till_date'
    """
    Good-till-date requires an order, if not executed, to expire at the
    specified date.
    """

    AT_OPEN = 'at_open'
    """
    At-open requires an order to be executed at the opening or not at all.

    All or part of any order not executed at the opening is treated as canceled.
    """

    AT_CLOSE = 'at_close'
    """
    At-close requires an order to be executed at the closing or not at all.

    All or part of any order not executed at the closing is treated as canceled.

    Indicated price is to be around the closing price, however,
    not held to the closing price.
    """

class OrderSide(Enum):
    """Enumerates sides of an order."""

    BUY = 'buy'
    """
    Buy order side refers to the buying of a security.
    """

    SELL = 'sell'
    """
    Sell order side refers to the selling of a security.
    """

    BUY_MINUS = 'buy_minus'
    """
    Buy-minus (buy "minus") is a buy order provided that the price is not higher than
    the last sale if the last sale was a 'minus' or 'zero minus' tick and not higher
    than the last sale minus the minimum fractional change in the security if the last
    sale was a 'plus' or 'zero plus' tick.

    'Minus tick' is a trade executed at a lower price than the preceding trade.

    'Zero minus tick' is a trade executed at the same price as the preceding
    trade, but at a lower price than the last trade of a different price.

    For example, if a succession of trades occurs at $10.25, $10.00 and $10.00 again,
    the second trade is a "minus tick" trade and the latter trade is a 'zero minus tick'
    or 'zero downtick' trade.
    """

    Sell_Plus = 'sell_plus'
    """
    Sell-plus (sell 'plus') is a sell order provided that the price is not lower than
    the last sale if the last sale was a 'plus' or 'zero plus' tick and not lower
    than the last sale minus the minimum fractional change in the security if the last
    sale was a 'minus' or 'zero minus' tick.

    'Plus tick' is a trade that is executed at a higher price than the preceding trade.

    'Zero plus tick' is a trade that is executed at the same price as the preceding
    trade but at a higher price than the last trade of a different price.

    For example, if a succession of trades occurs at $10.00, $10.25 and $10.25 again,
    the second trade is a "plus tick" trade and the latter trade is a 'zero plus tick'
    or 'zero uptick' trade.
    """

    Sell_Short = 'sell_short'
    """
    Sell-short is an order to sell a security that the seller does not own.

    Since 1938 there was an 'uptick rule' established by the U.S. Securities
    and Exchange Commission (SEC). The rule stated that securities could be
    shorted only on an 'uptick' or a 'zero plus' tick, not on a 'downtick'
    or a 'zero minus' tick. This rule was lifted in 2007, allowing short
    sales to occur on any price tick in the market, whether up or down.

    However, in 2010 the SEC adopted the alternative uptick rule applies to
    all securities, which is triggered when the price of a security has dropped
    by 10% or more from the previous day's close. When the rule is in effect,
    short selling is permitted if the price is above the current best bid and
    stays in effect for the rest of the day and the following trading session.

    'Plus tick' is a trade that is executed at a higher price than the preceding trade.

    'Zero plus tick' is a trade that is executed at the same price as the preceding
    trade but at a higher price than the last trade of a different price.

    For example, if a succession of trades occurs at $10, $10.25 and $10.25 again,
    the second trade is a "plus tick" trade and the latter trade is a 'zero plus tick'
    or 'zero uptick' trade.

    'Minus tick' is a trade executed at a lower price than the preceding trade.

    'Zero minus tick' is a trade executed at the same price as the preceding
    trade, but at a lower price than the last trade of a different price.

    For example, if a succession of trades occurs at $10.25, $10.00 and $10.00 again,
    the second trade is a "minus tick" trade and the latter trade is a 'zero minus tick'
    or 'zero downtick' trade.
    """

    SELL_SHORT_EXEMPT = 'sell_short_exempt'
    """
    Sell-short-exempt refers to a special trading situation where a short sale is
    allowed on a minustick.
    """

class OrderStatus(Enum):
    """Enumerates the states an order runs through during its lifetime."""

    ACCEPTED = 'accepted'
    """
    Accepted indicates the order has been received by the broker and is being evaluated.

    The order will proceed to the Pending-new status.
    """

    PENDING_NEW = 'pending_new'
    """
    Pending-new indicates the order has been accepted by the broker but not yet acknowledged
    for execution.

    The order will proceed to the either New or the Rejected status.
    """

    NEW = 'new'
    """
    New indicates the order has been acknowledged by the broker and becomes the
    outstanding order with no executions.

    The order can proceed to the Filled, the Partially-filled, the Expired,
    the Pending-cancel, the Pending-replace, or to the Rejected status.
    """

    REJECTED = 'rejected'
    """
    Rejected indicates the order has been rejected by the broker. No executions were done.

    This is a terminal state of an order, no further changes are allowed.
    """

    PARTIALLY_FILLED = 'partially_filled'
    """
    Partially-filled indicates the order has been partially filled and has remaining quantity.

    The order can proceed to the Filled, the Pending-cancel, or to the
    Pending-replace status.
    """

    FILLED = 'filled'
    """
    Filled indicates the order has been completely filled.

    This is a terminal state of an order, no further changes are allowed.
    """

    EXPIRED = 'expired'
    """
    Expired indicates the order (with or without executions) has been canceled
    in broker's system due to time in force instructions.

    The only exceptions are Fill-or-kill and Immediate-or-cancel
    orders that have Canceled as terminal order state.

    This is a terminal state of an order, no further changes are allowed.
    """

    PENDING_REPLACE = 'pending_replace'
    """
    Pending-replace indicates a replace request has been sent to the broker, but the broker
    hasn't replaced the order yet.

    The order will proceed back to the previous status.
    """

    PENDING_CANCEL = 'pending_cancel'
    """
    Pending-cancel indicates a cancel request has been sent to the broker, but
    the broker hasn't canceled the order yet.

    The order will proceed to the either Canceled
    or back to the previous status.
    """

    CANCELED = 'canceled'
    """
    Canceled indicates the order (with or without executions)
    has been canceled by the broker.

    The order may still be partially filled.
    This is a terminal state of an order, no further changes are allowed.
    """

class OrderReportType(Enum):
    """Enumerates an order report event types."""

    PENDING_NEW = 'pending_new'
    """
    Accepted indicates the order has been received by the broker and is being evaluated.

    The order will proceed to the PendingNew status.
    """

    NEW = 'new'
    """
    New reports a transition to the "new" order status.
    """

    REJECTED = 'rejected'
    """
    Rejected reports a transition to the "rejected" order status.
    """

    PARTIALLY_FILLED = 'partially_filled'
    """
    Partially-filled reports a transition to the "partially filled" order status.
    """

    FILLED = 'filled'
    """
    Filled reports a transition to the "filled" order status.
    """

    EXPIRED = 'expired'
    """
    Expired reports a transition to the "expired" order status.
    """

    PENDING_REPLACE = 'pending_replace'
    """
    Pending-replace reports a transition to the "pending replace" order status.
    """

    REPLACED = 'replaced'
    """
    Replaced reports that an order has been replaced.
    """

    REPLACE_REJECTED = 'replace_rejected'
    """
    Replace-rejected reports that an order replacement has been rejected.
    """

    PENDING_CANCEL = 'pending_cancel'
    """
    Pending-cancel reports a transition to the "pending cancel" order status.
    """

    CANCELED = 'canceled'
    """
    Canceled reports a transition to the "canceled" order status.
    """

    CANCEL_REJECTED = 'cancel_rejected'
    """
    Cancel-rejected reports that an order cancellation has been rejected.
    """

    ORDER_STATUS = 'order_status'
    """
    Order-status reports an order status.
    """

class OrderSingle:
    """A request to place an order in a single instrument.

    Parameters
    ----------
    instrument : Instrument
        The instrument to be traded.
    type : OrderType
        The type of the order.
    side : OrderSide
        The side of the order.
    time_in_force : OrderTimeInForce
        The time in force condition of the order.
    quantity : float
        The total order quantity (in units) to execute.
    minimum_quantity : float, optional
        The minimum quantity (in units) to be filled.
    limit_price : float, optional
        The limit price in instrument's currency per unit of quantity.
        Zero if not set.

        Required for limit, stop-limit, and limit-if-touched orders.

        For FX orders, should be the "all-in" rate (spot rate adjusted for forward points).
    stop_price : float, optional
        The stop price in instrument's currency per unit of quantity.
        Zero if not set.

        Required for stop, stop-limit, and limit-if-touched orders.
    trailing_distance : float, optional
        The trailing distance in instrument's currency.
        Zero if not set.

        Required for trailing-stop orders.
    creation_time : datetime, optional
        The date and time when the order was created by a trader, trading system, or intermediary.
    expiration_time : datetime, optional
        The order expiration date and time for the good-till-date orders
    note : str, optional
        A free-format text with notes on the order.
    """
    def __init__(self,
                 instrument: Instrument,
                 type: OrderType = OrderType.MARKET,
                 side: OrderSide = OrderSide.BUY,
                 time_in_force: OrderTimeInForce = OrderTimeInForce.DAY,
                 quantity: float = 1,
                 minimum_quantity: float = None,
                 limit_price: float = None,
                 stop_price: float = None,
                 trailing_distance: float = None,
                 creation_time: datetime = None,
                 expiration_time: datetime = None,
                 note: str = '',
                 ):
        self.instrument = instrument
        self.type = type
        self.side = side
        self.time_in_force = time_in_force
        self.quantity = quantity
        self.minimum_quantity = minimum_quantity
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.trailing_distance = trailing_distance
        self.creation_time = creation_time
        self.expiration_time = expiration_time
        self.note = note

    def __str__(self):
        return f'{self.instrument.symbol} {self.side} {self.type} {self.time_in_force} {self.quantity} {self.instrument.symbol} {self.type}'

    def __repr__(self):
        attributes = ['instrument', 'type', 'side', 'time_in_force', 'quantity',
            'minimum_quantity', 'limit_price', 'stop_price', 'trailing_distance',
            'creation_time', 'expiration_time', 'note']
        attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes ]
        #attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes if getattr(self, attr) is not None]
        return 'OrderSingle(' + ', '.join(attr_strings) + ')'

class OrderSingleExecutionReport:
    """A report event for an order in a single instrument.

    Parameters
    ----------
    order : OrderSingle
        The underlying order for this execution report.

        If there were any successful order replacements, this will be the most recent version.
    transaction_time : datetime
        The date and time when the business represented by this report occurred.
    status : OrderStatus
        The current state of an order as understood by the broker.
    report_type : OrderReportType
        Identifies an action of this report.
    ID : str
        The unique identifier of this report as assigned by the sell-side.
    note : str
        A free-format text that accompany this report.
    replace_source_order : OrderSingle, optional
        The replace source order. Filled when report type is Replaced or Replace-rejected.
    replace_target_order : OrderSingle, optional
        The replace target order. Filled when report type is Replaced or Replace-rejected.
    last_filled_price : float
        The price (in order instrument's currency) of the last fill.
    average_price : float
        The average price (in order instrument's currency) of all fills.
    last_filled_quantity : float
        the quantity bought or sold on the last fill.
    leaves_quantity : float
        The remaining quantity to be filled.

        If the order status is Canceled, Expired or Rejected (in which case
        the order is no longer active) then this could be 0, otherwise it is
        
        order.quantity - cumulative_quantity.
    cumulative_quantity : float
        The total quantity filled.
    last_fill_commission : float
        The commission (in commission currency) of the last fill.
    cumulative_commission : float
        The total commission (in commission currency) of all fills.
    commission_currency : Currency
        The currency in which the commission is denominated.
    """
    def __init__(self,
                 order: OrderSingle,
                 transaction_time: datetime,
                 status: OrderStatus = OrderStatus.ACCEPTED,
                 report_type: OrderReportType = OrderReportType.PENDING_NEW,
                 ID: str = '',
                 note: str = '',
                 replace_source_order: OrderSingle = None,
                 replace_target_order: OrderSingle = None,
                 last_filled_price: float = 0,
                 average_price: float = 0,
                 last_filled_quantity: float = 0,
                 leaves_quantity: float = 0,
                 cumulative_quantity: float = 0,
                 last_fill_commission: float = 0,
                 cumulative_commission: float = 0,
                 commission_currency: Currency = Currencies.USD,
                 ):
        self.order = order
        self.transaction_time = transaction_time
        self.status = status
        self.report_type = report_type
        self.ID = ID
        self.note = note
        self.replace_source_order = replace_source_order
        self.replace_target_order = replace_target_order
        self.last_filled_price = last_filled_price
        self.average_price = average_price
        self.last_filled_quantity = last_filled_quantity
        self.leaves_quantity = leaves_quantity
        self.cumulative_quantity = cumulative_quantity
        self.last_fill_commission = last_fill_commission
        self.cumulative_commission = cumulative_commission
        self.commission_currency = commission_currency
        
    def __str__(self):
        return f'{self.order.instrument.symbol} {self.status} {self.report_type}'

    def __repr__(self):
        attributes = ['order', 'transaction_time', 'status', 'report_type', 'ID',
            'note', 'replace_source_order', 'replace_target_order', 'last_filled_price',
            'average_price', 'last_filled_quantity', 'leaves_quantity', 'cumulative_quantity',
            'last_fill_commission', 'cumulative_commission', 'commission_currency']
        attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes ]
        #attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes if getattr(self, attr) is not None]
        return 'OrderSingleExecutionReport(' + ', '.join(attr_strings) + ')'

class OrderSingleTicket(ABC):
    """A ticket to track a new order in a single instrument.

    Parameters
    ----------
    order : OrderSingle
        The underlying order for this ticket.

        If there were any successful order replacements,
        this will be the most recent version.
    client_order_ID : str
        The unique identifier for an order as assigned
        by the buy-side (institution, broker, intermediary etc.).
    order_ID : str
        The unique identifier for an order as assigned by the sell-side.
    status : OrderStatus
        The current state of an order as understood by the broker.
    last_report : OrderSingleExecutionReport
        The last execution report for this order, None if not any.
    reports : List[OrderSingleExecutionReport]
        The list of execution reports in the chronological order.
    """
    def __init__(self,
                 order: OrderSingle,
                 client_order_ID: str = '',
                 order_ID: str = '',
                 status: OrderStatus = OrderStatus.ACCEPTED,
                 last_report: OrderSingleExecutionReport = None,
                 reports: List[OrderSingleExecutionReport] = [],
                 ):
        self.order = order
        self.client_order_ID = client_order_ID
        self.order_ID = order_ID
        self.status = status
        self.last_report = last_report
        self.reports = reports

    def __str__(self):
        return f'{self.order.instrument.symbol} {self.status} {self.client_order_ID} {self.order_ID}'

    def __repr__(self):
        attributes = ['order', 'client_order_ID', 'order_ID', 'status',
            'last_report', 'reports']
        attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes ]
        #attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes if getattr(self, attr) is not None]
        return 'OrderSingleTicket(' + ', '.join(attr_strings) + ')'

    @abstractmethod
    def cancel(self):
        """Cancels this order

        If order has been already completed (successfully or not), does nothing.

        Produces an execution report on completion.
      """

    @abstractmethod
    def cancel_replace(self, replacement_order: OrderSingle):
        """Used to change the parameters of an existing order.

        Allowed modifications to an order include:
        - reducing or increasing order quantity
        - changing a limit order to a market order
        - changing the limit price
        - changing time in force

        Modifications cannot include:
        - changing order side
        - changing series
        - reducing quantity to zero (canceling the order)
        - re-opening a filled order by increasing quantity

        Unchanging attributes to be carried over from the
        original order must be specified in the replacement.

        If the order has been completed (successfully or not), does nothing.

        Produces an execution report on completion.
        """        
