from enum import Enum

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
