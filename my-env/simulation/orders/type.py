from enum import Enum

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
