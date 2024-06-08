from enum import Enum

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

    SELL_PLUS = 'sell_plus'
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

    SELL_SHORT = 'sell_short'
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

    def is_buy(self):
        return self in (OrderSide.BUY, OrderSide.BUY_MINUS)

    def is_sell(self):
        return self in (OrderSide.SELL, OrderSide.SELL_SHORT, OrderSide.SELL_SHORT_EXEMPT)

    def is_short(self):
        return self in (OrderSide.SELL_PLUS, OrderSide.SELL_SHORT, OrderSide.SELL_SHORT_EXEMPT)
