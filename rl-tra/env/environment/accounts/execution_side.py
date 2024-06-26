from enum import Enum

class ExecutionSide(Enum):
    """Enumerates sides of an execution."""

    BUY = 'buy'
    """
    Buy order side refers to the buying of a security.
    """

    SELL = 'sell'
    """
    Sell order side refers to the selling of a security.
    """

    def is_buy(self):
        return self == ExecutionSide.BUY

    def is_sell(self):
        return self == ExecutionSide.SELL
