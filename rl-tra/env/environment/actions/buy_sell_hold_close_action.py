from datetime import datetime
from numbers import Real
import numpy as np
from gymnasium import spaces

from ..accounts import Account
from ..broker import Broker
from ..orders import MarketOrder
from .action_scheme import ActionScheme

BUY = 0
SELL = 1
HOLD = 2
CLOSE = 3
NUMBER_OF_ACTIONS = 4

class BuySellHoldCloseAction(ActionScheme):
    """
    The action scheme, where an agent can choose from:

    `{Buy, Sell, Hold, Close}`
    
    This scheme can operate with or without short positions.
    
    If `allow_short_positions` is `True`, you can open both long
    and short positions on some asset.
    Otherwise, only long positions can be opened.
    
    - If action is `Buy`, it increases position by predefined `quantity`
        by creating a new `MarketOrder` to buy the asset.

    - If action is `Sell`, it decreases position by predefined `quantity`
        by createing a new `MarketOrder` to sell the asset,
        unless `allow_short_positions` is `False` and selling will lead
        to the short position. In this case, `Sell` action does nothing.

    - If action is `Hold`, does nothing (holds).

    - If action is `Close`, it closes any open position by creating a new
        `MarketOrder` to close the position.
        The order will sell or buy, depending on the current position
        being long or short.
    
    Note that because of the predefined quantity, the position cannot go
    from long to short or vice versa in a single `Buy` or `Sell` action.

    It always goes through the zero (closed) position,
    which is the behavior enforced by most markets.
    """
    space = spaces.Discrete(NUMBER_OF_ACTIONS)

    def __init__(self,
                 order_quantity: Real = 1,
                 allow_short_positions: bool = True):
        """
        Initializes `BuySellHoldCloseAction` instance.
        
        Args:
            oredr_quantity Real:
                The quantity to be traded on `Buy` and `Sell` actions.
                Default: 1
            allow_short_positions bool:
                Whether to allow short positions.
                Default: True  
        """
        super().__init__()

        if not isinstance(order_quantity, Real) or order_quantity <= 0:
            raise ValueError(f'order_quantity {order_quantity} must be a positive number')
        self.order_quantity_abs = order_quantity
        self.allow_short_positions = allow_short_positions

    def reset(self):
        pass

    def get_random_action(self) -> int:
        """
        Returns a random action, one of: {`Buy`, `Sell`, `Hold`, `Close`}.
        """
        return np.random.randint(0, NUMBER_OF_ACTIONS)

    def get_default_action(self) -> int:
        """
        Returns the default action: `Hold`.
        """
        return HOLD
    
    def process_action(self, broker: Broker, account: Account, action, time: datetime):
        assert (0 <= action < NUMBER_OF_ACTIONS)

        # Calculate order quantity based on action.
        if action == BUY:
            order_quantity_signed = self.order_quantity_abs
        elif action == SELL:
            if not self.allow_short_positions and \
                account.position.quantity_signed < self.order_quantity_abs:
                order_quantity_signed = 0
            else:
                order_quantity_signed = -self.order_quantity_abs
        elif action == CLOSE:
            order_quantity_signed = -account.position.quantity_signed
        else: # HOLD
            order_quantity_signed = 0

        # Create an order if needed.
        if order_quantity_signed != 0:
            order = MarketOrder(
                account=account,
                operation=('B' if (order_quantity_signed > 0) else 'S'),
                amount=abs(order_quantity_signed),
                time_init=time,
                time_kill=None
            )
            broker.add_order(order)
