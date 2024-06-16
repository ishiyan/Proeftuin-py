from datetime import datetime
from numbers import Real
import numpy as np
from gymnasium import spaces

from ..accounts import Account
from ..broker import Broker
from ..orders import MarketOrder
from .action_scheme import ActionScheme
    
class BuySellCloseAction(ActionScheme):
    """
    Basic action scheme, where agent can choose from: {Buy, Sell, Close}
    
    Notes
    -----
    This scheme assumes you can open both long and short positions on some asset.
    It opens long positions of some predefined amount upon Buy signal.
    It opens short positions upon Sell signal.
    It closes any open position upon Close signal.
    
    Parameters
    ----------
    amount : Real
        The amount to be traded. Default: 1
    """
    space = spaces.Discrete(3)

    def __init__(self, amount: Real = 1, **kwargs):
        super().__init__(**kwargs)
        self.amount = abs(amount)

    def reset(self):
        """Automatically invoked by Environment when it is being reset"""
        pass

    def get_random_action(self) -> int:
        """
        Returns random action, one of: {Buy, Sell, Close}.

        Returns
        -------
        action : int
            random value [0 .. 2]
        """
        return np.random.randint(0, 3)

    def get_default_action(self) -> int:
        """
        Returns default action: Close.

        Returns
        -------
        action : int
            Value = 2
        """
        return 2
    
    def process_action(self, broker: Broker, account: Account, action, time: datetime):
        """
        Called by environment instance to actually perform action, chosen by an agent.

        Notes
        -----
        If agent has chosen Buy, this method creates new MarketOrder to buy some fixed number of assets.
        If agent has chosen Sell, this method creates new MarketOrder to sell some fixed number of assets.
        If agent has chosen Close and there is an open position (long or short),
        this method creates new MarketOrder to close that position.
        
        The amount for long or short positions is specified at initialization.

        Parameters
        ----------
        broker : Broker
            An underlying broker object, which accepts and executes orders produced by an action scheme.
        account : Account
            An account instance associated with a particular agent, which issued an action.
        action : Any
            Action issued by an agent
        time : datetime
            A moment in time at which action has arrived.
        """
        assert (0 <= action <= 2)
        # Calculate target position based on action
        target_position = self.amount if (action == 0) else -self.amount if (action == 1) else 0
        delta = (target_position - account.position.quantity_signed)
        # Issue a market order if needed
        if delta != 0:
            order = MarketOrder(
                account=account,
                operation=('B' if (delta > 0) else 'S'),
                amount=abs(delta),
                time_init=time,
                time_kill=None
            )
            broker.add_order(order)
