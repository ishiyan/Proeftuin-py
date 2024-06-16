from typing import Sequence
from collections import defaultdict
from datetime import datetime
import numpy as np
from gymnasium import spaces

from ..accounts import Account
from ..broker import Broker
from ..orders import StopOrder, TakeProfitOrder
from .action_scheme import ActionScheme
    
class PingPongAction(ActionScheme):
    """
    A specialized action scheme for mean reversion trading
    
    Notes
    -----
    Market can be described as being in one of two modes: trend or consolidation.
    On a global scale markets are in trend mode for most of the time.
    But as you go deeper to a smaller timeframes you may find out that consolidation mode becomes significant.
    
    When market is in consolidation mode price goes up and down in some range.
    This is a good time for mean reversion trading.
    
    It means you open short position when price is near the upper bound,
    and then you open long position when price is near the lower bound.
    
    This action scheme performs it automatically for you.
    All you need is to provide four values as an action:
    - lower price bound,
    - upper price bound,
    - trail delta value to close position with profit, if price goes in your direction,
    - stop delta value to close position with stop-loss, if price goes not in your direction.
    
    When price goes up, crosses the upper bound and falls down more than trail delta:
    1. A trailing sell TakeProfitOrder is executed and short position is opened.
    2. A trailing buy TakeProfitOrder is created to automatically close (buy) position if price raises up
       more than trail delta from the local minimum.
    3. A StopOrder is created to automatically close (buy) position by stop loss if price goes up
       more than stop delta from the position price.

    When price goes down, crosses the lower bound and then rises up more than trail delta:
    1. A trailing buy TakeProfitOrder is executed and long position is opened.
    2. A trailing sell TakeProfitOrder is created to automatically close (sell) position if price falls down
       more than trail delta from the local minimum.
    3. A StopOrder is created to automatically close (sell) position by stop loss if price goes down
       more than stop delta from the position price.
    
    On each step an agent should provide an action as a tuple (or list, or numpy array) of four values:
    >>> (lower_price, upper_price, trail_delta, stop_delta)
    
    On each step agent can change these values according to its policy.
    Existing take-profit or stop-loss orders will be updated for new values.
    """
    space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(4,),
        dtype=np.float32
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.position = defaultdict(lambda: None)
        self.lower_price = defaultdict(lambda: None)
        self.upper_price = defaultdict(lambda: None)
        self.trail_delta = defaultdict(lambda: None)
        self.stop_delta = defaultdict(lambda: None)
        self.buy_order_id = defaultdict(lambda: None)
        self.sell_order_id = defaultdict(lambda: None)
        self.buy_stop_order_id = defaultdict(lambda: None)
        self.sell_stop_order_id = defaultdict(lambda: None)

    def reset(self):
        """Automatically invoked by Environment when it is being reset."""
        self.position.clear()
        self.lower_price.clear()
        self.upper_price.clear()
        self.trail_delta.clear()
        self.stop_delta.clear()
        self.buy_order_id.clear()
        self.sell_order_id.clear()
        self.buy_stop_order_id.clear()
        self.sell_stop_order_id.clear()

    def get_random_action(self) -> np.ndarray:
        """
        Returns random action
        
        Notes
        -----
        In fact, we can't make any realistic random action because here we don't know
        the price and its range.
        That is why using get_random_action with PingPongActionScheme is useless.
        """
        return np.array((-np.inf, np.inf, 0, 0))

    def get_default_action(self) -> np.ndarray:
        """
        Returns default action

        Notes
        -----
        In fact, we can't make any realistic default action because here we don't know
        the price and its range.
        That is why using get_default_action with PingPongActionScheme is useless.
        """
        return np.array((-np.inf, np.inf, 0, 0))

    def process_action(self, broker: Broker, account: Account, action, time: datetime):
        """
        Called by environment instance to actually perform action, chosen by an agent.

        Notes
        -----
        On each step an agent should provide an action as a tuple (or list, or numpy array) of four values:
        >>> (lower_price, upper_price, trail_delta, stop_delta)

        The amount for long or short positions is always: 1.

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
        # Get new values for buy, sell prices and stop delta for this account
        assert isinstance(action, (Sequence, np.ndarray)) and (len(action) == 4)
        lower_price, upper_price, trail_delta, stop_delta = action
        
        # Setup account update callback if not yet
        if self.lower_price[account] is None:
            account.subscribe(self, lambda ex, acc, t: self.update(ex, acc, t))

        # Find out what values have been changed
        position_changed = (self.position[account] is None) or (self.position[account] != account.position)
        lower_price_changed = (self.lower_price[account] is None) or (self.lower_price[account] != lower_price)
        upper_price_changed = (self.upper_price[account] is None) or (self.upper_price[account] != upper_price)
        trail_delta_changed = (self.trail_delta[account] is None) or (self.trail_delta[account] != trail_delta)
        stop_delta_changed = (self.stop_delta[account] is None) or (self.stop_delta[account] != stop_delta)

        # Update values
        self.lower_price[account] = lower_price
        self.upper_price[account] = upper_price
        self.trail_delta[account] = trail_delta
        self.stop_delta[account] = stop_delta
        self.position[account] = account.position

        # Find out what orders should we change
        if account.position == 0:
            if lower_price_changed or trail_delta_changed or position_changed:
                self._open_buy(broker, account, time)
            if upper_price_changed or trail_delta_changed or position_changed:
                self._open_sell(broker, account, time)
        elif account.position > 0:
            if upper_price_changed or trail_delta_changed or position_changed:
                self._open_sell(broker, account, time)
            if upper_price_changed or stop_delta_changed or position_changed:
                self._open_stop_sell(broker, account, time)
        elif account.position < 0:
            if lower_price_changed or trail_delta_changed or position_changed:
                self._open_buy(broker, account, time)
            if lower_price_changed or stop_delta_changed or position_changed:
                self._open_stop_buy(broker, account, time)
            
    def update(self, broker: Broker, account: Account, time: datetime):
        # Get old and new position
        old_position = self.position[account] or 0
        new_position = account.position
        
        if (old_position < 0) and (new_position == 0):
            # Short position was closed by stop buy order
            self._kill_stop_buy(broker, account, time)
        elif (old_position < 0) and (new_position > 0):
            # Short position was reverted to long by buy order
            self._kill_stop_buy(broker, account, time)
            self._open_stop_sell(broker, account, time)
            self._open_sell(broker, account, time)
        elif (old_position > 0) and (new_position == 0):
            # Long position was closed by stop sell order
            self._kill_stop_sell(broker, account, time)
        elif (old_position > 0) and (new_position < 0):
            # Long position was reverted to short by sell order
            self._kill_stop_sell(broker, account, time)
            self._open_stop_buy(broker, account, time)
            self._open_buy(broker, account, time)
        elif (old_position == 0) and (new_position < 0):
            # We had no position and now we are in short
            self._open_stop_buy(broker, account, time)
            self._open_buy(broker, account, time)
        elif (old_position == 0) and (new_position > 0):
            # We had no position and now we are in long
            self._open_stop_sell(broker, account, time)
            self._open_sell(broker, account, time)
        elif (old_position == 0) and (new_position == 0):
            # We had no position and still we have none
            pass
        
        # Save new position
        self.position[account] = new_position
            
    def _open_sell(self, broker: Broker, account: Account, time: datetime):
        # Initialize take profit order to sell at a higher price
        if 1 + account.position > 0:
            self.sell_order_id[account] = broker.replace_order(
                id=self.sell_order_id[account],
                new_order=TakeProfitOrder(
                    account=account,
                    operation='S',
                    amount=(1 + account.position),
                    time_init=time,
                    time_kill=None,
                    target_price=self.upper_price[account],
                    trail_delta=self.trail_delta[account],
                    best_price=None
                )
            )
            
    def _open_buy(self, broker: Broker, account: Account, time: datetime):
        # Initialize take profit order to buy at a lower price
        if 1 - account.position > 0:
            self.buy_order_id[account] = broker.replace_order(
                id=self.buy_order_id[account],
                new_order=TakeProfitOrder(
                    account=account,
                    operation='B',
                    amount=(1 - account.position),
                    time_init=time,
                    time_kill=None,
                    target_price=self.lower_price[account],
                    trail_delta=self.trail_delta[account],
                    best_price=None
                )
            )
    
    def _kill_buy(self, broker: Broker, account: Account, time: datetime):
        # Kill take profit order to buy
        if self.buy_order_id[account] is not None:
            broker.kill_order(self.buy_order_id[account], time_kill=time)
            self.buy_order_id[account] = None

    def _kill_sell(self, broker: Broker, account: Account, time: datetime):
        # Kill take profit order to sell
        if self.sell_order_id[account] is not None:
            broker.kill_order(self.sell_order_id[account], time_kill=time)
            self.sell_order_id[account] = None
        
    def _open_stop_buy(self, broker: Broker, account: Account, time: datetime):
        # Initialize stop order to buy if price raises beyond some level
        self.buy_stop_order_id[account] = broker.replace_order(
            id=self.buy_stop_order_id[account],
            new_order=StopOrder(
                account=account,
                operation='B',
                amount=abs(account.position),
                price=(account.position_price + self.stop_delta[account]),
                time_init=time,
                time_kill=None
            )
        )

    def _open_stop_sell(self, broker: Broker, account: Account, time: datetime):
        # Initialize stop order to sell if price drops below some level
        self.sell_stop_order_id[account] = broker.replace_order(
            id=self.sell_stop_order_id[account],
            new_order=StopOrder(
                account=account,
                operation='S',
                amount=account.position,
                price=(account.position_price - self.stop_delta[account]),
                time_init=time,
                time_kill=None
            )
        )
        
    def _kill_stop_buy(self, broker: Broker, account: Account, time: datetime):
        # Kill stop order to buy
        if self.buy_stop_order_id[account] is not None:
            broker.kill_order(self.buy_stop_order_id[account], time_kill=time)
            self.buy_stop_order_id[account] = None

    def _kill_stop_sell(self, broker: Broker, account: Account, time: datetime):
        # Kill stop order to sell
        if self.sell_stop_order_id[account] is not None:
            broker.kill_order(self.sell_stop_order_id[account], time_kill=time)
            self.sell_stop_order_id[account] = None
