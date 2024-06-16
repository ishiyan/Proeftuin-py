from abc import ABC, abstractmethod
from datetime import datetime
from gymnasium import spaces

from ..accounts import Account
from ..broker import Broker

class ActionScheme(ABC):

    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def reset(self):
        """Automatically invoked by environment when it is being reset."""
        raise NotImplementedError()
    
    @abstractmethod
    def get_random_action(self):
        """Returns some random action."""
        raise NotImplementedError()

    @abstractmethod
    def get_default_action(self):
        """Returns some default action."""
        raise NotImplementedError()

    @abstractmethod
    def process_action(self, broker: Broker, account: Account, action, time: datetime):
        """
        Called by environment instance to actually perform action, chosen by an agent.

        For example, in case of {Buy, Sell, Close} action scheme, if agent has chosen Buy,
        this method might create new MarketOrder to buy some fixed number of stocks.
        
        Parameters
        ----------
        exchange : Broker
            An underlying broker object, which accepts and executes orders produced by an action scheme.
        account : Account
            An account instance associated with a particular agent, which issued an action.
        action : Any
            Action issued by an agent
        time : datetime
            A moment in time at which action has arrived.
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def space(self) -> spaces.Space:
        raise NotImplementedError()

    def __repr__(self):
        return f'{self.__class__.__name__}()'
