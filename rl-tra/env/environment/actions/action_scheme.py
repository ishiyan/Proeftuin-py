from abc import ABC, abstractmethod
from datetime import datetime
from gymnasium import spaces

from ..accounts import Account
from ..broker import Broker

class ActionScheme(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    def reset(self):
        """Automatically invoked by environment when it is being reset."""
        pass
    
    @abstractmethod
    def get_random_action(self):
        """Returns some random action."""
        pass

    @abstractmethod
    def get_default_action(self):
        """Returns some default action."""
        pass

    @abstractmethod
    def process_action(self, broker: Broker, account: Account, action, time: datetime):
        """
        Called by environment to actually perform action chosen by an agent.

        For example, in case of {`Buy`, `Sell`, `Hold`} action scheme,
        if agent has chosen `Buy`, this method might create a new `MarketOrder`
        to buy some fixed quantity of the asset.
        
        Args:
            broker Broker:
                An underlying broker object, which accepts and executes
                orders produced by an action scheme.
            account Account:
                An account associated with a particular agent,
                which issued an action.
            action Any:
                Action issued by an agent.
            time datetime:
                A moment in time at which action has arrived.
        """
        pass
    
    @property
    @abstractmethod
    def space(self) -> spaces.Space:
        pass
