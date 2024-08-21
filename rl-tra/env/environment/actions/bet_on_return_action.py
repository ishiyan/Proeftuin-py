from datetime import datetime
import numpy as np
from gymnasium import spaces

from ..accounts import Account
from ..broker import Broker
from .action_scheme import ActionScheme

BET_ON_RISE = 0
BET_ON_FALL = 1
NUMBER_OF_ACTIONS = 2

class BetOnReturnAction(ActionScheme):
    """
    The action scheme, where an agent can choose from:

    `{Bet on rise, Bet on fall}`
    
    This scheme doesn't create any orders, position is always empty.    
    """
    space = spaces.Discrete(NUMBER_OF_ACTIONS)

    def __init__(self):
        """
        Initializes `BetOnReturnAction` instance.
        """
        super().__init__()

    def reset(self):
        pass

    def get_random_action(self) -> int:
        """
        Returns a random action, one of: {Bet on rise, Bet on fall}.
        """
        return np.random.randint(0, NUMBER_OF_ACTIONS)

    def get_default_action(self) -> int:
        """
        Returns the default action: `Bet on rise`.
        """
        return BET_ON_RISE
    
    def process_action(self, broker: Broker, account: Account, action, time: datetime):        
        pass
