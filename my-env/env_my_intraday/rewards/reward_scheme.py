from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..environment import Environment
from ..accounts import Account

class RewardScheme(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    def reset(self):
        """Automatically invoked by environment when it is being reset."""
        pass
    
    @abstractmethod
    def get_reward(self, env: 'Environment', account: Account) -> float:
        """
        Called by environment to get reward.
        
        Args:
            env Environment:
                The environment object.
            account Account:
                An account associated with the agent
                for which we are getting reward.
        """
        pass

