from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..environment import Environment
from ..accounts import Account
from .reward_scheme import RewardScheme

class ConstantReward(RewardScheme):
    """
    A reward scheme that returns a constant reward value.
    """

    def __init__(self, value: float = 1.0):
        """
        Initializes the reward scheme.

        Args:
            value float:
                The value of the reward.
        """
        super().__init__()
        self.value = value
    
    def reset(self):
        pass
    
    def get_reward(self, env: 'Environment', account: Account) -> float:
        return self.value
