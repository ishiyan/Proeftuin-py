from __future__ import annotations
from typing import Union, TYPE_CHECKING
from numbers import Real

if TYPE_CHECKING:
    from ..environment import Environment
from ..accounts import Account
from .reward_scheme import RewardScheme

class BalanceReward(RewardScheme):
    """
    A reward scheme that returns a reward equal to the difference between
    the current balance and the balance at the previous step.

    The reward can be normalized by dividing it by a constant value or by
    a value from the frame.
    """

    def __init__(self, norm_factor: Union[None, Real, str] = None):
        """
        Initializes the reward scheme.

        Args:
            norm_factor Union[None, Real, str]:
                A normalization factor.
                
                If it is a number, the reward is divided by this number.
                
                If it is a string, the reward is divided by the value of
                the attribute with this name in the last frame.

                If it is `None`, the reward is not normalized.
        """
        super().__init__()
        self.norm_factor = norm_factor
        self.last_balance = {}
    
    def reset(self):
        self.last_balance.clear()
        
    def get_reward(self, env: 'Environment', account: Account) -> float:
        if account not in self.last_balance:
            self.last_balance[account] = account.initial_balance
        reward = account.balance - self.last_balance[account]
        self.last_balance[account] = account.balance

        if isinstance(self.norm_factor, Real):
            reward = reward / float(self.norm_factor)
        elif isinstance(self.norm_factor, str):
            last_frame = env.frames[-1]
            norm = getattr(last_frame, self.norm_factor)
            reward = (reward / float(norm)) if (norm > 1e-8) else 0.0
        return reward
