from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..environment import Environment
from ..actions import BetOnReturnAction
from ..accounts import Account
from .reward_scheme import RewardScheme

class BetOnReturnReward(RewardScheme):
    """
    A reward scheme that rewards a betting on return.

    If the last betting action was BetOnReturnAction.BET_ON_RISE,
    the reward is the current return. If return is positive,
    the reward is also positive. If the return is negative, the
    reward is negative.

    If the last betting action was BetOnReturnAction.BET_ON_FALL,
    the reward is the negative of the current return. If return is
    positive, the reward is negative. If the return is negative,
    the reward is positive.

    The return may be arithmetic (the total reward will be equal
    to the arithmetic sum of returns) or geometric (the total reward
    will be equal to the cumulative return).
    """

    def __init__(self, geometric: bool = True):
        """
        Initializes the reward scheme.
        Args:
            geometric: bool
                If true, reward is an increment in cumulative
                return.
                
                If false, reward is a raw return.
        """
        super().__init__()
        self.geometric = geometric
        self.previous_price: float = None
        self._cumulative_return_plus_1: float = None
    
    def reset(self):
        self.previous_price = None
        self._cumulative_return_plus_1 = 1.0
        
    def get_reward(self, env: 'Environment', account: Account) -> float:
        if self.previous_price is None:
            self.previous_price = env.frames[-2].close
        price = env.frames[-1].close
        ret = price / self.previous_price - 1.0
        self.previous_price = price

        if env.last_action == BetOnReturnAction.BET_ON_FALL:
            ret = -ret

        if not self.geometric:
            return ret
        
        prev = self._cumulative_return_plus_1
        self._cumulative_return_plus_1 *= 1.0 + ret
        return self._cumulative_return_plus_1 - prev
