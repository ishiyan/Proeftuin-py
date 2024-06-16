from typing import Union
from numbers import Real

from ..accounts import Account
from .reward_scheme import RewardScheme

class BalanceReward(RewardScheme):
    def __init__(self, norm_factor: Union[None, Real, str] = None, **kwargs):
        super().__init__(**kwargs)
        self.norm_factor = norm_factor
        self.last_balance = {}
    
    def reset(self):
        self.last_balance.clear()
        
    def get_reward(self, env, account: Account) -> float:
        reward = (account.balance - self.last_balance[account]) if (account in self.last_balance) else 0
        self.last_balance[account] = account.balance
        if isinstance(self.norm_factor, Real):
            reward = reward / float(self.norm_factor)
        elif isinstance(self.norm_factor, str):
            last_frame = env.frames[-1]
            norm = getattr(last_frame, self.norm_factor)
            reward = (reward / float(norm)) if (norm > 1e-8) else 0.0
        return reward
