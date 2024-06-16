from ..accounts import Account
from .reward_scheme import RewardScheme

class ConstantReward(RewardScheme):
    def __init__(self, value: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.value = value
    
    def reset(self):
        pass
    
    def get_reward(self, env, account: Account) -> float:
        return self.value
