from ..accounts import Account

class RewardScheme(object):
    def __init__(self, **kwargs):
        pass
    
    def reset(self):
        raise NotImplementedError()
    
    def get_reward(self, env, account: Account) -> float:
        raise NotImplementedError()
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'

