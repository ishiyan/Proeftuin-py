from collections import OrderedDict
from typing import Sequence, Literal
import math

import gymnasium as gym

from .feature import Feature
from ..frame import Frame

class AccountSharpe(Feature):
    """
    Account performance: Sharpe ratio.
    """
    def __init__(self,
                 write_to: Literal['state', 'frame', 'both'] = 'state'
                ):
        super().__init__(write_to=write_to)
        self.names = ['sharpe']
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        if self.account is None:
            raise ValueError('this feature requires an account to be set by the environment')
        value = self.account.performance.daily.sharpe_ratio()
        if value is None:
            value = 0.0
        frame = frames[-1]
        if self.write_to_frame:
            setattr(frame, self.names[0], value)
        if self.write_to_state:
            state[self.names[0]] = value
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(write_to={self.write_to})')
