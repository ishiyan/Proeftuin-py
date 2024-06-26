from collections import OrderedDict
from typing import Sequence, Literal
import math

import gymnasium as gym

from .feature import Feature
from ..frame import Frame

class AccountROI(Feature):
    """
    Account performance: Return On Investment.
    """
    def __init__(self,
                 write_to: Literal['state', 'frame', 'both'] = 'state'
                ):
        super().__init__(write_to=write_to)
        self.names = ['roi','zscore_roi']
        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(-math.inf, math.inf, shape=(1,)) for name in self.names})

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        if self.account is None:
            raise ValueError('this feature requires an account to be set by the environment')
        if len(self.account.report.returns_on_investments) == 0:
            roi = 0.0
            roi_zscore = 0.0
        else:
            roi = self.account.report.returns_on_investments[-1]
            if roi is None:
                roi = 0.0
            roi_mean = self.account.report.roi_mean()
            if roi_mean is None:
                roi_mean = 0.0
            roi_std = self.account.report.roi_std()
            if roi_std is None:
                roi_std = 0.0
            if roi_std > 1e-8:
                roi_zscore = (roi - roi_mean) / roi_std
            else:
                roi_zscore = 0.0
        frame = frames[-1]
        if self.write_to_frame:
            setattr(frame, self.names[0], roi)
            setattr(frame, self.names[1], roi_zscore)
        if self.write_to_state:
            state[self.names[0]] = roi
            state[self.names[1]] = roi_zscore
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(write_to={self.write_to})')
