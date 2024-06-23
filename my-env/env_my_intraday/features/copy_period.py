from typing import Sequence, Union
from collections import OrderedDict
import math

import gymnasium as gym

from ..frame import Frame
from .feature import Feature

class CopyPeriod(Feature):
    """
    Copies some attributes from a period of last frames to state.

    The `copy_period` argument specifies how many frames to copy to the output and defaults to 1.

    If `copy_period` is greater than 1, each attribute is copied as an array.

    If `copy_period` is 1, each attribute is copied as an individual value.

    Args:
    ----------
    source : str or Sequence[str] or None
        Names of Frame's attributes to be copied into state object.
        If None - all attributes are copied into state.
    """
    
    def __init__(self,
                 source: Union[str, Sequence[str]],
                 copy_period: int = 1):
        """
        Initializes the feature processor.

        Args:
            source str or Sequence[str]:
                Names of Frame's attributes to copy.
            copy_period int:
                Number of frames to copy to the state.

                If `copy_period` is greater than 1, each attribute is copied as an array.

                If `copy_period` is 1, each attribute is copied as an individual value.

                Default: 1
        """
        if not isinstance(copy_period, int) or copy_period < 1:
            raise ValueError(f'copy_period {copy_period} must be a positive integer')

        super().__init__(period=copy_period, write_to='state')

        if isinstance(source, str):
            self.source = [source]
        elif isinstance(source, Sequence):
            self.source = source
        else:
            raise ValueError(f'source {source} must be a string '
                             'or a sequence of strings')

        prefixes_zero_one = [
            'minmax', 'yday_', 'wday_', 'tday_', 'cl_hl', 'ol_hl',
            'market_dimension_', 'efficiency_ratio_'
        ]
        prefixes_min_point_five_point_five = ['stoch_']        
        prefixes_min_one_one = ['snapshot_', 'cmf_']        
        prefixes_zero_inf = ['abnormal_', 'abs_']        
        mins = []
        maxs = []
        for name in self.source:
            assert isinstance(name, str)
            # The right way is to pass the min and max values as arguments,
            # but for now we will just check for known names.
            if any(name.startswith(prefix) for prefix in prefixes_zero_one):
                mi = 0.0
                ma = 1.0
            elif any(name.startswith(prefix) for prefix in prefixes_min_one_one):
                mi = -1.0
                ma = 1.0
            elif any(name.startswith(prefix) for prefix in prefixes_min_point_five_point_five):
                mi = -0.5
                ma = 0.5
            elif any(name.startswith(prefix) for prefix in prefixes_zero_inf):
                mi = 0.0
                ma = math.inf
            else:
                mi = -math.inf
                ma = math.inf
            mins.append(mi)
            maxs.append(ma)
            if copy_period == 1:
                self.names.append(name)
            else:
                self.names.append(f'{name}_{copy_period}')

        self.spaces = OrderedDict({name: gym.spaces.Box(low, high, \
            shape=(copy_period,)) for name, low, high in \
            zip(self.names, mins, maxs)})

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        if self.period == 1:
            last_frame = frames[-1]
            for name in self.source:
                state[name] = getattr(last_frame, name)
        else:
            window = frames[-self.period:]
            for i, name in enumerate(self.source):
                result = [getattr(frame, name) for frame in window]
                state[self.names[i]] = result
    
    def __repr__(self):
        return f'{self.__class__.__name__}(source={self.source}, period={self.period})'
