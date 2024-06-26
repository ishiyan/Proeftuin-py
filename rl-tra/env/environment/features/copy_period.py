from numbers import Real
from typing import Sequence, Tuple, Union
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
                 source: Union[Tuple[str, float, float], Sequence[Tuple[str, float, float]]],
                 copy_period: int = 1):
        """
        Initializes the feature processor.

        Args:
            source (str, float, float) or Sequence[(str, float, float)]:
                Sequence of tuples of three values:                

                First: is the name of the attribute.
                Second: is the min value for `gym.spaces.Box`.
                Third: is the max value for `gym.spaces.Box`.
            copy_period int:
                Number of frames to copy to the state.

                If `copy_period` is greater than 1, each attribute is copied as an array.

                If `copy_period` is 1, each attribute is copied as an individual value.

                Default: 1
        """
        if not isinstance(copy_period, int) or copy_period < 1:
            raise ValueError(f'copy_period {copy_period} must be a positive integer')

        super().__init__(period=copy_period, write_to='state')

        if isinstance(source, Tuple) and (len(source) == 3):
            source = [source]
        elif not isinstance(source, Sequence):
            raise ValueError(f'source {source} must be a tuple of three '
                'values or a sequence of tuples of three values')

        minvals = []
        maxvals = []
        self.source = []
        for (name, minval, maxval) in source:
            if not isinstance(name, str):
                raise ValueError(f'name {name} must be a string')
            if not (isinstance(minval, float) and \
                isinstance(maxval, float) and (minval < maxval)):
                raise ValueError(f'minval {minval} and maxval {maxval} must '
                                 'be floats and minval must be less than maxval')
            self.source.append(name)
            if copy_period == 1:
                self.names.append(name)
            else:
                self.names.append(f'{name}_{copy_period}')
            minvals.append(minval)
            maxvals.append(maxval)

        self.spaces = OrderedDict({name: gym.spaces.Box(low, high, \
            shape=(copy_period,)) for name, low, high in \
            zip(self.names, minvals, maxvals)})

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
