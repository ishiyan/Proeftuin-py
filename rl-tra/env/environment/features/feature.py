from abc import ABC, abstractmethod
from typing import Optional, Sequence, Literal, List
from collections import OrderedDict

import gymnasium as gym

from ..accounts import Account
from ..frame import Frame

class Feature(ABC):
    """
    Base class for all feature processors
    
    Args:
        write_to {'state', 'frame', 'both'}:
            Specifies where you should put your computed values into.
            `state`: put values into state's OrderedDict
            `frame`: put values into latest (newest) frame
            `both`: put values into both state and frame
        write_to_frame (bool):
            Derived from `write_to` value. If True you should write to frame.
        write_to_state (bool):
            Derived from `write_to` value. If True you should write to state.
        period (Optional[int]):
            How many latest frames you need to compute you feature's values?

            Not all features require previous frames. Specify `1` or `0` or
            `None` in this case.
        names (List[str]):
            A list of all names of values this feature instance produces.

            The order of names should match the order of their appearance in
            state (OrderedDict)
        spaces (OrderedDict[str, gym.Space]):
            An ordered dict of all names of values this feature instance outputs
            to state and their Space definitions.

            The order of names should match the order of their appearance in
            state (OrderedDict)

            Note this dict must be empty in case when `write_to='frame'`.

            It contains only names which are actually being written into state.
    """
    def __init__(self,
                 write_to: Literal['state', 'frame', 'both'] = 'state',
                 period: Optional[int] = None):
        if not isinstance(write_to, str) or write_to not in \
                                            {'state', 'frame', 'both'}:
            raise ValueError(f"write_to {write_to} must be one of 'state', "
                             "'frame', or 'both'")

        if period is not None and (not isinstance(period, int) or period < 0):
            raise ValueError(f'period {period} must be a positive integer or None')

        self.write_to = write_to
        self.write_to_frame: bool = write_to in {'frame', 'both'}
        self.write_to_state: bool = write_to in {'state', 'both'}
        self.period = period
        self.names: List[str] = []
        self.spaces: OrderedDict[str, gym.Space] = OrderedDict()
        self.account: Optional[Account] = None
    
    def reset(self):
        """
        Cleanup and reset internal state between episodes.
        
        This method is automatically invoked by Environment
        instance upon its reset.
        """
        pass

    @abstractmethod
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        """
        Compute and write features values when a new frame arrives.
        
        Important: this method is invoked only ONCE for each new frame.
        
        Args:
            frames (Sequence[Frame]):
                Contains a list of latest frames.
                `frames[-1]`: the newest frame.
                `frames[-2]`: the frame before the newest one.
            state (OrderedDict):
                The common state dict to write feature values to.
                
                This dict collects values from all the features of
                the features pipeline, and then passes those values
                to an agent for it to make next action.

                Write your values directly into this dict, for example:
                ```python
                state['ema_price'] = self.ema_price
                ```

                Ensure that you only use names as they are specified in
                `self.names` and `self.spaces`.
        """
        pass
    
    def __str__(self):
        return self.__repr__()

