from typing import Optional, Sequence, Literal, Tuple
from collections import OrderedDict

import gymnasium as gym

from ..frame import Frame
from .feature import Feature

class OhlcRatios(Feature):
    """
    Ratios of OHLC values, all three bounded to `[0, 1]`:
    
    - `l_h`: `low/high` ratio
    - `cl_hl`: `(close - low)/(high - low)` ratio
    - `ol_hl`: `(open - low)/(high - low)` ratio

    If `high == low`, `cl_hl` and `ol_hl` are set to `0`.
    """

    def __init__(self,
                 source: Optional[Tuple[str, str, str, str]] = None,
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        """
        Initializes the feature processor

        Args:
            source Tuple[str, str, str, str], optional:
                Names for `open`, `high`, `low` and `close` of frame's attributes.                

                If None, defaults to ('open', 'high', 'low', 'close').

                When `source` is provided, the names are used to access the
                corresponding attributes of the frame.
                
                If at least one of the names is different from
                `('open', 'high', 'low', 'close')`,
                the name of the features will look like this:

                - `l_h_{source[0]}_{source[1]}_{source[2]}_{source[3]}`
                - `cl_hl_{source[0]}_{source[1]}_{source[2]}_{source[3]}`
                - `ol_hl_{source[0]}_{source[1]}_{source[2]}_{source[3]}`

                If all names are the same as the default ones or `source` is
                `None`, the names of the features will be:

                - `l_h`
                - `cl_hl`
                - `ol_hl`
            write_to str {'frame','state','both'}:
                Destination of where to put computed values.
        """
        super().__init__(write_to=write_to)
        standard_names = True
        if source is None:
            source = ('open', 'high', 'low', 'close')
        else:            
            for name in source:
                if name not in ('open', 'high', 'low', 'close'):
                    standard_names = False
                    break
        self.source = source
        if standard_names:
            self.names = ['l_h','cl_hl', 'ol_hl']
        else:
            self.names = [
                f'l_h_{self.source[0]}_{self.source[1]}_{self.source[2]}_{self.source[3]}',
                f'cl_hl_{self.source[0]}_{self.source[1]}_{self.source[2]}_{self.source[3]}',
                f'ol_hl_{self.source[0]}_{self.source[1]}_{self.source[2]}_{self.source[3]}'
            ]

        if write_to in {'state', 'both'}:
            self.spaces = OrderedDict({name: gym.spaces.Box(0., 1., shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        frame = frames[-1]
        open, high = getattr(frame, self.source[0]), getattr(frame, self.source[1])
        low, close = getattr(frame, self.source[2]), getattr(frame, self.source[3])
        delta = high - low
        if delta == 0:
            l_h = 1.0
            cl_hl = 0.0
            ol_hl = 0.0
        else:
            l_h = low / high
            cl_hl = (close - low) / delta
            ol_hl = (open - low) / delta

        if self.write_to_frame:
            setattr(frame, self.names[0], l_h)
            setattr(frame, self.names[1], cl_hl)
            setattr(frame, self.names[2], ol_hl)
        if self.write_to_state:
            state[self.names[0]] = l_h
            state[self.names[1]] = cl_hl
            state[self.names[2]] = ol_hl

    def __repr__(self):
        return f'{self.__class__.__name__}(write_to={self.write_to})'
