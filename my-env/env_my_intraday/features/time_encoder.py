from typing import Sequence, Union, Literal
from collections import OrderedDict

import gymnasium as gym

from ..frame import Frame
from .feature import Feature

class TimeEncoder(Feature):
    """
    Encodes timestamp into format a model can understand.

    Machine Learning models can't efficiently utilize raw timestamp values.

    Given a timestamp they can hardly say:
    - is it summer or winter?
    - is it monday or friday?
    - is it morning or noon?

    We can convert timestamp into some floating numbers to make it easier
    for models to use it:
    - `day of year` offset: [0,1], where 0 - January 1st and 1- December 31
    - `day of week` offset: [0,1], where 0 - Monday, 1 - Sunday
    - `time of day` offset: [0,1], where 0 - 00:00:00, 1 - 23:59:59

    Args:
        source (str or Sequence[str]):
            Names of Frame's time attributes for which to compute the values.
        yday (bool):
            If output the day of year. Default: True
        wday (bool):
            If output the day of week. Default: True
        tday (bool):
            If output the time of day. Default: True
        write_to {'frame','state','both'}:
            Destination of where to put computed values.
    """
    
    def __init__(self,
                 source: Union[str, Sequence[str]] = 'time_start',
                 yday=True,
                 wday=True,
                 tday=True,
                 write_to: Literal['frame', 'state', 'both'] = 'state'):
        super().__init__(write_to=write_to)
        if isinstance(source, str):
            self.source = [source]
        elif isinstance(source, Sequence):
            self.source = source
        else:
            raise ValueError(f'source {source} must be a string '
                             'or a sequence of strings')
        self.yday = yday
        self.wday = wday
        self.tday = tday
        self.length = 0
        self.length += 1 if yday else 0
        self.length += 1 if wday else 0
        self.length += 1 if tday else 0
        for name in self.source:
            assert isinstance(name, str)
            if self.yday:
                self.names.append(f'yday_{name}')
            if self.wday:
                self.names.append(f'wday_{name}')
            if self.tday:
                self.names.append(f'tday_{name}')
        if write_to in {'state', 'both'} and self.length > 0:
            self.spaces = OrderedDict({name: gym.spaces.Box(0, 1, shape=(1,)) for name in self.names})
        else:
            self.spaces = OrderedDict()
    
    def process(self, frames: Sequence[Frame], state: OrderedDict):
        if self.length == 0:
            return
        last_frame = frames[-1]
        for i, name in enumerate(self.source):
            datetime = getattr(last_frame, name)
            # Calculate day of year offset.
            year_start = datetime.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            year_end = datetime.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
            day_start = datetime.replace(hour=0, minute=0, second=0, microsecond=0)
            if self.yday:
                yday = (day_start - year_start) / (year_end - year_start)
            else:
                yday = 0.0
            # Calculate day of week offset.
            if self.wday:
                wday = datetime.isoweekday() / 7
            else:
                wday = 0.0
            # Calculate time of day offset.
            if self.tday:
                time = (datetime.timestamp() - day_start.timestamp()) / float(24 * 60 * 60)
            else:
                time = 0.0
            # Write out values.
            off = self.length * i
            if self.write_to_frame:
                cnt = 0
                if self.yday:
                    setattr(last_frame, self.names[off], yday)
                    cnt += 1
                if self.wday:
                    setattr(last_frame, self.names[off + cnt], wday)
                    cnt += 1
                if self.tday:
                    setattr(last_frame, self.names[off + cnt], time)
            if self.write_to_state:
                cnt = 0
                if self.yday:
                    state[self.names[off]] = yday
                    cnt += 1
                if self.wday:
                    state[self.names[off + cnt]] = wday
                    cnt += 1
                if self.tday:
                    state[self.names[off + cnt]] = time
    
    def __repr__(self):
        return f'{self.__class__.__name__}(source={self.source}, write_to={self.write_to})'
