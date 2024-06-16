from typing import Union, Optional, Type, NamedTuple
from numbers import Real
from collections import namedtuple
import numpy as np
from datetime import timedelta, datetime

class Provider(object):
    def __init__(self, **kwargs):
        pass

    def reset(self,
              episode_min_duration: Union[None, Real, timedelta] = None,
              rng: Optional[np.random.RandomState] = None,
              **kwargs
              ) -> datetime:
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def __iter__(self):
        self.reset()
        return self
    
    def __next__(self):
        raise NotImplementedError()
    
    @property
    def kind(self) -> Type[NamedTuple]:
        """Returns type of objects which this provider outputs. One of of namedtuple: Trade, TradeOI, Candle, Kline"""
        raise NotImplementedError()

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def session_start_datetime(self) -> Union[datetime, None]:
        raise NotImplementedError()

    @property
    def session_end_datetime(self) -> Union[datetime, None]:
        raise NotImplementedError()

    @property
    def episode_start_datetime(self) -> Union[datetime, None]:
        raise NotImplementedError()
