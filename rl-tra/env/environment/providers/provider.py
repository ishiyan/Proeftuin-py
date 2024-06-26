from abc import ABC, abstractmethod
from typing import Union, Optional
from numbers import Real
import numpy as np
from datetime import timedelta, datetime

class Provider(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def reset(self,
              episode_min_duration: Union[None, Real, timedelta] = None,
              rng: Optional[np.random.RandomState] = None) -> datetime:
        pass

    @abstractmethod
    def close(self):
        pass

    def __iter__(self):
        self.reset()
        return self

    @abstractmethod    
    def __next__(self):
        pass

    @property
    @abstractmethod    
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def session_start_datetime(self) -> Optional[datetime]:
        pass

    @property
    @abstractmethod
    def session_end_datetime(self) -> Optional[datetime]:
        pass

    @property
    @abstractmethod
    def episode_start_datetime(self) -> Optional[datetime]:
        pass
