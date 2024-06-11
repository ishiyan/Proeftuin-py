from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np
import gymnasium as gym

class Trader(ABC):
    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Space:
        raise NotImplementedError()

    @abstractmethod
    def act(self, action: Any, new_price_window: np.array, finished: bool) -> Tuple[float, dict]:
        raise NotImplementedError()

    @abstractmethod
    def reset(self, initial_price_window: np.array) -> dict:
        raise NotImplementedError()
 