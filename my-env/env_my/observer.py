from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import gymnasium as gym

#from .environment import Environment

class Observer(ABC):
    """Generates an observation at each step of an episode."""

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        """The observation space of the environment. (`Space`, read-only)"""
        raise NotImplementedError()

    @abstractmethod
    #def observe(self, env: 'Environment') -> Tuple[np.array, np.array, bool]:
    def observe(self, env) -> Tuple[np.array, np.array, bool]:
        """Gets the observation at the current step of an episode."""
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """Resets the observer."""
        raise NotImplementedError()
 