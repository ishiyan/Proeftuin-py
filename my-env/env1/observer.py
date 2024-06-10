from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import gymnasium as gym

class Observer(ABC):
    """Generates an observation at each step of an episode."""

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        """The observation space of the environment. (`Space`, read-only)"""
        raise NotImplementedError()

    @abstractmethod
    def observe(self) -> Tuple[np.array, np.array]:
        """Gets the observation at the current step of an episode."""
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """Resets the observer."""
        raise NotImplementedError()

class DataObserver(Observer):
    def __init__(self, df, window_size, frame_bound):
        assert df.ndim == 2
        self.df = df
        self.window_size = window_size # window size for observation lookback
        self.frame_bound = frame_bound

        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])
        self._start_idx = self.window_size
        self._end_idx = len(self.prices) - 1
        self._current_idx = None

        INF = 1e10
        self.observation_space = gym.spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float32,
        )

    def _process_data(self):
        prices = self.df.loc[:, 'close'].to_numpy()
        open = self.df.loc[:, 'f_open'].to_numpy()
        high = self.df.loc[:, 'f_high'].to_numpy()
        low = self.df.loc[:, 'f_low'].to_numpy()
        close = self.df.loc[:, 'f_close'].to_numpy()
        volume = self.df.loc[:, 'f_volume'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0] - self.window_size : self.frame_bound[1]]
        open = open[self.frame_bound[0] - self.window_size : self.frame_bound[1]]
        high = high[self.frame_bound[0] - self.window_size : self.frame_bound[1]]
        low = low[self.frame_bound[0] - self.window_size : self.frame_bound[1]]
        close = close[self.frame_bound[0] - self.window_size : self.frame_bound[1]]
        volume = volume[self.frame_bound[0] - self.window_size : self.frame_bound[1]]

        signal_features = np.column_stack((open, high, low, close, volume))

        return prices.astype(np.float32), signal_features.astype(np.float32)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    def observe(self) -> Tuple[np.array, np.array]:
        obs = self.signal_features[(self._current_idx - self.window_size+1):self._current_idx+1]
        prices = self.prices[(self._current_idx - self.window_size+1):self._current_idx+1]
        self._current_idx += 1
        return obs, prices

    def reset(self) -> None:
        self._current_idx = self._start_idx
 