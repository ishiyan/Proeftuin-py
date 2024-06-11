from typing import Tuple
import numpy as np
import gymnasium as gym

from .observer import Observer
from .environment import Environment

class DataframeObserver(Observer):
    def __init__(self, df, frame_bound, window_size):
        assert df.ndim == 2
        self.df = df
        self.window_size = window_size # window size for observation lookback
        self.frame_bound = frame_bound

        self.features, self.ohlcvs = self._process_data()
        self.shape = (window_size, self.features.shape[1])
        self._start_idx = self.window_size
        self._end_idx = len(self.ohlcvs) - 1
        self._current_idx = None

        INF = 1 # 1e10
        self._observation_space = gym.spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float32,
        )

    def _process_data(self) -> Tuple[np.array, np.array]:
        print('observer process data')
        df = self.df.copy()
        df = df[self.frame_bound[0] - self.window_size : self.frame_bound[1]]

        open = df.loc[:, 'f_open'].to_numpy()
        high = df.loc[:, 'f_high'].to_numpy()
        low = df.loc[:, 'f_low'].to_numpy()
        close = df.loc[:, 'f_close'].to_numpy()
        volume = df.loc[:, 'f_volume'].to_numpy()
        features = np.column_stack((open, high, low, close, volume))

        open = df.loc[:, 'open'].to_numpy()
        high = df.loc[:, 'high'].to_numpy()
        low = df.loc[:, 'low'].to_numpy()
        close = df.loc[:, 'close'].to_numpy()
        volume = df.loc[:, 'volume'].to_numpy()
        ohlcvs = np.column_stack((open, high, low, close, volume))

        return features.astype(np.float32), ohlcvs.astype(np.float32)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    def observe(self, env: 'Environment') -> Tuple[np.array, np.array, bool]:
        #print('observe:', env.__class__.__name__)
        features = self.features[(self._current_idx - self.window_size+1):self._current_idx+1]
        ohlcvs = self.ohlcvs[(self._current_idx - self.window_size+1):self._current_idx+1]
        self._current_idx += 1
        truncated = self._current_idx > self._end_idx
        return features, ohlcvs, truncated

    def reset(self) -> None:
        self._current_idx = self._start_idx
 