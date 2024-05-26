from cmath import inf
from typing import Any, List
from abc import abstractmethod
from enum import Enum
import numbers

import os
import copy
import pandas as pd
import numpy as np
import torch

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

class Actions(int, Enum):
    DOUBLE_SELL = 0
    SELL = 1
    HOLD = 2
    BUY = 3
    DOUBLE_BUY = 4

class Positions(int, Enum):
    SHORT = -1.
    FLAT = 0.
    LONG = 1.

def transform(position: Positions, action: int): # -> Any: #tuple[Positions, bool]:
    '''
    This func is used to transform the env's position from
    the input (position, action) pair according to the state machine.
    
    Arguments:
        - position(Positions) : Long, Short, Flat
        - action(int) : Buy, Hold, Sell
    
    Returns:
        - tuple containing the position after transformation and a boolean indicating if a trade should be done.
    '''
    if action == Actions.SELL:
        if position == Positions.LONG:
            return Positions.FLAT, False
        if position == Positions.FLAT:
            return Positions.SHORT, True

    if action == Actions.BUY:
        if position == Positions.SHORT:
            return Positions.FLAT, False
        if position == Positions.FLAT:
            return Positions.LONG, True

    if action == Actions.DOUBLE_SELL and (position == Positions.LONG or position == Positions.FLAT):
        return Positions.SHORT, True

    if action == Actions.DOUBLE_BUY and (position == Positions.SHORT or position == Positions.FLAT):
        return Positions.LONG, True

    return position, False

def to_ndarray(item: Any, dtype: np.dtype = None) -> Any:
    """
    Overview:
        Convert ``torch.Tensor`` to ``numpy.ndarray``.
    Arguments:
        - item (:obj:`Any`): The ``torch.Tensor`` objects to be converted. It can be exactly a ``torch.Tensor`` \
            object or a container (list, tuple or dict) that contains several ``torch.Tensor`` objects.
        - dtype (:obj:`np.dtype`): The type of wanted array. If set to ``None``, its dtype will be unchanged.
    Returns:
        - item (:obj:`object`): The changed arrays.

    Examples (ndarray):
        >>> t = torch.randn(3, 5)
        >>> tarray1 = to_ndarray(t)
        >>> assert tarray1.shape == (3, 5)
        >>> assert isinstance(tarray1, np.ndarray)

    Examples (list):
        >>> t = [torch.randn(5, ) for i in range(3)]
        >>> tarray1 = to_ndarray(t, np.float32)
        >>> assert isinstance(tarray1, list)
        >>> assert tarray1[0].shape == (5, )
        >>> assert isinstance(tarray1[0], np.ndarray)

    .. note:

        Now supports item type: :obj:`torch.Tensor`,  :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`.
    """

    def transform(d):
        if dtype is None:
            return np.array(d)
        else:
            return np.array(d, dtype=dtype)

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            new_data[k] = to_ndarray(v, dtype)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return None
        elif isinstance(item[0], numbers.Integral) or isinstance(item[0], numbers.Real):
            return transform(item)
        elif hasattr(item, '_fields'):  # namedtuple
            return type(item)(*[to_ndarray(t, dtype) for t in item])
        else:
            new_data = []
            for t in item:
                new_data.append(to_ndarray(t, dtype))
            return new_data
    elif isinstance(item, torch.Tensor):
        if dtype is None:
            return item.numpy()
        else:
            return item.numpy().astype(dtype)
    elif isinstance(item, np.ndarray):
        if dtype is None:
            return item
        else:
            return item.astype(dtype)
    elif isinstance(item, bool) or isinstance(item, str):
        return item
    elif np.isscalar(item):
        if dtype is None:
            return np.array(item)
        else:
            return np.array(item, dtype=dtype)
    elif item is None:
        return None
    else:
        raise TypeError("not support item type: {}".format(type(item)))

class TradingEnv(gym.Env):

    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(self,
        env_id='stocks-v0',
        eps_length=253, # one trading year, episode length
        window_size=20, # associated with the feature length
        plot_save_path='./', # the path to save result image
        plot_freq = 10,
        # the stocks range percentage used by train/test.
        # if one of them is None, train & test set will use all data by default.
        train_range=None,
        test_range=None,
        start_idx=None,
        render_mode=None):

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        
        self._env_id = env_id
        #======== plotting params =========
        self.cnt = 0
        self.plot_freq = plot_freq
        self.plot_save_path = plot_save_path
        #================================
        self.start_idx = start_idx
        self.train_range = train_range
        self.test_range =test_range
        self.eps_length = eps_length
        self.window_size = window_size
        self.prices = None
        self.signal_features = None
        self.feature_dim_len = None
        self.shape = (window_size, 3)
        #======== episode params =========
        self._start_tick = 0
        self._end_tick = 0
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._truncated = None
        #======================================
        self._init_flag = True
    
        # init the following variables variable at first reset
        self._action_space = None
        self._observation_space = None
        self._reward_space = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.cnt += 1
        self.prices, self.signal_features, self.feature_dim_len = self._process_data()
        if self._init_flag:
            self.shape = (self.window_size, self.feature_dim_len)
            self._action_space = spaces.Discrete(len(Actions))
            self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)
            self._reward_space = gym.spaces.Box(-inf, inf, shape=(1, ), dtype=np.float32)
            self._init_flag = False
        self._done = False
        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.FLAT
        self._position_history = [self._position]
        self._profit_history = [1.]
        self._total_reward = 0.
        #self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))

        return self._get_observation()

    def random_action(self): # -> Any:
        return np.array([self.action_space.sample()])

    def step(self, action): #np.ndarray):
        #assert isinstance(action, np.ndarray), type(action)
        if action.shape == (1, ):
            action = action.item()  # 0-dim array

        self._done = False
        self._truncated = False
        self._current_tick += 1

        if self._current_tick >= self._end_tick:
            self._truncated = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._position, trade = transform(self._position, action)

        if trade:
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        self._profit_history.append(float(np.exp(self._total_reward)))
        observation = self._get_observation()
        info = dict(total_reward=self._total_reward, position=self._position.value)

        if self._done:
            if self._env_id[-1] == 'e' and self.cnt % self.plot_freq == 0:
                self.render()
            info['max_possible_profit'] = np.log(self.max_possible_profit())
            info['eval_episode_return'] = self._total_reward

        step_reward = to_ndarray([step_reward]).astype(np.float32)
        return observation, step_reward, False, self._truncated, info

    def _get_observation(self): # -> np.ndarray:
        obs = to_ndarray(self.signal_features[(self._current_tick - self.window_size + 1):self._current_tick + 1]
                         ).reshape(-1).astype(np.float32)

        tick = (self._current_tick - self._last_trade_tick) / self.eps_length
        obs = np.hstack([obs, to_ndarray([self._position.value]), to_ndarray([tick])]).astype(np.float32)
        return obs

    def render(self): # -> None:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.xlabel('trading days')
        plt.ylabel('profit')
        plt.plot(self._profit_history)
        plt.savefig(self.plot_save_path + str(self._env_id) + "-profit.png")

        plt.clf()
        plt.xlabel('trading days')
        plt.ylabel('close price')
        window_ticks = np.arange(len(self._position_history))
        eps_price = self.raw_prices[self._start_tick:self._end_tick + 1]
        plt.plot(eps_price)

        short_ticks = []
        long_ticks = []
        flat_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.SHORT:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.LONG:
                long_ticks.append(tick)
            else:
                flat_ticks.append(tick)

        plt.plot(long_ticks, eps_price[long_ticks], 'g^', markersize=3, label="Long")
        plt.plot(flat_ticks, eps_price[flat_ticks], 'bo', markersize=3, label="Flat")
        plt.plot(short_ticks, eps_price[short_ticks], 'rv', markersize=3, label="Short")
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
        plt.savefig(self.plot_save_path + str(self._env_id) + '-price.png')

    def close(self):
        import matplotlib.pyplot as plt
        plt.close()

    @abstractmethod
    def _process_data(self):
        raise NotImplementedError

    @abstractmethod
    def _calculate_reward(self, action):
        raise NotImplementedError

    @abstractmethod
    def max_possible_profit(self):
        raise NotImplementedError

    @property
    def observation_space(self): # -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self): # -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self): # -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self): # -> str:
        return "Trading Env"
