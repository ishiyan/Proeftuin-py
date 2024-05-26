from time import time
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

class Actions(Enum):
    Sell = 0
    Buy = 1

class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class TradingEnv(gym.Env):
    """
    TradingEnv is an abstract class which inherits `gym.Env`.

    * Properties:

    `df`: An abbreviation for `DataFrame`. It's a `pandas' DataFrame` which contains your dataset and is passed in the class' constructor.

    `prices`: Real prices over time. Used to calculate profit and render the environment.

    `signal_features`: Extracted features over time. Used to create `Gym observations`.

    `window_size`: Number of ticks (current and previous ticks) returned as a `Gym observation`. It is passed in the class' constructor.

    `action_space`: The `Gym action_space` property. Containing discrete values of `Sell` and `Buy`.

    `observation_space`: The `Gym observation_space` property. Each observation is a window on `signal_features` from index `current_tick - window_size + 1` to `current_tick`.
    So `_start_tick` of the environment would be equal to `window_size`. In addition, initial value for `_last_trade_tick` is `window_size - 1` .

    `shape`: Shape of a single observation.

    `history`: Stores the information of all steps.

    * Methods:

    `seed`: Typical `Gym seed` method.

    `reset`: Typical `Gym reset` method.

    `step`: Typical `Gym step` method.

    `render`: Typical `Gym render` method. Renders the information of the environment's current tick.

    `render_all`: Renders the whole environment.

    `close`: Typical `Gym close` method.

    * Abstract Methods:

    `_process_data`: It is called in the constructor and returns `prices` and `signal_features` as a tuple.
    In different trading markets, different features need to be obtained. So this method enables our `TradingEnv`
    to be a general-purpose environment and specific features can be returned for specific environments.

    `_calculate_reward`: The reward function for the RL agent.

    `_update_profit`: Calculates and updates total profit which the RL agent has achieved so far.
    Profit indicates the amount of units of currency you have achieved by starting with
    `1.0* unit (Profit = FinalMoney / StartingMoney)`.

    `max_possible_profit`: The maximum possible profit that an RL agent can obtain regardless of trade fees.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(self, df, window_size, render_mode=None):
        assert df.ndim == 2

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.df = df
        self.window_size = window_size # window size for observation lookback
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = gym.spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float32,
        )

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))

        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action):
        self._truncated = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._truncated = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, step_reward, False, self._truncated, info

    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position
        )

    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _render_frame(self):
        self.render()

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        start_time = time()

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        end_time = time()
        process_time = end_time - start_time

        pause_time = (1 / self.metadata['render_fps']) - process_time
        assert pause_time > 0., "High FPS! Try to reduce the 'render_fps' value."
        plt.pause(pause_time)

    def render_all(self, title=None):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError

    def max_possible_profit(self): # trade fees are ignored
        raise NotImplementedError
