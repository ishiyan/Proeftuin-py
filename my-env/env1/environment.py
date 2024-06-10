import uuid
import logging

from typing import Dict, Any, Tuple
from random import randint

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import datetime

from tensortrade.core import TimeIndexed, Clock, Component
from tensortrade.env.generic import (
    ActionScheme,
    RewardScheme,
    Observer,
    Stopper,
    Informer,
    Renderer
)

class TradingEnv(gym.Env):
    """A trading environment made for use with Gym-compatible reinforcement
    learning algorithms.

    Parameters
    ----------
    action_scheme : `ActionScheme`
        A component for generating an action to perform at each step of the
        environment.
    reward_scheme : `RewardScheme`
        A component for computing reward after each step of the environment.
    observer : `Observer`
        A component for generating observations after each step of the
        environment.
    informer : `Informer`
        A component for providing information after each step of the
        environment.
    renderer : `Renderer`
        A component for rendering the environment.
    kwargs : keyword arguments
        Additional keyword arguments needed to create the environment.
    """

    agent_id: str = None
    episode_id: str = None

    def __init__(self,
                 action_scheme: ActionScheme,
                 reward_scheme: RewardScheme,
                 observer: Observer,
                 stopper: Stopper,
                 informer: Informer,
                 renderer: Renderer,
                 min_periods: int = None,
                 max_episode_steps: int = None,
                 random_start_pct: float = 0.00,
                 **kwargs) -> None:
        super().__init__()

        self.action_scheme = action_scheme
        self.reward_scheme = reward_scheme
        self.observer = observer
        self.stopper = stopper
        self.informer = informer
        self.renderer = renderer
        self.min_periods = min_periods
        self.random_start_pct = random_start_pct

        # Register the environment in Gym and fetch spec
        gym.envs.register(
            id='Mbrane-v0',
            max_episode_steps=max_episode_steps,
        )
        self.spec = gym.spec(env_id='Mbrane-v0')

        self.action_space = action_scheme.action_space
        self.observation_space = observer.observation_space

        self._enable_logger = kwargs.get('enable_logger', False)
        if self._enable_logger:
            self.logger = logging.getLogger(kwargs.get('logger_name', __name__))
            self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))

    def step(self, action: Any) -> 'Tuple[np.array, float, bool, dict]':
        """Makes one step through the environment.
gymnasium.Env.step(self, action: ActType) → tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]
        Parameters
        ----------
        action : Any
            An action to perform on the environment.

        Returns
        -------
        `np.array`
            The observation of the environment after the action being
            performed.
        float
            The computed reward for performing the action.
        bool
            Whether or not the episode is complete.
        dict
            The information gathered after completing the step.
        """
        self.action_scheme.perform(self, action)

        obs = self.observer.observe(self)
        reward = self.reward_scheme.reward(self)
        done = self.stopper.stop(self)
        info = self.informer.info(self)

        self.clock.increment()

        return obs, reward, done, info

    # gymnasium.Env.reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) → tuple[ObsType, dict[str, Any]]
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.array, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        if self.random_start_pct > 0.00:
            size = len(self.observer.feed.process[-1].inputs[0].iterable)
            random_start = randint(0, int(size * self.random_start_pct))
        else:
            random_start = 0

        self.episode_id = str(uuid.uuid4())

        for c in self.components.values():
            if hasattr(c, "reset"):
                if isinstance(c, Observer):
                    c.reset(random_start=random_start)
                else:
                    c.reset()

        obs = self.observer.observe(self)

        return obs

    def render(self, **kwargs) -> None:
        """Renders the environment."""
        self.renderer.render(self, **kwargs)

    def save(self) -> None:
        """Saves the rendered view of the environment."""
        self.renderer.save()

    def close(self) -> None:
        """Closes the environment."""
        self.renderer.close()
