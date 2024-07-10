from typing import Any, Optional, SupportsFloat

import os
import imageio as iio # pip install imageio
import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame

NOTSET = 0

class RecordAnimatedGIF(
    gym.Wrapper[ObsType, ActType, ObsType, ActType],
    gym.utils.RecordConstructorArgs):
    """
    Records animated GIFs of environment episodes and steps using the
    environment's render function.

    There are several usage patterns for this wrapper:
    
    - Record the last step of every episode
    - Record the last step of every i-th episode
    - Record all steps of every selected episode
    - Record every i-th steps of every selected episode

    To record the last step of every episode, set
    `record_episodes_end=True` and `record_episodes_delta=1`.

    To record the last step of every hundredth episode, set
    `record_episodes_end=True` and `record_episodes_delta=100`.

    To record all steps of selected episode, set
    `record_episodes_steps=True` and `record_steps_delta=1`.
    You can also set `separate_steps_file_per_episode=True`
    to record steps of each episode in a separate file.

    To record every thousandth steps of selected episode, set
    `record_episodes_steps=True` and `record_steps_delta=1000`.
    You can also set `separate_steps_file_per_episode=True`
    to record steps of each episode in a separate file.

    To limit the number of frames in the animated GIF to 1000,
    set `max_gif_frames=1000`. In this case, the wrapper will
    create a new file every 1000 frames.

    No vector version of the wrapper exists.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        output_folder: str,
        name_prefix: str = 'rl',
        max_gif_frames: int = 1000,
        record_episodes_end: bool = True,
        record_episodes_steps: bool = False,
        record_episodes_delta: int = 1,
        record_steps_delta: int = 1,
        separate_steps_file_per_episode: bool = True,
        fps_episode_end: int | float = 0.5,
        fps_episode_step: int | float = 3,
        vec_env_index: Optional[int] = None):
        """
        Args:
            env:
                The environment that will be wrapped.
            optput_folder (str):
                The folder where the animated GIFs will be stored.
            name_prefix (str):
                Will be prepended to the filename of the recordings.
                Default: 'rl'.
            max_gif_frames (int):
                The maximum number of frames in a single animated GIF.
                Default: 1000.
            record_episodes_end (bool):
                Whether to record the last step of selected episodes.
                Default: True.
            record_episodes_steps (bool):
                Whether to record steps of every selected episode.
                Default: False.
            record_episodes_delta (int):
                The interval between recorded episodes.
                Default: 1.
            record_steps_delta (int):
                The interval between recorded steps.
                Default: 1.
            separate_steps_file_per_episode (bool):
                Whether to record steps of each episode in a separate file.
                Default: True.
            fps_episode_end (int|float):
                The frames per second of the episode GIF.
                Default: 0.5.
            fps_episode_step (int|float):
                The frames per second of the step GIF.
                Default: 3.
            vec_env_index (int, optional):
                The index of the vector environment.
                Default: None.
        """
        gym.Wrapper.__init__(self, env)
        if env.render_mode != 'rgb_array':
            raise ValueError(
                f'Render mode is {env.render_mode}, which is incompatible ',
                'with RecordAnimatedGIF. Initialize your environment with ',
                'the "rgb_array" render_mode.')

        self.vec_env_index_str = '' if vec_env_index is None else f'-{vec_env_index}'
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        if not os.path.exists(self.output_folder):
            raise ValueError(f'Cannot create directory {output_folder}')

        self.record_episodes: bool = record_episodes_end
        self.record_steps: bool = record_episodes_steps
        if not self.record_episodes and not self.record_steps:
            raise ValueError(f'At least one of record_episodes_end {self.record_episodes} ' \
                             f'or record_episodes_steps {self.record_steps} must be True.')

        self.separate_steps_file_per_episode: bool = separate_steps_file_per_episode
        self.name_prefix: str = name_prefix
        self.max_gif_frames: int = max_gif_frames
        self.record_episodes_delta: int = record_episodes_delta
        self.record_steps_delta: int = record_steps_delta
        self.fps_step: int|float = fps_episode_step
        self.fps_episode: int|float = fps_episode_end

        self.recorded_step_frames: list[RenderFrame] = []
        self.recorded_episode_frames: list[RenderFrame] = []
        self.step_id = NOTSET
        self.episode_id = NOTSET

        self.animation_episode_first: int = NOTSET
        self.animation_episode_last: int = NOTSET
        self.animation_step_first: int = NOTSET
        self.animation_step_first_episode: int = NOTSET
        self.animation_step_last: int = NOTSET
        self.animation_step_last_episode: int = NOTSET

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment and records the animation frame."""
        obs, info = super().reset(seed=seed, options=options)
        self.episode_id += 1
        self.step_id = 1
        self.capture_frame()
        return obs, info

    def step(self, action: ActType
        ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment and records the animation frame."""
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.step_id += 1
        self.capture_frame(episode_end=terminated | truncated)
        return obs, rew, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame]:
        """Returns the rendered frame as specified by render_mode attribute during initialization of the environment."""
        return super().render()

    def close(self):
        """Closes the wrapper writing all unterminated animated GIFs."""
        super().close()
        if self.record_steps:
            self.flush_steps()
        if self.record_episodes:
            self.flush_episodes()

    def capture_frame(self, episode_end: bool = False):
        frame = self.env.render()
        if isinstance(frame, np.ndarray):
            if self.record_steps:
                if self.record_steps_delta == 1 or \
                    self.step_id % self.record_steps_delta == 0:
                    self.recorded_step_frames.append(frame)
                if self.animation_step_first_episode == NOTSET:
                    self.animation_step_first_episode = self.episode_id
                if self.animation_step_first == NOTSET:
                    self.animation_step_first = self.step_id
                self.animation_step_last_episode = self.episode_id
                self.animation_step_last = self.step_id
                if (episode_end and self.separate_steps_file_per_episode) or \
                    len(self.recorded_step_frames) > self.max_gif_frames:
                    self.flush_steps()

            if self.record_episodes and episode_end:
                if self.record_episodes_delta == 1 or \
                    self.episode_id % self.record_episodes_delta == 0:
                    self.recorded_episode_frames.append(frame)
                if self.animation_episode_first == NOTSET:
                    self.animation_episode_first = self.episode_id
                self.animation_episode_last = self.episode_id
                if len(self.recorded_episode_frames) > self.max_gif_frames:
                    self.flush_episodes()
        else:
            raise ValueError('The render function of the environment should return '
                             f'a numpy array, not a {type(frame)}.')

    def flush_episodes(self):
        name = f'{self.name_prefix}{self.vec_env_index_str}-episodes_' \
            f'{self.animation_episode_first}_{self.animation_episode_last}.gif'
        path = os.path.join(self.output_folder, name)
        print(f'Flushing {len(self.recorded_episode_frames)} episodes to: {path}')
        if len(self.recorded_episode_frames) > 0:
            iio.mimsave(uri=path, ims=self.recorded_episode_frames,
                        fps=self.fps_episode, loop=0, subrectangles=True)
            self.recorded_episode_frames = []
        self.animation_first_episode = NOTSET
        self.animation_last_episode = NOTSET

    def flush_steps(self):
        name = f'{self.name_prefix}{self.vec_env_index_str}-steps_' \
            f'({self.animation_step_first_episode}-{self.animation_step_first})_' \
            f'({self.animation_step_last_episode}-{self.animation_step_last}).gif'
        path = os.path.join(self.output_folder, name)
        print(f'Flushing {len(self.recorded_step_frames)} steps to: {path}')
        if len(self.recorded_step_frames) > 0:
            iio.mimsave(uri=path, ims=self.recorded_step_frames,
                        fps=self.fps_step, loop=0, subrectangles=True)
            self.recorded_step_frames = []
        self.animation_step_first_episode = NOTSET
        self.animation_step_first = NOTSET
        self.animation_step_last_episode = NOTSET
        self.animation_step_last = NOTSET
