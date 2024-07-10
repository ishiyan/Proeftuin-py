from typing import Any
import warnings
import os
import imageio as iio # pip install imageio
import numpy as np

from gymnasium.core import RenderFrame
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn,  VecEnvWrapper

class VecRecordAnimatedGIF(VecEnvWrapper):
    """
    Wraps a VecEnv or VecEnvWrapper object to record
    animated GIFs of environment episodes and steps
    using the environment's render function.

    There are several usage patterns for this wrapper:
    
    - Record the last step of every episode
    - Record all steps of every selected episode

    To record the last step of every episode, set
    `record_episodes_end=True`.

    To record all steps of selected episode, set
    `record_episodes_steps=True`.

    To limit the number of frames in the animated GIF to 1000,
    set `max_gif_frames=1000`. In this case, the wrapper will
    create a new file every 1000 frames.
    """

    def __init__(
        self,
        env: VecEnv,
        output_folder: str,
        name_prefix: str = 'rl',
        max_gif_frames: int = 1000,
        record_episodes_end: bool = True,
        record_episodes_steps: bool = False,
        fps_episode_end: int | float = 0.5,
        fps_episode_step: int | float = 3,
        separate_steps_per_environment: bool = False,
        ):
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
            fps_episode_end (int|float):
                The frames per second of the episode GIF.
                Default: 0.5.
            fps_episode_step (int|float):
                The frames per second of the step GIF.
                Default: 3.
            separate_steps_per_environment (bool):
                Whether to record steps of every environment separately.
                Default: False.
        """
        VecEnvWrapper.__init__(self, env)
        self.env = env

        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        if not os.path.exists(self.output_folder):
            raise ValueError(f'Cannot create directory {output_folder}')

        self.record_episodes: bool = record_episodes_end
        self.record_steps: bool = record_episodes_steps
        if not self.record_episodes and not self.record_steps:
            raise ValueError(f'At least one of record_episodes_end {self.record_episodes} ' \
                             f'or record_episodes_steps {self.record_steps} must be True.')

        self.name_prefix: str = name_prefix
        self.max_gif_frames: int = max_gif_frames
        self.fps_episode: int|float = fps_episode_end
        self.fps_step: int|float = fps_episode_step
        self.separate_steps_per_environment: bool = separate_steps_per_environment

        self.recorded_episode_frames: list[RenderFrame] = []
        self.recorded_episode_frame_per_env: list[RenderFrame] = \
            [None for _ in range(self.env.num_envs)]
        self.recorded_step_frames: list[RenderFrame] = []
        self.recorded_step_frames_per_env: list[list[RenderFrame]] = \
            [[] for _ in range(self.env.num_envs)]
        self.recorded_step_counts_per_env: list[int] = \
            [0 for _ in range(self.env.num_envs)]
        self.step_count = 0
        self.episode_count = 0

    def reset(self) -> VecEnvObs:
        obs = self.env.reset()
        try:
            self.capture_frames(dones=None)
        except Exception as e:
            warnings.warn(f'capture_frames error on reset: {e}')
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()
        try:
            self.capture_frames(dones=dones)
        except Exception as e:
            warnings.warn(f'capture_frames error on step_wait: {e}')
        return obs, rews, dones, infos

    def close(self) -> None:
        self.flush_remaining()
        VecEnvWrapper.close(self)

    def __del__(self):
        self.flush_remaining()

    def flush_remaining(self):
        if self.record_episodes:
            self.record_episodes = False
            self.flush_episodes()

        if self.record_steps:
            self.record_steps = False
            if self.separate_steps_per_environment:
                for i in range(self.env.num_envs):
                    self.flush_steps_env(i)
            else:
                # Flush completed episode steps.
                self.flush_steps()
                # Flush incomplete episode steps separately
                for list in self.recorded_step_frames_per_env:
                    if len(list) > 0:
                        for frame in list:
                            self.recorded_step_frames.append(frame)
                        list.clear()
                self.flush_steps(incomplete=True)

    def capture_frames(self, dones: list[bool] = None):
        frames = self.env.env_method('render')
        assert len(frames) == len(dones) if dones is not None else True, \
            f'len(frames)={len(frames)} != len(dones)={len(dones)}'
        assert len(frames) == len(self.recorded_step_frames_per_env), \
            f'len(rows)={len(frames)} != recorded_step_frames_per_env=' \
            f'{len(self.recorded_step_frames_per_env)}'

        for i, frame in enumerate(frames):
            assert isinstance(frame, np.ndarray), \
                f'frame {i} is not a numpy array but {type(frame)}'
            done = False if dones is None else dones[i]
            episode_end_frame = self.recorded_episode_frame_per_env[i] if done else None
            self.recorded_episode_frame_per_env[i] = frame
            if self.record_steps:
                self.recorded_step_frames_per_env[i].append(frame)
                if self.separate_steps_per_environment:
                    if len(self.recorded_step_frames_per_env[i]) > self.max_gif_frames:
                        self.flush_steps_env(i)
                elif done:
                    for f in self.recorded_step_frames_per_env[i]:
                        self.recorded_step_frames.append(f)
                    self.recorded_step_frames_per_env[i].clear()
                    if len(self.recorded_step_frames) > self.max_gif_frames:
                        self.flush_steps()
            if self.record_episodes:
                if episode_end_frame is not None:
                    self.recorded_episode_frames.append(episode_end_frame)
                    if len(self.recorded_episode_frames) > self.max_gif_frames:
                        self.flush_episodes()

    def flush_episodes(self):
        if len(self.recorded_episode_frames) > 0:
            self.episode_count += 1
            name = f'{self.name_prefix}-episodes.{self.episode_count}.gif'
            path = os.path.join(self.output_folder, name)
            iio.mimsave(uri=path, ims=self.recorded_episode_frames,
                        fps=self.fps_episode, loop=0, subrectangles=True)
            self.recorded_episode_frames.clear()

    def flush_steps(self, incomplete: bool=False):
        if len(self.recorded_step_frames) > 0:
            self.step_count += 1
            s = '-incomplete' if incomplete else ''
            name = f'{self.name_prefix}-steps.{self.step_count}{s}.gif'
            path = os.path.join(self.output_folder, name)
            iio.mimsave(uri=path, ims=self.recorded_step_frames,
                        fps=self.fps_step, loop=0, subrectangles=True)
            self.recorded_step_frames.clear()

    def flush_steps_env(self, env_index: int):
        frames = self.recorded_step_frames_per_env[env_index]
        if len(frames) > 0:
            self.recorded_step_counts_per_env[env_index] += 1
            count = self.recorded_step_counts_per_env[env_index]
            name = f'{self.name_prefix}-steps-env_{env_index}.{count}.gif'
            path = os.path.join(self.output_folder, name)
            iio.mimsave(uri=path, ims=frames,
                        fps=self.fps_step, loop=0, subrectangles=True)
        self.recorded_step_frames_per_env[env_index].clear()
