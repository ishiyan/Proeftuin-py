from typing import Any, Optional, SupportsFloat

import os
import gzip

import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame

NOTSET = 0

class CsvFileLogger(
    gym.Wrapper[ObsType, ActType, ObsType, ActType],
    gym.utils.RecordConstructorArgs):
    """
    Logs information from the environment's render function to a CSV file.

    There are several usage patterns for this wrapper:

    - Log the last step of every episode
    - Log the last step of every i-th episode
    - Log all steps of every selected episode
    - Log every i-th steps of every selected episode

    To log the last step of every episode, set
    `log_episodes_end=True` and `log_episodes_delta=1`.

    To log the last step of every hundredth episode, set
    `log_episodes_end=True` and `log_episodes_delta=100`.

    To log all steps of selected episode, set
    `log_episodes_steps=True` and `log_steps_delta=1`.
    You can also set `separate_steps_file_per_episode=True`
    to log steps of each episode in a separate file.

    To log every thousandth steps of selected episode, set
    `log_episodes_steps=True` and `log_steps_delta=1000`.
    You can also set `separate_steps_file_per_episode=True`
    to log steps of each episode in a separate file.

    To limit the number of rows in the CSV fole to 1000,
    set `max_cvs_rows=1000`. In this case, the wrapper will
    create a new file every 1000 rows.

    No vector version of the wrapper exists.
    """
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        output_folder: str,
        name_prefix: str = 'rl',
        max_csv_rows: int = 1048000, # Excel row limit 1_048_576
        log_episodes_end: bool = True,
        log_episodes_steps: bool = False,
        log_episodes_delta: int = 1,
        log_steps_delta: int = 1,
        separate_steps_file_per_episode: bool = False,
        compress_gzip: bool = False,
        vec_env_index: Optional[int] = None):
        """
        Args:
            env:
                The environment that will be wrapped.
            optput_folder (str):
                The folder where the CSV files will be stored.
            name_prefix (str):
                Will be prepended to the CSV filename.
                Default: 'rl'.
            max_csv_rows (int):
                The maximum number of rows in a single CSV file.
                Default: 1048000, slightly less than Excel row limit 1_048_576.
            log_episodes_end (bool):
                Whether to log the last step of selected episodes.
                Default: True.
            log_episodes_steps (bool):
                Whether to log steps of every selected episode.
                Default: False.
            log_episodes_delta (int):
                The interval between logged episodes.
                Default: 1.
            log_steps_delta (int):
                The interval between logged steps.
                Default: 1.
            separate_steps_file_per_episode (bool):
                Whether to log steps of each episode in a separate file.
                Default: False.
            compress_gzip (bool):
                Whether to compress the CSV files with gzip.
            vec_env_index (int, optional):
                The index of the vector environment.
                Default: None.
        """
        gym.Wrapper.__init__(self, env)
        if env.render_mode != 'ansi':
            raise ValueError(
                f'Render mode is {env.render_mode}, which is incompatible ',
                'with CsvFileWrapper. Initialize your environment with ',
                'the "ansi" render_mode.')

        self.vec_env_index_str = '' if vec_env_index is None else f'-{vec_env_index}'
        self.compress_gzip = compress_gzip
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        if not os.path.exists(self.output_folder):
            raise ValueError(f'Cannot create directory {output_folder}')

        self.log_episodes: bool = log_episodes_end
        self.log_steps: bool = log_episodes_steps
        if not self.log_episodes and not self.log_steps:
            raise ValueError('At least one of log_episodes_end or ' \
                             'log_episodes_steps must be True.')

        self.separate_steps_file_per_episode: bool = separate_steps_file_per_episode
        self.name_prefix: str = name_prefix
        self.max_csv_rows: int = max_csv_rows
        self.log_episodes_delta: int = log_episodes_delta
        self.log_steps_delta: int = log_steps_delta

        self.csv_hearder: str = env.render()
        self.step_rows: list[str] = []
        self.episode_rows: list[str] = []
        self.step_id = NOTSET
        self.episode_id = NOTSET

        self.log_episode_first: int = NOTSET
        self.log_episode_last: int = NOTSET
        self.log_step_first: int = NOTSET
        self.log_step_first_episode: int = NOTSET
        self.log_step_last: int = NOTSET
        self.log_step_last_episode: int = NOTSET

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment and writes the CSV row."""
        obs, info = super().reset(seed=seed, options=options)
        self.episode_id += 1
        self.step_id = 1
        self.log_row()
        return obs, info

    def step(self, action: ActType
        ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment and writes the CSV row."""
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.step_id += 1
        self.log_row(episode_end=terminated | truncated)
        return obs, rew, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame]:
        """Returns the rendered frame as specified by render_mode attribute during initialization of the environment."""
        return super().render()

    def close(self):
        """Closes the wrapper writing all unterminated CSV rows."""
        super().close()
        if self.log_steps:
            self.flush_steps()
        if self.log_episodes:
            self.flush_episodes()

    def log_row(self, episode_end: bool = False):
        row = self.env.render()
        if isinstance(row, str):
            if self.log_steps:
                if self.log_steps_delta == 1 or \
                    self.step_id % self.log_steps_delta == 0:
                    self.step_rows.append(row)
                if self.log_step_first_episode == NOTSET:
                    self.log_step_first_episode = self.episode_id
                if self.log_step_first == NOTSET:
                    self.log_step_first = self.step_id
                self.log_step_last_episode = self.episode_id
                self.log_step_last = self.step_id
                if (episode_end and self.separate_steps_file_per_episode) or \
                    len(self.step_rows) > self.max_csv_rows:
                    self.flush_steps()

            if self.log_episodes and episode_end:
                if self.log_episodes_delta == 1 or \
                    self.episode_id % self.log_episodes_delta == 0:
                    self.episode_rows.append(row)
                if self.log_episode_first == NOTSET:
                    self.log_episode_first = self.episode_id
                self.log_episode_last = self.episode_id
                if len(self.episode_rows) > self.max_csv_rows:
                    self.flush_episodes()
        else:
            raise ValueError('The render function of the environment should return '
                             f'a string, not a {type(row)}.')

    def flush_episodes(self):
        name = f'{self.name_prefix}{self.vec_env_index_str}-episodes_' \
            f'{self.log_episode_first}_{self.log_episode_last}.csv'
        path = os.path.join(self.output_folder, name)
        if len(self.episode_rows) > 0:
            if self.compress_gzip:
                with gzip.open(path+'.gz', 'wt', compresslevel=9, newline='\n') as file:
                    file.write(self.csv_hearder + '\n')
                    for row in self.step_rows:
                        file.write(row + '\n')
            else:
                with open(path, 'w', newline='\n') as file:
                    file.write(self.csv_hearder + '\n')
                    for row in self.episode_rows:
                        file.write(row + '\n')
            self.episode_rows = []
        self.log_first_episode = NOTSET
        self.log_last_episode = NOTSET

    def flush_steps(self):
        name = f'{self.name_prefix}{self.vec_env_index_str}-steps_' \
            f'({self.log_step_first_episode}-{self.log_step_first})_' \
            f'({self.log_step_last_episode}-{self.log_step_last}).csv'
        path = os.path.join(self.output_folder, name)
        if len(self.step_rows) > 0:
            if self.compress_gzip:
                with gzip.open(path+'.gz', 'wt', compresslevel=9, newline='\n') as file:
                    file.write(self.csv_hearder + '\n')
                    for row in self.step_rows:
                        file.write(row + '\n')
            else:
                with open(path, 'w', newline='\n') as file:
                    file.write(self.csv_hearder + '\n')
                    for row in self.step_rows:
                        file.write(row + '\n')
            self.step_rows = []
        self.log_step_first_episode = NOTSET
        self.log_step_first = NOTSET
        self.log_step_last_episode = NOTSET
        self.log_step_last = NOTSET
