from typing import Any, Optional, SupportsFloat

import os
import gzip

import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame

class CsvFileLogger(
    gym.Wrapper[ObsType, ActType, ObsType, ActType],
    gym.utils.RecordConstructorArgs):
    """
    Logs information from the environment's render function to a CSV file.

    There are several usage patterns for this wrapper:

    - Log the last step of every episode
    - Log all steps of every selected episode

    To log the last step of every episode, set
    `log_episodes_end=True`.

    To log all steps of selected episode, set
    `log_episodes_steps=True`.

    To limit the number of rows in the CSV fole to 1000,
    set `max_cvs_rows=1000`. In this case, the wrapper will
    create a new file every 1000 rows.

    To limit the number of bytes in the CSV fole to 20 MB,
    set `max_cvs_bytes=20*1024*1024`.

    No vector version of the wrapper exists.
    """
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        output_folder: str,
        name_prefix: str = 'rl',
        max_csv_rows: int = 1048000, # Excel row limit 1_048_576
        max_csv_bytes: int = 20*1024*1024, # 20 MB
        log_episodes_end: bool = True,
        log_episodes_steps: bool = False,
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
            max_csv_bytes (int):
                The maximum number of bytes in a single CSV file.
                Default: 20 MB.
            log_episodes_end (bool):
                Whether to log the last step of selected episodes.
                Default: True.
            log_episodes_steps (bool):
                Whether to log steps of every selected episode.
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

        self.vec_env_index_str = '' if vec_env_index is None else f'-env_{vec_env_index}'
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        if not os.path.exists(self.output_folder):
            raise ValueError(f'Cannot create directory {output_folder}')

        self.log_episodes: bool = log_episodes_end
        self.log_steps: bool = log_episodes_steps
        if not self.log_episodes and not self.log_steps:
            raise ValueError('At least one of log_episodes_end or ' \
                             'log_episodes_steps must be True.')

        self.name_prefix: str = name_prefix
        self.max_csv_rows: int = max_csv_rows
        self.max_csv_bytes: int = max_csv_bytes
        self.compress_gzip = compress_gzip
        self.step_rows: list[str] = []
        self.episode_rows: list[str] = []
        self.step_rows_bytes: int = 0 
        self.episode_rows_bytes: int = 0 
        self.log_step_count: int = 0
        self.log_episode_count: int = 0
        self.csv_hearder: str = env.render()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment and writes the CSV row."""
        obs, info = super().reset(seed=seed, options=options)
        self.log_row()
        return obs, info

    def step(self, action: ActType
        ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment and writes the CSV row."""
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.log_row(episode_end=terminated | truncated)
        return obs, rew, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame]:
        """Returns the rendered frame as specified by render_mode attribute during initialization of the environment."""
        return super().render()

    def close(self):
        """Closes the wrapper writing all unterminated CSV rows."""
        self.flush_remaining()
        super().close()

    def __del__(self):
        self.flush_remaining()

    def flush_remaining(self):
        if self.log_steps:
            self.flush_steps()
        if self.log_episodes:
            self.flush_episodes()

    def log_row(self, episode_end: bool = False):
        row = self.env.render()
        if isinstance(row, str):
            row_bytes = len(row.encode('utf-8'))
            if self.log_steps:
                self.step_rows.append(row)
                self.step_rows_bytes += row_bytes
                if len(self.step_rows) > self.max_csv_rows or \
                    self.step_rows_bytes > self.max_csv_bytes:
                    self.flush_steps()

            if self.log_episodes and episode_end:
                self.episode_rows.append(row)
                self.episode_rows_bytes += row_bytes
                if len(self.episode_rows) > self.max_csv_rows or \
                    self.episode_rows_bytes > self.max_csv_bytes:
                    self.flush_episodes()
        else:
            raise ValueError('The render function of the environment should return '
                             f'a string, not a {type(row)}.')

    def flush_episodes(self):
        if len(self.episode_rows) > 0:
            self.log_episode_count += 1
            name = f'{self.name_prefix}-episodes{self.vec_env_index_str}.{self.log_episode_count}.csv'
            path = os.path.join(self.output_folder, name)
            if self.compress_gzip:
                with gzip.open(path+'.gz', 'wt', compresslevel=9, newline='\n') as file:
                    file.write(self.csv_hearder + '\n')
                    for row in self.episode_rows:
                        file.write(row + '\n')
            else:
                with open(path, 'w', newline='\n') as file:
                    file.write(self.csv_hearder + '\n')
                    for row in self.episode_rows:
                        file.write(row + '\n')
            self.episode_rows.clear()
            self.episode_rows_bytes = 0

    def flush_steps(self):
        if len(self.step_rows) > 0:
            self.log_step_count += 1
            name = f'{self.name_prefix}-steps{self.vec_env_index_str}.{self.log_step_count}.csv'
            path = os.path.join(self.output_folder, name)
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
            self.step_rows.clear()
            self.step_rows_bytes = 0
