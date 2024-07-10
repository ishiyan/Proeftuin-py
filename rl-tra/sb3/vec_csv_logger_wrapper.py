import warnings
import os
import gzip

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn,  VecEnvWrapper

class VecCsvFileLogger(VecEnvWrapper):
    """
    Wraps a VecEnv or VecEnvWrapper object to log information
    from the environment's render function to a CSV file.

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
    """
    def __init__(
        self,
        env: VecEnv,
        output_folder: str,
        name_prefix: str = 'rl',
        max_csv_rows: int = 1_048_000, # Excel row limit 1_048_576
        max_csv_bytes: int = 20*1024*1024, # 20 MB
        log_episodes_end: bool = True,
        log_episodes_steps: bool = False,
        compress_gzip: bool = False):
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
        """
        VecEnvWrapper.__init__(self, env)
        self.env = env

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
        self.csv_hearder: str = None

    def reset(self) -> VecEnvObs:
        if self.csv_hearder is None:
            try:
                headers = self.env.env_method('render')
                for i, header in enumerate(headers):
                    if header is not None and len(header) > 0:
                        self.csv_hearder = header                        
                        break
            except Exception as e:
                warnings.warn(f'Failed to get CSV header from environment: {e}')
        obs = self.env.reset()
        try:
            self.log_rows(dones=None)
        except Exception as e:
            warnings.warn(f'Failed to log rows on reset: {e}')
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()
        try:
            self.log_rows(dones=dones)
        except Exception as e:
            warnings.warn(f'Failed to log rows on step_wait: {e}')
        return obs, rews, dones, infos

    def close(self) -> None:
        self.flush_remaining()
        VecEnvWrapper.close(self)

    def __del__(self):
        self.flush_remaining()

    def flush_remaining(self):
        if self.log_steps:
            self.flush_steps()
        if self.log_episodes:
            self.flush_episodes()

    def log_rows(self, dones: list[bool] = None):
        rows = self.env.env_method('render')
        assert len(rows) == len(dones) if dones is not None else True, \
            f'len(rows)={len(rows)} != len(dones)={len(dones)}'
        for i, row in enumerate(rows):
            done = False if dones is None else dones[i]
            row_bytes = len(row.encode('utf-8'))
            if self.log_steps:
                self.step_rows.append(row)
                self.step_rows_bytes += row_bytes
                if len(self.step_rows) > self.max_csv_rows or \
                    self.step_rows_bytes > self.max_csv_bytes:
                    self.flush_steps()

            if self.log_episodes and done:
                self.episode_rows.append(row)
                self.episode_rows_bytes += row_bytes
                if len(self.episode_rows) > self.max_csv_rows or \
                    self.episode_rows_bytes > self.max_csv_bytes:
                    self.flush_episodes()

    def flush_episodes(self):
        if len(self.episode_rows) > 0:
            self.log_episode_count += 1
            name = f'{self.name_prefix}-episodes.{self.log_episode_count}.csv'
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
            self.episode_rows = []
            self.episode_rows_bytes = 0

    def flush_steps(self):
        if len(self.step_rows) > 0:
            self.log_step_count += 1
            name = f'{self.name_prefix}-steps.{self.log_step_count}.csv'
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
            self.step_rows = []
            self.step_rows_bytes = 0
