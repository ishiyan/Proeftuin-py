from typing import Optional, Sequence, Union
import os
import multiprocessing

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from env import Env01, CsvFileLogger, RecordAnimatedGIF
from sb3 import SaveOnBestTrainingRewardCallback, VecCsvFileLogger, VecRecordAnimatedGIF

def _create_dir_and_prefix(dir: str, iteration: int, name: str, eval: bool):
    prefix = 'eval' if eval else 'iter'
    prefix = f'{prefix}-{iteration}_{name}'
    if eval:
        dir = f'{dir}evaluate-policy/'
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    return dir, prefix

def save_evaluation_statistics(
    mean_reward: float,
    std_reward: float,
    name: str='rl',
    dir: str='./',
    iteration: int=1
    ):
    dir, _ = _create_dir_and_prefix(dir, iteration, name, True)
    file_name = f'{dir}evaluate_policy.csv'
    file_exists = os.path.exists(file_name)
    file_handler = open(file_name, f'at', newline='\n')
    if not file_exists:
        file_handler.write('iteration,mean_reward,std_reward\n')
    file_handler.write(f'{iteration},{mean_reward:.4f},{std_reward:.4f}\n')
    file_handler.close()

def cleanup_model_env(model, env):
    env.close()
    del model
    del env

def _create_env01(
    vec_env_index: Optional[int]=None,
    symbol: Union[str, Sequence[str]]='SINE',
    episode_max_steps: int=128,
    name: str='rl',
    dir: str='./',
    iteration: int=1,
    eval: bool=False,
    render: Optional[str]=None, # 'log','gif',None
    render_mode: Optional[str]=None, # 'ansi','rgb_array',None
    monitor: bool=True,
    ):
    dir, prefix = _create_dir_and_prefix(dir, iteration, name, eval)
    env = Env01(
        symbol=symbol,
        time_frame='1m',
        scale_period = episode_max_steps,
        copy_period = episode_max_steps,
        episode_max_steps=episode_max_steps,
        render_mode=render_mode,
        vec_env_index=vec_env_index
        )
    if render == 'gif':
        env = RecordAnimatedGIF(env,
            output_folder=dir,
            name_prefix=prefix,
            max_gif_frames=episode_max_steps*1000,
            record_episodes_end=True,
            record_episodes_steps=True,
            fps_episode_end=0.5,
            fps_episode_step=3,
            vec_env_index=vec_env_index
            )
    if render == 'log':
        env = CsvFileLogger(env,
            output_folder=dir,
            name_prefix=prefix,
            max_csv_rows=1048000, # Excel row limit 1_048_576
            max_csv_bytes=120*1024*1024, # 120 MB
            log_episodes_end=True,
            log_episodes_steps=True,
            compress_gzip=True, # Compresses 12 MB to 3 MB
            vec_env_index=vec_env_index
            )
    if monitor:
        s = '' if vec_env_index is None else f'.env{vec_env_index}'
        env = Monitor(env,
            filename=f'{dir}{prefix}{s}',
            allow_early_resets=False,
            info_keywords=(
                'vec_env_index',
                'episode_number',
                'step_number',
                'provider_name',
                'aggregator_name',
                'account_halted')
            )
    return env

def create_single_env01(
    symbol: Union[str, Sequence[str]]='SINE',
    episode_max_steps: int=128,
    name: str='rl',
    dir: str='./',
    iteration: int=1,
    eval: bool=False,
    render: Optional[str]=None, # 'log','gif',None
    monitor: bool=True
    ):
    if render == 'gif':
        render_mode = 'rgb_array'
    elif render == 'log':
        render_mode = 'ansi'
    else:
        render_mode = None
    return _create_env01(
        vec_env_index=None,
        symbol=symbol,
        episode_max_steps=episode_max_steps,
        name=name,
        dir=dir,
        iteration=iteration,
        eval=eval,
        render=render,
        render_mode=render_mode,
        monitor=monitor)

def _create_envs01(
    symbol: Union[str, Sequence[str]]='SINE',
    episode_max_steps: int=128,
    name: str='rl',
    dir: str='./',
    iteration: int=1,
    eval: bool=False,
    render: Optional[str]=None, # 'log','gif',None
    render_mode: Optional[str]=None, # 'ansi','rgb_array',None
    monitor: bool=True,
    max_envs: Optional[int]=None,
    verbose: int=1
    ):
    num_cores = multiprocessing.cpu_count()
    if verbose > 0:
        print(f'number of cores: {num_cores}')
    if max_envs is not None and num_cores > max_envs:
        num_cores = max_envs
        if verbose > 0:
            print(f'limited to: {num_cores}')
    # Return a list of lambda functions, each calling create_env
    # with a unique vec_env_index.
    return [lambda i=i: _create_env01(
        vec_env_index=i,
        symbol=symbol,
        episode_max_steps=episode_max_steps,
        name=name,
        dir=dir,
        iteration=iteration,
        eval=eval,
        render=render,
        render_mode=render_mode,
        monitor=monitor
        ) for i in range(num_cores)]

def create_vec_env01(
    vec_env: str='subproc', # 'subproc','dummy'
    symbol: Union[str, Sequence[str]]='SINE',
    episode_max_steps: int=128,
    name: str='rl',
    dir: str='./',
    iteration: int=1,
    eval: bool=False,
    render: Optional[str]=None, # 'log','gif',None
    vectorized_render: bool=True,
    vectorized_monitor: bool=True,
    max_envs: Optional[int]=None,
    verbose: int=1
    ):
    if render == 'gif':
        render_mode = 'rgb_array'
    elif render == 'log':
        render_mode = 'ansi'
    else:
        render_mode = None
    original_dir = dir
    dir, prefix = _create_dir_and_prefix(dir, iteration, name, eval)

    if vec_env == 'subproc':
        env = SubprocVecEnv(_create_envs01(
            symbol=symbol,
            episode_max_steps=episode_max_steps,
            name=name,
            dir=original_dir,
            iteration=iteration,
            eval=eval,
            render=None if vectorized_render else render,
            render_mode=render_mode,
            monitor=not vectorized_monitor,
            max_envs=max_envs,
            verbose=verbose
            ))
    elif vec_env == 'dummy':
        env = DummyVecEnv(_create_envs01(
            symbol=symbol,
            episode_max_steps=episode_max_steps,
            name=name,
            dir=original_dir,
            iteration=iteration,
            eval=eval,
            render=None if vectorized_render else render,
            render_mode=render_mode,
            monitor=not vectorized_monitor,
            max_envs=max_envs,
            verbose=verbose
            ))
    else:
        raise ValueError(f'vec_env={vec_env} not supported')
    if vectorized_render and render == 'gif':
        env = VecRecordAnimatedGIF(env,
            output_folder=dir,
            name_prefix=prefix,
            max_gif_frames=episode_max_steps*1000,
            record_episodes_end=True,
            record_episodes_steps=True,
            fps_episode_end=0.5,
            fps_episode_step=3,
            separate_steps_per_environment=False
            )
    if vectorized_render and render == 'log':
        env = VecCsvFileLogger(env,
            output_folder=dir,
            name_prefix=prefix,
            max_csv_rows=1048000, # Excel row limit 1_048_576
            max_csv_bytes=120*1024*1024, # 120 MB
            log_episodes_end=True,
            log_episodes_steps=True,
            compress_gzip=True # Compresses 12 MB to 3 MB
            )
    if vectorized_monitor:
        env = VecMonitor(env,
            filename=f'{dir}{prefix}',
            info_keywords=(
                'vec_env_index',
                'episode_number',
                'step_number',
                'provider_name',
                'aggregator_name',
                'account_halted')
            )
    return env

def create_subproc_env01(
    symbol: Union[str, Sequence[str]]='SINE',
    episode_max_steps: int=128,
    name: str='rl',
    dir: str='./',
    iteration: int=1,
    eval: bool=False,
    render: Optional[str]=None, # 'log','gif',None
    vectorized_render: bool=True,
    vectorized_monitor: bool=True,
    max_envs: Optional[int]=None,
    verbose: int=1
    ):
    return create_vec_env01(
        vec_env='subproc',
        symbol=symbol,
        episode_max_steps=episode_max_steps,
        name=name,
        dir=dir,
        iteration=iteration,
        eval=eval,
        render=render,
        vectorized_render=vectorized_render,
        vectorized_monitor=vectorized_monitor,
        max_envs=max_envs,
        verbose=verbose)

def create_dummyvec_env01(
    symbol: Union[str, Sequence[str]]='SINE',
    episode_max_steps: int=128,
    name: str='rl',
    dir: str='./',
    iteration: int=1,
    eval: bool=False,
    render: Optional[str]=None, # 'log','gif',None
    vectorized_render: bool=True,
    vectorized_monitor: bool=True,
    max_envs: Optional[int]=None,
    verbose: int=1
    ):
    return create_vec_env01(
        vec_env='dummy',
        symbol=symbol,
        episode_max_steps=episode_max_steps,
        name=name,
        dir=dir,
        iteration=iteration,
        eval=eval,
        render=render,
        vectorized_render=vectorized_render,
        vectorized_monitor=vectorized_monitor,
        max_envs=max_envs,
        verbose=verbose)

def create_save_on_best_training_reward_callback(
    check_freq: int=128*10,
    name: str='rl',
    dir: str='./',
    save_replay_buffer: bool=False,
    verbose: int=0):
    return SaveOnBestTrainingRewardCallback(
        check_freq=check_freq,
        log_dir=dir,
        algo_name=name,
        csv_suffix=None,
        save_replay_buffer=save_replay_buffer,
        verbose=verbose
        )