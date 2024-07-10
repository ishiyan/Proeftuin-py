import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from env import Env01, CsvFileLogger, RecordAnimatedGIF
from sb3 import SaveOnBestTrainingRewardCallback, VecCsvFileLogger, VecRecordAnimatedGIF

# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
# https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html
# https://github.com/DLR-RM/stable-baselines3/issues/597

evaluate_episodes_every_iteration=4 # Set to 0 to disable
verbose=1
zip_logs=False

name = 'ppo_01_xxx'
dir = f'./sb3/{name}-new/'

start_iteration_number = 1
total_iterations = 3 #1000

learn_episodes = 100#1000#10000
episode_max_steps = 180

save_on_best_training_reward=True
save_on_best_training_reward_episode_freq=10#episode_max_steps*1 # 50 = every 50th episode

if not os.path.exists(dir):
    os.makedirs(dir, exist_ok=True)
if evaluate_episodes_every_iteration > 0:
    eval_dir = dir + 'evaluate_policy/'
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir, exist_ok=True)

callback = SaveOnBestTrainingRewardCallback(
    check_freq=episode_max_steps*save_on_best_training_reward_episode_freq,
    log_dir=dir,
    algo_name=name,
    csv_suffix=None,
    save_replay_buffer=False,
    verbose=0)

def create_env(vec_env_index: int=None, symbol: str='SINE',
               iteration: int=1, eval: bool=False,
               monitor: bool=True, logger: bool=True,
               animator: bool=True):
    env = Env01(symbol=symbol,
        time_frame='1m',
        scale_period = episode_max_steps,
        copy_period = episode_max_steps,
        episode_max_steps=episode_max_steps,
        render_mode='rgb_array' if eval else 'ansi',
        vec_env_index=vec_env_index)
    if eval:
        if animator:
            env = RecordAnimatedGIF(env,
                output_folder=eval_dir,
                name_prefix=f'eval_{iteration}_{name}',
                max_gif_frames=100000,
                record_episodes_end=True,
                record_episodes_steps=True,
                record_episodes_delta=1,
                record_steps_delta=1,
                separate_steps_file_per_episode=False,
                fps_episode_end=0.5,
                fps_episode_step=3,
                vec_env_index=vec_env_index)
    else:
        if logger:
            env = CsvFileLogger(env,
                output_folder=dir,
                name_prefix=f'iter_{iteration}_{name}',
                max_csv_rows=1048000, # Excel row limit 1_048_576
                log_episodes_end=True,
                log_episodes_steps=True,
                log_episodes_delta=1,
                log_steps_delta=1,
                separate_steps_file_per_episode=False,
                compress_gzip=False,
                vec_env_index=vec_env_index)
    if monitor:
        s = '' if vec_env_index is None else f'.{vec_env_index}'
        env = Monitor(env,
            filename=f'{eval_dir}eval_{iteration}_{name}{s}' \
                if eval else f'{dir}iter_{iteration}_{name}{s}',
            allow_early_resets=False,
            info_keywords=('vec_env_index','episode_number',
                'step_number','provider_name','aggregator_name','account_halted')
            )
    return env

def create_envs(symbol: str='SINE', iteration: int=1,
                eval: bool=False, monitor: bool=True,
                logger: bool=True, animator: bool=True,
                max_envs: int=None):
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    if verbose > 0:
        print(f'number of cores: {num_cores}')
    if max_envs is not None and num_cores > max_envs:
        num_cores = max_envs
        if verbose > 0:
            print(f'limited to: {num_cores}')
    # Return a list of lambda functions, each calling create_env
    # with a unique vec_env_index.
    return [lambda i=i: create_env(vec_env_index=i,
        symbol=symbol, iteration=iteration, eval=eval,
        monitor=monitor, logger=logger,
        animator=animator) for i in range(num_cores)]

def create_subproc_env(symbol: str='SINE', iteration: int=1,
                        eval: bool=False, max_envs: int=None):
    env = SubprocVecEnv(create_envs(symbol=symbol,
            iteration=iteration, eval=eval, monitor=False,
            logger=False, animator=False, max_envs=max_envs))
    if eval:
        env = VecRecordAnimatedGIF(env,
            output_folder=eval_dir,
            name_prefix=f'eval_{iteration}_{name}',
            max_gif_frames=episode_max_steps*max(100, evaluate_episodes_every_iteration),
            record_episodes_end=True,
            record_episodes_steps=True,
            fps_episode_end=0.5,
            fps_episode_step=3)
    else:
        env = VecCsvFileLogger(env,
            output_folder=dir,
            name_prefix=f'iter_{iteration}_{name}',
            max_csv_rows=1048000, # Excel row limit 1_048_576
            max_csv_bytes=80*1024*1024, # 80 MB
            log_episodes_end=True,
            log_episodes_steps=True,
            compress_gzip=True # Compresses 12 MB to 3 MB
            ) 
    env = VecMonitor(env,
        filename=f'{eval_dir}eval_{iteration}_{name}' \
            if eval else f'{dir}iter_{iteration}_{name}',
        info_keywords=('vec_env_index','episode_number',
            'step_number','provider_name','aggregator_name','account_halted'))
    return env
                        
def zip_files(dir_path: str, file_prefix: str, zip_file_name: str):
    import zipfile
    with zipfile.ZipFile(f'{dir_path}{zip_file_name}.zip', 'w') as zipf:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.startswith(file_prefix) and file.endswith('.csv'):
                    full_path = os.path.join(root, file)
                    print(f'Adding {os.path.relpath(full_path, dir_path)} ({full_path}) to {dir}{zip_file_name}.zip')
                    #if os.path.getsize(full_path) >= 1024 * 1024:
                    zipf.write(full_path,
                        arcname=os.path.relpath(full_path, dir_path))
                    # Delete the file after adding it to the ZIP
                    os.remove(full_path)

if __name__=="__main__":
    for iteration in range(start_iteration_number, total_iterations):
        #env = SubprocVecEnv(create_envs(symbol=['ETHUSDT','BTCUSDT'],
        #    iteration=iteration, eval=False, max_envs=4))
        env = create_subproc_env(symbol=['ETHUSDT','BTCUSDT'],
            iteration=iteration, eval=False, max_envs=4)
        #env = DummyVecEnv(create_envs(symbol=['ETHUSDT','BTCUSDT'], iteration=iteration, eval=False))
        #env = create_env(vec_env_index=None, symbol='SINE', iteration=iteration, eval=False)

        saved_model_path = os.path.join(dir, f'{name}_model.zip')
        if os.path.exists(saved_model_path):
            model = PPO.load(saved_model_path, env=env, verbose=verbose, print_system_info=True)
            model.set_random_seed(iteration)
            model._last_obs = None
        else:
            model = PPO('MultiInputPolicy', env, seed=iteration, verbose=verbose)

        try:
            model.learn(total_timesteps=int(episode_max_steps*learn_episodes),# 24576 12288 6144 3072 1536 768 384 192 96 48 24 12 6 3
                log_interval=4,
                callback=callback if save_on_best_training_reward else None,
                reset_num_timesteps = False,
                progress_bar=True)
            model.save(saved_model_path)
            del model
            env.close()
            del env
        except Exception as e:
            del model
            env.close()
            del env
            raise e
        if zip_logs:
            zip_files(dir_path=dir, file_prefix=f'iter_{iteration}', zip_file_name=f'iter_{iteration}_logs')

        if evaluate_episodes_every_iteration > 0:
            if verbose > 0:
                print(f'Evaluate policy at iteration {iteration} ({evaluate_episodes_every_iteration} episodes)')
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            #env = SubprocVecEnv(create_envs(symbol='BTCEUR',
            #    iteration=iteration, eval=True, max_envs=4))
            env = create_subproc_env(symbol=['ETHUSDT','BTCUSDT'],
                iteration=iteration, eval=True, max_envs=4)
            #env = DummyVecEnv(create_envs(symbol='BTCEUR', iteration=iteration, eval=True))
            #env = create_env(vec_env_index=None, symbol='BTCEUR', iteration=iteration, eval=True)
            try:
                model = PPO.load(saved_model_path, env=env, verbose=verbose, print_system_info=True)
                model.set_random_seed(iteration)
                model._last_obs = None
                mean_reward, std_reward = evaluate_policy(model, env,
                    n_eval_episodes=evaluate_episodes_every_iteration)
                if verbose > 0:
                    print(f'Mean reward: {mean_reward:.4f} +/- {std_reward:.4f}')
                env.close()
                if zip_logs:
                    zip_files(dir_path=eval_dir, file_prefix=f'eval_{iteration}', zip_file_name=f'eval_{iteration}_logs')
                file_name = f'{eval_dir}evaluate_policy.csv'
                file_exists = os.path.exists(file_name)
                file_handler = open(file_name, f'at', newline='\n')
                if not file_exists:
                    file_handler.write('iteration,mean_reward,std_reward\n')
                file_handler.write(f'{iteration},{mean_reward:.4f},{std_reward:.4f}\n')
                file_handler.close()
                del model
                del env
            except Exception as e:
                del model
                env.close()
                del env
                raise e
