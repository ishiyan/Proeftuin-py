import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from env import Env01, CsvFileLogger, RecordAnimatedGIF
from sb3 import SaveOnBestTrainingRewardCallback

# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
# https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html
# https://github.com/DLR-RM/stable-baselines3/issues/597

# Replay buffer takes about 16 gigabytes of space.
save_replay_buffer=False
evaluate_every_iteration=True

name = 'ppo_01_sine'
dir = f'./sb3/{name}/'

start_iteration_number = 1
total_iterations = 1000

learn_episodes = 1000#10000
episode_max_steps = 128

save_on_best_training_reward=True
save_on_best_training_reward_episode_freq=episode_max_steps*1 # 50 = every 50th episode

if not os.path.exists(dir):
    os.makedirs(dir, exist_ok=True)

callback = SaveOnBestTrainingRewardCallback(
    check_freq=episode_max_steps*save_on_best_training_reward_episode_freq,
    log_dir=dir,
    algo_name=name,
    csv_suffix=None,
    save_replay_buffer=save_replay_buffer,
    verbose=0)

for iteration in range(start_iteration_number, total_iterations):
    env = Env01(symbol='sine',#'ETHUSDT',
        time_frame='1m',
        scale_period = 196,
        copy_period = 196,
        episode_max_steps=episode_max_steps,
        render_mode='ansi')
    env = CsvFileLogger(env,
        output_folder=dir,
        name_prefix=f'iter_{iteration}_{name}',
        max_csv_rows=1048000, # Excel row limit 1_048_576
        log_episodes_end=True,
        log_episodes_steps=True,
        log_episodes_delta=1,
        log_steps_delta=1,
        separate_steps_file_per_episode=False)
    env = Monitor(env,
        filename=f'{dir}iter_{iteration}',
        allow_early_resets=False,
        info_keywords=('episode_number', 'step_number'))

    saved_model_path = os.path.join(dir, f'{name}_model.zip')
    saved_replay_buffer_path = os.path.join(dir, f'{name}_replay_buffer.pkl')

    if os.path.exists(saved_model_path):
        model = PPO.load(saved_model_path, env=env, verbose=1, print_system_info=True)
        if os.path.exists(saved_replay_buffer_path):
            model.load_replay_buffer(saved_replay_buffer_path)
        model.set_random_seed(iteration)
        model._last_obs = None
    else:
        model = PPO('MultiInputPolicy', env,
            seed=iteration,
            verbose=1)
    try:
        model.learn(total_timesteps=int(episode_max_steps*learn_episodes),
            log_interval=4,
            callback=callback if save_on_best_training_reward else None,
            reset_num_timesteps = False,
            progress_bar=True)
 
        model.save(saved_model_path)
        if save_replay_buffer and hasattr(model, "replay_buffer") \
                          and model.replay_buffer is not None:
            model.save_replay_buffer(saved_replay_buffer_path)
        env.close()
    except Exception as e:
        env.close()
        raise e
    if evaluate_every_iteration:
        ev_dir = dir + 'evaluate_policy/'
        if not os.path.exists(ev_dir):
            os.makedirs(ev_dir, exist_ok=True)
        env = Env01(symbol='sine',#'ETHUSDT',
            time_frame='1m',
            scale_period = 196,
            copy_period = 196,
            episode_max_steps=episode_max_steps,
            render_mode='rgb_array')
        env = RecordAnimatedGIF(env,
            output_folder=ev_dir,
            name_prefix=f'evaluate_policy_iter_{iteration}_{name}',
            max_gif_frames=100000,
            record_episodes_end=True,
            record_episodes_steps=True,
            record_episodes_delta=1,
            record_steps_delta=1,
            separate_steps_file_per_episode=False,
            fps_episode_end=0.5,
            fps_episode_step=3)
        env = Monitor(env,
            filename=f'{ev_dir}evaluate_policy_iter_{iteration}',
            allow_early_resets=False,
            info_keywords=('episode_number', 'step_number'))

        print(f'Evaluate policy at iteration {iteration}')
        try:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
            print(f'Mean reward: {mean_reward:.4f} +/- {std_reward:.4f}')
            env.close()
            file_name = f'{ev_dir}evaluate_policy.csv'
            file_exists = os.path.exists(file_name)
            file_handler = open(file_name, f'at', newline='\n')
            if not file_exists:
                file_handler.write('iteration,mean_reward,std_reward\n')
            file_handler.write(f'{iteration},{mean_reward:.4f},{std_reward:.4f}\n')
            file_handler.close()
        except Exception as e:
            env.close()
            raise e
    del model
    del env
