import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from env import Env01
from sb3 import SaveOnBestTrainingRewardCallback

# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
# https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html
# https://github.com/DLR-RM/stable-baselines3/issues/597

# This takes about 20 gigabytes of space.
save_replay_buffer=False
evaluate_every_epoch=True

name = 'ppo_01'
dir = f'./sb3/{name}/'
if not os.path.exists(dir):
    os.makedirs(dir, exist_ok=True)

# 128 * 50 = every 50th episode
callback = SaveOnBestTrainingRewardCallback(
    check_freq=128*50,    
    log_dir=dir,
    model_name='best_reward_model',
    save_replay_buffer=False,
    verbose=0)

start_epoch_number = 4
for epoch in range(start_epoch_number, 1000):
    env = Env01(symbol='ETHUSDT',
            time_frame='1m',
            scale_period = 196,
            copy_period = 196,
            episode_max_steps=128,
            render_mode=None)
    env = Monitor(env, dir+"epoch_"+str(epoch), allow_early_resets=False)

    saved_model_path = os.path.join(dir, 'ppo.zip')
    saved_replay_buffer_path = os.path.join(dir, 'ppo_replay_buffer.pkl')

    if os.path.exists(saved_model_path):
        model = PPO.load(name, env=env, verbose=1, print_system_info=True)
        if os.path.exists(saved_replay_buffer_path):
            model.load_replay_buffer(saved_replay_buffer_path)
        model.set_random_seed(epoch)
        model._last_obs = None
    else:
        model = PPO('MultiInputPolicy', env,
            seed=epoch,
            verbose=1)

    model.learn(total_timesteps=int(128*10000),
        log_interval=4,
        callback=callback,
        reset_num_timesteps = False,
        progress_bar=True)
 
    model.save(saved_model_path)
    if save_replay_buffer:
        model.save_replay_buffer(saved_replay_buffer_path)
    if not evaluate_every_epoch:
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        print(f'Epoch {epoch} mean_reward: {mean_reward:.2f} std_reward: {std_reward:.2f}')
    del model
    del env
