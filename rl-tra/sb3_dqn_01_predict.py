import os
import numpy as np

from stable_baselines3 import DQN

from env import Env01, HumanRendering

name = 'dqn_01'
dir = f'./sb3/{name}/'
if not os.path.exists(dir):
    os.makedirs(dir, exist_ok=True)

saved_model_path = os.path.join(dir, 'dqn.zip')
saved_replay_buffer_path = os.path.join(dir, 'dqn_replay_buffer.pkl')


env = Env01(symbol='ETHUSDT',
        time_frame='1m',
        scale_period = 196,
        copy_period = 196,
        episode_max_steps=128,
        render_mode='rgb_array')
env = HumanRendering(env)

if os.path.exists(saved_model_path):
    model = DQN.load(name, env=env, verbose=1, print_system_info=True)
    if os.path.exists(saved_replay_buffer_path):
        model.load_replay_buffer(saved_replay_buffer_path)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render('human')
