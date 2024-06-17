# https://wandb.ai/mukilan/intro_to_gym/reports/A-Gentle-Introduction-to-OpenAI-Gym--VmlldzozMjg5MTA3
# https://wandb.ai/mukilan/intro_to_rl/reports/A-Gentle-Introduction-to-Reinforcement-Learning-With-An-Example--VmlldzoyODc4NzY0

import random
import numpy as np
import gymnasium as gym

class ThermoEnv(gym.Env):
    def __init__(self):
        # Actions our pod can take: increase, decrease or stay same
        self.action_space = gym.spaces.Discrete(3)
        # Temperature array
        self.observation_space = gym.spaces.Box(low=np.array([50]), high=np.array([70]))
        
        self.max_episode_steps = 1000
        self.episode_step = None
        self.state = None # Temperature
        self.reward = None

    def reset(self, seed, options):
        super().reset(seed, options)
        # Set random temperature
        self.state = np.array([60 + random.randint(-10,10)]).astype(float)
        self.episode_step = 0
        self.reward = 0
        return self.state

    def step(self, action):
        self.episode_step += 1

        # Perform action: 0=decrease, 1=hold, 2=increase
        self.state += (action - 1)*5

        # Calculate reward
        reward = 100 if self.state >=60 and self.state <=67 else -10
        self.reward += reward

        # Observe next state (apply temperature noise)
        self.state += random.randint(-5,5)
        
        # Check if episode is over
        truncated = True if self.episode_step > self.max_episode_steps else False        
        # Check if we failed to maintain temperature
        terminated = True if self.state < 50 or self.state > 70 else False
        
        return self.state, self.reward, terminated | truncated, {}


    def render(self):
        pass
    
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env = ThermoEnv()
#env = Monitor(env)
#env = DummyVecEnv([lambda:env])

model = PPO(
            policy = "MlpPolicy",
            env = env,
            verbose = 1,
            tensorboard_log="runs/thermo")
model.learn(
            total_timesteps = 25000,
            log_interval = 1,
            tb_log_name = "PPO_Thermo",
            progress_bar = True)

model.save('PPO_Thermo_25k')

evaluate_policy(model, env, n_eval_episodes=10)
