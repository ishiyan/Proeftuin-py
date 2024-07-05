import os
import numpy as np

# ---------------------------------- env
from env import Env01
env = Env01(symbol='ETHUSDT',
            time_frame='1m',
            scale_period = 196,
            copy_period = 196,
            episode_max_steps=128,
            render_mode=None)

# ---------------------------------- model
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor

dir = './dqn_02/'
if not os.path.exists(dir):
    os.makedirs(dir, exist_ok=True)
#env = Monitor(env, dir, allow_early_resets=False)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int =100, log_dir: str ='./dqn_01/', save_replay_buffer: bool = False, verbose: int =0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_replay_buffer = save_replay_buffer
        self.save_path = os.path.join(log_dir, 'best_model')
        self.save_path_replay_buffer = os.path.join(log_dir, 'best_model_replay_buffer')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f'Num timesteps: {self.num_timesteps}')
                print(f'Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}')

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f'Saving new best model to {self.save_path}')
                  self.model.save(self.save_path)
                  if self.save_replay_buffer:
                    if self.verbose >= 1:
                        print(f'Saving replay buffer to {self.save_path_replay_buffer}')
                    self.model.save_replay_buffer(self.save_path_replay_buffer)

        return True
callback = SaveOnBestTrainingRewardCallback(check_freq=1280, log_dir=dir, verbose=1)

for epo in range(1000):
    epoch = 2 + epo
    env = Env01(symbol='ETHUSDT',
            time_frame='1m',
            scale_period = 196,
            copy_period = 196,
            episode_max_steps=128,
            render_mode=None)
    env = Monitor(env, dir+"epoch_"+str(epoch+1), allow_early_resets=False)

    name='./dqn_02.zip'
    if os.path.exists(name):
        model = DQN.load(name, env=env, verbose=1, print_system_info=True)
        #model.load_replay_buffer("dqn_02_replay_buffer")
        model.set_random_seed(epoch+1)
    else:
        model = DQN('MultiInputPolicy', env, verbose=1)

    # , reset_num_timesteps=False
    model.learn(total_timesteps=int(128*10000), log_interval=4, progress_bar=True)#, callback=callback)
    model.save('dqn_02')
    #model.save_replay_buffer("dqn_02_replay_buffer")
    del model
    del env


# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
#model = DQN.load('dqn_01', env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
#vec_env = model.get_env()
#obs = vec_env.reset()
#for i in range(1000):
#    action, _states = model.predict(obs, deterministic=True)
#    obs, rewards, dones, info = vec_env.step(action)
#    vec_env.render("human")

