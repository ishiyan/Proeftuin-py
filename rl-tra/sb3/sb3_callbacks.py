import os
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    Args:
        check_freq:
            Check frequency (in timesteps).
        log_dir:
            Path to the folder where the model will be saved.
            It must contain the file created by the ``Monitor`` wrapper.
        verbose:
            Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages.
    """
    def __init__(self, check_freq: int=12800, log_dir: str ='./sb3/dqn/',
                 model_name: str='best_model', save_replay_buffer: bool=False, verbose: int=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_replay_buffer = save_replay_buffer
        self.save_path = os.path.join(log_dir, model_name)
        self.save_path_replay_buffer = os.path.join(log_dir, model_name+'_replay_buffer')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed.
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          # Retrieve training reward.
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes.
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f'Num timesteps: {self.num_timesteps}')
                print(f'Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}')
              # New best model, you could save the agent here.
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  if self.verbose >= 1:
                    print(f'Saving new best model to {self.save_path}')
                  self.model.save(self.save_path)
                  if self.save_replay_buffer:
                    if self.verbose >= 1:
                        print(f'Saving replay buffer to {self.save_path_replay_buffer}')
                    self.model.save_replay_buffer(self.save_path_replay_buffer)
        return True
