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
    def __init__(self,
                 check_freq: int=128*10,
                 log_dir: str ='./sb3/algo/',
                 algo_name: str='algo',
                 csv_suffix: str=None,
                 save_replay_buffer: bool=False,
                 verbose: int=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir if log_dir is not None else './'
        self.save_replay_buffer = save_replay_buffer
        self.csv_path = os.path.join(log_dir,
            f'{algo_name}_best_reward_model.csv') if csv_suffix is None \
            else os.path.join(log_dir,
                f'{algo_name}_best_reward_model_{csv_suffix}')
        self.save_path = os.path.join(log_dir,
            f'{algo_name}_best_reward_model.zip')
        self.save_path_replay_buffer = os.path.join(log_dir,
            f'{algo_name}_best_reward_replay_buffer.pkl')
        self.best_mean_reward = -np.inf
        self.csv_file_handler = None
        
    def _init_callback(self) -> None:
        # Create (if any) missing filename directories.
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        file_exists = os.path.exists(self.csv_path)
        # Prevent newline issue on Windows, see GH issue #692.
        self.csv_file_handler = open(self.csv_path, f'at', newline='\n')
        if not file_exists:
            self.csv_file_handler.write('r,t,c\n')
            self.csv_file_handler.flush()
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward.
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes.
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f'Num timesteps: {self.num_timesteps}')
                    print(f'Best mean reward: {self.best_mean_reward:.4f} - Last mean reward per episode: {mean_reward:.4f}')
                # New best model, you could save the agent here.
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward

                    if self.csv_file_handler is not None:
                        self.csv_file_handler.write(f'{mean_reward:.4f},{self.num_timesteps},{self.n_calls}\n')
                        self.csv_file_handler.flush()

                    if self.verbose > 0:
                        print(f'Saving new best model to {self.save_path}')
                    self.model.save(self.save_path)
                    if self.save_replay_buffer and hasattr(self.model, "replay_buffer") \
                                                and self.model.replay_buffer is not None:
                        if self.verbose > 0:
                            print(f'Saving replay buffer to {self.save_path_replay_buffer}')
                        self.model.save_replay_buffer(self.save_path_replay_buffer)
        return True

    def _on_training_end(self) -> None:
        if self.csv_file_handler is not None:
            self.csv_file_handler.close()
            self.csv_file_handler = None