import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from sb3 import create_single_env02, create_vec_env02, save_evaluation_statistics
from sb3 import create_save_on_best_training_reward_callback, cleanup_model_env

name = 'ppo_02_bet_on_return_2'
dir = f'./sb3/{name}/'

start_iteration_number = 1
total_iterations = 100#1000

episode_max_steps = 180
learn_episodes = 10000#100000#10#1000#10000

evaluate_episodes_every_iteration=4 # Set to 0 to disable
verbose=1

#save_on_best_training_reward_callback=None
save_on_best_training_reward_callback=create_save_on_best_training_reward_callback(
    check_freq=episode_max_steps*5, # 50 = every 50th episode
    name=name, dir=dir, verbose=0)

def create_env(which='subproc', iteration: int=1, eval: bool=False):
    #symbol = 'BTCEUR' if eval else ['ETHUSDT','BTCUSDT']
    symbol = ['ETHUSDT','BTCUSDT'] # 'BTCEUR' sometimes has no data intervals
    render='gif' if eval else 'log'
    if which in ['subproc','dummy']:
        return create_vec_env02(vec_env=which, symbol=symbol,
            episode_max_steps=episode_max_steps, name=name, dir=dir,
            iteration=iteration, eval=eval, render=render,
            vectorized_render=True, vectorized_monitor=True,
            max_envs=4, verbose=verbose)
    else:
        return create_single_env02(symbol=symbol,
            episode_max_steps=episode_max_steps, name=name, dir=dir,
            iteration=iteration, eval=eval, render=render,
            monitor=True)

if __name__=="__main__":
    for iteration in range(start_iteration_number, total_iterations):
        env = create_env(which='subproc', iteration=iteration, eval=False)

        saved_model_path = os.path.join(dir, f'{name}_model.zip')
        if os.path.exists(saved_model_path):
            model = PPO.load(saved_model_path, env=env, verbose=verbose, print_system_info=True)
            #model.set_random_seed(iteration)
            model._last_obs = None
        else:
            #model = PPO('MultiInputPolicy', env, seed=iteration, verbose=verbose)
            model = PPO('MultiInputPolicy', env, verbose=verbose)
            #model = PPO('MlpPolicy', env, verbose=verbose)

        try:
            # 24576 12288 6144 3072 1536 768 384 192 96 48 24 12 6 3
            model.learn(total_timesteps=int(episode_max_steps*learn_episodes),
                log_interval=4,
                callback=save_on_best_training_reward_callback,
                reset_num_timesteps = False,
                progress_bar=True)
            model.save(saved_model_path)
            cleanup_model_env(model, env)
        except Exception as e:
            cleanup_model_env(model, env)
            raise e

        if evaluate_episodes_every_iteration > 0:
            if verbose > 0:
                print(f'Evaluate policy at iteration {iteration} '
                      f'({evaluate_episodes_every_iteration} episodes)')
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            env = create_env(which='subproc', iteration=iteration, eval=True)
            try:
                model = PPO.load(saved_model_path, env=env, verbose=verbose, print_system_info=True)
                #model.set_random_seed(iteration)
                model._last_obs = None
                mean_reward, std_reward = evaluate_policy(model, env=env, deterministic=True,
                    n_eval_episodes=evaluate_episodes_every_iteration)
                save_evaluation_statistics(mean_reward, std_reward, name, dir, iteration)
                cleanup_model_env(model, env)
            except Exception as e:
                cleanup_model_env(model, env)
                raise e
