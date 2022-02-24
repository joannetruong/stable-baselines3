import os

import gym
import numpy as np


from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.succ_save_path = os.path.join(log_dir, 'best_succ_model')
        self.spl_save_path = os.path.join(log_dir, 'best_spl_model')
        self.best_mean_success = -np.inf
        self.best_mean_spl = -np.inf

    # def _init_callback(self) -> None:
        # # Create folder if needed
        # if self.succ_save_path is not None:
        #     os.makedirs(self.succ_save_path, exist_ok=True)
        # if self.spl_save_path is not None:
        #     os.makedirs(self.spl_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, succs = ts2xy(load_results(self.log_dir), 'timesteps', 'success')
          _, spls = ts2xy(load_results(self.log_dir), 'timesteps', 'spl')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_success = np.mean(succs[-100:])
              mean_spl = np.mean(spls[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps} - Curr mean success: {mean_success:.2f} - Curr mean spl: {mean_spl:.2f}")
                print(f"Best mean success: {self.best_mean_success:.2f} - Last mean success per episode: {mean_success:.2f}")
                print(f"Best mean spl: {self.best_mean_spl:.2f} - Last mean spl per episode: {mean_spl:.2f}")

              # New best model, you could save the agent here
              if mean_success > self.best_mean_success:
                  self.best_mean_success = mean_success
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.succ_save_path}")
                  self.model.save(self.succ_save_path)
              if mean_spl > self.best_mean_spl:
                  self.best_mean_spl = mean_spl
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.spl_save_path}")
                  self.model.save(self.spl_save_path)
        return True
