import gymnasium
import numpy as np
import os
import time
import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

from src.environments.standard_env import Standard2048Env
from src.environments.constrained_env import Constrained2048Env

from src.environments.wrapper import Log2Wrapper

# --- Arg Configurations ---
parser = argparse.ArgumentParser(description="Trains a DQN agent for 2048.")
parser.add_argument(
    "--group",
    type=str,
    default="DefaultGroup",
    help="High-level experiment group (e.g., 'RewardExperiments')."
)
parser.add_argument(
    "--reward",
    type=str,
    default="raw_score",
    help="Name of the reward structure (e.g., 'merge_penalty')."
)
parser.add_argument(
    "--state",
    type=str,
    default="raw_values",
    help="Name of the state representation (e.g., 'log2_normalized')."
)
parser.add_argument(
    "--timesteps",
    type=int,
    default=1_000_000,
    help="Total number of timesteps to train for."
)
parser.add_argument(
    "--log_name",
    type=str,
    default="raw_state_raw_reward",
    help="Name of the log directory for this experiment."
)
parser.add_argument(
    "--checkpoint_freq",
    type=int,
    default=100_000,
    help="Save a model checkpoint every N steps."
)
parser.add_argument(
    "--env",
    choices=["standard", "constrained"],
    default="standard",
    help="Which 2048 environment to use."
)

args = parser.parse_args()

run_name_tags = f"reward-{args.reward}_state-{args.state}"

# --- Configuration ---
TOTAL_TIMESTEPS = args.timesteps
LOG_NAME = args.log_name
CHECKPOINT_FREQ = args.checkpoint_freq


BASE_LOG_DIR = "logs"
FINAL_LOG_DIR = os.path.join(BASE_LOG_DIR, args.group, run_name_tags)

# Creates the final directory
os.makedirs(FINAL_LOG_DIR, exist_ok=True)

# --- Environment Setup ---
print(f"Setting up environment... Logging to {FINAL_LOG_DIR}")
print(f"Using state representation: {args.state}")

print(f"Selected environment: {args.env}")
EnvClass = Standard2048Env if args.env == "standard" else Constrained2048Env

env = EnvClass(reward_mode=args.reward)


# Conditionally applies the wrapper based on the --state argument
if args.state == "log2":
    print("Applying Log2Wrapper...")
    env = Log2Wrapper(env)
else:
    print(f"State Representation: {args.state}")

# Applys the Monitor wrapper
info_keywords = ("score", "max_tile")
env = Monitor(env, FINAL_LOG_DIR, info_keywords=info_keywords)
print("Environment setup complete.")

# --- Model Setup ---
print("Initializing DQN model...")
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=FINAL_LOG_DIR,
    buffer_size=100_000,     # Size of the replay buffer
    learning_rate=1e-4,
    batch_size=128,
    gamma=0.99,
    train_freq=4,            # Trains the model every 4 steps
    gradient_steps=1,
    target_update_interval=1000, # Update the target network
    exploration_fraction=0.1,    # 10% of training is exploration
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[256, 256]) # Network architecture
)

print("Model initialized.")

# --- Callback Class ---
class EpisodeLogCallback(BaseCallback):
    """
    Custom callback for plotting final episode values
    directly to tensorboard in the 'rollout/' section.
    """
    def __init__(self, verbose=0):
        super(EpisodeLogCallback, self).__init__(verbose)
        self.writer = None

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # Finds the TensorBoardOutputFormat
        for formatter in self.logger.output_formats:
            if isinstance(formatter, TensorBoardOutputFormat):
                self.writer = formatter.writer
                break
        
        if self.writer is None:
            raise RuntimeError(
                "TensorBoardOutputFormat not found."
                "Ensure you have set tensorboard_log in your model."
            )

    def _on_step(self) -> bool:
        """
        Check if the episode is done and log the final info.
        """
        if self.locals['dones'][0]:
            
            # Gets the info dict from the last step
            info = self.locals['infos'][0]
            
            # Logs score and max_tile to 'rollout/' namespace
            # Uses self.n_calls as the step counter
            if 'score' in info:
                self.writer.add_scalar('rollout/score', info['score'], self.n_calls)
            if 'max_tile' in info:
                self.writer.add_scalar('rollout/max_tile', info['max_tile'], self.n_calls)
            
        return True

# --- Callbacks ---
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=os.path.join(FINAL_LOG_DIR, "checkpoints"),
    name_prefix="dqn_baseline"
)

# Create the new callback
episode_log_callback = EpisodeLogCallback()

# Combine all callbacks into a list
callback_list = [checkpoint_callback, episode_log_callback]

# --- Training ---
print(f"--- Starting Training ---")
print(f"Log Name: {LOG_NAME}")
print(f"Total Timesteps: {TOTAL_TIMESTEPS}")
print(f"TensorBoard logs: {FINAL_LOG_DIR}")
print("---------------------------")
start_time = time.time()

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callback_list,
    reset_num_timesteps=False
)

end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds.")

# --- Save Final Model ---
FINAL_MODEL_PATH = os.path.join(FINAL_LOG_DIR, "dqn_baseline_final.zip")
model.save(FINAL_MODEL_PATH)
print(f"Final model saved to {FINAL_MODEL_PATH}")
env.close()

# Run Test:
# python -m src.agents.train_baseline --group="Rewards" --reward="raw_score" --state="raw" --timesteps=50000 --log_name="smoke_test" --checkpoint_freq=10000

# Run 1:
# python -m src.agents.train_baseline --group="States" --reward="raw_score" --state="raw"

# Run 2:
# python -m src.agents.train_baseline --group="States" --reward="raw_score" --state="log2"

# View Logs:
# tensorboard --logdir=logs