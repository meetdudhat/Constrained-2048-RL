"""
Tuned Training Entry Point: DQN with Custom CNN.

This script advances the baseline by replacing the simple MLP (Multi-Layer Perceptron)
with a Custom Convolutional Neural Network (CNN).

Key Architectural Differences:
1.  Spatial Awareness: Unlike the MLP which treats the board as a flat list of 16 numbers,
    the CNN preserves the 4x4 grid structure, allowing the agent to detect spatial patterns.
2.  Custom Feature Extractor: We define `CustomCnnFor2048` because standard RL libraries
    use large kernels (stride 4, 8x8 filters) designed for Atari games. Using those on a 4x4
    grid would reduce the entire board to a single pixel instantly, losing all information.
3.  Tuned Hyperparameters: Uses a significantly longer exploration phase (50% of training)
    to prevent the agent from converging too early on suboptimal strategies.

Example Usage:
    python -m src.agents.train_tuned --group="Tuned_CNN" --reward="potential_log" --state="log2"
"""


import gymnasium
import numpy as np
import os
import time
import argparse
import torch
import torch.nn as nn

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.environments.standard_env import Standard2048Env
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
args = parser.parse_args()


# --- Logging Configuration ---
run_name_tags = f"reward-{args.reward}_state-{args.state}"
TOTAL_TIMESTEPS = args.timesteps
LOG_NAME = args.log_name
CHECKPOINT_FREQ = args.checkpoint_freq


BASE_LOG_DIR = "logs"
FINAL_LOG_DIR = os.path.join(BASE_LOG_DIR, args.group, run_name_tags)

# Creates the final directory
os.makedirs(FINAL_LOG_DIR, exist_ok=True)

class CustomCnnFor2048(BaseFeaturesExtractor):
    """
    A custom CNN (brain) for our 4x4 2048 board.
    The default "NatureCNN" is too large (built for 84x84 images)
    and crashes when given a 4x4 input.
    
    This network uses 2x2 and 3x3 convolutions that *can*
    process a 4x4 grid.
    
    Input: (batch_size, 1, 4, 4)
    """
    def __init__(self, observation_space: gymnasium.spaces.Box, feature_dim: int = 256):
        super().__init__(observation_space, feature_dim)
        
        # Input shape is (Channels, Height, Width) -> (1, 4, 4)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            # Layer 1: 3x3 kernel. Padding=1 preserves 4x4 size.
            nn.Conv2d(n_input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 2: 2x2 kernel. Padding=0 reduces 4x4 -> 3x3.
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            # Layer 3: 2x2 kernel. Reduces 3x3 -> 2x2.
            nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # The output of the CNN is flattened. 
        # 256 filters * 2x2 spatial size = 1024 features.
        self.linear = nn.Sequential(
            nn.Linear(1024, feature_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn(observations)
        return self.linear(cnn_features)


# --- Environment Setup ---
print(f"Setting up environment... Logging to {FINAL_LOG_DIR}")
print(f"Using state representation: {args.state}")
env_base = Standard2048Env(reward_mode=args.reward)


# When using a CNN, the state must be formatted as an image channel (C, H, W).
# The Log2Wrapper handles this reshaping when policy_type="cnn" is passed.
if args.state == "log2":
    print("Applying Log2Wrapper for CNN...")
    env = Log2Wrapper(env_base, policy_type="cnn")
else:
    print(f"State Representation: {args.state}")
    env_wrapped = env_base

# Applys the Monitor wrapper
info_keywords = ("score", "max_tile")
env = Monitor(env, FINAL_LOG_DIR, info_keywords=info_keywords)
print("Environment setup complete.")

# We tell the CnnPolicy to NOT normalize our images,
# because our Log2Wrapper already "normalizes" them into floats.
custom_policy_kwargs = dict(
    features_extractor_class=CustomCnnFor2048,
    features_extractor_kwargs=dict(feature_dim=256),
    normalize_images=False
)

# --- Model Setup ---
print("Initializing DQN model with CnnPolicy and Tuned Hyperparameters...")
model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=FINAL_LOG_DIR,
    buffer_size=100_000,     # Size of the replay buffer
    learning_rate=5e-4,
    batch_size=128,
    gamma=0.99,
    train_freq=4,            # Trains the model every 4 steps
    gradient_steps=1,
    target_update_interval=1000, # Update the target network
    exploration_fraction=0.5,    # 50% of training is exploration
    exploration_final_eps=0.1,
    policy_kwargs=custom_policy_kwargs
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
            info = self.locals['infos'][0]
            
            if 'score' in info:
                self.writer.add_scalar('rollout/score', info['score'], self.n_calls)
            if 'max_tile' in info:
                self.writer.add_scalar('rollout/max_tile', info['max_tile'], self.n_calls)
            
        return True

# --- Callbacks ---
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=os.path.join(FINAL_LOG_DIR, "checkpoints"),
    name_prefix="dqn_cnn"
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
# python -m src.agents.train_tuned --group="Tests" --reward="potential_log" --state="log2" --timesteps=50000 --log_name="smoke_test" --checkpoint_freq=10000

# Run 1:
# python -m src.agents.train_tuned --group="Tuned_CNN" --reward="raw_score" --state="raw"

# Run 2:
# python -m src.agents.train_tuned --group="Tuned_CNN" --reward="raw_score" --state="log2"

# Run 3:
# python -m src.agents.train_tuned --group="Tuned_CNN" --reward="log_merge" --state="log2"

# Run 4:
# python -m src.agents.train_tuned --group="Tuned_CNN" --reward="potential_log" --state="log2"

# Run 5:
# python -m src.agents.train_tuned --group="Tuned_CNN" --reward="potential_log" --state="log2" --env="constrained"

# View Logs:
# tensorboard --logdir=logs