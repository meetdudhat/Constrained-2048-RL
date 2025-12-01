"""
Hyperparameter Configuration for AlphaZero Training.

This module acts as the central control panel for the reinforcement learning pipeline.
It balances the trade-off between:
1.  Search Depth (MCTS): Quality of the "teacher" policy.
2.  Throughput: Speed of data generation (Self-Play).
3.  Model Convergence: Learning rate and batch stability.

Architecture Note:
    The "Batch Size <= Num Self Play Games" constraint exists because our specific
    ReplayBuffer implementation samples *games* first, then *moves*. If the batch
    size exceeds the number of games played per iteration, the buffer sample
    logic may fail or introduce excessive correlation.
"""

import torch

# --- System Configuration ---
# Auto-detect CUDA for GPU acceleration; fallback to CPU for compatibility.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The objective tile. Reaching this terminates the episode with a Win signal (+1).
WIN_TILE = 2048

# ==============================================================================
# ACTIVE CONFIGURATION PROFILE: "High Quality" (~10 Hour Runtime w no GPU)
# ==============================================================================
NUM_SIMULATIONS = 150
C_PUCT = 1.0                
DIRICHLET_ALPHA = 0.3
EXPLORATION_FRACTION = 0.25

# --- Self-Play Config ---
NUM_ITERATIONS = 12 
NUM_SELF_PLAY_GAMES = 20    # Play 20 games per cycle to get stable data
REPLAY_BUFFER_SIZE = 50000
ACTION_TEMPERATURE_MOVES = 10
ACTION_TEMPERATURE = 1.0

# --- Network & Training Config ---
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_TRAIN_STEPS = 200
CHECKPOINT_INTERVAL = 1     
PIT_GAMES = 6               # Play 6 games to reduce luck in evaluation
UPDATE_THRESHOLD = 0.51


# # --- ~4 Hour Config ---
# NUM_SIMULATIONS = 75
# C_PUCT = 1.0                
# DIRICHLET_ALPHA = 0.3
# EXPLORATION_FRACTION = 0.25

# # --- Self-Play Config ---
# NUM_ITERATIONS = 8
# NUM_SELF_PLAY_GAMES = 15    # Play 15 games per cycle
# REPLAY_BUFFER_SIZE = 50000
# ACTION_TEMPERATURE_MOVES = 10
# ACTION_TEMPERATURE = 1.0

# # --- Network & Training Config ---
# BATCH_SIZE = 12
# LEARNING_RATE = 1e-4
# WEIGHT_DECAY = 1e-4
# NUM_TRAIN_STEPS = 150       # 150 updates per cycle
# CHECKPOINT_INTERVAL = 1     
# PIT_GAMES = 4               # Play 4 evaluation games
# UPDATE_THRESHOLD = 0.51



# # --- ~1 Hour Config ---
# NUM_SIMULATIONS = 50
# C_PUCT = 1.0
# DIRICHLET_ALPHA = 0.3
# EXPLORATION_FRACTION = 0.25

# # --- Self-Play Config ---
# NUM_ITERATIONS = 5
# NUM_SELF_PLAY_GAMES = 10
# REPLAY_BUFFER_SIZE = 50000
# ACTION_TEMPERATURE_MOVES = 10
# ACTION_TEMPERATURE = 1.0

# # --- Network & Training Config ---
# BATCH_SIZE = 8
# LEARNING_RATE = 1e-4 
# WEIGHT_DECAY = 1e-4
# NUM_TRAIN_STEPS = 100
# CHECKPOINT_INTERVAL = 1
# PIT_GAMES = 4
# UPDATE_THRESHOLD = 0.51




# --- "5 Minute Test" MCTS Config ---
# NUM_SIMULATIONS = 25        # Extremely fast, just to check if MCTS runs
# C_PUCT = 1.0
# DIRICHLET_ALPHA = 0.3
# EXPLORATION_FRACTION = 0.25

# # --- Self-Play Config ---
# NUM_ITERATIONS = 1          # Just run 1 loop to verify the whole cycle
# NUM_SELF_PLAY_GAMES = 2     # Play only 2 games to generate a tiny bit of data
# REPLAY_BUFFER_SIZE = 50000
# ACTION_TEMPERATURE_MOVES = 5
# ACTION_TEMPERATURE = 1.0

# # --- Network & Training Config ---
# BATCH_SIZE = 2              # Tiny batch size
# LEARNING_RATE = 1e-4
# WEIGHT_DECAY = 1e-4
# NUM_TRAIN_STEPS = 5        # Train for a few seconds
# CHECKPOINT_INTERVAL = 1     
# PIT_GAMES = 2               # Play 2 games (1 as Player A, 1 as Player B)
# UPDATE_THRESHOLD = 0.51




# MOST OPTIMAL SETTINGS FOR BEST RESULTS (VERY SLOW TRAINING NEED GPU)

# # --- Global Config ---
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# WIN_TILE = 2048  # The target tile to be considered a "win"

# # --- MCTS Config ---
# NUM_SIMULATIONS = 800  # "Thinking time" per move, 800-1600 for strong play
# C_PUCT = 1.25          # UCB exploration constant
# DIRICHLET_ALPHA = 0.3  # Noise parameter for root node exploration
# EXPLORATION_FRACTION = 0.25 # Fraction of noise to add

# # --- Self-Play Config ---
# NUM_ITERATIONS = 200        # Total training iterations (self-play + train)
# NUM_SELF_PLAY_GAMES = 5000  # Games generated per iteration
# REPLAY_BUFFER_SIZE = 500000 # Max games (not steps) in buffer
# ACTION_TEMPERATURE_MOVES = 10 # Number of moves to use temperature for exploration
# ACTION_TEMPERATURE = 1.0    # Exploration temperature

# # --- Network & Training Config ---
# BATCH_SIZE = 2048           # MCTS-hybrids require large batch sizes
# LEARNING_RATE = 1e-4        # Initial learning rate
# WEIGHT_DECAY = 1e-4         # L2 regularization
# NUM_TRAIN_STEPS = 1000      # Gradient descent steps per iteration
# CHECKPOINT_INTERVAL = 1     # Save model every N iterations
# PIT_GAMES = 20              # Games to play to evaluate new model
# UPDATE_THRESHOLD = 0.55     # Win rate needed to replace old model