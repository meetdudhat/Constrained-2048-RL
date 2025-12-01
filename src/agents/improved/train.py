"""
AlphaZero Training Manager for 2048.

This script implements the core reinforcement learning loop inspired by AlphaZero.
It manages the cycle of:
1.  Self-Play: Generating training data using MCTS as a policy improvement operator.
2.  Training: Updating the Neural Network weights to match MCTS probabilities (Policy Head)
    and predict game outcomes (Value Head).
3.  Evaluation (Pitting): Comparing the new network against the previous best to ensure
    monotonous improvement and prevent catastrophic forgetting.

Usage:
    python -m src.agents.improved.train
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import time
from datetime import timedelta

from src.environments.game_env import Game2048Env
from src.agents.improved.network import Network
from src.agents.improved.mcts import MCTS
from src.agents.improved.config \
import WIN_TILE, ACTION_TEMPERATURE, ACTION_TEMPERATURE_MOVES, \
    BATCH_SIZE, NUM_TRAIN_STEPS, DEVICE, PIT_GAMES, NUM_ITERATIONS, \
    REPLAY_BUFFER_SIZE, NUM_SELF_PLAY_GAMES, LEARNING_RATE, WEIGHT_DECAY, \
    UPDATE_THRESHOLD, CHECKPOINT_INTERVAL


class ReplayBuffer:
    """
    Experience Replay for storing Self-Play data.
    
    Stores tuples of (Canonical State, MCTS Policy, Final Value).
    
    Architectural Note:
    We store 'Canonical' states (rotation/reflection invariant) rather than raw states.
    This effectively multiplies the dataset size by 8x, allowing the network to generalize 
    across symmetric board positions.
    """
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        
    def add_game(self, game_history):
        """
        Ingests a full game's worth of transitions.
        """
        self.buffer.append(game_history)
        
    def sample(self, batch_size):
        """
        Samples a batch of transitions for training.
        
        Optimization:
        Instead of treating the buffer as a flat list of millions of moves (which is slow to manage),
        we store lists of games. We first sample N games, then pick one random move from each.
        This breaks temporal correlations between consecutive moves, which is crucial for SGD stability.
        """
        actual_batch_size = min(len(self.buffer), batch_size)
        game_histories = random.sample(self.buffer, k=actual_batch_size)
        
        states, policies, values = [], [], []
        
        for game_history in game_histories:
            state, policy, value = random.choice(game_history)
            states.append(state)
            policies.append(policy)
            values.append(value)
            
        return np.array(states), np.array(policies), np.array(values)

    def __len__(self):
        return len(self.buffer)


def self_play(network, env, replay_buffer, writer, iteration_idx, game_idx):
    """
    Executes one episode of Self-Play to generate training data.
    
    The Agent plays against the Environment (Chance).
    - The Network acts as the 'Student'.
    - The MCTS acts as the 'Teacher', providing a better policy target (pi) than the raw network output.
    """
    print("Starting self-play game...")
    game_history = []
    env.reset()
    move_count = 0
    
    while True:
        if env.game.is_game_over():
            break
        
        # Snapshot current state
        state_tensor = env.get_state()
        
        # Initialize MCTS with the current network
        mcts = MCTS(network)
        
        # Temperature Schedule:
        # - Early game (Temperature > 0): Explore diverse starting positions.
        # - Late game (Temperature = 0): Play greedily to master specific tactical endpoints.
        temperature = ACTION_TEMPERATURE if move_count < ACTION_TEMPERATURE_MOVES else 0
        
        # Execute Search to get improved policy (pi)
        action, mcts_policy = mcts.search(env, temperature=temperature)
        
        if np.sum(mcts_policy) == 0:
            # Fallback for rare terminal states not caught by is_game_over
            break
        
        # Convert to Canonical Form (Symmetry) before storage
        canonical_state, canonical_policy = env.get_canonical_form(state_tensor, mcts_policy)
        
        # Store (s, pi). We cannot store 'z' (value) yet because the game isn't over.
        game_history.append((canonical_state, canonical_policy))

        _, reward, done = env.step(action)
        move_count += 1
        
        if done:
            break
            
    # Retrospective Value Assignment:
    # Once the game ends, we propagate the final result (Win/Loss) back to all steps.
    final_value = reward
    
    full_game_history = []
    for state, policy in game_history:
        full_game_history.append((state, policy, final_value))
        
    replay_buffer.add_game(full_game_history)
    
    # Logging metrics
    max_tile = env.game.get_max_tile()
    score = env.game.score
    
    # Calculate global step for linear tensorboard x-axis
    global_step = (iteration_idx * NUM_SELF_PLAY_GAMES) + game_idx
    
    writer.add_scalar("SelfPlay/MaxTile", max_tile, global_step)
    writer.add_scalar("SelfPlay/Score", score, global_step)
    writer.add_scalar("SelfPlay/Moves", move_count, global_step)

    print(f"Game {game_idx+1}: Max Tile: {max_tile}, Score: {score}")
    
    print(f"Self-play game finished. Max tile: {env.game.get_max_tile()}, Result: {final_value}")


def train_network(network, replay_buffer, optimizer, writer, iteration_idx):
    """
    Performs Neural Network optimization using the generated dataset.
    
    Objective Function:
        Loss = (z - v)^2  -  pi * log(p) + c * ||theta||^2
        
        1. MSE Loss (z - v): Minimizes error between predicted value (v) and actual outcome (z).
        2. Cross-Entropy (-pi * log(p)): Minimizes difference between Network policy (p) and MCTS policy (pi).
        3. L2 Regularization (theta): Handled by the optimizer's weight_decay.
    """
    if len(replay_buffer) < BATCH_SIZE:
        print(f"Buffer size ({len(replay_buffer)}) is too small. Need at least {BATCH_SIZE}. Skipping training.")
        return

    print("Starting training phase...")
    network.train()
    
    total_loss_v = 0
    total_loss_p = 0
    
    for _ in tqdm(range(NUM_TRAIN_STEPS), desc="Training Steps"):
        # 1. Sample Batch
        states, target_policies, target_values = replay_buffer.sample(BATCH_SIZE)
        
        # Move to GPU/Device
        states = torch.from_numpy(states).float().to(DEVICE)
        target_policies = torch.from_numpy(target_policies).float().to(DEVICE)
        target_values = torch.from_numpy(target_values).float().unsqueeze(1).to(DEVICE)
        
        # 2. Forward Pass
        pred_policies, pred_values = network(states)
        
        # 3. Calculate Loss
        loss_v = F.mse_loss(pred_values, target_values)
        
        # Use log_softmax for numerical stability with Cross Entropy
        loss_p = -torch.sum(target_policies * F.log_softmax(pred_policies, dim=1), dim=1).mean()
        
        total_loss = loss_v + loss_p
        
        # 4. Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        total_loss_v += loss_v.item()
        total_loss_p += loss_p.item()
        
        avg_loss_v = total_loss_v / NUM_TRAIN_STEPS
        avg_loss_p = total_loss_p / NUM_TRAIN_STEPS
        writer.add_scalar("Train/ValueLoss", avg_loss_v, iteration_idx)
        writer.add_scalar("Train/PolicyLoss", avg_loss_p, iteration_idx)
    
    print(f"Training complete. Avg Value Loss: {total_loss_v / NUM_TRAIN_STEPS:.4f}, Avg Policy Loss: {total_loss_p / NUM_TRAIN_STEPS:.4f}")


def pit(new_model, old_model, env):
    """
    Evaluates the new model against the current best model.
    
    To accept a new model, it must demonstrate a statistically significant 
    improvement (defined by UPDATE_THRESHOLD) over the previous iteration.
    This prevents the model from chasing unstable local optima.
    """
    print("Pitting new model vs. old model...")
    new_wins = 0
    old_wins = 0
    
    for i in tqdm(range(PIT_GAMES), desc="Pitting Games"):
        # Although 2048 is single-player, we treat the 'old' and 'new' models
        # as competitors to normalize against environment variance (RNG of tile spawns).
        model_a = new_model if i % 2 == 0 else old_model
        model_b = old_model if i % 2 == 0 else new_model

        max_tile_a = run_game(model_a, env)
        max_tile_b = run_game(model_b, env)
        
        win_a = max_tile_a >= WIN_TILE
        win_b = max_tile_b >= WIN_TILE
        
        # Evaluation Logic:
        # 1. Did A reach 2048 and B fail? -> A wins.
        # 2. Did both fail? -> Tie-break on Max Tile.
        if win_a and not win_b:
            score_a = 1
            score_b = 0
        elif win_b and not win_a:
            score_a = 0
            score_b = 1
        elif max_tile_a > max_tile_b:
            score_a = 1
            score_b = 0
        else:
            # Conservative Update: In exact ties, the Old model retains the title.
            score_a = 0
            score_b = 1
            
        if i % 2 == 0:
            new_wins += score_a
            old_wins += score_b
        else:
            new_wins += score_b
            old_wins += score_a
            
    win_rate = new_wins / PIT_GAMES if PIT_GAMES > 0 else 0
    print(f"Pit results: New: {new_wins} / Old: {old_wins}. Win rate: {win_rate:.2%}")
    return win_rate

def run_game(network, env):
    """Helper for the evaluation phase. Runs a greedy (deterministic) game."""
    env.reset()
    while True:
        mcts = MCTS(network)
        # Temperature = 0 forces the agent to pick the absolute best move found by MCTS
        action, _ = mcts.search(env, temperature=0)
        _, _, done = env.step(action)
        if done:
            return env.game.get_max_tile()


def main():
    """
    Main Execution Pipeline.
    
    Structure:
    1. Initialize Environment and Best Model.
    2. Loop N Iterations:
       a. Self-Play: Populate buffer with new game data.
       b. Train: Optimize a *copy* of the model on the buffer.
       c. Pit: Compare Copy vs Original. Update if Copy is better.
       d. Checkpoint: Save progress.
    """
    # --- Configuration ---
    writer = SummaryWriter(log_dir="logs/improved_mcts")
    env = Game2048Env(immovable_cell=(3,0))
    
    current_best_net = Network(input_channels=env.num_channels).to(DEVICE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    total_start_time = time.time()
    print(f"--- Training Started ---")

    # --- Main AlphaZero Loop ---
    for i in range(NUM_ITERATIONS):
        iter_start_time = time.time()
        
        print(f"\n--- Iteration {i+1} / {NUM_ITERATIONS} ---")
        
        # 1. Data Generation
        print("Generating self-play data...")
        current_best_net.eval()
        for j in tqdm(range(NUM_SELF_PLAY_GAMES), desc="Self-Play Games"):
            self_play(current_best_net, env, replay_buffer, writer, i, j)
            
        # 2. Optimization
        print("Training network...")
        new_net = Network(input_channels=env.num_channels).to(DEVICE)
        new_net.load_state_dict(current_best_net.state_dict())
        
        optimizer = optim.Adam(new_net.parameters(), 
                               lr=LEARNING_RATE, 
                               weight_decay=WEIGHT_DECAY)
                               
        train_network(new_net, replay_buffer, optimizer, writer, i)
        
        # 3. Model Selection
        print("Evaluating new model...")
        new_net.eval()
        current_best_net.eval()
        
        win_rate = pit(new_net, current_best_net, env)
        
        writer.add_scalar("Evaluation/WinRate_vs_Old", win_rate, i)
        print(f"Pit Win Rate: {win_rate:.2%}")
        
        if win_rate > UPDATE_THRESHOLD:
            print("--- NEW MODEL IS BETTER, UPDATING ---")
            current_best_net.load_state_dict(new_net.state_dict())
            torch.save(current_best_net.state_dict(), f"{model_dir}/best_model.pth")
        else:
            print("--- New model not better, discarding. ---")

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(current_best_net.state_dict(), f"{model_dir}/checkpoint_iter_{i+1}.pth")

        iter_end_time = time.time()
        iter_duration = iter_end_time - iter_start_time
        print(f"Iteration {i+1} Finished in: {str(timedelta(seconds=int(iter_duration)))}")
        
    writer.close()
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\n===========================================")
    print(f"TRAINING COMPLETE")
    print(f"Total Run Time: {str(timedelta(seconds=int(total_duration)))}")
    print(f"===========================================")
    
if __name__ == "__main__":
    main()
    
    
# python -m src.agents.improved.train

# View Logs:
# tensorboard --logdir=logs