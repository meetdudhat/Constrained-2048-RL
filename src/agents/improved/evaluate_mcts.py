"""
Evaluation Script for AlphaZero-based 2048 Agent.

This script benchmarks the performance of a trained AlphaZero model.
Unlike evaluate.py (which tests the DQN baseline), this script integrates the 
Neural Network with the Monte Carlo Tree Search (MCTS) engine.

Key Evaluation Logic:
1.  **Deterministic Execution**: We use a temperature of 0.0 during MCTS search.
    This forces the agent to always select the move with the highest visit count
    (the "best" move), removing the randomness used during training.
2.  **Fresh Search**: A new search tree is built for every move to ensure the 
    decision is based solely on the current board configuration, independent of 
    previous stochastic outcomes.

Usage:
    python -m src.agents.improved.evaluate_mcts --model_path models/best_model.pth --env standard
"""


import argparse
import torch
import numpy as np
from tqdm import tqdm
import os
import sys
import time
from datetime import timedelta

from src.environments.game_env import Game2048Env
from src.agents.improved.network import Network
from src.agents.improved.mcts import MCTS
from src.agents.improved.config import DEVICE, WIN_TILE

def evaluate_agent():
    """
    Runs a series of test episodes to gather statistical performance metrics
    (Win Rate, Average Score, Max Tile).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth file")
    parser.add_argument("--env", type=str, choices=["standard", "constrained"], default="standard")
    parser.add_argument("--n_episodes", type=int, default=100)
    args = parser.parse_args()

    # --- Environment Setup ---
    # The 'Constrained' variant places an immovable tile at (3,0) to test
    # the agent's ability to adapt to blocked corners.
    immovable = (3,0) if args.env == "constrained" else None
    env = Game2048Env(immovable_cell=immovable)

    # --- Model Loading ---
    print(f"Loading model from {args.model_path}...")
    network = Network(input_channels=env.num_channels).to(DEVICE)
    
    # map_location ensures a model trained on a GPU can still be evaluated on a CPU
    
    # network.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    state_dict = torch.load(args.model_path, map_location="cpu")
    network.load_state_dict(state_dict)
    
    network.eval() # Set to evaluation mode (disables BatchNorm tracking/Dropout)

    all_scores = []
    all_max_tiles = []
    wins = 0

    print(f"*** Starting MCTS Evaluation ***")
    print(f"Env: {args.env}, Episodes: {args.n_episodes}\n")

    start_time = time.time()

    for i in tqdm(range(args.n_episodes)):
        env.reset()
        done = False
        
        while not done:
            
            # We instantiate a fresh MCTS for every move.
            # While computationally expensive, this ensures the search tree is
            # perfectly synchronized with the current actual board state.
            mcts = MCTS(network)
            
            # Temperature = 0 effectively turns MCTS into a pure Argmax function.
            # We want the agent to play its absolute best move, not explore.
            action, _ = mcts.search(env, temperature=0)
            
            if action is None:
                # Handle edge cases where the game is technically over but
                # the loop condition hasn't caught it yet.
                break
                
            _, _, done = env.step(action)

        # --- Metrics Collection ---
        score = env.game.score
        max_tile = env.game.get_max_tile()
        
        if max_tile >= 1024:  # Only print interesting boards
            print(f"\nEpisode {i} Final Board (Max: {max_tile}):")
            print(env.game.board)
            print("-" * 20)
        
        all_scores.append(score)
        all_max_tiles.append(max_tile)
        
        # In 2048, reaching the target tile counts as a "Win"
        if max_tile >= WIN_TILE:
            wins += 1
        
    end_time = time.time()
    total_duration = end_time - start_time

    print("\n*** Final Results ***")
    print(f"Avg Score:      {np.mean(all_scores):.2f}")
    print(f"Avg Max Tile:   {np.mean(all_max_tiles):.2f}")
    print(f"Win Rate:       {wins / args.n_episodes:.2%}")
    print(f"Best Score:     {np.max(all_scores)}")
    print(f"Highest Tile:   {np.max(all_max_tiles)}")
    print(f"Total Time:     {str(timedelta(seconds=int(total_duration)))}")

if __name__ == "__main__":
    evaluate_agent()
    
    
# python -m src.agents.improved.evaluate_mcts --model_path models/checkpoint_iter_11.pth --env standard