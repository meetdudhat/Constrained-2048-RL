"""
Evaluation Script for 2048 DQN Agents.

This script loads a pre-trained reinforcement learning model and subjects it to
a series of evaluation episodes. It aggregates key performance metrics (Win Rate,
Average Score, Max Tile) to validate the agent's ability to generalize beyond
the training phase.
"""

import argparse
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from src.environments.standard_env import Standard2048Env
from src.environments.constrained_env import Constrained2048Env
from src.environments.wrapper import Log2Wrapper

# Safety cap to prevent infinite loops in broken agents or stuck states
MAX_STEPS_PER_EPISODE = 5000

def evaluate_agent():
    """
    Orchestrates the evaluation of a saved agent.

    Process:
    1. Reconstructs the specific environment configuration (Standard/Constrained, Raw/Log2).
    2. Loads the checkpointed model.
    3. Runs N evaluation episodes to gather statistical significance.
    4. Reports aggregate metrics including the specific "Win Rate" (reaching 2048).

    Args:
        None (Parses command line arguments directly).
    
    Returns:
        None (Prints metrics to stdout).
    """

    # Argument Parser setup that takes arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True) # path to zipped folder
    parser.add_argument("--env", type=str, choices=["standard", "constrained"], required=True) # choose standard vs constrained version
    parser.add_argument("--state", type=str, choices=["raw_values", "log2"], required=True) # raw or log2 board
    parser.add_argument("--n_episodes", type=int, default=100) # how many evaluation episodes to run
    args = parser.parse_args()

    if args.env == "standard":
        env_base = Standard2048Env() # build standard env
    else:
        env_base = Constrained2048Env() # other, constrained env

    if args.state == "log2":
        env = Log2Wrapper(env_base) # load Log2Wrapper state wrapper
    else:
        env = env_base

    env = Monitor(env, filename=None) # add the monitor to tracks things like final score, max tile, 
    model = DQN.load(args.model_path, env=env) # load model and attach the environment

    # variables to store results for each episode
    all_scores = []
    all_max_tiles = []
    wins = 0 # start with zero
    
    print(f"*** Starting Evaluation ***")
    print(f"Model: {args.model_path}")
    print(f"Env: {args.env}, State: {args.state}, Episodes: {args.n_episodes}\n")

    # Run Evaluation Episodes
    for i in range(args.n_episodes):
        obs, info = env.reset() # reset the env to start a new episode
        done = False
        
        MAX_STEPS_PER_EPISODE = 5000
        current_step = 0
        while not done:
            current_step += 1
            # take a look at current board and chooses best action deterministically
            action, _ = model.predict(obs)
            action = int(action) # convert action to int
            # obs(state) - new state after making the move
            # reward - ignore the reward
            # done - whether its finished
            # info - extra details
            obs, reward, terminated, truncated, info = env.step(action)

            done =  terminated or truncated
            
            if current_step >= MAX_STEPS_PER_EPISODE:
                done = True
                print("Reached max steps for episode, terminating.")

        # Extract episode stats - like total score and largest tile reached
        score = info.get("score", 0)
        max_tile = info.get("max_tile", 0)

        all_scores.append(score)
        all_max_tiles.append(max_tile)

        if max_tile >= 2048:
            wins += 1 # count a win

    print("Avg Score:", np.mean(all_scores))
    print("Avg Max Tile:", np.mean(all_max_tiles))
    print("Win Rate:", wins / args.n_episodes)
    print("Best Score:", np.max(all_scores))
    print("Highest Max Tile:", np.max(all_max_tiles))

if __name__ == "__main__":
    evaluate_agent()

