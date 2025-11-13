import gymnasium
from gymnasium import spaces
import numpy as np
from src.game.standard_game import StandardGame2048

class Standard2048Env(gymnasium.Env):
    
    def __init__(self, render_mode=None, reward_mode="raw_score"):
        super().__init__()
        self.game = StandardGame2048()
        self.action_space = spaces.Discrete(4)
        self.reward_mode = reward_mode
        
        # Observation space is 0 to infinity (no -1 block)
        self.observation_space = spaces.Box(low=0, 
                                            high=np.inf, 
                                            shape=(4, 4), 
                                            dtype=np.int32)
        
        self._action_to_direction = {
            0: 'up',
            1: 'right',
            2: 'down',
            3: 'left'
        }
        self.render_mode = render_mode
    
    def _get_episode_info(self):
        """Helper to safely get logging info."""
        positive_tiles = self.game.board[self.game.board > 0]
        max_tile = int(np.max(positive_tiles)) if positive_tiles.size > 0 else 0
        return {
            "score": self.game.score,
            "max_tile": max_tile
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = StandardGame2048()
        observation = self.game.board
        
        info = self._get_episode_info()
        
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        score_before_move = self.game.score
        
        valid_move = self.game.move(direction)
        observation = self.game.board
        
        # ensure info is always a dict
        info = {}

        # initialize additive reward components
        potential_bonus = 0.0
        cost_of_living_penalty = 0.0
        
        if not valid_move:
            reward = -1 # Punish invalid moves
            merged_tiles = []
        else:
            merged_tiles = getattr(self.game, "last_merged_tiles", [])
            
            # Determine base reward based on reward mode
            if self.reward_mode == "log_merge" or self.reward_mode == "potential_log": # Use log2 of merged tiles as reward if specified in arguments
                # Use log2 of merged tiles as base reward
                base_reward = float(np.sum(np.log2(merged_tiles))) if merged_tiles else 0.0
            else:
                base_reward = self.game.score - score_before_move

            # Apply potential reward and cost of living penalty in "potential_log" mode
            if self.reward_mode == "potential_log":
                # Potential Reward: small bonus for every empty cell on the board after the move
                num_empty_cells = int(np.sum(self.game.board == 0))
                potential_bonus = 0.01 * num_empty_cells
                
                # Cost of Living: small penalty for any valid move that results in zero merges
                if not merged_tiles:
                    cost_of_living_penalty = -0.1

            # Final reward is base reward plus potential bonus minus cost of living penalty
            reward = base_reward + potential_bonus + cost_of_living_penalty

        terminated = self.game.has_won() or self.game.is_game_over()
        truncated = False
        info = {
            "num_empty_cells": int(np.sum(self.game.board == 0)),
            "potential_bonus": float(potential_bonus),
            "cost_of_living_penalty": float(cost_of_living_penalty),
            "merged_tiles": merged_tiles,
            "raw_score_delta": self.game.score - score_before_move,
            "reward_mode": self.reward_mode
        }
        
        if terminated:
            info.update(self._get_episode_info())
        
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(self.game)
