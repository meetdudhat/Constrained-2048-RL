import gymnasium
from gymnasium import spaces
import numpy as np
from src.game.constrained_game import ConstrainedGame2048

class Constrained2048Env(gymnasium.Env):
    """
    A custom Gymnasium environment for our Contrained 2048 game.
    This class wraps the game logic.
    """
    
    def __init__(self, render_mode=None, reward_mode="raw_score"):
        """
        Initializes the environment, defining the state and action spaces.
        """
        
        super().__init__()
        
        self.game = ConstrainedGame2048()
        self.reward_mode = reward_mode

        # Defines the action space: 4 discrete actions (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Maps the actions to the game's string directions
        self._aciton_to_direction = {
            0: 'up',
            1: 'right',
            2: 'down',
            3: 'left'
        }
        
        # Defines state/observation space:
            # 4x4 grid, -1 for immovable block, infinity for 2048 or higher
        self.observation_space = spaces.Box(low=-1, high=np.inf, shape=(4, 4), dtype=int)
        
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
        """
        Resets the environment for a new game/episode.
        """
        
        super().reset(seed=seed)
        
        # Start a new game
        self.game = ConstrainedGame2048()
        
        observation = self.game.board
        
        info = self._get_episode_info()
        
        if self.render_mode == "human":
            self.render()
        
        
        return observation, info
    
    def step(self, action):
        """
        Performs on time-step in the enviroment.
        """
        
        # Gets game direction from action
        direction = self._aciton_to_direction[action]
        
        # stores the score before the move to calculate the reward
        score_before_move = self.game.score
        
        # performs the action
        valid_move = self.game.move(direction)
        
        # gets the new state
        observation = self.game.board

        # ensure info is always a dict
        info = {}

        # initialize additive reward components
        potential_bonus = 0.0
        cost_of_living_penalty = 0.0

        # Added penalty for invalid moves to help the agent learn
        if not valid_move:
            reward = -1  # Punish invalid moves
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
        
        # terminated is True if the game is won or over
        terminated = self.game.has_won() or self.game.is_game_over()
        
        # truncated is False (no time limit)
        truncated = False
        
        # populate info with reward components
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
        """
        Renders the environment.
        """
        if self.render_mode == "human":
            print(self.game)