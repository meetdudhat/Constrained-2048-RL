import gymnasium
from gymnasium import spaces
import numpy as np
from src.game.standard_game import StandardGame2048

class Standard2048Env(gymnasium.Env):
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.game = StandardGame2048()
        self.action_space = spaces.Discrete(4)
        
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
        
        if not valid_move:
            reward = -1 # Punish invalid moves
        else:
            reward = self.game.score - score_before_move

        terminated = self.game.has_won() or self.game.is_game_over()
        truncated = False
        info = {}
        
        if terminated:
            info = self._get_episode_info()
        
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(self.game)
