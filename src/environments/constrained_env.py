import gymnasium
from gymnasium import spaces
import numpy as np
from src.game.constrained_game import ConstrainedGame2048

class Constrained2048Env(gymnasium.Env):
    """
    A custom Gymnasium environment for our Contrained 2048 game.
    This class wraps the game logic.
    """
    
    def __init__(self, render_mode=None):
        """
        Initializes the environment, defining the state and action spaces.
        """
        
        super().__init__()
        
        self.game = ConstrainedGame2048()
        
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
        
        # Added penalty for invalid moves to help the agent learn
        if not valid_move:
            reward = -1  # Punish invalid moves
        else:
            # calulates the reward as thes score difference
            reward = self.game.score - score_before_move
        
        # terminated is True if the game is won or over
        terminated = self.game.has_won() or self.game.is_game_over()
        
        # truncated is False (no time limit)
        truncated = False
        
        if terminated:
            info = self._get_episode_info()
        
        if self.render_mode == "human":
            self.render()
            
        return observation, reward, terminated, truncated, info
    

    def render(self):
        """
        Renders the environment.
        """
        if self.render_mode == "human":
            print(self.game)