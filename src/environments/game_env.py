import numpy as np
from src.game.game_2048 import Game2048

from src.agents.improved.config import WIN_TILE

class Game2048Env:
    """
    Reinforcement Learning Wrapper for the 2048 Game Engine.

    This class serves as the interface between the raw game logic and the AlphaZero agent.
    Its primary responsibilities are:
    1.  State Encoding: Converting the 4x4 integer grid into an 18-channel One-Hot tensor.
    2.  Reward Shaping: providing a sparse, binary reward signal (+1 Win, -1 Loss) rather
        than the traditional score-based reward.
    3.  Canonicalization: Exploiting the game's rotational symmetry to reduce the effective
        state space the network must learn.
    """
    def __init__(self, size=4, immovable_cell=None):
        self.size = size
        self.immovable_cell = immovable_cell
        self.game = Game2048(self.size, self.immovable_cell)
        
        # --- One-Hot Encoding Setup ---
        # The neural network cannot natively understand that '4' is twice as good as '2'
        # if we just feed it the numbers. Instead, we treat each tile value as a 
        # distinct "class" or channel.
        # Channel 0: Empty
        # Channel 1..16: Powers of 2 (2^1 to 2^16)
        # Channel 17: Immovable Obstacle
        self.tile_map = {0: 0}
        for i in range(1, 17): # 2^1 to 2^16
            self.tile_map[2**i] = i
        
        self.tile_map[-1] = 17 # Use channel 17 for the block
        self.num_channels = 18 # 0, 2^1...2^16, block
        self.render_mode = "human"
    
    def reset(self):
        """Resets the game and returns the initial tensor state."""
        self.game.reset()
        return self.get_state()

    def get_state(self):
        """
        Converts the current board into a One-Hot Encoded tensor.
        
        Returns:
            np.ndarray: Shape (18, 4, 4).
        """
        state = np.zeros((self.num_channels, self.size, self.size), dtype=np.float32)
        for r in range(self.size):
            for c in range(self.size):
                tile_val = self.game.board[r, c]
                channel = self.tile_map.get(tile_val, 0) # Default to 0 (Empty) if unknown
                state[channel, r, c] = 1
        return state
    
    def get_state_from_board(self, board):
        """
        Static helper to encode a hypothetical board state during MCTS simulation.
        """
        state = np.zeros((self.num_channels, self.size, self.size), dtype=np.float32)
        for r in range(self.size):
            for c in range(self.size):
                tile_val = board[r, c]
                channel = self.tile_map.get(tile_val, 0) # Default to 0
                state[channel, r, c] = 1
        return state
    
    def step(self, action):
        """
        Executes an action and returns the transition tuple.
        
        Reward Strategy:
            AlphaZero relies on predicting the final outcome of the game (Value Head).
            Therefore, we do not reward points for merging tiles (like the baseline).
            We only reward:
            - +1.0 for reaching the WIN_TILE.
            - -1.0 for Game Over without winning.
            -  0.0 for all intermediate steps.
        """
        if not self.game.is_move_possible(action):
            # Penalize invalid moves heavily if they result in a terminal state check failure,
            # though MCTS usually masks these out beforehand.
            state = self.get_state()
            done = self.game.is_game_over()
            reward = -1.0 if done else 0.0
            return state, reward, done

        self.game.move(action)

        done = self.game.is_game_over()
        max_tile = self.game.get_max_tile()

        # Sparse Binary Reward
        reward = 0.0
        if done:
            if max_tile >= WIN_TILE:
                reward = 1.0  # Win
            else:
                reward = -1.0 # Loss
        
        state = self.get_state()
        return state, reward, done

    def get_valid_moves_mask(self):
        """
        Returns a boolean mask [Up, Down, Left, Right] indicating legal moves.
        Used by MCTS to zero out probabilities for illegal actions.
        """
        return [self.game.is_move_possible(i) for i in range(4)]
    
    def get_canonical_form(self, state_tensor, policy_vector):
        """
        Reduces the state space by canonicalizing the board.
        
        Rationale:
            In 2048, the logic is rotationally invariant. A board rotated 90 degrees
            is strategically identical if the policy vector is also rotated.
            
            This function iterates through all 8 symmetries (4 rotations + 4 flips)
            and selects the 'smallest' byte representation as the canonical form.
            This ensures the neural network always sees a consistent orientation,
            effectively multiplying the training data by 8x.

        Args:
            state_tensor: (C, H, W) board representation.
            policy_vector: (4,) action probabilities [Up, Down, Left, Right].

        Returns:
            (state, policy): The canonical pair.
        """
        # Mapping of action indices for each transformation
        # 0:Up, 1:Down, 2:Left, 3:Right
        policy_transforms = [
            [0, 1, 2, 3], # Original
            [3, 2, 0, 1], # Rot90 (Up<-Right, Down<-Left, Left<-Up, Right<-Down)
            [1, 0, 3, 2], # Rot180
            [2, 3, 1, 0], # Rot270
            [0, 1, 3, 2], # Flip (Left/Right swap)
            [3, 2, 1, 0], # Flip + R90
            [1, 0, 2, 3], # Flip + R180
            [2, 3, 0, 1]  # Flip + R270
        ]
        
        canonical_state = None
        canonical_policy = None
        
        # We use the byte representation of the array to determine "order"
        current_board_repr = state_tensor.tobytes()

        for i in range(8):
            transformed_state = state_tensor
            
            # Apply geometric transformations
            if i >= 4:
                transformed_state = np.flip(transformed_state, axis=2) # Flip Width
            
            rot_k = i % 4
            transformed_state = np.rot90(transformed_state, k=rot_k, axes=(1, 2))
            
            # Apply corresponding permutation to the policy vector
            transformed_policy = policy_vector[policy_transforms[i]]
            
            new_board_repr = transformed_state.tobytes()

            # Selection: Pick the lexicographically smallest representation
            if canonical_state is None or new_board_repr < current_board_repr:
                current_board_repr = new_board_repr
                # Ensure contiguous memory layout for PyTorch efficiency
                canonical_state = np.ascontiguousarray(transformed_state)
                canonical_policy = np.ascontiguousarray(transformed_policy)

        return canonical_state, canonical_policy
    
    def get_game_state_for_mcts(self):
        """Returns a fast copy of the internal game state for simulation."""
        return self.game.fast_copy()
    
    def render(self):
        if self.render_mode == "human":
            print(self.game)
