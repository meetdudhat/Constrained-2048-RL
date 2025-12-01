"""
Core Game Logic for 2048 (Optimized for MCTS).

This module implements the rules of 2048 with critical performance optimizations
designed to support the high requirements of Monte Carlo Tree Search.

Key Optimizations:
1.  Numba JIT Compilation: The heavy lifting of sliding and merging arrays
    is compiled to machine code using `@njit`, bypassing Python's interpreter overhead.
2.  Logic Separation: The `_move_logic` method performs the calculation *without*
    mutating the state or spawning tiles. This allows MCTS to "imagine" moves
    rapidly without polluting the actual game board.
3.  Fast Copying: `fast_copy()` bypasses the standard object initialization
    steps to create lightweight clones for tree expansion.
"""

import numpy as np
import random
from numba import njit

from src.agents.improved.config import WIN_TILE

@njit(fastmath=True)
def _merge_part(section):
    """
    Core Logic: Compresses and merges a 1D array of tiles.
    
    This function simulates gravity pulling tiles to the left (index 0).
    It handles the 2048 specific merge rule: 
    - [2, 2, 4, 8] -> [4, 4, 8, 0] (Merge once per move)
    - [2, 2, 2, 2] -> [4, 4, 0, 0] (Double merge)
    
    Args:
        section (np.array): A 1D row or column from the board.
        
    Returns:
        (np.array, int): The new line configuration and the score gained.
    """
    length = section.shape[0]
    temp = np.zeros(length, dtype=np.int32)
    
    # 1. Compress Phase: Remove zeros/gaps
    count = 0
    for i in range(length):
        if section[i] != 0:
            temp[count] = section[i]
            count += 1
    
    # 2. Merge Phase: Combine identical neighbors
    result = np.zeros(length, dtype=np.int32)
    score = 0
    write_idx = 0
    read_idx = 0
    
    while read_idx < count:
        current_val = temp[read_idx]
        
        # Check adjacent tile
        if read_idx + 1 < count:
            next_val = temp[read_idx + 1]
            if current_val == next_val:
                # Merge
                merged_val = current_val * 2
                result[write_idx] = merged_val
                score += merged_val
                read_idx += 2 # Skip the next one since we merged it
            else:
                # No merge
                result[write_idx] = current_val
                read_idx += 1
        else:
            # Last element
            result[write_idx] = current_val
            read_idx += 1
            
        write_idx += 1
        
    return result, score

@njit(fastmath=True)
def fast_slide_row(row):
    """
    Handles row sliding logic, respecting Immovable Obstacles (-1).
    
    If the environment contains a blocked cell (Value -1), the row is split
    into two independent segments around the block. The block itself never moves
    and never merges.
    """
    size = row.shape[0]
    block_idx = -1
    
    # Locate obstacle
    for i in range(size):
        if row[i] == -1:
            block_idx = i
            break
            
    # Standard Case (No Block)
    if block_idx == -1:
        return _merge_part(row)
    
    # Constrained Case (Block Exists)
    else:
        # Split row into [Before Block] and [After Block] segments
        # Strict copy ensures we don't modify the input view
        part_before = row[:block_idx].copy()
        part_after = row[block_idx+1:].copy()
        
        # Process segments independently
        new_before, score_before = _merge_part(part_before)
        new_after, score_after = _merge_part(part_after)
        
        # Reassemble: [Processed Before] + [-1] + [Processed After]
        final_row = np.zeros(size, dtype=np.int32)
        
        for i in range(len(new_before)):
            final_row[i] = new_before[i]
            
        final_row[block_idx] = -1
        
        start_after = block_idx + 1
        for i in range(len(new_after)):
            final_row[start_after + i] = new_after[i]
            
        return final_row, (score_before + score_after)

class Game2048:
    def __init__(self, size=4, immovable_cell=None):
        """
        The Game Engine. Manages state, moves, and win/loss conditions.
        """
        self.size = size
        self.immovable_cell = immovable_cell
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.reset()
        
        if self.immovable_cell:
            r, c = self.immovable_cell
            self.board[r, c] = -1
            
        self._add_new_tile()
        self._add_new_tile()
    
    def reset(self):
        """
        Resets the game state in-place.
        
        Optimization:
            Reuses the existing `self.board` memory allocation rather than
            creating a new numpy array object, reducing GC pressure during training.
        """
        self.board.fill(0)
        self.score = 0
        
        if self.immovable_cell:
            r, c = self.immovable_cell
            self.board[r, c] = -1
            
        self._add_new_tile()
        self._add_new_tile()
    
    def _add_new_tile(self):
        """
        Spawns a new tile in a random empty location.
        Probabilities: 90% chance of '2', 10% chance of '4'.
        """
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.board[row, col] = np.random.choice([2, 4], p=[0.9, 0.1])
    
    def _move_logic(self, direction):
        """
        Calculates the result of a move *without* spawning new tiles.
        
        This method is the "Brain" of the MCTS simulation. It allows the agent
        to preview the consequences of an action (board state change, score gain)
        deterministically before the environment introduces randomness.
        
        Args:
            direction (int): 0:Up, 1:Down, 2:Left, 3:Right
            
        Returns:
            (np.ndarray, int, bool): The new board, score gained, and whether the board changed.
        """
        # Strategy:
        # Instead of writing 4 separate merge functions (Up, Down, Left, Right),
        # we rotate the board so the desired direction always points "Left",
        # apply the generic "Merge Left" function, and then rotate back.
        
        # Rotations needed to orient 'direction' to the Left (Vector 2)
        # Up(0) -> Left requires 1 rotation (CCW)
        rotations_needed = {0: 1, 1: 3, 2: 0, 3: 2} 
        k = rotations_needed[direction]
        
        original_board = self.board # No copy needed, we're not saving
        rotated_board = np.rot90(original_board, k=k)
        
        move_score = 0
        temp_board = np.zeros_like(rotated_board)
        
        # Apply compiled JIT logic to each row
        for r in range(self.size):
            new_row, row_score = fast_slide_row(rotated_board[r])
            temp_board[r] = new_row
            move_score += row_score

        # Restore original orientation
        final_board = np.rot90(temp_board, k=-k)
        
        board_changed = not np.array_equal(original_board, final_board)
        
        return final_board, move_score, board_changed
    
    def move(self, direction):
        """
        Executes a move in the actual game environment.
        
        Side Effects:
            1. Updates self.board
            2. Updates self.score
            3. Spawns a new tile (if the board changed)
        """
        final_board, move_score, board_changed = self._move_logic(direction)
        
        if board_changed:
            self.board = final_board
            self.score += move_score
            self._add_new_tile()
        
        return board_changed
    
    def fast_copy(self):
        """
        Creates a lightweight clone of the game instance.
        
        Optimization:
            Bypasses `__init__` to avoid unnecessary setup overhead.
            Used heavily by MCTS to create tree nodes.
        """
        # We bypass __init__ to avoid adding new tiles
        new_game = self.__class__.__new__(self.__class__)
        new_game.size = self.size
        new_game.immovable_cell = self.immovable_cell
        new_game.score = self.score
        new_game.board = np.copy(self.board) # The only deep copy needed
        return new_game
    
    def is_move_possible(self, direction):
        """Checks if a specific move is legal."""
        # Simulate move on a temporary copy
        temp_game = self.fast_copy()
        # Note: calling .move() on the copy is safe
        move_made = temp_game.move(direction)
        return move_made
    
    def get_valid_moves(self):
        """Returns a list of all legal move indices [0, 1, 2, 3]."""
        moves = []
        for direction in range(4):
            if self.is_move_possible(direction):
                moves.append(direction)
        return moves
    
    def is_game_over(self):
        """True if no moves are possible."""
        return not self.get_valid_moves()
    
    def get_max_tile(self):
        return np.max(self.board)
    
    def has_won(self):
        return np.any(self.board == WIN_TILE)
    
    def __str__(self):
        """String representation for printing the board."""
        score_str = "Score: {}\n".format(self.score)
        board_str = str(self.board)
        return score_str + board_str
    

if __name__ == "__main__":
    # --- Interactive Terminal Mode ---
    game = Game2048()
    print("Welcome to 2048!")
    print("Use W (up), A (left), S (down), D (right) to play. Type 'q' to quit.")
    
    move_map = {
        'w': 0, 
        's': 1, 
        'a': 2, 
        'd': 3
    }
    
    while True:
        print(game)
        
        if game.has_won():
            print("Congratulations! You've reached 2048! You win! 🎉")
            break
        
        if game.is_game_over():
            print("Game Over!")
            break
            
        move_input = input("Enter your move (w/a/s/d): ").lower()
        
        if move_input == 'q':
            break
        
        if move_input in move_map:
            direction = move_map[move_input]
            if not game.move(direction):
                print("Invalid move. Try another direction.")
        else:
            print("Invalid input. Please use w, a, s, or d.")