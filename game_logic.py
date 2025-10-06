import numpy as np
import random

class Game2048:
    def __init__(self):
        """
        Initializes the game board and score.
        """
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        # Starts the game with two random tiles
        self._add_new_tile()
        self._add_new_tile()

    def _get_empty_cells(self):
        """Finds all empty cells (where value is 0)."""
        rows, cols = np.where(self.board == 0)
        return list(zip(rows, cols))

    def _add_new_tile(self):
        """
        Adds a new tile (90% chance of 2, 10% chance of 4)
        to a random empty cell.
        """
        empty_cells = self._get_empty_cells()
        if empty_cells:
            row, col = random.choice(empty_cells)
            # Uses weighted choice for 2 or 4
            self.board[row, col] = np.random.choice([2, 4], p=[0.9, 0.1])
        
    def __str__(self):
        """String representation for printing the board."""
        score_str = "Score: {}\n".format(self.score)
        board_str = str(self.board)
        return score_str + board_str