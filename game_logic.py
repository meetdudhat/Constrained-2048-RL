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
    
    def _process_line(self, line):
        """
        Processes a single line (row or column) for a move.
        It handles both compressing and merging tiles.
        
        Example: [2, 0, 2, 4] -> [4, 4, 0, 0]
        """
        # 1. Compression: Moves all non-zero tiles to the left
        non_zero_tiles = line[line != 0]
        
        new_line = np.zeros(4, dtype=int)
        line_score = 0
        
        # 2. Merging
        i = 0
        j = 0
        while i < len(non_zero_tiles):
            if i + 1 < len(non_zero_tiles) and non_zero_tiles[i] == non_zero_tiles[i+1]:
                # Merge tiles
                merged_value = non_zero_tiles[i] * 2
                new_line[j] = merged_value
                line_score += merged_value
                # Skip the next tile because it was merged
                i += 2
            else:
                # Move tile
                new_line[j] = non_zero_tiles[i]
                i += 1
            j += 1
            
        return new_line, line_score
    
    def move(self, direction):
        """
        Performs a move in the given direction ('left', 'right', 'up', 'down').
        Returns True if the board changed, False otherwise.
        """
        original_board = self.board.copy()
        
        # Rotates the board so we can always process it as a left move
        if direction == 'left':
            # 0 rotations
            rotated_board = self.board
        elif direction == 'right':
            # 2 rotations counter-clockwise (180 degrees)
            rotated_board = np.rot90(self.board, 2)
        elif direction == 'up':
            # 1 rotation counter-clockwise
            rotated_board = np.rot90(self.board, 1)
        elif direction == 'down':
            # 1 rotation clockwise
            rotated_board = np.rot90(self.board, -1)
        else:
            # Invalid direction
            return False

        new_board = np.zeros((4, 4), dtype=int)
        move_score = 0
        
        for i in range(4):
            new_line, line_score = self._process_line(rotated_board[i])
            new_board[i] = new_line
            move_score += line_score
        
        self.score += move_score
        
        # Rotates the board back to its original orientation
        if direction == 'left':
            self.board = new_board
        elif direction == 'right':
            self.board = np.rot90(new_board, -2)
        elif direction == 'up':
            self.board = np.rot90(new_board, -1)
        elif direction == 'down':
            self.board = np.rot90(new_board, 1)

        # Checks if the move changed anything
        if not np.array_equal(original_board, self.board):
            self._add_new_tile()
            return True
        return False
    
    def is_game_over(self):
        """Checks if the game is over (no more possible moves)."""
        if self._get_empty_cells():
            # Can still add tiles
            return False

        # Checks for possible horizontal merges
        for r in range(4):
            for c in range(3):
                if self.board[r, c] == self.board[r, c+1]:
                    return False

        # Checks for possible vertical merges
        for c in range(4):
            for r in range(3):
                if self.board[r, c] == self.board[r+1, c]:
                    return False
        
        return True
    
    def __str__(self):
        """String representation for printing the board."""
        score_str = "Score: {}\n".format(self.score)
        board_str = str(self.board)
        return score_str + board_str
    

# Main block to play the game in the terminal
if __name__ == "__main__":
    game = Game2048()
    print("Welcome to 2048!")
    print("Use W (up), A (left), S (down), D (right) to play. Type 'q' to quit.")
    
    while True:
        print(game)
        
        if game.is_game_over():
            print("Game Over!")
            break
            
        move_input = input("Enter your move (w/a/s/d): ").lower()
        
        if move_input == 'q':
            break
            
        move_map = {'w': 'up', 'a': 'left', 's': 'down', 'd': 'right'}
        
        if move_input in move_map:
            direction = move_map[move_input]
            if not game.move(direction):
                print("Invalid move. Try another direction.")
        else:
            print("Invalid input. Please use w, a, s, or d.")