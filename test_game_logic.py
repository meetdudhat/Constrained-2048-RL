import unittest
import numpy as np
from game_logic import Game2048

class TestGame2048(unittest.TestCase):

    """Initialization Tests"""
    def test_initial_board_not_empty(self):
        game = Game2048()
        non_zero = np.count_nonzero(game.board)
        self.assertEqual(non_zero, 2, "Board should start with exactly 2 tiles")

    def test_tiles_are_only_2_or_4(self):
        game = Game2048()
        unique_tiles = set(game.board.flatten())
        unique_tiles.discard(0)
        for tile in unique_tiles:
            self.assertIn(tile, [2, 4], "Initial tiles should be 2 or 4")

    def test_score_starts_at_zero(self):
        game = Game2048()
        self.assertEqual(game.score, 0, "Initial score should be zero")

    """Movement Tests (Basic)"""
    def test_move_changes_board(self):
        game = Game2048()
        game.board = np.array([
            [2, 2, 0, 0],
            [4, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        old_board = game.board.copy()

        changed = game.move('left')
        self.assertTrue(changed, "A valid move should return True")
        self.assertFalse(np.array_equal(old_board, game.board),
                         "A valid move should actually change the board state")

    def test_valid_move_no_change(self):
        game = Game2048()
        game.board = np.array([
            [2, 4, 0, 0],
            [4, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        old_board = game.board.copy()

        changed = game.move('left')
        self.assertFalse(changed, "Valid move should return False if no tile moves or merges")
        self.assertTrue(np.array_equal(old_board, game.board),
                        "Board should remain the same when no movement is possible")

    def test_invalid_move_does_not_change_board(self):
        game = Game2048()
        old_board = game.board.copy()

        changed = game.move('null')  
        self.assertFalse(changed, "An invalid move should return False")
        self.assertTrue(np.array_equal(old_board, game.board),
                        "An invalid move should not change the board")

    """Movement Tests (All Directions)"""
    def test_move_right_merges_correctly(self):
        game = Game2048()
        game.board = np.array([
            [0, 0, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        game.move('right')
        self.assertEqual(game.board[0, 3], 4, "Tiles should merge when moving right")

    def test_move_up_merges_correctly(self):
        game = Game2048()
        game.board = np.array([
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        game.move('up')
        self.assertEqual(game.board[0, 0], 4, "Tiles should merge when moving up")

    def test_move_down_merges_correctly(self):
        game = Game2048()
        game.board = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0]
        ])
        game.move('down')
        self.assertEqual(game.board[3, 0], 4, "Tiles should merge when moving down")

    """Merging Behavior Tests"""
    def test_double_merge_in_one_row(self):
        game = Game2048()
        game.board = np.array([
            [2, 2, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        game.move('left')
        self.assertEqual(game.board[0, 0], 4,
                        "Tiles should merge once per move, not chain merge immediately")        
        self.assertEqual(game.board[0, 1], 4,
                        "Tiles should merge once per move, not chain merge immediately")
        
    def test_tiles_merge_once_per_move(self):
        game = Game2048()
        game.board = np.array([
            [4, 4, 8, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        game.move('left')
        self.assertEqual(game.board[0, 0], 8,
                        "Tiles should merge once per move, not chain merge immediately")
        self.assertEqual(game.board[0, 1], 8,
                        "Tiles should merge once per move, not chain merge immediately")

    """Scoring Tests"""
    def test_score_increases_after_merge(self):
        game = Game2048()
        game.board = np.array([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        game.move('left')
        self.assertEqual(game.score, 4, "Merging 2 and 2 should add 4 to score")

    def test_score_adds_multiple_merges(self):
        game = Game2048()
        game.board = np.array([
            [2, 2, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        game.move('left')
        self.assertEqual(game.score, 8, "Two merges should add 8 to score total")

    """Tile Spawning Tests"""
    def test_new_tile_spawn_after_valid_move(self):
        game = Game2048()
        game.board = np.array([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        non_zero_before = np.count_nonzero(game.board)
        game.move('left')
        non_zero_after = np.count_nonzero(game.board)
        self.assertGreaterEqual(non_zero_after, non_zero_before, "A new tile should spawn after valid move")

    def test_no_new_tile_after_invalid_move(self):
        game = Game2048()
        game.board = np.array([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        non_zero_before = np.count_nonzero(game.board)
        game.move('null')
        non_zero_after = np.count_nonzero(game.board)
        self.assertEqual(non_zero_after, non_zero_before, "No tile should spawn after invalid move")

    """Win / Game Over Tests"""
    def test_winning_condition(self):
        game = Game2048()
        game.board[0, 0] = 2048  
        self.assertTrue(game.has_won(), "Should detect win when 2048 tile is on board")

    def test_game_over_condition(self):
        game = Game2048()
        game.board = np.array([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2]
        ])
        self.assertTrue(game.is_game_over(), "Should detect game over when no moves left")

    def test_game_not_over_if_merge_possible(self):
        game = Game2048()
        game.board = np.array([
            [2, 2, 4, 8],
            [16, 32, 64, 128],
            [256, 512, 1024, 2],
            [4, 8, 16, 32]
        ])
        self.assertFalse(game.is_game_over(), "Should not detect game over if merge is possible")

if __name__ == "__main__":
    unittest.main()
