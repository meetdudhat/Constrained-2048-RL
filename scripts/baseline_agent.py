import random
import time
from src.game.constrained_game import Game2048

'''
A baseline agent that interacts with our 2048 game by choosing random moves.
'''
class RandomBaselineAgent:

    def __init__(self):
        self.actions = ['up', 'down', 'left', 'right']

    def choose_action(self, game):
        '''
        Picks random move to change the board.
        '''
        valid_actions = []

        for action in self.actions:

            temp_game = Game2048()

            temp_game.board = game.board.copy()

            if temp_game.move(action):
                valid_actions.append(action) # if board moved based on the choosen action, append to valid actions

        return random.choice(valid_actions) if valid_actions else None # return random valid action


'''
Plays a full around of game using the given agent,
until either are no valid moves or we have won the game.
'''
def play_game(agent, print_board=True, delay=0.2):

    game = Game2048()

    while not game.is_game_over():
        if print_board: # if true, print board every turn 
            print(game)
            print("-" * 20)
            time.sleep(delay)
        
        action = agent.choose_action(game)

        if action is None:
            break # that means no valid move

        game.move(action) # based on the random action, move the board

        if game.has_won():
            print("Congrats, Agent reached 2048 block!")
            break

    print("Final Board:")
    print(game)
    print(f"Final Score: {game.score}")
    print("=" * 30)
    return game.score

if __name__ == "__main__":
    agent = RandomBaselineAgent()
    total_score = 0
    num_games = 1


    for i in range (num_games):
        print(f"Starting Game - {i+1}")
        score = play_game(agent)
        total_score += score

    print(f"\nAverage Score after {num_games} games: {total_score / num_games:.2f}")






