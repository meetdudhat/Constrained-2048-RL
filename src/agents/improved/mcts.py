"""
Monte Carlo Tree Search (MCTS) Engine for 2048.

This module implements a modified version of the AlphaZero search algorithm
adapted for a single-player stochastic domain.

Architectural Deviations from Standard AlphaZero:
1.  Expectimax Tree Structure: Instead of Min/Max nodes (Player vs. Adversary),
    the tree alternates between Player Nodes (Action Decision) and Chance Nodes
    (Environment Stochasticity).
2.  No Value Inversion: In 2-player games, values are inverted (-v) during
    backpropagation. Here, the environment is not an opponent but a probability
    distribution, so values are backpropagated directly to average out outcomes.
3.  Stochastic Expansion: Chance nodes expand based on the game rules
    (90% prob of 2, 10% prob of 4) rather than a policy vector.

Usage:
    mcts = MCTS(network=neural_net)
    action, policy = mcts.search(env, temperature=1.0)
"""


import numpy as np
import math
import random
from src.agents.improved.config import C_PUCT, DIRICHLET_ALPHA, EXPLORATION_FRACTION, NUM_SIMULATIONS, WIN_TILE


class Node:
    """
    Represents a specific state in the Monte Carlo Search Tree.
    
    The tree creates a distinct distinction between decision states (Player)
    and reaction states (Environment/Chance).
    """

    def __init__(self, parent=None, is_player_node=True):
        """
        Args:
            parent (Node): Predecessor node.
            is_player_node (bool): 
                - True: Agent needs to select an Action (Up, Down, Left, Right).
                - False: Environment needs to spawn a tile (2 or 4).
        """
        self.parent = parent
        self.is_player_node = is_player_node
        
        # Maps:
        #   Action (int) -> Node (if is_player_node=True)
        #   Spawn (tuple) -> Node (if is_player_node=False)
        self.children = {}
        
        self.N = 0      # Visit count
        self.W = 0.0    # Total win/loss value (-1 to 1)
        self.P = 0.0    # Prior probability (from Policy Head or Spawn Probability)

    def get_q_value(self):
        """
        Returns the mean action value Q(s, a).
        """
        if self.N == 0:
            return 0.0
        return self.W / self.N
    
    def get_ucb_score(self):
        """
        Calculates the Predictor + Upper Confidence Bound (PUCT) score.
        
        Formula:
            UCB = Q + C_puct * P * (sqrt(Parent_Visits) / (1 + Child_Visits))
        
        This balances exploitation (high Q) with exploration (high P or low visit count).
        """

        q_value = self.get_q_value()
        
        if self.parent is None:
            parent_n_sqrt = 1.0
        else:
            parent_n_sqrt = math.sqrt(self.parent.N)
            
        u_score = (C_PUCT * self.P * (parent_n_sqrt / (1 + self.N)))
        
        return q_value + u_score
    
    def select_child(self):
        """
        Selects the child node with the highest UCB score.
        
        Applicable only to Player Nodes (Selection Phase).
        """
        assert self.is_player_node
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            score = child.get_ucb_score()
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
    
    def select_chance_outcome(self):
        """
        Simulates the stochastic environment by sampling a spawn outcome.
        
        Applicable only to Chance Nodes. Uses the weighted probabilities of
        tile spawning rules rather than UCB maximization.
        
        Returns:
            (tuple, Node): The spawn configuration ((r,c), value) and the next node.
        """
        assert not self.is_player_node

        items = list(self.children.items())
        spawns = [item[0] for item in items]
        
        # P values in chance nodes are explicit probabilities (0.9 or 0.1 / n_cells)
        probs = [node.P for node in self.children.values()]
        
        prob_sum = sum(probs)
        if prob_sum == 0 or len(spawns) == 0:
            # If no children (game over/full board) or unvisited, return None or random
            if len(spawns) == 0:
                return None, None
            
            # Fallback for uninitialized edge cases
            idx = random.randrange(len(spawns))
            return spawns[idx], self.children[spawns[idx]]

        # Normalize probabilities to ensure sum equals 1.0 for np.choice
        normalized_probs = [p / prob_sum for p in probs]
        idx = np.random.choice(len(spawns), p=normalized_probs)
        
        spawn = spawns[idx]
        child_node = self.children[spawn]
        return spawn, child_node
    
    def expand_player_node(self, game_state, policy_priors, valid_moves):
        """
        Expands a Player Node by creating Chance Node children for all legal moves.
        """
        assert self.is_player_node
        for action in valid_moves:
            self.children[action] = Node(parent=self, is_player_node=False)
            self.children[action].P = policy_priors[action]
            
    
    def expand_chance_node(self, game_state_after_move):
        """
        Expands a Chance Node by generating all possible tile spawn outcomes.
        The children of these nodes are Player Nodes (new board states).
        """
        assert not self.is_player_node
        empty_cells = list(zip(*np.where(game_state_after_move.board == 0)))
        
        # If no empty cells, this path is terminal (Game Over)
        if not empty_cells:
            return

        n_empty = len(empty_cells)
        
        # 2048 Rules: 90% chance of 2, 10% chance of 4.
        # Probability is distributed uniformly across all empty cells.
        prob_2 = 0.9 / n_empty
        prob_4 = 0.1 / n_empty
        
        for r, c in empty_cells:
            # Spawn '2' branch
            spawn_2 = ((r, c), 2)
            self.children[spawn_2] = Node(parent=self, is_player_node=True)
            self.children[spawn_2].P = prob_2
            
            # Spawn '4' branch
            spawn_4 = ((r, c), 4)
            self.children[spawn_4] = Node(parent=self, is_player_node=True)
            self.children[spawn_4].P = prob_4
            
    def backpropagate(self, v):
        """
        Propagates the evaluation value `v` up the tree.
        
        Note: No value inversion (-v) is performed because the environment (Chance Node)
        is not an adversarial player trying to minimize our score.
        """
        self.N += 1
        self.W += v
        if self.parent:
            self.parent.backpropagate(v)
            
    def is_leaf(self):
        return len(self.children) == 0
    
    

class MCTS:
    """
    Manages the Monte Carlo Tree Search.
    
    This implementation uses a Neural Network to guide:
    1. Expansion (Policy Head -> Priors P)
    2. Evaluation (Value Head -> Value v)
    """
    def __init__(self, network):
        self.network = network
        self.root = Node(is_player_node=True)
    
    def _run_simulation(self, game_state, env):
        """
        Executes one MCTS iteration: Selection -> Expansion -> Evaluation -> Backpropagation.
        
        Side Effects:
            Mutates `game_state` temporarily during traversal. 
            The caller must provide a COPY of the game state.
        """
        node = self.root
        
        # --- 1. SELECTION ---
        # Traverse down the tree until a leaf is reached.
        while not node.is_leaf():
            if node.is_player_node:
                # Player chooses action (Maximize UCB)
                action, node = node.select_child()
                # Apply move logic without spawning (environment's turn next)
                new_board, _, _ = game_state._move_logic(action)
                game_state.board = new_board
            else:
                # Environment spawns tile (Sample Probability)
                spawn, node = node.select_chance_outcome()
                (r, c), tile = spawn
                game_state.board[r, c] = tile
        
        
        # --- 2. EXPANSION & EVALUATION ---
        v = 0.0
        
        # Verify terminal state
        valid_moves = game_state.get_valid_moves()
        is_terminal = (len(valid_moves) == 0)

        if is_terminal:
            max_tile = game_state.get_max_tile()
            v = 1.0 if max_tile >= WIN_TILE else -1.0
        else:
            # Query the Neural Network
            state_tensor = env.get_state_from_board(game_state.board)
            policy_priors, v_array = self.network.predict(state_tensor)
            v = v_array.item()
            
            # Leaf node reached is a Player Node. Expand it.
            assert node.is_player_node, "Leaf node must be a player node to expand"
            node.expand_player_node(game_state, policy_priors, valid_moves)
            
            # OPTIMIZATION: Pre-expand Chance Nodes immediately.
            # This ensures that when we visit these nodes in the next simulation,
            # the probabilities are already set up for Expectimax averaging.
            original_board_state = game_state.board.copy()
            
            for action, chance_node in node.children.items():
                new_board, _, _ = game_state._move_logic(action)
                game_state.board = new_board
                chance_node.expand_chance_node(game_state)
                # Reset board for next iteration
                game_state.board = original_board_state

        # --- 3. BACKPROPAGATION ---
        node.backpropagate(v)
        
        
    def _add_dirichlet_noise(self, env):
        """
        Injects Dirichlet noise into the root node's priors.
        
        Essential for self-play training to ensure the agent explores 
        different starting moves rather than deterministically following 
        the neural network's initial bias.
        """
        valid_moves = env.game.get_valid_moves()
        if not valid_moves:
            return
            
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(valid_moves))
        
        for i, move in enumerate(valid_moves):
            if move in self.root.children:
                child = self.root.children[move]
                child.P = (1 - EXPLORATION_FRACTION) * child.P + \
                                EXPLORATION_FRACTION * noise[i]
                                
    def get_action_policy(self, temperature=1.0):
        """
        Converts the root node's visit counts (N) into a probability distribution (pi).
        
        Args:
            temperature (float): 
                - 1.0: Proportional to visits (Training/Exploration).
                - 0.0: Argmax (Evaluation/Competitive).
        """
        visit_counts = np.zeros(4, dtype=np.float32)
        for action, node in self.root.children.items():
            visit_counts[action] = node.N
            
        if np.sum(visit_counts) == 0:
            return visit_counts

        if temperature == 0:
            policy = np.zeros(4, dtype=np.float32)
            policy[np.argmax(visit_counts)] = 1.0
        else:
            visit_counts_temp = visit_counts ** (1.0 / temperature)
            policy = visit_counts_temp / np.sum(visit_counts_temp)
            
        return policy
    
    
    def search(self, env, temperature=1.0):
        """
        Main entry point for MCTS.
        
        Returns:
            action (int): The selected move to play.
            policy (np.array): The probability distribution over moves (target for NN training).
        """
        
        # Reset Root for new search
        self.root = Node(is_player_node=True)
        
        # Initial Expansion
        state_tensor = env.get_state()
        policy_priors, _ = self.network.predict(state_tensor)
        
        valid_moves = env.game.get_valid_moves()
        self.root.expand_player_node(env.game, policy_priors, valid_moves)
        
        # Pre-expand immediate chance nodes
        for action, chance_node in self.root.children.items():
            board_after_move, _, _ = env.game._move_logic(action)
            temp_game = env.game.fast_copy()
            temp_game.board = board_after_move
            chance_node.expand_chance_node(temp_game)
        
        # Add exploration noise
        self._add_dirichlet_noise(env)
        
        # Execute Simulations
        for _ in range(NUM_SIMULATIONS):
            # Use a fast copy to treat the game state as immutable during simulation
            game_state_copy = env.game.fast_copy()
            self._run_simulation(game_state_copy, env)
            
        policy = self.get_action_policy(temperature=temperature)
        
        if np.sum(policy) == 0:
            return 0, policy
            
        action = np.random.choice(range(4), p=policy)
        
        return action, policy