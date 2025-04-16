'''
AI Disclaimer:
This agent was created in close collaboration with OpenAI's GPT-4.1 preview
model. Some of the code was generated based on my work in reversi_bot.py, and
other parts were generated as a result of my prompting. The overall structure
of the algorithm is of my own making, and the heuristic function is not
AI-designed. Credit for much of the heuristic function is due to Kartik
Kukreja and his work published at the following link:
https://kartikkukreja.wordpress.com/2013/03/30/heuristic-function-for-reversiothello/
'''

import numpy as np
import random as rand
import reversi
import copy
import time
from reversi_moves import change_colors

class ReversiBot:
    def __init__(self, move_num):
        self.move_num = move_num

    def make_move(self, state):
        '''
        This is the only function that needs to be implemented for the lab!
        The bot should take a game state and return a move.

        The parameter "state" is of type ReversiGameState and has two useful
        member variables. The first is "board", which is an 8x8 numpy array
        of 0s, 1s, and 2s. If a spot has a 0 that means it is unoccupied. If
        there is a 1 that means the spot has one of player 1's stones. If
        there is a 2 on the spot that means that spot has one of player 2's
        stones. The other useful member variable is "turn", which is 1 if it's
        player 1's turn and 2 if it's player 2's turn.

        ReversiGameState objects have a nice method called get_valid_moves.
        When you invoke it on a ReversiGameState object a list of valid
        moves for that state is returned in the form of a list of tuples.

        Move should be a tuple (row, col) of the move you want the bot to make.
        '''
        min_depth = 5
        max_depth = 10
        alpha = float('-inf')
        beta = float('inf')
        maximizing = (state.turn == self.move_num)
        # best_score, best_move = self.alphabeta(state, depth, alpha, beta, maximizing)

        start_time = time.time()
        time_limit = 2.0  # seconds
        best_move = None
        for d in range(min_depth, max_depth + 1):
            if time.time() - start_time > time_limit:
                print(f"depth: {d - 1}")
                break
            best_score_at_depth, best_move_at_depth = self.alphabeta(state, d, alpha, beta, maximizing)
            if best_move_at_depth:
                best_move = best_move_at_depth

        return best_move

    def alphabeta(self, state, depth, alpha, beta, maximizing):
        valid_moves = state.get_valid_moves()
        if depth == 0 or not valid_moves:
            return self.heuristic(state), None

        best_move = None
        if maximizing:
            value = float('-inf')
            for move in valid_moves:
                new_state = copy.deepcopy(state)
                self.simulate_move(new_state, move)
                score, _ = self.alphabeta(new_state, depth - 1, alpha, beta, False)
                if score > value:
                    value = score
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, best_move
        else:
            value = float('inf')
            for move in valid_moves:
                new_state = copy.deepcopy(state)
                self.simulate_move(new_state, move)
                score, _ = self.alphabeta(new_state, depth - 1, alpha, beta, True)
                if score < value:
                    value = score
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value, best_move

    def simulate_move(self, state, move):
        row, col = move
        change_colors(row, col, state.turn - 1, state)
        state.board[row][col] = state.turn
        state.turn = 3 - state.turn

    def heuristic(self, state):
        board = state.board
        my_num = self.move_num
        opp_num = 3 - self.move_num

        my_coins = np.sum(board == my_num)
        opp_coins = np.sum(board == opp_num)
        parity_weight = 50 + ((my_coins + opp_coins) / state.board_dim ** 2) * 100
        parity = (my_coins - opp_coins) / (my_coins + opp_coins + 1)

        # Corners
        corners = [(0,0), (0,7), (7,0), (7,7)]
        my_corners = sum([1 for r,c in corners if board[r][c] == my_num])
        opp_corners = sum([1 for r,c in corners if board[r][c] == opp_num])
        corner_score = 25 * (my_corners - opp_corners)

        # Corner adjacency penalty
        adjacents = [(0,1),(1,0),(1,1),(0,6),(1,6),(1,7),(6,0),(6,1),(7,1),(6,6),(6,7),(7,6)]
        my_adj = sum([1 for r,c in adjacents if board[r][c] == my_num])
        opp_adj = sum([1 for r,c in adjacents if board[r][c] == opp_num])
        adj_score = -12.5 * (my_adj - opp_adj)

        # Mobility
        orig_turn = state.turn
        state.turn = my_num
        my_moves = len(state.get_valid_moves())
        state.turn = opp_num
        opp_moves = len(state.get_valid_moves())
        state.turn = orig_turn
        if my_moves + opp_moves > 0:
            mobility = 100 * (my_moves - opp_moves) / (my_moves + opp_moves)
        else:
            mobility = 0

        # Edge bonus
        my_edges = 0
        opp_edges = 0
        for i in range(8):
            for j in [0,7]:
                if board[i][j] == my_num:
                    my_edges += 1
                elif board[i][j] == opp_num:
                    opp_edges += 1
                if board[j][i] == my_num:
                    my_edges += 1
                elif board[j][i] == opp_num:
                    opp_edges += 1
        edge_score = 5 * (my_edges - opp_edges)

        return parity_weight * parity + corner_score + adj_score + mobility + edge_score
