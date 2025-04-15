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
        time_limit = 2.0
        start_time = time.time()
        best_move = None
        depth = 1
        maximizing = (state.turn == self.move_num)
        while True:
            if time.time() - start_time > time_limit:
                break
            score, move = self.alphabeta_with_time(state, depth, float('-inf'), float('inf'), maximizing, start_time, time_limit)
            if move is not None:
                best_move = move
            depth += 1
        return best_move

    def alphabeta_with_time(self, state, depth, alpha, beta, maximizing, start_time, time_limit):
        if time.time() - start_time > time_limit:
            return self.heuristic(state), None
        valid_moves = state.get_valid_moves()
        if depth == 0 or not valid_moves:
            return self.heuristic(state), None
        best_move = None
        if maximizing:
            value = float('-inf')
            for move in valid_moves:
                new_state = copy.deepcopy(state)
                self.simulate_move(new_state, move)
                score, _ = self.alphabeta_with_time(new_state, depth - 1, alpha, beta, False, start_time, time_limit)
                if score > value:
                    value = score
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
                if time.time() - start_time > time_limit:
                    break
            return value, best_move
        else:
            value = float('inf')
            for move in valid_moves:
                new_state = copy.deepcopy(state)
                self.simulate_move(new_state, move)
                score, _ = self.alphabeta_with_time(new_state, depth - 1, alpha, beta, True, start_time, time_limit)
                if score < value:
                    value = score
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    break
                if time.time() - start_time > time_limit:
                    break
            return value, best_move

    def simulate_move(self, state, move):
        row, col = move
        change_colors(row, col, state.turn - 1, state)  # flip stones correctly
        state.board[row][col] = state.turn
        state.turn = 3 - state.turn

    def heuristic(self, state):
        board = state.board
        my_num = self.move_num
        opp_num = 3 - self.move_num
        my_coins = np.sum(board == my_num)
        opp_coins = np.sum(board == opp_num)
        parity = 100 * (my_coins - opp_coins) / (my_coins + opp_coins + 1)

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

        return parity + corner_score + adj_score + mobility + edge_score
