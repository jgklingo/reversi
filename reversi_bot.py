import numpy as np
import random as rand
import copy
from reversi import *
from reversi_moves import *


class GameNode:
    def __init__(self, state: ReversiGameState, move: tuple[int, int]=None):
        self.state = state
        self.move = move
        self.children: list[GameNode] = []
        self.score = None
    
    @staticmethod
    def generate_children(node: 'GameNode'):
        valid_moves = node.state.get_valid_moves()
        for move in valid_moves:
            new_state = copy.deepcopy(node.state)
            change_colors(move[0], move[1], new_state.turn, new_state)
            new_state.turn = 3 - new_state.turn
            node.children.append(GameNode(new_state, move))


class ReversiBot:
    DEPTH = 10

    def __init__(self, move_num):
        self.move_num = move_num

    def make_move(self, state: ReversiGameState):
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
        # valid_moves = state.get_valid_moves()
        # move = rand.choice(valid_moves) # Moves randomly...for now

        # print(self.heuristic(state))
        root = GameNode(state)
        maximizing = state.turn == self.move_num
        best_node = None
        start_time = time.time()
        time_limit = 2.0  # seconds, for example
        
        for d in range(1, self.DEPTH + 1):
            if time.time() - start_time > time_limit:
                # print(f"depth: {d - 1}")
                break
            best_node_at_depth = self.alphabeta(root, d, float('-inf'), float('inf'), maximizing)
            if best_node_at_depth:
                best_node = best_node_at_depth
        
        print(f"Current heuristic: {self.heuristic(state)}")
        print(f"New heuristic after move: {best_node.score}")
        print("Score change:", best_node.score - self.heuristic(state))
        return best_node.move
    
    def alphabeta(self, node: GameNode, depth: int, alpha: float, beta: float, maximizing: bool):
        if depth == 0 or node.state.turn == -999:
            node.score = self.heuristic(node.state)
            return node
        
        GameNode.generate_children(node)

        if not node.children:
            node.score = self.heuristic(node.state)
            return node
        
        if maximizing:
            value = float('-inf')
            bssf = None
            for child in node.children:
                candidate = self.alphabeta(child, depth - 1, alpha, beta, False)
                child.score = candidate.score
                if child.score > value:
                    value = child.score
                    bssf = child
                if value >= beta:
                    break
                alpha = max(alpha, value)
            node.score = value
            return bssf
        else:
            value = float('inf')
            bssf = None
            for child in node.children:
                candidate = self.alphabeta(child, depth - 1, alpha, beta, True)
                child.score = candidate.score  # negate because the heuristic is always from the maximizer's perspective?
                if child.score < value:
                    value = child.score
                    bssf = child
                if value <= alpha:
                    break
                beta = min(beta, value)
            node.score = value
            return bssf

    
    def heuristic(self, state: ReversiGameState):
        # if state.board[1][3] == state.turn:  # (in)sanity test
        #     return -99999

        maximizing_corners, minimizing_corners = self.count_corners(state)
        CORNERS = 100
        h_corners = CORNERS * maximizing_corners - CORNERS * minimizing_corners

        maximizing_corner_adjacent, minimizing_corner_adjacent = self.count_corner_adjacent(state)
        CORNERADJ = -50
        h_corner_adjacent = CORNERADJ * maximizing_corner_adjacent

        maximizing_borders, minimizing_borders = self.count_borders(state)
        BORDERS = 20
        h_borders = BORDERS * maximizing_borders - BORDERS * minimizing_borders

        maximizing_coins, minimizing_coins = self.count_coins(state)
        COINS = 5
        h_coins = COINS * maximizing_coins - COINS * minimizing_coins

        maximizing_moves, minimizing_moves = self.count_moves(state)
        MOVES = 30
        h_moves = MOVES * maximizing_moves - MOVES * minimizing_moves

        # dynamically increase parity_weight as the game goes on
        # parity_weight = ((maximizing_coins + minimizing_coins) / state.board_dim ** 2) * 10
        
        return h_corners + h_corner_adjacent + h_borders + h_coins + h_moves

    def count_coins(self, state: ReversiGameState):
        # returns the number of coins of each player, returning a tuple 
        # (maximizing, minimizing)
        maximizing = 0
        minimizing = 0
        for r in range(state.board_dim):
            for c in range(state.board_dim):
                if state.board[r][c] == self.move_num:
                    maximizing += 1
                elif state.board[r][c] != 0:
                    minimizing += 1
        return (maximizing, minimizing)
    
    def count_moves(self, state: ReversiGameState):
        turn = state.turn
        state.turn = self.move_num
        maximizing = len(state.get_valid_moves())
        state.turn = 2 if self.move_num == 1 else 1
        minimizing = len(state.get_valid_moves())
        state.turn = turn
        return (maximizing, minimizing)
    
    def count_corners(self, state: ReversiGameState):
        corners = [
            (0, 0), 
            (0, state.board_dim - 1), 
            (state.board_dim - 1, 0), 
            (state.board_dim - 1, state.board_dim - 1)
            ]
        maximizing = 0
        minimizing = 0
        for r, c in corners:
            if state.board[r][c] == self.move_num:
                maximizing += 1
            elif state.board[r][c] != 0:
                minimizing += 1
        return (maximizing, minimizing)
    
    def count_corner_adjacent(self, state: ReversiGameState):
        corner_adjacent = [
            (0, 1),
            (1, 0),
            (1, 1),
            (1, state.board_dim - 1),
            (0, state.board_dim - 2),
            (1, state.board_dim - 2),
            (state.board_dim - 1, 1), 
            (state.board_dim - 2, 0), 
            (state.board_dim - 2, 1),
            (state.board_dim - 2, state.board_dim - 1),
            (state.board_dim - 1, state.board_dim - 2),
            (state.board_dim - 2, state.board_dim - 2)
        ]
        maximizing = 0
        minimizing = 0
        for r, c in corner_adjacent:
            if state.board[r][c] == self.move_num:
                maximizing += 1
            elif state.board[r][c] != 0:
                minimizing += 1
        return (maximizing, minimizing)
    
    def count_borders(self, state: ReversiGameState):
        # only count non-corner borders
        maximizing = 0
        minimizing = 0
        for i in range(1, state.board_dim - 1):
            if state.board[i][0] == self.move_num:
                maximizing += 1
            elif state.board[i][0] != 0:
                minimizing += 1
            if state.board[i][state.board_dim - 1] == self.move_num:
                maximizing += 1
            elif state.board[i][state.board_dim - 1] != 0:
                minimizing += 1
            if state.board[0][i] == self.move_num:
                maximizing += 1
            elif state.board[0][i] != 0:
                minimizing += 1
            if state.board[state.board_dim - 1][i] == self.move_num:
                maximizing += 1
            elif state.board[state.board_dim - 1][i] != 0:
                minimizing += 1
        return (maximizing, minimizing)

    # def count_stability(self, state: ReversiGameState):
    #     maximizing = 0
    #     minimizing = 0
    #     for r in range(state.board_dim):
    #         for c in range(state.board_dim):
    #             if state.board[r][c] == self.move_num:
    #                 maximizing += self.determine_stability(state, r, c)
    #             elif state.board[r][c] != 0:
    #                 minimizing += self.determine_stability(state, r, c)
    #     return (maximizing, minimizing)

    # def determine_stability(self, state: ReversiGameState, r: int, c: int):
    #     corners = [
    #         (0, 0), 
    #         (0, state.board_dim - 1), 
    #         (state.board_dim - 1, 0), 
    #         (state.board_dim - 1, state.board_dim - 1)
    #         ]
    #     if (r, c) in corners:
    #         return 1
    #     directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
    #                   (-1, -1), (-1, 1), (1, -1), (1, 1)]
    #     for dr, dc in directions:
    #         nr, nc = r + dr, c + dc
    #         if 0 <= nr < state.board_dim and 0 <= nc < state.board_dim:
    #             if state.board[nr][nc] == 0:
    #                 return -1
    #     return 0
