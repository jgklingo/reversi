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
    DEPTH = 6

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

        root = GameNode(state)
        maximizing = state.turn == self.move_num
        best_node = self.alphabeta(root, self.DEPTH, float('-inf'), float('inf'), maximizing=maximizing)
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
                if candidate.score > value:
                    value = candidate.score
                    bssf = child
                if value > beta:
                    break
                alpha = max(alpha, value)
            node.score = value
            return bssf
        else:
            value = float('inf')
            bssf = None
            for child in node.children:
                candidate = self.alphabeta(child, depth - 1, alpha, beta, True)
                if candidate.score < value:
                    value = candidate.score
                    bssf = child
                if value < alpha:
                    break
                beta = min(beta, value)
            node.score = value
            return bssf

    
    def heuristic(self, state: ReversiGameState):
        maximizing_coins, minimizing_coins = self.count_coins(state)
        if maximizing_coins + minimizing_coins != 0:
            parity_h = 100 * (maximizing_coins - minimizing_coins) / (maximizing_coins + minimizing_coins)
        else:
            parity_h = 0

        maximizing_moves, minimizing_moves = self.count_moves(state)
        if maximizing_moves + minimizing_moves != 0:
            mobility_h = 100 * (maximizing_moves - minimizing_moves) / (maximizing_moves + minimizing_moves)
        else:
            mobility_h = 0

        maximizing_corners, minimizing_corners = self.count_corners(state)
        if maximizing_corners + minimizing_corners != 0:
            corners_h = 100 * (maximizing_corners - minimizing_corners) / (maximizing_corners + minimizing_corners)
        else:
            corners_h = 0

        maximizing_stability, minimizing_stability = self.count_stability(state)
        if maximizing_stability + minimizing_stability != 0:
            stability_h = 100 * (maximizing_stability - minimizing_stability) / (maximizing_stability + minimizing_stability)
        else:
            stability_h = 0
        
        # TODO: tweak weights
        return 1 * parity_h + 2 * mobility_h + 10 * corners_h + 4 * stability_h

    def count_coins(self, state: ReversiGameState):
        # returns the number of coins of each player, returning a tuple 
        # (maximizing, minimizing) (or (1, 2))
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
        state.turn = 2 if self.move_num == 1 else 2
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

    def count_stability(self, state: ReversiGameState):
        maximizing = 0
        minimizing = 0
        for r in range(state.board_dim):
            for c in range(state.board_dim):
                if state.board[r][c] == self.move_num:
                    maximizing += self.determine_stability(state, r, c)
                elif state.board[r][c] != 0:
                    minimizing += self.determine_stability(state, r, c)
        return (maximizing, minimizing)

    def determine_stability(self, state: ReversiGameState, r: int, c: int):
        corners = [
            (0, 0), 
            (0, state.board_dim - 1), 
            (state.board_dim - 1, 0), 
            (state.board_dim - 1, state.board_dim - 1)
            ]
        if (r, c) in corners:
            return 1
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < state.board_dim and 0 <= nc < state.board_dim:
                if state.board[nr][nc] == 0:
                    return -1
        return 0
        