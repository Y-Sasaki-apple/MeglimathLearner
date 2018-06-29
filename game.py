# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np
import MeglimathPy as meg


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 6))
        self.height = int(kwargs.get('height', 6))
        self.board = meg.Board()
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        ### self.states = {}
        # need how many pieces in a row to win
        ### self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        self.board.init_board(start_player)
        self.current_player = self.board.get_current_player()
        self.availables = self.board.availables


    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 5*width*height
        """

        square_state = np.zeros((8, self.width, self.height))   # {0}層に変更
        curr_state = self.board.get_current_state()
        curr_player = self.get_current_player()
        player_state = self.board.get_player_state()
        agents = ((1,2),(3,4))[curr_player]
        enemy_agents = ((3,4),(1,2))[curr_player]
        square_state[0][curr_state == curr_player] = 1.0
        square_state[1][curr_state != curr_player] = 1.0
        square_state[2][player_state == agents[0]] = 1.0    # エージェントの座標1
        square_state[3][player_state == agents[1]] = 1.0    # エージェントの座標2
        square_state[4][player_state == enemy_agents[0]] = 1.0    # 敵エージェントの座標1
        square_state[5][player_state == enemy_agents[1]] = 1.0    # 敵エージェントの座標2
        square_state[6] = (self.board.get_board_state()+16)/np.max(self.board.get_board_state()+16)
        square_state[7][:,:] = curr_player
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.board.do_move(move)
        self.availables = self.board.availables
        self.current_player = self.board.get_current_player()
        """
        moveの仕様
        D=方向数。下を0として左回り。[0,7]
        A=行動数。移動=0、除去=1
        Nx=各エージェントの行動。A*8+D または 17(停滞) [0,17]
        move=2エージェントの行動。N1*17+N2 [0,17*17)
        """

    def has_a_winner(self):
        return self.board.has_a_winner()

    def game_end(self):
        """Check whether the game is ended or not"""
        return self.board.game_end()

    def get_current_player(self):
        return self.board.get_current_player()


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
