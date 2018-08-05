# -*- coding: utf-8 -*-

import numpy as np
import MeglimathPy as meg


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 6))
        self.height = int(kwargs.get('height', 6))
        self.board = meg.Board()
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0,turn=60):
        self.board.init_board(turn,start_player)
        self.current_player = self.get_current_player()
        self.availables = self.board.availables

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 5*width*height
        """

        square_state = np.zeros((8, self.width, self.height))   # {0}層に変更
        curr_state = self.board.get_current_state()
        curr_player = self.board.get_current_player()
        other_player = 1 if curr_player==0 else 0
        player_state = self.board.get_player_state()
        agents = ((1,2),(3,4))[curr_player]
        enemy_agents = ((3,4),(1,2))[curr_player]
        square_state[0][curr_state == curr_player] = 1.0
        square_state[1][curr_state == other_player] = 1.0
        square_state[2][player_state == agents[0]] = 1.0    # エージェントの座標1
        square_state[3][player_state == agents[1]] = 1.0    # エージェントの座標2
        square_state[4][(player_state == enemy_agents[0] )   # 敵エージェントの座標1
           | (player_state == enemy_agents[1])] = 1.0    # 敵エージェントの座標2
        square_state[5] = self.board.get_board_state()/16.0
        square_state[6] = np.abs(self.board.get_board_state())/16.0
        square_state[7] = self.board.remain_turn/60.0
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.board.do_move(move)
        self.availables = self.board.availables
        self.current_player = self.get_current_player()
        """
        moveの仕様
        D=方向数。下を0として左回り。[0,7]
        A=行動数。移動=0、除去=1
        Nx=各エージェントの行動。A*8+D または 17(停滞) [0,17]
        move=2エージェントの行動。N1*17+N2 [0,17*17)
        """

    def has_a_winner(self):
        ret = self.board.has_a_winner()
        if ret[1]!=-1:
            ret=(ret[0],ret[1]+1)
        return ret

    def game_end(self):
        """Check whether the game is ended or not"""
        ret = self.board.game_end()
        if ret[1]!=-1:
            ret=(ret[0],ret[1]+1)
        return ret

    def get_current_player(self):
        return self.players[self.board.get_current_player()]
