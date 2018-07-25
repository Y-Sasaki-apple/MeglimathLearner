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
        square_state[4][player_state == enemy_agents[0]] = 1.0    # 敵エージェントの座標1
        square_state[5][player_state == enemy_agents[1]] = 1.0    # 敵エージェントの座標2
        square_state[6] = self.board.get_board_state()/16.0
        square_state[7][:,:] = curr_player
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


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with 1,2,'O'", board.board.get_point(0))
        print("Player", player2, "with 3,4,'X'", board.board.get_point(1))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                printed=''
                if board.board.get_current_state()[i,j] ==0 :
                    printed += 'O'
                elif board.board.get_current_state()[i,j] ==1:
                    printed+='X' 
                printed += str(board.board.get_board_state()[i,j])
                if board.board.get_player_state()[i,j] != 0:
                    printed += '('+str(board.board.get_player_state()[i,j])+')'
                print(printed.center(8),end='')
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
            if is_shown and current_player == p2:
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
