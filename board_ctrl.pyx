# -*- coding: utf-8 -*-

import numpy as np
import MeglimathPy as meg


class Board(object):
    """board for the game"""
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 6))
        self.height = int(kwargs.get('height', 6))
        self.board = meg.Board()

    def init_board(self,int start_player=0,int turn=60):
        self.board.init_board(turn,start_player)

    @property
    def availables(self):
        return self.board.availables

    @property
    def current_player(self):
        return self.board.get_current_player()

    @property
    def remain_turn(self):
        return self.board.remain_turn

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 5*width*height
        """
        square_state = np.zeros((8, self.width, self.height))   # {0}層に変更
        curr_state = self.board.get_current_state()
        curr_player = self.current_player
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

    def do_move(self,int move):
        self.board.do_move(move)
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

import unittest
import random

class boardTest(unittest.TestCase):
    def setUp(self):
        self.bd=Board()

    def test_construct(self):
        self.assertEqual(self.bd.height,6)
        self.assertEqual(self.bd.width,6)
        b = Board(width=11,height=12)
        self.assertEqual(b.height,12)
        self.assertEqual(b.width,11)

    def test_init_board(self):
        b = self.bd
        b.init_board()
        self.assertEqual(b.remain_turn,60)
        self.assertEqual(b.game_end(),(False,-1))
        self.assertEqual(b.has_a_winner(),(False,-1))
        self.assertEqual(b.current_player,0)
        self.assertEqual(b.current_state().shape,(8,6,6))
        self.check_available_and_position(b)

    def check_available_and_position(self,b):
        pos1=np.where(b.current_state()[2]==1)
        self.assertEqual(len(pos1),2)
        self.assertEqual(len(pos1[0]),1)
        self.assertEqual(len(pos1[1]),1)
        pos2=np.where(b.current_state()[3]==1)
        self.assertEqual(len(pos2),2)
        self.assertEqual(len(pos2[0]),1)
        self.assertEqual(len(pos2[1]),1)     
        """
        availablesの数は位置に依る
        1個あたり　中央9 ヘリ6 角4 が最低値
        それに周辺の自分マスが加算される
        それ２個分の乗算 
        """
        if (pos1[0]==0 or pos1[0]==5) and (pos1[1]==0 or pos1[1]==5):
            num_availables1 = 4
        elif (pos1[0]==0 or pos1[0]==5) or (pos1[1]==0 or pos1[1]==5):
            num_availables1 = 6
        else:
            num_availables1 = 9
        if (pos2[0]==0 or pos2[0]==5) and (pos2[1]==0 or pos2[1]==5):
            num_availables2 = 4
        elif (pos2[0]==0 or pos2[0]==5) or (pos2[1]==0 or pos2[1]==5):
            num_availables2 = 6
        else:
            num_availables2 = 9
        self.assertGreaterEqual(len(b.availables),num_availables1*num_availables2,
            msg = str(b.current_state()))
        self.assertLessEqual(len(b.availables),(num_availables1+8)*(num_availables2+8),
            msg = str(b.current_state()))
        enemypos=np.where(b.current_state()[4]==1)
        self.assertEqual(len(enemypos),2)
        self.assertEqual(len(enemypos[0]),2)
        self.assertEqual(len(enemypos[1]),2)     

    def test_move(self):
        b = self.bd
        b.init_board()
        self.check_move(b)

    def check_move(self,b):
        """
        ゲーム終了までmoveする
        """
        for turn in range(60,0,-1):
            self.assertEqual(b.remain_turn,turn)
            self.assertEqual(b.game_end(),(False,-1))
            self.assertEqual(b.has_a_winner(),(False,-1))

            self.assertEqual(b.current_player,0)
            b.do_move(random.choice(b.availables))
            self.check_available_and_position(b)

            self.assertEqual(b.remain_turn,turn)
            self.assertEqual(b.game_end(),(False,-1))
            self.assertEqual(b.has_a_winner(),(False,-1))

            self.assertEqual(b.current_player,1)
            b.do_move(random.choice(b.availables))
            self.check_available_and_position(b)

        self.assertEqual(b.remain_turn,0)
        p0 = b.board.get_point(0) 
        p1 = b.board.get_point(1) 
        if p0 > p1:
            self.assertEqual(b.game_end(),(True,0))
            self.assertEqual(b.has_a_winner(),(True,0))
        elif p1 > p0:
            self.assertEqual(b.game_end(),(True,1))
            self.assertEqual(b.has_a_winner(),(True,1))
        else:
            self.assertEqual(b.game_end(),(True,-1))
            self.assertEqual(b.has_a_winner(),(False,-1))
        

if __name__=="__main__":
    unittest.main()