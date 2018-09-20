# -*- coding: utf-8 -*-

import numpy as np
import MeglimathPy as meg
from os import system
import random

class Board(object):
    """board for the game"""
    def __init__(self):
        self.board = meg.Board()

    def init_board(self,start_player=0,turn=60,size=None):
        if size==None:
            size=random.choice([(11,12)])
            #サイズ決める
        #サイズ決めてランダム初期化
        self.board.init_board(turn,start_player,size[0],size[1])

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
        square_state = np.zeros((9, 12, 12))   # {0}層に変更
        data_size=(12,12)
        curr_state = self.board.get_current_state()
        input_size=curr_state.shape
        raw_state = np.zeros((9,input_size[0],input_size[1]))
        pad_size=[(0,0),(0,data_size[0]-input_size[0]),(0,data_size[1]-input_size[1])]
        curr_player = self.current_player
        other_player = 1 if curr_player==0 else 0
        player_state = self.board.get_player_state()
        agents = ((1,2),(3,4))[curr_player]
        enemy_agents = ((3,4),(1,2))[curr_player]
        raw_state[0][curr_state == curr_player] = 1.0
        raw_state[1][curr_state == other_player] = 1.0
        raw_state[2][player_state == agents[0]] = 1.0    # エージェントの座標1
        raw_state[3][player_state == agents[1]] = 1.0    # エージェントの座標2
        raw_state[4][(player_state == enemy_agents[0] )   # 敵エージェントの座標1
           | (player_state == enemy_agents[1])] = 1.0    # 敵エージェントの座標2
        raw_state[5] = self.board.get_board_state()/16.0 #得点
        raw_state[6] = np.abs(self.board.get_board_state())/16.0 #得点の絶対値
        raw_state[7] = self.board.remain_turn/60.0
        raw_state[8] = np.ones(input_size)
        square_state = np.pad(raw_state,pad_size,"constant")
        return square_state[:, ::-1, :]

    def do_move(self,move):
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

    def make_board(self,size,points,agent_points,tile,remain_turn):
        self.width = size[0]
        self.height = size[1]
        self.board.make_board(size,points,
            agent_points[0],agent_points[1],agent_points[2],agent_points[3],
            tile,remain_turn)

    def __str__(self):
        """
        boardの情報を表示する。縦横逆なので注意
        """
        return (str(self.board.get_board_state())+'\n'+
                str(self.board.get_current_state())+'\n'+
                str(self.board.get_player_state()))

    def graphic(self,start_player=None,cls=False):
        if (start_player is not None) and (self.current_player != start_player): return
        curr_state = self.board.get_current_state()
        input_size=curr_state.shape
        width,height=input_size
        if cls:system('cls')
        print("Turn:",self.remain_turn)
        print("Player with 1,2,'O'", self.board.get_point(0))
        print("Player with 3,4,'X'", self.board.get_point(1))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                printed=''
                if self.board.get_current_state()[j,i] ==0 :
                    printed += 'O'
                elif self.board.get_current_state()[j,i] ==1:
                    printed+='X' 
                printed += str(self.board.get_board_state()[j,i])
                if self.board.get_player_state()[j,i] != 0:
                    printed += '('+str(self.board.get_player_state()[j,i])+')'
                print(printed.center(8),end='')
            print('\r\n\r\n')

import unittest
import random

class boardTest(unittest.TestCase):
    def setUp(self):
        self.bd=Board()

    def test_construct(self):
        b = Board()
        b.init_board(0,60,(12,11))

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

    def test_make_board(self):
        b=self.bd
        size=(7,5)
        points=np.array([0,5,4,16,
                         2,-10,0,5,
                         3,4,2,-16])
        agent_points=[(0,0),(4,2),(6,2),(6,4)]
        tile=np.array(['a','-','-','-','-','-','-',
                       'a','-','-','-','-','-','-',
                       'a','-','-','-','a','-','b',
                       'a','-','-','-','-','-','-',
                       'a','-','-','-','-','-','b'])
        remain=50
        b.make_board(size,points,agent_points,tile,remain)
        print(b)


    def test_move(self):
        return
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