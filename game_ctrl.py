# -*- coding: utf-8 -*-
import numpy as np
import MeglimathPy as meg
from board_ctrl import Board
from os import system
def print_result(winner):
    if winner != -1:
        print("Game end. Winner is " + str(winner))
    else:
        print("Game end. Tie")

def graphic(board,start_player,cls=False):
    if board.current_player != start_player: return
    width = board.width
    height = board.height
    if cls:system('cls')
    print("Player with 1,2,'O'", board.board.get_point(0))
    print("Player with 3,4,'X'", board.board.get_point(1))
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

def start_play(player1, player2, start_player=0, is_shown=1):
    """
    冒険手のないプレイ
    """
    if start_player not in (0, 1):
        raise Exception()
    board = Board()
    board.init_board(start_player)
    if is_shown:graphic(board,start_player)
    while True:
        current_player = board.current_player
        move = [player1,player2][current_player].get_action(board)
        board.do_move(move)
        if is_shown : graphic(board,start_player)
        end, winner = board.game_end()
        if end:
            if is_shown:print_result(winner)
            return winner

def start_self_play(player, is_shown=0, temp=1e-3):
    """
    冒険手プレイ
    """
    board = Board()
    board.init_board()
    states, mcts_probs, current_players = [], [], []
    while True:
        move, move_probs = player.get_action(board,
                                                temp=temp,
                                                return_prob=1)
        states.append(board.current_state())
        mcts_probs.append(move_probs)
        current_players.append(board.current_player)
        board.do_move(move)
        if is_shown:graphic(board,0)
        end, winner = board.game_end()
        if end:
            winners_z = np.zeros(len(current_players))
            if winner != -1:
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0
            if is_shown:print_result(winner)
            return winner, zip(states, mcts_probs, winners_z)

import unittest
import random

class random_player():
    """
    Playerのスタブ
    """
    def get_action(self,board,temp=None,return_prob=0):
        if return_prob==1:
            prob = np.zeros(17*17)
            act=random.choice(board.availables)
            prob[act]=1
            return act,prob
        else:
            return random.choice(board.availables)

class gameTest(unittest.TestCase):
    def test_self_play(self):
        win,result = start_self_play(random_player(),is_shown=0)
        batch = list(result)
        self.assertEqual(len(batch),120)
        for data in batch:
            state = data[0]
            self.assertEqual(state.shape,(8,6,6))
            mcts_probs = data[1]
            self.assertEqual(mcts_probs.shape,(17*17,))
            winner = data[2]
            self.assertEqual(winner.shape,())
        winner_batch = [data[2] for data in batch]
        self.assertEqual(winner_batch[win],1.0)
        self.assertEqual(winner_batch[win+1],-1.0)

    def test_play(self):
        win=start_play(random_player(),random_player(),0,0)
        win=start_play(random_player(),random_player(),1,0)
        self.assertRaises(Exception,start_play,random_player(),random_player(),2)


if __name__=="__main__":
    unittest.main()