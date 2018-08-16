# -*- coding: utf-8 -*-
import numpy as np
from board_ctrl import Board

def print_result(winner):
    if winner != -1:
        print("Game end. Winner is " + str(winner))
    else:
        print("Game end. Tie")

def start_play(player1, player2, start_player=0, is_shown=1):
    """
    冒険手のないプレイ
    """
    if start_player not in (0, 1):
        raise Exception()
    board = Board()
    board.init_board(start_player)
    if is_shown:board.graphic(start_player)
    while True:
        current_player = board.current_player
        move = [player1,player2][current_player].get_action(board)
        board.do_move(move)
        if is_shown : board.graphic(start_player)
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
        if is_shown:board.graphic(0)
        end, winner = board.game_end()
        if end:
            winners_z = np.zeros(len(current_players))
            if winner != -1:
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0
            if is_shown:print_result(winner)
            return winner, zip(states, mcts_probs, winners_z)

import unittest
from Util import random_player,random_network
from Player import MCTSPlayer

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
        if win!=-1:
            self.assertEqual(winner_batch[win],1.0)
            self.assertEqual(winner_batch[win+1],-1.0)

    def test_play(self):
        start_play(random_player(),random_player(),0,0)
        start_play(random_player(),random_player(),1,0)
        #self.assertRaises(Exception,start_play,random_player(),random_player(),2)
        #self.player_self_play()
        self.player_play()

    def player_play(self):
        from AZNet import PolicyValueNet
        policy_value_net = PolicyValueNet(6,6,
                                                model_file='models/current_policy.model')
        pl1=MCTSPlayer(policy_value_net,1,400,0)
        pl2=MCTSPlayer(random_network(),1,400,0)
        start_play(pl1,pl2,start_player=0,is_shown=1)      

    def player_self_play(self):
        pl1=MCTSPlayer(random_network(),0.01,100,1)
        start_self_play(pl1,1,1.0)

if __name__=="__main__":
    unittest.main()