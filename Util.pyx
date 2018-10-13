# -*- coding: utf-8 -*-
import random
from board import Board
import numpy as np

class random_network():
    """
    ニューラルネットワークのスタブ
    """
    def policy_value_fn(self, board):
        legal_positions = board.availables
        value = 0
        value = (board.board.get_point(0)-board.board.get_point(1))/100
        if board.current_player==1:
            value = -value
        act_probs = [1.0/len(legal_positions) for _ in legal_positions]
        # act_probs = np.array([random.random() for _ in legal_positions])
        # def softmax(x):
        #     probs = np.exp(x - np.max(x))
        #     probs /= np.sum(probs)
        #     return probs
        # act_probs=softmax(act_probs)
        act_probs = zip(legal_positions,act_probs)
        return act_probs,value

class random_player():
    """
    Playerのスタブ
    """
    def get_action(self,board,temp=None,return_prob=0):
        if return_prob==1:
            prob = np.zeros(12*12*2*12*12*2)
            act=random.choice(board.availables)
            prob[act]=1
            return act,prob
        else:
            return random.choice(board.availables)

class Human():
    def __init__(self):
        pass

    def get_action(self, board):
        try:
            moves = input("Your move: ")
            moves = [int(n, 10) for n in moves.split(",")]
            dirdir = {1:5,2:6,3:7,4:4,5:0,6:0,7:3,8:2,9:1}
            if moves[1]==2:moves[0]=5
            if moves[3]==2:moves[2]=5
            move = dirdir[moves[2]]+moves[3]*8+dirdir[moves[0]]*17+moves[1]*17*8
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move