# -*- coding: utf-8 -*-
import random
from board_ctrl import Board
import numpy as np

class random_network():
    """
    ニューラルネットワークのスタブ
    """
    def policy_value_fn(self, board):
        legal_positions = board.availables
        value = 0
        # value = (board.board.get_point(0)-board.board.get_point(1))/100
        # if board.current_player==1:
        #     value = -value
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
            prob = np.zeros(17*17)
            act=random.choice(board.availables)
            prob[act]=1
            return act,prob
        else:
            return random.choice(board.availables)