# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

from AZMCTS import MCTS
#from az_mcts import MCTS
import numpy as np
import copy

class MCTSPlayer:
    def __init__(self,network,
                    c_puct=5, n_playout=2000, is_selfplay=0):
        if hasattr(network,'policy_value'):
            policy_value_fn = self.make_policy_value_fn(network.policy_value)
        else:
            policy_value_fn = network.policy_value_fn
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def make_policy_value_fn(self,policy_value):
        def policy_value_fn(board):
            """
            input: board
            output: a list of (action, probability) tuples for each available
            action and the score of the board state
            """
            legal_positions = board.availables
            data=board.current_state().reshape(
                    -1, 9, 12, 12)
            current_state = np.ascontiguousarray(data)
            act_probs, value = policy_value(current_state)
            act_probs = zip(legal_positions, act_probs[0][legal_positions])
            return act_probs, value
        return policy_value_fn

    def get_action(self,board, temp=1e-3, return_prob=0):
        move_probs = np.zeros(12*12*2*12*12*2)
        acts, probs = self.mcts.get_move_probs(board, temp)
        move_probs[list(acts)] = probs
        if self._is_selfplay:
            move = np.random.choice(
                acts,
                p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
            )
            # これまで構築した木を再利用する
            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(acts, p=probs)
            # 木のリセット
            self.mcts.update_with_move(-1)

        if return_prob:
            return move, move_probs
        else:
            return move
            
import unittest
from Util import random_network

class mctsTest(unittest.TestCase):
    def test_init(self):
        pl=MCTSPlayer(random_network,1,400,0)
        self.assertEqual(pl._is_selfplay,0)

if __name__=="__main__":
    unittest.main()