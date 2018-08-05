# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

from AZMCTS import MCTS
import numpy as np
import copy

class MCTSPlayer:
    def __init__(self,policy_value,
                    c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value.policy_value_fn, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self,board, temp=1e-3, return_prob=0):
        move_probs = np.zeros(17*17)
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
            