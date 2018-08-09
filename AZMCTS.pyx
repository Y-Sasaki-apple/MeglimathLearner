# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

#cdef struct NodeData:
#    NodeData* parent
#    NodeData* children[17*17]
#    float Q
#    float u
#   float P
#    int n_visits

class TreeNode:
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _reach_leaf(self,state):
        node = self._root
        while(1):
            if node.is_leaf():
                return node,state
            action, node = node.select(self._c_puct)
            state.do_move(action)

    def _state_eval(self,state):
        end, winner = state.game_end()    
        if end:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.current_player else -1.0
                )
            action_probs = None
        else:
            action_probs, leaf_value = self._policy(state)
        return action_probs,leaf_value

    def _playout(self, state):
        state = copy.deepcopy(state)
        node,state = self._reach_leaf(state)
        action_probs,leaf_value = self._state_eval(state)
        if action_probs:#ゲーム終了判定
            node.expand(action_probs)
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        for _ in range(self._n_playout):
            self._playout(state)

        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


import unittest
from Util import random_network

class mctsTest(unittest.TestCase):
    def setUp(self):
        network = random_network()
        self.mcts = MCTS(network.policy_value_fn,5,400)

    def testinit(self):
        mcts=self.mcts
        self.assertEqual(mcts._c_puct,5)
        self.assertEqual(mcts._n_playout,400)
        self.check_is_root(mcts._root)
        self.check_is_leaf(mcts._root)

    def _reach_leaf_with_stopping(self,state):       
        node = self.mcts._root
        while(1):
            if node.is_leaf():
                return node,state
            action, node = 17*17-1,node._children[17*17-1]
            state.do_move(action)

    def test_state_eval(self):
        network = random_network()
        mcts = MCTS(network.policy_value_fn,0,400)
        from board_ctrl import Board
        board = Board()
        board.init_board(turn=2)
        
        while(1):
            state = copy.deepcopy(board) 
            node,state = mcts._reach_leaf(state)
            self.check_is_leaf(node)
            action_probs,leaf_value = mcts._state_eval(state)
            if action_probs:#ゲーム終了判定
                node.expand(action_probs)
            node.update_recursive(-leaf_value)

            if action_probs is None:
                if state.has_a_winner()[0]:
                    if state.has_a_winner()[1]==0:
                        self.assertEqual(leaf_value,-1.0)
                    else:
                        self.assertEqual(leaf_value,1.0)
                else:
                    self.assertEqual(leaf_value,0.0)
                break

    def check_is_root(self,node):
        self.assertTrue(node.is_root())

    def check_is_leaf(self,node):
        self.assertTrue(node.is_leaf())

if __name__=="__main__":
    unittest.main()