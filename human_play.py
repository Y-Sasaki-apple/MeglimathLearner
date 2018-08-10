# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

import pickle
import game_ctrl as game
from Player import MCTSPlayer
from AZNet import PolicyValueNet
from Util import Human
def run():
    model_file = './models/current_policy.model'
    try:
        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        best_policy = PolicyValueNet(6, 6, model_file = model_file)
        mcts_player = MCTSPlayer(best_policy, c_puct=5, n_playout=400)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
