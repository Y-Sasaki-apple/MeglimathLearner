# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

#from Player import MCTSPlayer
from pl import MCTSPlayer
#from AZNet import PolicyValueNet
from alphazero_net import PolicyValueNet
import numpy as np
import random
#from board_ctrl import Board
from board import Board
#import game_ctrl as Game
import game as Game

def policy_view(c_puct,n_playout,n_games=2):
    """
    自分と戦って閲覧
    """
    current_mcts_player1 = MCTSPlayer(policy_value_net,
                                        c_puct=c_puct,
                                        n_playout=n_playout)
    current_mcts_player2 = MCTSPlayer(policy_value_net,
                                        c_puct=c_puct,
                                        n_playout=n_playout)
    for i in range(n_games):
        Game.start_play(current_mcts_player1,
                                 current_mcts_player2,
                                        start_player=i % 2,
                                        is_shown=1)

policy_value_net = None
mcts_player = None
def init(init_model,c_puct,n_playout):
    global policy_value_net
    global mcts_player
    if init_model:
        policy_value_net = PolicyValueNet(model_file=init_model)
    else:
        policy_value_net = PolicyValueNet()

    mcts_player = MCTSPlayer(policy_value_net,
                c_puct=c_puct,
                n_playout=n_playout,
                is_selfplay=1)

def get_equi_data(play_data):
    extend_data = play_data
    return extend_data

def collect_selfplay_data(temp):
    winner, play_data = Game.start_self_play(mcts_player,temp=temp)
    mcts_player.reset_player()
    play_data = list(play_data)[:]
    play_data = get_equi_data(play_data)
    return play_data

lr_multiplier = 1.0 
def policy_update(data_buffer,batch_size,learn_rate,kl_targ):
    global lr_multiplier
    epochs = 5
    mini_batch = random.sample(data_buffer, batch_size)
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]
    old_probs, old_v = policy_value_net.policy_value(state_batch)
    for _ in range(epochs):
        loss, entropy = policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                learn_rate*lr_multiplier)
        new_probs, new_v = policy_value_net.policy_value(state_batch)
        kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                axis=1)
        )
        if kl > kl_targ * 4:
            break
    if kl > kl_targ * 2 and lr_multiplier > 0.1:
        lr_multiplier /= 1.5
    elif kl < kl_targ / 2 and lr_multiplier < 10:
        lr_multiplier *= 1.5

    explained_var_old = (1 -
                            np.var(np.array(winner_batch) - old_v.flatten()) /
                            np.var(np.array(winner_batch)))
    explained_var_new = (1 -
                            np.var(np.array(winner_batch) - new_v.flatten()) /
                            np.var(np.array(winner_batch)))
    print(("kl:{:.5f},"
            "lr_multiplier:{:.3f},"
            "loss:{},"
            "entropy:{},"
            "explained_var_old:{:.3f},"
            "explained_var_new:{:.3f}"
            ).format(kl,
                    lr_multiplier,
                    loss,
                    entropy,
                    explained_var_old,
                    explained_var_new))
    return loss, entropy
    
best_win_ratio=0.0
pure_mcts_playout_num=500
def test_policy_and_save_best_policy():
    global best_win_ratio
    global pure_mcts_playout_num
    # win_ratio = policy_evaluate()
    win_ratio = None
    if win_ratio > best_win_ratio:
        print("New best policy!!!!!!!!")
        best_win_ratio = win_ratio
        policy_value_net.save_model('./models/best_policy.model')
        if (best_win_ratio == 1.0 and
                pure_mcts_playout_num < 5000):
            pure_mcts_playout_num += 1000
            best_win_ratio = 0.0

def save_model(filename):
    policy_value_net.save_model('./models/current_policy.model')