#-*- coding:utf-8 -*-
"""
Author : bamboowonsstring
"""
import numpy as np
from collections import defaultdict, deque
# from mcts_pure import MCTSPlayer as MCTS_Pure
import AZ

#パラメータの設定
learn_rate = 2e-3
temp = 1.0  #変化温度
n_playout = 400 
c_puct = 5 #puct定数
buffer_size = 10000
batch_size = 1024
data_buffer = deque(maxlen=buffer_size)
play_batch_size = 1
kl_targ = 0.02
check_freq = 100

init_model=None
AZ.init(init_model,6,6,c_puct,n_playout)

try:
    for i in range(1):
        play_data = AZ.collect_selfplay_data(temp)
        data_buffer.extend(play_data)
        episode_len = len(play_data)
        print("batch i:{}, episode_len:{}".format(i+1, episode_len))
        if len(data_buffer) > batch_size:
            loss, entropy = AZ.policy_update(data_buffer,batch_size,learn_rate,kl_targ)
        if (i+1) % check_freq == 0:
            print("current self-play batch: {}".format(i+1))
            AZ.save_model('./models/current_policy.model')
            AZ.policy_view(5,400)
            # test_policy_and_save_best_policy()
except KeyboardInterrupt:
    print('\n\rquit')
