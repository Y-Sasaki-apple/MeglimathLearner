#-*- coding:utf-8 -*-
"""
Author : bamboowonsstring
"""
import numpy as np
from collections import defaultdict, deque
# import AZ
import concurrent.futures
from multiprocessing import Process, freeze_support
import AZ
import pyximport
pyximport.install()

play_batch_size = 1
play_parallel_size = 3
check_freq = 1
#init_model = None
init_model='models/current_policy.model'

#パラメータの設定
learn_rate = 2e-3
temp = 1.0  #変化温度
n_playout = 400 
c_puct = 5 #puct定数
buffer_size = 20000
batch_size = 1024
data_buffer = deque(maxlen=buffer_size)
kl_targ = 0.02

AZ.init(init_model,c_puct,n_playout)

if __name__=="__main__":
    try:
        freeze_support()
        for i in range(play_batch_size):
            executor = concurrent.futures.ProcessPoolExecutor()
            results = []

            # executor.submit(a)    
            for i in range(play_parallel_size):
                results.append(executor.submit(AZ.collect_selfplay_data,temp))
                print("開始"+str(i))
            for r in results:
                play_data = r.result()        
                data_buffer.extend(play_data)
            
            # data_buffer.extend(play_data)
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
