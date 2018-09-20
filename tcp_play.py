# -*- coding: utf-8 -*-
import socket
from contextlib import closing
import json
import numpy as np
from ast import literal_eval
from board_ctrl import Board
from Player import MCTSPlayer
host = '127.0.0.1'
port = 31400
bufsize = 4096
init_model='models/current_policy.model'
board_width = 12
board_height = 12
c_puct = 0
n_playout = 400
if init_model:
    from AZNet import PolicyValueNet
    policy_value_net = PolicyValueNet(board_width,board_height,
                                            model_file=init_model)
else:
    from Util import random_network
    policy_value_net = random_network()
mcts_player = MCTSPlayer(policy_value_net,
            c_puct=c_puct,
            n_playout=n_playout,
            is_selfplay=0)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
board = Board()

def json_make_board(board,receive):
    resdata=json.loads(receive)
    size = literal_eval(resdata["Size"])
    points = np.array(resdata["Points"])
    tiles = np.array(''.join(resdata["Tiles"]))
    poss = [literal_eval(p) for p in resdata["AgentPosA"]]
    poss.extend(
            [literal_eval(p) for p in resdata["AgentPosB"]])
    point = [resdata["TotalPointA"],resdata["TotalPointB"]]
    remain = resdata["RemainingTurn"]
    board.make_board(size,points,poss,tiles,remain)

def act_to_json(act):
    move1 = act // 17
    move2 = act % 17
    act1 = move1 // 8
    act2 = move2 // 8
    dir1 = 8 if act1 == 2 else move1 % 8 
    dir2 = 8 if act2 == 2 else move2 % 8 
    actdict = {0:"Move",1:"Remove",2:"Stop"}
    dirdict = {0:"Right",1:"RightUp",2:"Up",
                3:"LeftUp",4:"Left",5:"LeftDown",
                6:"Down",7:"RightDown",8:""}
    data = {"Action":[actdict[act1],actdict[act2]],
            "Direction":[dirdict[dir1],dirdict[dir2]]}
    return json.dumps(data)

with closing(sock):
    sock.connect((host, port))
    print('Connected')

    receive=sock.recv(bufsize).decode('utf-8')
    resdata=json.loads(receive)
    print(resdata["TeamType"])
    while(1):
        receive=sock.recv(bufsize).decode('utf-8')
        if receive=='' : break
        json_make_board(board,receive)
        board.graphic()
        act = mcts_player.get_action(board)
        senddata = (act_to_json(act)+'\n').encode('utf-8')
        sock.send(senddata)