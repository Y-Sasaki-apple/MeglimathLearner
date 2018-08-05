# -*- coding: utf-8 -*-
import numpy as np
import MeglimathPy as meg
class Game:
    def __init__(self, board):
        self.board = board

    def graphic(self, board):
        width = board.width
        height = board.height

        print("Player with 1,2,'O'", board.board.get_point(0))
        print("Player with 3,4,'X'", board.board.get_point(1))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                printed=''
                if board.board.get_current_state()[i,j] ==0 :
                    printed += 'O'
                elif board.board.get_current_state()[i,j] ==1:
                    printed+='X' 
                printed += str(board.board.get_board_state()[i,j])
                if board.board.get_player_state()[i,j] != 0:
                    printed += '('+str(board.board.get_player_state()[i,j])+')'
                print(printed.center(8),end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """
        冒険手のないプレイ
        """
        if start_player not in (0, 1):
            raise Exception()

        self.board.init_board(start_player)
        if is_shown:
            self.graphic(self.board)
        while True:
            current_player = self.board.get_current_player()
            move = [player1,player2][current_player].get_action(self.board)
            self.board.do_move(move)
            if is_shown and current_player != start_player:
                self.graphic(self.board)

            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is")
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """
        冒険手プレイ
        """
        self.board.init_board()
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board)
            end, winner = self.board.game_end()
            if end:
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
