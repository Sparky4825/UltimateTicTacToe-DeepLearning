import Arena
from UltimateTicTacToe.UltimateTicTacToeGame import TicTacToeGame
from MCTS import MCTS

# from othello.OthelloGame import OthelloGame
# from othello.OthelloPlayers import *
# from othello.pytorch.NNet import NNetWrapper as NNet


import numpy as np

from UltimateTicTacToe.UltimateTicTacToePlayers import *
from UltimateTicTacToe.keras.NNet import NNetWrapper
from main import args
from utils import *

import tensorflow as tf

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True

ray.init()

g = TicTacToeGame()

# all players
rp = RandomPlayer(g).play
# gp = GreedyOthelloPlayer(g).play
hp = HumanTicTacToePlayer(g).play


# nnet players
# if mini_othello:
#     n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
# else:
#     n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
#
#
# n1.load_checkpoint("tictactoe\\pretrained", "best.pth.tar")

modelPath = "litemodels/20210326-092010.tflite"


with open(modelPath, "rb") as f:
    modelContent = f.read()

n1p = NNetPlayer(g, modelContent, args)


if human_vs_cpu:
    player2 = hp
else:
    pass

if __name__ == "__main__":

    arena = Arena.Arena(n1p.get_move, player2, g, display=TicTacToeGame.display)

    print(arena.playGames(2, verbose=True))
