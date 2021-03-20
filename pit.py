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

NNactor = NNetWrapper.remote(g)
ray.get(NNactor.load_checkpoint.remote("./temp/", "best.pth.tar"))

weights = ray.get(NNactor.get_weights.remote())


n1p = NNetPlayer(g, NNactor, weights, args)


ray.get(NNactor.load_checkpoint.remote(".\\temp", "temp.pth.tar"))


if human_vs_cpu:
    player2 = hp
else:
    pass


arena = Arena.Arena(n1p.get_move, player2, g, display=TicTacToeGame.display)

print(arena.playGames(2, verbose=True))
