import Arena
from UltimateTicTacToe.UltimateTicTacToeGame import TicTacToeGame
from OldMCTS import MCTS

# from othello.OthelloGame import OthelloGame
# from othello.OthelloPlayers import *
# from othello.pytorch.NNet import NNetWrapper as NNet


import numpy as np

from UltimateTicTacToe.UltimateTicTacToePlayers import *
from UltimateTicTacToe.keras.NNet import NNetWrapper
from main import args

import logging
from Coach import Coach

from main import args

from UltimateTicTacToe.keras.NNet import NNetWrapper as nn


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
    log = logging.getLogger(__name__)
    log.info("Loading %s...", TicTacToeGame.__name__)
    g = TicTacToeGame()

    log.info("Loading Neural Network (Ray actor)...")
    nnet = nn(g)

    previous_weights = nnet.get_weights()

    if True:
        log.info(
            'Loading checkpoint "%s/%s"...',
            "./temp/",
            "best.ckpt",
        )
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    new_weights = nnet.get_weights()

    log.info("Starting Arena Round 1")

    c = Coach(g, nnet, args)

    winNew1, winOld1, draw1 = c.runArenaInline(new_weights, previous_weights)

    log.info("Starting Arena Round 2")
    winOld2, winNew2, draw2 = c.runArenaInline(previous_weights, new_weights)

    pwins = winOld1 + winOld2
    nwins = winNew1 + winNew2
    draws = draw1 + draw2

    log.info("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))
