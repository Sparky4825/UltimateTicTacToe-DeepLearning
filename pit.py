import Arena
from UltimateTicTacToe.UltimateTicTacToeGame import TicTacToeGame

# from othello.OthelloGame import OthelloGame
# from othello.OthelloPlayers import *
# from othello.pytorch.NNet import NNetWrapper as NNet


import numpy as np

# from UltimateTicTacToe.UltimateTicTacToePlayers import *
from UltimateTicTacToe.keras.NNet import NNetWrapper
from main import args

import logging
from Coach import Coach

from main import args

from UltimateTicTacToe.keras.NNet import NNetWrapper as nn


from MCTS import PyMCTS, PyGameState, prepareBatch, batchResults
import tensorflow as tf
import time

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True


# nnet players
# if mini_othello:
#     n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
# else:
#     n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
#
#
# n1.load_checkpoint("tictactoe\\pretrained", "best.pth.tar")

# modelPath = "litemodels/20210326-092010.tflite"


def vsHuman(args, nnet, weights):
    """
    Returns in format (player1Wins, player2Wins, draws)
    :param player1Weights:
    :param player2Weights:
    :return:
    """
    p1Episodes = []
    p2Episodes = []
    results = []
    # Start all episodes
    for i in range(1):
        p1Episodes.append(PyMCTS(args.cpuct))
        g = PyGameState()
        p1Episodes[-1].startNewSearch(g)

        p2Episodes.append(PyMCTS(args.cpuct))
        g = PyGameState()
        p2Episodes[-1].startNewSearch(g)

    actionsTaken = 0

    # Loop until all episodes are complete
    while len(p1Episodes) > 0:
        overallStart = time.time()

        nnet.set_weights(weights)
        for _ in range(2000):
            # print("Stargin simulation")
            needsEval = prepareBatch(p1Episodes)

            pi, v = nnet.predict_on_batch(needsEval)
            batchResults(p1Episodes, pi, v)
        index = -1

        actionsTaken += 1
        print(f"Taking action {actionsTaken}")

        for ep in p1Episodes:
            index += 1
            ep2 = p2Episodes[index]

            pi = np.array(ep.getActionProb())
            pi /= sum(pi)

            # Choose action randomly if within tempThreshold
            if actionsTaken <= args.tempThreshold:
                # Correct a slight rounding error if necessary
                if sum(pi) != 1:
                    # print("CORRECTING ERROR")
                    # print(pi)
                    mostLikelyIndex = np.argmax(pi)
                    pi[mostLikelyIndex] += 1 - sum(pi)

                action = np.random.choice(len(pi), p=pi)
            else:
                # Take best action
                action = np.argmax(pi)

            ep.takeAction(action)

            status = ep.getStatus()
            # Remove episode and save results when the game is over
            if status != 0:
                print(ep.gameToString())
                print(f"Game over")
                if status == 1:
                    results.append(1)
                elif status == 2:
                    results.append(-1)
                else:
                    results.append(0)
                p1Episodes.remove(ep)
                p2Episodes.remove(p2Episodes[index])

        if len(p1Episodes) == 0:
            break

        ep = p1Episodes[0]

        # GET USER MOVE
        print(ep.gameToString())
        board = int(input("Board >> "))
        piece = int(input("Piece >> "))

        action = board * 9 + piece
        # MAKE USER MOVE

        ep.takeAction(action)

        print(ep.gameToString())

        status = ep.getStatus()
        # Remove episode and save results when the game is over
        if status != 0:
            print(f"Game over - {len(p1Episodes) - 1} remaining")
            if status == 1:
                results.append(1)
            elif status == 2:
                results.append(-1)
            else:
                results.append(0)
            p2Episodes.remove(ep)
            p1Episodes.remove(p1Episodes[index])

    return results.count(1), results.count(-1), results.count(0)


#
# with open(modelPath, "rb") as f:
#     modelContent = f.read()
#
# n1p = NNetPlayer(g, modelContent, args)

#
# if human_vs_cpu:
#     player2 = hp
# else:
#     pass

model2 = ["temp", "best.ckpt"]
# model1 = ["temp", "At Work (3 accepted)\\best.ckpt"]

model1 = None
if __name__ == "__main__":
    log = logging.getLogger(__name__)
    log.info("Loading %s...", TicTacToeGame.__name__)
    g = TicTacToeGame()

    log.info("Loading Neural Network (Ray actor)...")
    nnet = nn(g)

    # vsHuman(args, nnet, nnet.get_weights())

    if model1 is not None:
        log.info(
            'Loading checkpoint "%s/%s"...',
            model1[0],
            model1[1],
        )
        nnet.load_checkpoint(model1[0], model1[1])

    previous_weights = nnet.get_weights()

    if model2 is not None:
        log.info(
            'Loading checkpoint "%s/%s"...',
            model2[0],
            model2[1],
        )
        nnet.load_checkpoint(model2[0], model2[1])

    new_weights = nnet.get_weights()

    vsHuman(args, nnet, new_weights)

    log.info("Starting Arena Round 1")

    c = Coach(g, nnet, args)

    winNew1, winOld1, draw1 = c.runArenaInline(new_weights, previous_weights)

    log.info("Starting Arena Round 2")
    winOld2, winNew2, draw2 = c.runArenaInline(previous_weights, new_weights)

    pwins = winOld1 + winOld2
    nwins = winNew1 + winNew2
    draws = draw1 + draw2

    log.info("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))
