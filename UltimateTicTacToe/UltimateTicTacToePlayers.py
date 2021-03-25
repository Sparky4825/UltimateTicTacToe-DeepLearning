import asyncio

import numpy as np
import ray

from MCTS import MCTS

"""
Random and Human-ineracting players for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloPlayers by Surag Nair.

"""


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanTicTacToePlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i / self.game.n), int(i % self.game.n))
        while True:
            # Python 3.x
            a = input()
            # Python 2.x
            # a = raw_input()

            x, y = [int(x) for x in a.split(" ")]
            a = self.game.n * x + y if x != -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print("Invalid")

        return a


class NNetPlayer:
    def __init__(self, game, nnet_actor, weights, args):
        self.game = game
        self.nnet_actor = nnet_actor
        self.weights = weights

        self.mcts = MCTS(game, nnet_actor, args, False)

    def get_move(self, board):
        ray.get(self.nnet_actor.set_weights.remote(self.weights))

        results = asyncio.run(self.mcts.getActionProb(board, temp=0))
        return np.argmax(results)
