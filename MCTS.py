import logging
import math
import time

import coloredlogs
import numpy as np
import ray

import asyncio

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args, tflite, use_async=True):
        self.game = game
        self.nnet = nnet  # Reference to ray actor responsible for NN
        self.tflite = tflite
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

        self.use_async = use_async

        self.log = logging.getLogger(self.__class__.__name__)

        coloredlogs.install(level="INFO", logger=self.log)

    def getActionProb(self, canonicalBoard, temp=1, useNNPolicy=True):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard, useNNPolicy)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [
            self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
            for a in range(self.game.getActionSize())
        ]

        for index in range(len(counts)):
            if counts[index] >= 1 and canonicalBoard[2][int(index / 9)][index % 9] == 0:
                self.log.warning("MCTS suggesting invalid move")
                self.log.warning("Board: " + str(canonicalBoard))
                self.log.warning("COUNTS " + str(counts))
                self.game.display(canonicalBoard)
                self.log.warning(self.game.getValidMoves(canonicalBoard, 1))
                break

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, useNNPolicy=True):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            # Wait until the prediction is made, allowing other trees to be searched in the meantime

            t1 = time.time()
            self.Ps[s], v = self.tflite.predict(canonicalBoard)
            t2 = time.time()

            log.debug(f"Prediction made in {t2 - t1} seconds")

            valids = self.game.getValidMoves(canonicalBoard, 1)
            if useNNPolicy:
                self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
                sum_Ps_s = np.sum(self.Ps[s])
                if sum_Ps_s > 0:
                    self.Ps[s] /= sum_Ps_s  # renormalize
                else:
                    # if all valid moves were masked make all valid moves equally probable

                    # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                    # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                    log.error("All valid moves were masked, doing a workaround.")
                    # self.game.display(canonicalBoard)
                    self.Ps[s] = self.Ps[s] + valids
                    self.Ps[s] /= np.sum(self.Ps[s])

            # Don't use the NNet policy (because it isn't trained enough yet) just pick random valid moves
            else:
                self.Ps[s] = self.Ps[s] * 0
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float("inf")
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(
                        self.Ns[s]
                    ) / (1 + self.Nsa[(s, a)])
                else:
                    u = (
                        self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                    )  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s, useNNPolicy)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                self.Nsa[(s, a)] + 1
            )
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
