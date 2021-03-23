import logging
import os
import sys
from collections import deque
import time
from pickle import Pickler, Unpickler
from random import shuffle

import coloredlogs
import numpy as np
import ray
from ray.util.queue import Queue
from tqdm import tqdm

from Arena import Arena
from EpisodeWorker import ExecuteEpisodeActor
from MCTS import MCTS
from UltimateTicTacToe.UltimateTicTacToePlayers import NNetPlayer
from WorkerOrganizer import WorkerOrganizer

from args import args

import asyncio

log = logging.getLogger(__name__)


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(
        self,
        game,
        nnet,
        args,
        toNNQueue,
        fromNNQueue,
        resultsQueue,
    ):

        self.toNNQueue = toNNQueue
        self.fromNNQueue = fromNNQueue
        self.resultsQueue = resultsQueue

        self.game = game
        self.nnet = nnet  # Reference to ray actor responsible for processing NN
        # self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.trainExamplesHistory = (
            []
        )  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f"Starting Iter #{i} ...")
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                workerOrganizer = WorkerOrganizer(
                    self.game,
                    args.numCPUForMCTS,
                    self.toNNQueue,
                    self.fromNNQueue,
                    self.resultsQueue,
                    self.nnet,
                )

                workerOrganizer.runWorkers()

                log.info("Games complete, stopping workers")

                # Release the workers
                del workerOrganizer

                # Drain the results queue
                while not self.resultsQueue.empty():
                    iterationTrainExamples.extend(self.resultsQueue.get())

                log.info(
                    f"Self-games complete with {len(iterationTrainExamples)} positions to train from"
                )

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if (
                len(self.trainExamplesHistory)
                > self.args.numItersForTrainExamplesHistory
            ):
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}"
                )
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            log.info(f"About to begin training with {len(trainExamples)} samples")

            # training new network, keeping a copy of the old one
            previous_weights = ray.get(self.nnet.get_weights.remote())

            ray.get(
                self.nnet.save_checkpoint.remote(
                    folder=self.args.checkpoint, filename="temp.pth.tar"
                )
            )
            # self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            # previous_player = NNetPlayer(
            #     self.game, self.nnet, previous_weights, self.args
            # )

            ray.get(self.nnet.train.remote(trainExamples))
            log.info("TRAINING COMPLETE")

            new_weights = ray.get(self.nnet.get_weights.remote())

            # new_player = NNetPlayer(self.game, self.nnet, new_weights, self.args)

            log.info("PITTING AGAINST PREVIOUS VERSION")
            # arena = Arena(
            #     lambda x: previous_player.get_move(x),
            #     lambda x: new_player.get_move(x),
            #     self.game,
            # )

            # Have both models play as both sides
            log.info("Starting Arena round 1")
            arenaActor1 = ExecuteEpisodeActor.remote(
                self.game,
                self.nnet,
                self.args,
                arena=True,
                p1_weights=previous_weights,
                p2_weights=new_weights,
            )

            pwins1, nwins1, draws1 = ray.get(
                arenaActor1.executeMultipleEpisodes.remote(self.args.arenaCompare / 2)
            )

            ray.kill(arenaActor1)

            log.info("Starting Arena round 2")
            arenaActor2 = ExecuteEpisodeActor.remote(
                self.game,
                self.nnet,
                self.args,
                arena=True,
                p1_weights=new_weights,
                p2_weights=previous_weights,
            )

            nwins2, pwins2, draws2 = ray.get(
                arenaActor2.executeMultipleEpisodes.remote(self.args.arenaCompare / 2)
            )

            pwins = pwins1 + pwins2
            nwins = nwins1 + nwins2
            draws = draws1 + draws2

            log.info("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))
            if (
                pwins + nwins == 0
                or float(nwins) / (pwins + nwins) < self.args.updateThreshold
            ):
                log.info("REJECTING NEW MODEL")
                ray.get(self.nnet.set_weights.remote(previous_weights))
            else:
                log.info("ACCEPTING NEW MODEL")
                ray.get(self.nnet.set_weights.remote(previous_weights))

                ray.get(
                    self.nnet.save_checkpoint.remote(
                        folder=self.args.checkpoint, filename="best.pth.tar"
                    )
                )

    def getCheckpointFile(self, iteration):
        return "checkpoint_" + str(iteration) + ".pth.tar"

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(
            self.args.load_folder_file[0], self.args.load_folder_file[1]
        )
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            # r = input("Continue? [y|n]")
            r = "y"
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info("Loading done!")

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
