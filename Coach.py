import datetime
import logging
import os
import sys
import time
from pickle import Pickler, Unpickler

import coloredlogs
import numpy as np
from MCTS import (
    PyMCTS,
    PyGameState,
    prepareBatch,
    batchResults,
    runSelfPlayEpisodes,
)
from display import display

log = logging.getLogger(__name__)


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet  # Reference to ray actor responsible for processing NN
        self.args = args
        self.trainExamplesHistory = (
            []
        )  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overridden in loadTrainExamples()
        self.log = logging.getLogger(self.__class__.__name__)
        coloredlogs.install(level="INFO", logger=self.log)

        fh = logging.FileHandler(
            f'Training Log - {datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log'
        )
        fh.setLevel("DEBUG")
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s"
            )
        )

        self.log.addHandler(fh)

    def profileEpisodesInline(self):

        episodes = []
        # Start all episodes
        for i in range(self.args.CPUBatchSize):
            episodes.append(PyMCTS(self.args.cpuct))
            g = PyGameState()
            episodes[-1].startNewSearch(g)

        print("Staring move")
        overallStart = time.time()

        for _ in range(self.args.numMCTSSims):
            needsEval = prepareBatch(episodes)

            pi, v = self.nnet.predict_on_batch(needsEval)
            batchResults(episodes, pi, v)

        overallEnd = time.time()
        elapsed = round((overallEnd - overallStart), 2)
        total = self.args.numMCTSSims * self.args.CPUBatchSize
        print(
            f"{total} boards were processed in {elapsed} seconds at {round(total / elapsed, 2)} boards/sec [INLINE]"
        )

    def runArenaInline(self, player1Weights, player2Weights):
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
        for i in range(int(self.args.arenaCompare / 2)):
            p1Episodes.append(PyMCTS(self.args.cpuct))
            g = PyGameState()
            p1Episodes[-1].startNewSearch(g)

            p2Episodes.append(PyMCTS(self.args.cpuct))
            g = PyGameState()
            p2Episodes[-1].startNewSearch(g)

        actionsTaken = 0

        # Loop until all episodes are complete
        while len(p1Episodes) > 0:

            self.nnet.set_weights(player1Weights)
            for _ in range(self.args.numMCTSSims):
                needsEval = prepareBatch(p1Episodes)

                pi, v = self.nnet.predict_on_batch(needsEval)
                print("=" * 10)
                display(needsEval[0][0])
                # print(pi)
                # print(needsEval[1][0])
                print(v[0])

                # v *= -1
                batchResults(p1Episodes, pi, v)
            index = -1

            actionsTaken += 1
            print(f"Taking action {actionsTaken}")

            indexesToRemove = []

            for ep in p1Episodes:
                index += 1
                ep2 = p2Episodes[index]

                # If on first game, display results
                pi = np.array(ep.getActionProb())
                pi /= sum(pi)

                # Choose action randomly if within tempThreshold
                if actionsTaken <= self.args.tempThreshold:
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
                ep2.takeAction(action)

                status = ep.getStatus()
                # Remove episode and save results when the game is over
                if status != 0:
                    print(
                        f"Game over - {len(p1Episodes) - (len(indexesToRemove) + 1)} remaining"
                    )
                    if status == 1:
                        results.append(1)
                    elif status == 2:
                        results.append(-1)
                    else:
                        results.append(0)

                    indexesToRemove.append(index)

            for index in sorted(indexesToRemove, reverse=True):
                del p1Episodes[index]
                del p2Episodes[index]

            if len(p1Episodes) == 0:
                break

            self.nnet.set_weights(player2Weights)
            for _ in range(self.args.numMCTSSims):
                needsEval = prepareBatch(p2Episodes)
                pi, v = self.nnet.predict_on_batch(needsEval)

                # v *= -1
                batchResults(p2Episodes, pi, v)
            index = -1

            actionsTaken += 1
            print(f"Taking action {actionsTaken}")

            indexesToRemove = []

            for ep in p2Episodes:
                index += 1
                ep2 = p1Episodes[index]
                pi = np.array(ep.getActionProb())
                pi /= sum(pi)

                # Choose action randomly if within tempThreshold
                if actionsTaken <= self.args.tempThreshold:
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

                # print(pi)
                # print(ep.gameToString())

                ep.takeAction(action)
                ep2.takeAction(action)

                status = ep.getStatus()
                # Remove episode and save results when the game is over
                if status != 0:
                    print(
                        f"Game over - {len(p1Episodes) - (len(indexesToRemove) + 1)} remaining"
                    )
                    if status == 1:
                        results.append(1)
                    elif status == 2:
                        results.append(-1)
                    else:
                        results.append(0)
                    indexesToRemove.append(index)

            for index in sorted(indexesToRemove, reverse=True):
                del p1Episodes[index]
                del p2Episodes[index]

        return results.count(1), results.count(-1), results.count(0)

    def runEpisodesCpp(self):
        return runSelfPlayEpisodes(
            self.nnet.predict_on_batch,
            self.args.CPUBatchSize,
            self.args.numCPUForMCTS,
            self.args.numMCTSSims,
            self.args.pastTrainingIterations,
            self.args.cpuct,
            self.args.dir_a,
            self.args.dir_x,
            self.args.q_percent,
        )

    def learnIterations(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenOfQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            self.log.info(f"Starting Iter #{i} ...")

            self.log.info("Starting self-play")
            # inputs, pis, vs = self.runEpisodesRemote()

            if i > 1 or not self.args.skipFirstSelfPlay:
                inputs, pis, vs = self.runEpisodesCpp()

                with open("trainingData.examples", "wb+") as f:
                    Pickler(f).dump((inputs, pis, vs))

            else:
                self.log.info("Loading examples from file...")
                with open("trainingData.examples", "rb") as f:
                    inputs, pis, vs = Unpickler(f).load()

            self.log.info(f"About to begin training with {len(inputs[0])} samples")

            # TODO: Reduce the number of draws in training examples because they confuse the network
            # Save the rng_state to shuffle each array the say way
            assert inputs[0].shape[0] == pis.shape[0]
            assert inputs[0].shape[0] == vs.shape[0]
            rng_state = np.random.get_state()
            np.random.shuffle(inputs[0])
            np.random.set_state(rng_state)

            np.random.shuffle(inputs[1])
            np.random.set_state(rng_state)

            np.random.shuffle(pis)
            np.random.set_state(rng_state)

            np.random.shuffle(vs)

            if self.args.maxlenOfQueue < len(inputs):
                inputs = inputs[: self.args.maxlenOfQueue]
                pis = pis[: self.args.maxlenOfQueue]
                vs = vs[: self.args.maxlenOfQueue]
                self.log.info(f"Only training with {self.args.maxlenOfQueue} samples")

            # training new network, keeping a copy of the old one
            previous_weights = self.nnet.get_weights()

            self.nnet.save_checkpoint(folder="./temp", filename="temp.ckpt")

            self.nnet.train(inputs, pis, vs)

            # self.log.info("Starting NN train #2")
            #
            # self.nnet.train(inputs, pis, vs)

            new_weights = self.nnet.get_weights()

            log.info("PITTING AGAINST PREVIOUS VERSION")

            log.info("Starting Arena Round 1")

            winNew1, winOld1, draw1 = self.runArenaInline(new_weights, previous_weights)

            log.info(
                "ROUND 1 NEW/PREV WINS : %d / %d ; DRAWS : %d"
                % (winNew1, winOld1, draw1)
            )

            log.info("Starting Arena Round 2")
            winOld2, winNew2, draw2 = self.runArenaInline(previous_weights, new_weights)

            log.info(
                "ROUND 2 NEW/PREV WINS : %d / %d ; DRAWS : %d"
                % (winNew2, winOld2, draw2)
            )

            pwins = winOld1 + winOld2
            nwins = winNew1 + winNew2
            draws = draw1 + draw2

            log.info("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))

            if (
                pwins + nwins == 0
                or float(nwins) / (pwins + nwins) < self.args.updateThreshold
            ):
                log.info("REJECTING NEW MODEL")
                self.nnet.set_weights(previous_weights)
            else:
                log.info("ACCEPTING NEW MODEL")
                self.nnet.set_weights(new_weights)

                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename="best.ckpt"
                )

    @staticmethod
    def getCheckpointFile(iteration):
        return "checkpoint_" + str(iteration) + ".pth.tar"

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

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

    def loadExamplesIteration(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        examplesFile = os.path.join(
            folder, self.getCheckpointFile(iteration) + ".examples"
        )

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
