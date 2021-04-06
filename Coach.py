import datetime
import logging
import os
import random
import sys
from collections import deque
import time
from pickle import Pickler, Unpickler
from random import shuffle

import coloredlogs
import numpy as np
import ray
from ray.util.queue import Queue, Empty
from tqdm import tqdm

from Arena import Arena
from MCTS import (
    PyMCTS,
    PyGameState,
    prepareBatch,
    batchResults,
    compileExamples,
    runSelfPlayEpisodes,
)
from UltimateTicTacToe.UltimateTicTacToePlayers import NNetPlayer
from UltimateTicTacToe.keras.NNet import args

from LiteModel import LiteModel

import asyncio

log = logging.getLogger(__name__)


@ray.remote
class MCTSBatchActor:
    def __init__(self, id, options, toNNQueue, fromNNQueue, resultsQueue):
        self.id = id
        self.batchSize = options.CPUBatchSize
        self.cpuct = options.cpuct
        self.numMCTSSims = options.numMCTSSims

        self.toNNQueue = toNNQueue
        self.fromNNQueue = fromNNQueue
        self.resultsQueue = resultsQueue

        self.profile = False
        self.numIterations = 2

    def start(self):
        episodes = []
        # Start all episodes
        for i in range(self.batchSize):
            episodes.append(PyMCTS(self.cpuct))
            g = PyGameState()
            episodes[-1].startNewSearch(g)

        results = []
        actionsTaken = 0
        completeIterations = 0
        self.going = True
        # Loop until all episodes are complete
        while len(episodes) > 0 and self.going:
            for _ in range(self.numMCTSSims):
                # print("Stargin simulation")
                needsEval = prepareBatch(episodes)
                self.toNNQueue.put((self.id, needsEval))
                t1 = time.time()
                evalResult = self.fromNNQueue.get()
                t2 = time.time()
                # print(f"Worker waited on queue for {round((t2 - t1) * 1000, 4)} millis")
                pi, v = evalResult
                batchResults(episodes, pi, v)

            actionsTaken += 1
            print(f"Taking action {actionsTaken}")

            for ep in episodes:
                pi = np.array(ep.getActionProb())
                pi /= sum(pi)

                # Correct a slight rounding error if necessary
                if sum(pi) != 1:
                    # print("CORRECTING ERROR")
                    # print(pi)
                    mostLikelyIndex = np.argmax(pi)
                    pi[mostLikelyIndex] += 1 - sum(pi)

                action = np.random.choice(len(pi), p=pi)
                ep.takeAction(action)
                ep.saveTrainingExample(pi)

                status = ep.getStatus()
                # Remove episode and save results when the game is over
                if status != 0:
                    print(f"Game over ({self.id})- {len(episodes) - 1} remaining")
                    if status == 1:
                        results.append(ep.getTrainingExamples(1))
                    elif status == 2:
                        results.append(ep.getTrainingExamples(-1))
                    else:
                        results.append(ep.getTrainingExamples(0))
                    episodes.remove(ep)

            if self.profile:
                completeIterations += 1

                if completeIterations >= self.numIterations:
                    self.going = False
                    print("Exiting because profile is enabled")
                    break
        self.resultsQueue.put(results)


def test2():
    g = PyGameState()
    tree = PyMCTS(1)

    tree.startNewSearch(g)

    def evaluate(*args):
        policy = np.full(81, 1 / 81, dtype=np.float)

        value = round(random.random() * random.choice([-1, 1]), 4)
        return policy, value

    prepareBatch([tree])

    pi, v = evaluate()
    batchResults([tree], [pi], [v])

    tree.takeAction(0)

    print(tree.gameToString())

    print("Preparding 2nd batch")
    prepareBatch([tree])
    print("Done")

    pi, v = evaluate()
    print("BATCH RESULTS")
    batchResults([tree], [pi], [v])
    print("DONE")

    print("TAKING ACTION")
    tree.takeAction(1)

    print(tree.gameToString())


def testNewMCTS():
    def evaluate(*args):
        policy = np.full(81, 1 / 81, dtype=np.float)

        value = round(random.random() * random.choice([-1, 1]), 4)
        return policy, value

    class Options:
        def __init__(self):
            self.CPUBatchSize = 1
            self.cpuct = 1
            self.numMCTSSims = 81

    toNNQueue = Queue()
    fromNNQueue = Queue()

    worker = MCTSBatchActor.remote(Options(), toNNQueue, fromNNQueue)

    workerTask = worker.start.remote()

    while True:
        toBeEval = toNNQueue.get()
        pi, v = evaluate()
        fromNNQueue.put([[pi], [v]])


@ray.remote
class ExecuteEpisodeActor:
    def __init__(
        self,
        game,
        args,
        tfliteModel,
        arena=False,
        newTFliteModel=None,
    ):
        import tensorflow as tf

        self.game = game
        self.nnet_actor = None
        self.args = args

        self.interpreter = LiteModel(tf.lite.Interpreter(model_content=tfliteModel))

        # If arena is set, there must be a second tflite model
        assert not arena or newTFliteModel is not None

        if newTFliteModel is not None:
            self.interpreter2 = LiteModel(
                tf.lite.Interpreter(model_content=newTFliteModel)
            )

        self.prediction_timer_running = False
        self.last_prediction_time = time.time()

        self.run_evaluation = asyncio.Event()
        self.claim_evaluation = asyncio.Event()

        self.prediction_results = None
        self.unclaimed_results = None

        self.pending_evaluations = np.empty((0, 3, 9, 10))

        self.log = logging.getLogger(self.__class__.__name__)

        self.arena = arena
        self.current_weight_set = 1

        coloredlogs.install(level="INFO", logger=self.log)

    def executeMultipleEpisodes(self, num_episodes):
        """
        Will start multiple executeEpisodes at once, concurrently.
        :param num_episodes:
        :return:
        """

        # Update batch size to the correct size (default is equal to training value for NN)
        self.batch_size = num_episodes

        group = []
        for i in range(int(num_episodes)):
            group.append(self.executeEpisode())

        # group = await asyncio.gather(
        #     *[self.executeEpisode() for _ in range(int(num_episodes))]
        # )

        if not self.arena:
            return group

        else:
            return [group.count(1), group.count(-1), group.count(0)]

    def executeEpisodesFromQueue(self, queue, results):
        while not queue.empty():
            queue.get()
            results.put(self.executeEpisode(useNNPolicy=True))

    def executeEpisode(self, useNNPolicy=True):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        If arena is set, it will play out with temp=0 and return the game result (1 for player 1, -1 for player 2, 0 for draw)
        The function assumes that the weights will be shifted in between predictions.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        mcts = MCTS(self.game, self.args, self.interpreter)

        if self.arena:
            mcts2 = MCTS(self.game, self.args, self.interpreter2)

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)

            if not self.arena:
                temp = int(episodeStep < self.args.tempThreshold)
            else:
                temp = int(episodeStep < self.args.arenaTempThreshold)

            # Use the correct tree
            if not self.arena or curPlayer == 1:
                pi = mcts.getActionProb(
                    canonicalBoard, temp=temp, useNNPolicy=useNNPolicy
                )
            else:
                pi = mcts2.getActionProb(
                    canonicalBoard, temp=temp, useNNPolicy=useNNPolicy
                )

            if not self.arena:
                sym = self.game.getSymmetries(canonicalBoard, pi)
                for b, p in sym:
                    trainExamples.append([b, curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

            r = self.game.getGameEnded(board, curPlayer)

            if r != 0:

                # print(f"GAME COMPLETE - {self.batch_size} remaining")

                if self.arena:
                    if r == 1:
                        return curPlayer
                    elif r == -1:
                        return -1 * curPlayer
                    else:
                        return 0
                return [
                    (x[0], x[2], r * ((-1) ** (x[1] != curPlayer)))
                    for x in trainExamples
                ]


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet  # Reference to ray actor responsible for processing NN
        # self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.trainExamplesHistory = (
            []
        )  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
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

    def runEpisodes(
        self, remainingGamesQueue, resultsQueue, arena, numEps, numCPU, *args
    ):
        workers = []
        for _ in range(numCPU):
            workers.append(ExecuteEpisodeActor.remote(*args))

        for _ in range(self.args.numEps):
            remainingGamesQueue.put(1)

        for worker in workers:
            worker.executeEpisodesFromQueue.remote(remainingGamesQueue, resultsQueue)
        results = []

        for _ in tqdm(range(numEps), desc="Playing games"):
            if arena:
                results.append(resultsQueue.get())
            else:
                results.extend(resultsQueue.get())

        for worker in workers:
            ray.kill(worker)

        if arena:
            return [results.count(1), results.count(-1), results.count(0)]

        return results

    def profileEpisodesRemote(self):
        workers = []
        fromNNQueues = []
        toNNQueue = Queue()
        resultsQueue = Queue()
        for i in range(self.args.numCPUForMCTS):
            fromNNQueues.append(Queue())
            workers.append(
                MCTSBatchActor.remote(
                    len(workers), self.args, toNNQueue, fromNNQueues[-1], resultsQueue
                )
            )

            workers[-1].start.remote()

        # Loop until all games are done
        overallStart = time.time()
        count = 0
        while True:
            count += 1
            # if toNNQueue.empty():
            #     print("NNET MET WITH EMPTY QUEUE")
            allWorkerID = []
            evalLengths = []
            allNeedsEval = np.ndarray((0, 199))
            if self.args.GPUBatchSize > 0:
                t1 = time.time()
                for _ in range(self.args.GPUBatchSize):
                    workerID, needsEval = toNNQueue.get()
                    allWorkerID.append(workerID)
                    evalLengths.append(len(needsEval))

                    allNeedsEval = np.concatenate((allNeedsEval, needsEval), axis=0)
                t2 = time.time()

            else:
                workerID, allNeedsEval = toNNQueue.get()

            # print(
            #     f"GPU waited on queue for {round((t2 - t1) * 1000, 4)} millis"
            # )

            et1 = time.time()
            pi, v = self.nnet.predict_on_batch(allNeedsEval)
            et2 = time.time()
            print(f"Evaluation of {len(needsEval)} took {et2 - et1} seconds")
            evalLength = 0
            if self.args.GPUBatchSize > 0:
                for i in range(self.args.GPUBatchSize):
                    fromNNQueues[allWorkerID[i]].put(
                        (
                            pi[evalLength : evalLength + evalLengths[i]],
                            v[evalLength : evalLength + evalLengths[i]],
                        )
                    )
                    evalLength += evalLengths[i]

            else:
                fromNNQueues[workerID].put((pi, v))

            if (
                count
                == self.args.numMCTSSims
                * self.args.numCPUForMCTS
                / self.args.GPUBatchSize
            ):
                overallEnd = time.time()
                elapsed = round((overallEnd - overallStart), 2)
                total = (
                    self.args.numMCTSSims
                    * self.args.CPUBatchSize
                    * self.args.numCPUForMCTS
                )
                print(
                    f"{total} boards were processed in {elapsed} seconds at {round(total / elapsed, 2)} boards/sec"
                )
                exit()

    def profileEpisodesInline(self):

        episodes = []
        # Start all episodes
        for i in range(self.args.CPUBatchSize):
            episodes.append(PyMCTS(self.args.cpuct))
            g = PyGameState()
            episodes[-1].startNewSearch(g)

        # Loop until all episodes are complete
        while True:
            print("Staring move")
            overallStart = time.time()

            for _ in range(self.args.numMCTSSims):
                # print("Stargin simulation")
                needsEval = prepareBatch(episodes)

                pi, v = self.nnet.predict_on_batch(needsEval)
                batchResults(episodes, pi, v)

            overallEnd = time.time()
            elapsed = round((overallEnd - overallStart), 2)
            total = self.args.numMCTSSims * self.args.CPUBatchSize
            print(
                f"{total} boards were processed in {elapsed} seconds at {round(total / elapsed, 2)} boards/sec [INLINE]"
            )
            exit()

    def runArenaInline(self, player1Weights, player2Weights):
        """
        Returns in format (player1Wins, player2Wins, draws)
        :param player1Weights:
        :param player2Weights:
        :return:
        """
        from display import display

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
            overallStart = time.time()

            self.nnet.set_weights(player1Weights)
            for _ in range(self.args.numMCTSSims):
                # print("Stargin simulation")
                needsEval = prepareBatch(p1Episodes)

                pi, v = self.nnet.predict_on_batch(needsEval)
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
                    print(f"Game over - {len(p1Episodes) - 1} remaining")
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

            self.nnet.set_weights(player2Weights)
            for _ in range(self.args.numMCTSSims):
                # print("Stargin simulation")
                needsEval = prepareBatch(p2Episodes)

                pi, v = self.nnet.predict_on_batch(needsEval)
                batchResults(p2Episodes, pi, v)
            index = -1

            actionsTaken += 1
            print(f"Taking action {actionsTaken}")
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

                ep.takeAction(action)
                ep2.takeAction(action)

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

    def runEpisodesRemote(self):
        """
        Returns a list of training examples in the form
        [(board, pi, v)]
        """
        workers = []
        fromNNQueues = []
        toNNQueue = Queue()
        resultsQueue = Queue()
        for i in range(self.args.numCPUForMCTS):
            fromNNQueues.append(Queue())
            workers.append(
                MCTSBatchActor.remote(
                    len(workers), self.args, toNNQueue, fromNNQueues[-1], resultsQueue
                )
            )

            workers[-1].start.remote()

        # Loop until all workers have submitted results
        overallStart = time.time()
        count = 0
        done = False
        t1 = time.time()
        while resultsQueue.size() < len(workers):
            count += 1
            # if toNNQueue.empty():
            #     print("NNET MET WITH EMPTY QUEUE")
            allWorkerID = []
            evalLengths = []
            allNeedsEval = np.ndarray((0, 199))
            if self.args.GPUBatchSize > 0:
                for _ in range(self.args.GPUBatchSize):
                    try:
                        workerID, needsEval = toNNQueue.get(timeout=1)
                    except Empty:
                        done = True
                        break

                    allWorkerID.append(workerID)
                    evalLengths.append(len(needsEval))

                    allNeedsEval = np.concatenate((allNeedsEval, needsEval), axis=0)

                if done:
                    done = False
                    continue

            else:
                workerID, allNeedsEval = toNNQueue.get()

            # print(
            #     f"GPU waited on queue for {round((t2 - t1) * 1000, 4)} millis"
            # )

            et1 = time.time()
            pi, v = self.nnet.predict_on_batch(allNeedsEval)
            et2 = time.time()
            # print(f"Evaluation of {len(allNeedsEval)} took {et2 - et1} seconds")
            # print(f"Evaluation of {len(needsEval)} took {t2 - t1} seconds")
            evalLength = 0
            if self.args.GPUBatchSize > 0:
                for i in range(self.args.GPUBatchSize):
                    fromNNQueues[allWorkerID[i]].put(
                        (
                            pi[evalLength : evalLength + evalLengths[i]],
                            v[evalLength : evalLength + evalLengths[i]],
                        )
                    )
                    evalLength += evalLengths[i]

            else:
                fromNNQueues[workerID].put((pi, v))

        t2 = time.time()

        print(f"Execution complete in {t2 - t1} seconds")

        self.log.info("All workers submitted results")
        # Free the workers
        del workers

        trainingExamples = []

        while not resultsQueue.empty():
            self.log.info("Getting results...")
            examplesList = resultsQueue.get()

            for ex in examplesList:
                trainingExamples.append(ex)

        boards, pis, vs = compileExamples(trainingExamples)

        return boards, pis, vs

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
        examples in trainExamples (which has a maximum length of maxlenofQueue).
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

            self.log.info(f"About to begin training with {len(inputs)} samples")

            # TODO: Reduce the number of draws in training examples because they confuse the network
            # Save the rng_state to shuffle each array the say way
            assert inputs.shape[0] == pis.shape[0]
            assert inputs.shape[0] == vs.shape[0]
            rng_state = np.random.get_state()
            np.random.shuffle(inputs)
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
