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
from ray.util.queue import Queue
from tqdm import tqdm

from Arena import Arena
from MCTS import PyMCTS, PyGameState, prepareBatch, batchResults
from UltimateTicTacToe.UltimateTicTacToePlayers import NNetPlayer
from UltimateTicTacToe.keras.NNet import args

from LiteModel import LiteModel

import asyncio

log = logging.getLogger(__name__)


@ray.remote
class MCTSBatchActor:
    def __init__(self, id, options, toNNQueue, fromNNQueue):
        self.id = id
        self.batchSize = options.CPUBatchSize
        self.cpuct = options.cpuct
        self.numMCTSSims = options.numMCTSSims

        self.toNNQueue = toNNQueue
        self.fromNNQueue = fromNNQueue

    async def start(self):
        episodes = []
        # Start all episodes
        for i in range(self.batchSize):
            episodes.append(PyMCTS(self.cpuct))
            g = PyGameState()
            episodes[-1].startNewSearch(g)

        # Loop until all episodes are complete
        while True:
            print("Staring move")
            for _ in range(self.numMCTSSims):
                # print("Stargin simulation")
                needsEval = prepareBatch(episodes)
                self.toNNQueue.put((self.id, needsEval))

                evalResult = await self.fromNNQueue.get_async()
                pi, v = evalResult
                batchResults(episodes, pi, v)

            for ep in episodes:
                pi = np.array(ep.getActionProb())
                print("PI")
                print(pi)
                pi /= sum(pi)

                # Correct a slight rounding error if necessary
                if sum(pi) != 1:
                    print("CORRECTING ERROR")
                    print(pi)
                    mostLikelyIndex = np.argmax(pi)
                    pi[mostLikelyIndex] += 1 - sum(pi)

                print("Making move")
                action = np.random.choice(len(pi), p=pi)
                ep.takeAction(action)
                # TODO: Save boards for training
                print(ep.gameToString())

                status = ep.getStatus()
                if status != 0:
                    print("Game over removing it")
                    # TODO: Save results for training
                    episodes.remove(ep)


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

    def learnContinuous(self):
        """
        Continuously plays games against itself and learns after every game.
        """

        self.log.info("Starting continuous learning ...")

        self.log.info("Loading previous examples")
        self.loadExamplesIteration(4)

        gamesLoaded = sum(len(c) for c in self.trainExamplesHistory)

        if gamesLoaded >= self.args.maxlenOfQueue:
            self.log.info(
                f"{gamesLoaded} previous examples loaded, skipping pre-training games"
            )
        else:
            self.log.info(
                f"{gamesLoaded} previous examples loaded, pre-training games are needed"
            )

            self.log.info("Starting pre-training games")

            self.log.info("Creating TF-Lite model")
            tflite_model = self.nnet.convert_to_tflite()
            self.log.info("TF-Lite model done")

            with open(
                f"litemodels/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.tflite",
                "wb",
            ) as f:
                f.write(tflite_model)

            # Have enough games to start training with
            gamesQueue = Queue()
            resultsQueue = Queue()

            iterationTrainExamples = self.runEpisodes(
                gamesQueue,
                resultsQueue,
                False,
                self.args.numEps,
                self.args.numCPUForMCTS,
                self.game,
                self.args,
                tflite_model,
            )

            self.trainExamplesHistory.append(iterationTrainExamples)

            self.log.info("Pre-training games complete")

        gamesCount = 0
        while True:

            self.log.info("Creating TF-Lite model")
            tflite_model = self.nnet.convert_to_tflite()
            self.log.info("TF-Lite model done")

            with open(
                f"litemodels/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.tflite",
                "wb",
            ) as f:
                f.write(tflite_model)

            gamesCount += 1
            self.log.info(f"Start training game-set #{gamesCount}")

            gamesQueue = Queue()
            resultsQueue = Queue()

            iterationTrainExamples = self.runEpisodes(
                gamesQueue,
                resultsQueue,
                False,
                self.args.numCPUForMCTS,  # Run one game per CPU being used
                self.args.numCPUForMCTS,
                self.game,
                self.args,
                tflite_model,
            )

            self.log.info(
                f"Game-set complete with {len(iterationTrainExamples)} new positions"
            )

            self.log.info("Begin pre-fitting processing")

            # TODO: Remove duplicate positions and average their results

            self.trainExamplesHistory.append(iterationTrainExamples)
            del iterationTrainExamples

            trainExamples = []
            index = len(self.trainExamplesHistory)
            while len(trainExamples) < self.args.maxlenOfQueue and index > 0:
                index -= 1

                trainExamples.extend(self.trainExamplesHistory[index])

            # Remove positions that are too old to be useful
            self.trainExamplesHistory = self.trainExamplesHistory[index:]

            # Save examples for later use
            self.saveTrainExamples(gamesCount)

            shuffle(trainExamples)

            self.log.info("Pre-fitting processing done")

            self.log.info("Begin fitting")
            self.nnet.train(trainExamples, epochs=2)
            self.log.info("Fitting complete")

            self.nnet.save_checkpoint(
                folder=self.args.checkpoint,
                filename=f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.pth.tar",
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

            self.log.info("Creating TF-Lite model")
            tflite_model = self.nnet.convert_to_tflite()
            self.log.info("TF-Lite model done")

            with open(
                f"litemodels/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.tflite",
                "wb",
            ) as f:
                f.write(tflite_model)

            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                workers = []
                fromNNQueues = []
                toNNQueue = Queue()
                for i in range(self.args.numCPUForMCTS):
                    fromNNQueues.append(Queue())
                    workers.append(
                        MCTSBatchActor.remote(
                            len(workers), self.args, toNNQueue, fromNNQueues[-1]
                        )
                    )

                    workers[-1].start.remote()

                # Loop until all games are done
                while True:
                    workerID, needsEval = toNNQueue.get()
                    pi, v = self.nnet.predict_on_batch(needsEval)
                    fromNNQueues[workerID].put((pi, v))

                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                # for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                # Create the actors to run the episodes

                gamesQueue = Queue()
                resultsQueue = Queue()

                iterationTrainExamples = self.runEpisodes(
                    gamesQueue,
                    resultsQueue,
                    False,
                    self.args.numEps,
                    self.args.numCPUForMCTS,
                    self.game,
                    self.args,
                    tflite_model,
                )

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

            for j in range(1000):
                counts = trainExamples[j][1]
                board = trainExamples[j][0]

                for index in range(len(counts)):
                    if counts[index] >= 1 and board[2][int(index / 9)][index % 9] == 0:
                        self.log.warning("MCTS suggesting invalid move")
                        self.log.warning(trainExamples[j])
                        self.log.warning("Board: " + str(board))
                        self.log.warning("COUNTS " + str(counts))
                        self.game.display(board)
                        self.log.warning(self.game.getValidMoves(board, 1))
                        break

            # TODO: Reduce the number of draws in training examples because they confuse the network
            shuffle(trainExamples)

            log.info(f"About to begin training with {len(trainExamples)} samples")

            # training new network, keeping a copy of the old one
            previous_weights = self.nnet.get_weights()

            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )

            # self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            # previous_player = NNetPlayer(
            #     self.game, self.nnet, previous_weights, self.args
            # )

            self.nnet.train(trainExamples)

            # Release the RAM for use in the arena competition
            del trainExamples
            # self.trainExamplesHistory.pop(0)
            # TODO: this must be changed to ever use training examples from the past

            log.info("TRAINING COMPLETE")

            log.info("Creating new TF-Lite model")
            new_tflite_model = self.nnet.convert_to_tflite()
            self.log.info("TF-Lite model done")

            new_weights = self.nnet.get_weights()

            # new_player = NNetPlayer(self.game, self.nnet, new_weights, self.args)

            log.info("PITTING AGAINST PREVIOUS VERSION")
            # arena = Arena(
            #     lambda x: previous_player.get_move(x),
            #     lambda x: new_player.get_move(x),
            #     self.game,
            # )

            # Have both models play as both sides
            log.info("Starting Arena round 1")

            gamesQueue = Queue()
            resultsQueue = Queue()

            pwins1, nwins1, draws1 = self.runEpisodes(
                gamesQueue,
                resultsQueue,
                True,
                self.args.arenaCompare,
                self.args.numCPUForMCTS,
                self.game,
                self.args,
                tflite_model,
                True,
                new_tflite_model,
            )

            log.info("Starting Arena round 2")

            gamesQueue = Queue()
            resultsQueue = Queue()

            nwins2, pwins2, draws2 = self.runEpisodes(
                gamesQueue,
                resultsQueue,
                True,
                self.args.arenaCompare,
                self.args.numCPUForMCTS,
                self.game,
                self.args,
                new_tflite_model,
                True,
                tflite_model,
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
                self.nnet.set_weights(previous_weights)
            else:
                log.info("ACCEPTING NEW MODEL")
                self.nnet.set_weights(new_weights)

                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename="best.pth.tar"
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
