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
from MCTS import MCTS
from UltimateTicTacToe.UltimateTicTacToePlayers import NNetPlayer
from UltimateTicTacToe.keras.NNet import args

from LiteModel import LiteModel

import asyncio

log = logging.getLogger(__name__)


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

    # def run_batch(self):
    #     self.log.debug("Requesting batch prediction")
    #     self.prediction_results = self.nnet_actor.predict_batch(
    #         self.pending_evaluations
    #     )
    #
    #     # Clear out the old boards
    #     self.pending_evaluations = np.empty((0, 3, 9, 10))
    #
    #     # Add the boards need to be claimed
    #     self.unclaimed_results = np.full(len(self.prediction_results[0]), 1)
    #
    #     # Tell the awaiting functions that the batch has been processed and they need to claim results
    #     self.claim_evaluation.clear()
    #     self.run_evaluation.set()
    #
    #     self.last_prediction_time = time.time()
    #
    #     # Flip weights back and forth every time
    #     if self.arena:
    #         if self.current_weight_set == 1:
    #             self.nnet_actor.set_weights(self.player2_weights)
    #
    #         else:
    #             self.nnet_actor.set_weights(self.player1_weights)
    #
    #         self.current_weight_set = 2 / self.current_weight_set
    #
    # async def prediction_timer(self):
    #     """
    #     Checks every second to see if the prediction timer
    #     has run out and a prediction needs to be run, despite
    #     not having a full batch
    #     :return:
    #     """
    #
    #     self.log.info("PREDICTION TIMER STARTED")
    #
    #     self.prediction_timer_running = True
    #
    #     while self.prediction_timer_running:
    #         self.log.debug("Checking if prediction is needed")
    #         if (
    #             time.time() > self.last_prediction_time + 2
    #             and len(self.pending_evaluations) > 0
    #         ):
    #             self.log.info("Prediction is needed")
    #             self.run_batch()
    #
    #         else:
    #             self.log.debug(
    #                 f"Prediction is not needed - Pending evaluations: {self.pending_evaluations}"
    #             )
    #             await asyncio.sleep(2)
    #
    # async def request_prediction(self, board):
    #     """
    #     Adds the given board to be evaluated on the GPU with the next batch.
    #     Then it awaits for the result.
    #     :param board:
    #     :return:
    #     """
    #
    #     self.log.debug(
    #         f"Requesting prediction {len(self.pending_evaluations)} {self.batch_size}"
    #     )
    #
    #     async with self.sem:
    #         # Update the timer every time a new prediction is requested
    #         self.last_prediction_time = time.time()
    #
    #         if self.run_evaluation.is_set():
    #             self.log.debug(
    #                 "Waiting for another process to claim results to add prediction"
    #             )
    #             self.log.debug(f"Unclaimed results: {self.unclaimed_results}")
    #             await self.claim_evaluation.wait()
    #
    #         # Add the board to the list of predictions to be made
    #         self.pending_evaluations = np.append(
    #             self.pending_evaluations, board[np.newaxis, :, :], axis=0
    #         )
    #
    #         # Save the board index locally to remember which results go with this board after predictions are calculated
    #         board_index = len(self.pending_evaluations) - 1
    #
    #         # Check if its time to run a batch
    #         if len(self.pending_evaluations) >= self.batch_size:
    #             self.run_batch()
    #
    #         else:
    #             # Wait until the predictions have been made
    #             await self.run_evaluation.wait()
    #
    #         # Get and return the results
    #         self.log.debug(f"Prediction results: {self.prediction_results}")
    #         pi, v = (
    #             self.prediction_results[0][board_index],
    #             self.prediction_results[1][board_index],
    #         )
    #         self.unclaimed_results[board_index] = 0
    #
    #         # Check if all the results have been claimed
    #         if not np.any(self.unclaimed_results):
    #
    #             # If they have, allow the next set of boards to be setup
    #             self.claim_evaluation.set()
    #             self.run_evaluation.clear()
    #
    #         return pi, v

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
            results.put(self.executeEpisode())

    def executeEpisode(self):
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
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        mcts = MCTS(
            self,
            self.game,
            None,
            self.args,
            self.interpreter,
        )

        if self.arena:
            mcts2 = MCTS(
                self,
                self.game,
                None,
                self.args,
                self.interpreter2,
            )

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)

            if not self.arena:
                temp = int(episodeStep < self.args.tempThreshold)
            else:
                temp = 0

            # Use the correct tree
            if not self.arena or curPlayer == 1:
                pi = mcts.getActionProb(canonicalBoard, temp=temp)
            else:
                pi = mcts2.getActionProb(canonicalBoard, temp=temp)

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

        for _ in tqdm(range(numEps)):
            if arena:
                results.append(resultsQueue.get())
            else:
                results.extend(resultsQueue.get())

        for worker in workers:
            ray.kill(worker)

        if arena:
            return [results.count(1), results.count(-1), results.count(0)]

        return results

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

            self.log.info("Creating TF-Lite model")
            tflite_model = self.nnet.convert_to_tflite()
            self.log.info("TF-Lite model done")

            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
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
