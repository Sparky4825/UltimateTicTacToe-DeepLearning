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
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS
from UltimateTicTacToe.UltimateTicTacToePlayers import NNetPlayer
from UltimateTicTacToe.keras.NNet import args


import asyncio

log = logging.getLogger(__name__)


@ray.remote
class ExecuteEpisodeActor:
    def __init__(
        self, game, nnet_actor, args, arena=False, p1_weights=None, p2_weights=None
    ):
        self.game = game
        self.nnet_actor = nnet_actor
        self.args = args

        # Variables to keep track of batch processing of boards
        self.batch_size = ray.get(nnet_actor.get_batch_size.remote())

        # Allow enough concurrent processes to fill the batch
        self.sem = asyncio.Semaphore(self.batch_size + 5)

        self.prediction_timer_running = False
        self.last_prediction_time = time.time()

        self.run_evaluation = asyncio.Event()
        self.claim_evaluation = asyncio.Event()

        self.prediction_results = None
        self.unclaimed_results = None

        self.pending_evaluations = np.empty((0, 3, 9, 10))

        self.log = logging.getLogger(self.__class__.__name__)

        self.arena = arena
        self.player1_weights = p1_weights
        self.player2_weights = p2_weights
        self.current_weight_set = 1

        coloredlogs.install(level="INFO", logger=self.log)

        if not arena:
            asyncio.create_task(self.prediction_timer())

    def run_batch(self):
        self.log.debug("Requesting batch prediction")
        self.prediction_results = ray.get(
            self.nnet_actor.predict_batch.remote(self.pending_evaluations)
        )

        # Clear out the old boards
        self.pending_evaluations = np.empty((0, 3, 9, 10))

        # Add the boards need to be claimed
        self.unclaimed_results = np.full(len(self.prediction_results[0]), 1)

        # Tell the awaiting functions that the batch has been processed and they need to claim results
        self.claim_evaluation.clear()
        self.run_evaluation.set()

        self.last_prediction_time = time.time()

        # Flip weights back and forth every time
        if self.arena:
            if self.current_weight_set == 1:
                ray.get(self.nnet_actor.set_weights.remote(self.player2_weights))

            else:
                ray.get(self.nnet_actor.set_weights.remote(self.player1_weights))

            self.current_weight_set = 2 / self.current_weight_set

    async def prediction_timer(self):
        """
        Checks every second to see if the prediction timer
        has run out and a prediction needs to be run, despite
        not having a full batch
        :return:
        """

        self.log.info("PREDICTION TIMER STARTED")

        self.prediction_timer_running = True

        while self.prediction_timer_running:
            self.log.debug("Checking if prediction is needed")
            if (
                time.time() > self.last_prediction_time + 2
                and len(self.pending_evaluations) > 0
            ):
                self.log.info("Prediction is needed")
                self.run_batch()

            else:
                self.log.debug(
                    f"Prediction is not needed - Pending evaluations: {self.pending_evaluations}"
                )
                await asyncio.sleep(2)

    async def request_prediction(self, board):
        """
        Adds the given board to be evaluated on the GPU with the next batch.
        Then it awaits for the result.
        :param board:
        :return:
        """

        self.log.debug(
            f"Requesting prediction {len(self.pending_evaluations)} {self.batch_size}"
        )

        async with self.sem:
            # Update the timer every time a new prediction is requested
            self.last_prediction_time = time.time()

            if self.run_evaluation.is_set():
                self.log.debug(
                    "Waiting for another process to claim results to add prediction"
                )
                self.log.debug(f"Unclaimed results: {self.unclaimed_results}")
                await self.claim_evaluation.wait()

            # Add the board to the list of predictions to be made
            self.pending_evaluations = np.append(
                self.pending_evaluations, board[np.newaxis, :, :], axis=0
            )

            # Save the board index locally to remember which results go with this board after predictions are calculated
            board_index = len(self.pending_evaluations) - 1

            # Check if its time to run a batch
            if len(self.pending_evaluations) >= self.batch_size:
                self.run_batch()

            else:
                # Wait until the predictions have been made
                await self.run_evaluation.wait()

            # Get and return the results
            self.log.debug(f"Prediction results: {self.prediction_results}")
            pi, v = (
                self.prediction_results[0][board_index],
                self.prediction_results[1][board_index],
            )
            self.unclaimed_results[board_index] = 0

            # Check if all the results have been claimed
            if not np.any(self.unclaimed_results):

                # If they have, allow the next set of boards to be setup
                self.claim_evaluation.set()
                self.run_evaluation.clear()

            return pi, v

    async def executeMultipleEpisodes(self, num_episodes):
        """
        Will start multiple executeEpisodes at once, concurrently.
        :param num_episodes:
        :return:
        """

        # Update batch size to the correct size (default is equal to training value for NN)
        self.batch_size = num_episodes

        group = await asyncio.gather(
            *[self.executeEpisode() for _ in range(int(num_episodes))]
        )

        if not self.arena:
            return group

        else:
            return [group.count(1), group.count(-1), group.count(0)]

    async def executeEpisode(self):
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

        mcts = MCTS(self, self.game, self.nnet_actor, self.args)

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)

            if not self.arena:
                temp = int(episodeStep < self.args.tempThreshold)
            else:
                temp = 0

            pi = await mcts.getActionProb(canonicalBoard, temp=temp)

            if not self.arena:
                sym = self.game.getSymmetries(canonicalBoard, pi)
                for b, p in sym:
                    trainExamples.append([b, curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

            r = self.game.getGameEnded(board, curPlayer)

            if r != 0:
                self.batch_size -= (
                    1  # No longer wait for this game to be present in batch
                )
                print(f"GAME COMPLETE - {self.batch_size} remaining")

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

                # for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                # Create the actors to run the episodes
                workers = []
                for _ in range(self.args.numCPUForMCTS):
                    workers.append(
                        ExecuteEpisodeActor.remote(self.game, self.nnet, self.args)
                    )

                pool = ray.util.ActorPool(workers)

                # Each pool will execute BATCH_SIZE games in a session
                for poolResult in pool.map(
                    lambda a, v: a.executeMultipleEpisodes.remote(v),
                    [self.args.CPUBatchSize]
                    * int(self.args.numEps / self.args.CPUBatchSize),
                ):
                    # Run the episodes at once
                    for trainingPositions in poolResult:
                        log.info(f"New training positions {len(trainingPositions)}")
                        iterationTrainExamples.extend(trainingPositions)
                log.info(
                    f"Self-games complete with {len(iterationTrainExamples)} positions to train from"
                )

                # Kill the workers because they are no longer needed
                for worker in workers:
                    ray.kill(worker)

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
