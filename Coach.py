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

from args import args

import asyncio

log = logging.getLogger(__name__)


class NNetEvalTicket:
    """
    Contains the minimum amount of information the NNet needs
    to give an evaluation.
    """

    def __init__(self, canonicalBoard):
        self.canonicalBoard = canonicalBoard
        self.pi = None
        self.v = None


class Episode:
    """
    Keeps track on a single 'episode' (or game) that the
    computer is playing against itself. Will be passed between
    Episode Actors and NNetActors through a Ray queue.
    """

    def __init__(self, game):
        self.trainExamples = []
        self.board = game.getInitBoard()
        self.curPlayer = 1
        self.episodeStep = 0
        self.game = game

        self.mcts = MCTS(game)

        self.temp = 1


@ray.remote
class ExecuteEpisodeActor:
    def __init__(
        self,
        game,
        nnet_actor,
        args,
        toNNQueue,
        fromNNQueue,
        resultsQueue,
        arena=False,
        p1_weights=None,
        p2_weights=None,
    ):
        self.id = id

        self.game = game
        self.nnet_actor = nnet_actor
        self.args = args

        self.toNNQueue = toNNQueue
        self.fromNNQueue = fromNNQueue
        self.resultsQueue = resultsQueue

        # Variables to keep track of batch processing of boards
        self.batch_size = self.args.CPUBatchSize

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

    def startEpisode(self):
        """
        Starts an episode and adds it to the queue for the NNet Actor to evaluate.
        """

        # Preform the starting operations
        new_episode = Episode(self.game)

        new_episode.episodeStep += 1

        if not self.arena:
            new_episode.temp = int(new_episode.episodeStep < self.args.tempThreshold)
        else:
            new_episode.temp = 0

        assert new_episode.mcts.searchPreNN() is True

        # TODO: This function assumes that the NNet will be needed always for the first move, is this correct?

        # Add it to the queue for NN evaluation
        self.toNNQueue.put(new_episode)

    def checkMoveandMake(self, ep):
        """
        Return true if game is over at move
        :param ep:
        :return:
        """
        if ep.episodeStep >= self.args.numMCTSSims:
            # Get final results and make a move
            print("MAKING A MOVE")
            ep.mcts.getActionProb(ep.temp)

            if not self.arena:
                sym = self.game.getSymmetries(ep.mcts.canonicalBoard, ep.mcts.pi)
                for b, p in sym:
                    ep.trainExamples.append([b, ep.curPlayer, p, None])

            probs = ep.mcts.getActionProb(ep.temp)

            action = np.random.choice(len(probs), p=probs)
            ep.board, ep.curPlayer = self.game.getNextState(
                ep.board, ep.curPlayer, action
            )

            self.game.display(ep.board)

            result = self.game.getGameEnded(ep.board, ep.curPlayer)

            if result != 0:
                # Game complete
                if self.arena:
                    if result == 1:
                        self.resultsQueue.put(ep.curPlayer)
                    elif result == -1:
                        self.resultsQueue.put(-1 * ep.curPlayer)
                    else:
                        self.resultsQueue.put(1e-8)
                else:
                    self.resultsQueue.put(
                        [
                            (
                                x[0],
                                x[2],
                                result * ((-1) ** (x[1] != ep.curPlayer)),
                            )
                            for x in ep.trainExamples
                        ]
                    )
                # Game is over, exit function
                return True

            else:
                # Game not complete - start MCTS search again

                ep.mcts.moveStartBoard2Action(action)
                ep.episodeStep = 0
        return False

    def loopExecuteEpisodeFromQueue(self):
        while True:
            self.executeEpisodeTurnFromQueue()

    def executeEpisodeTurnFromQueue(self):
        """
        Gets a single episode from the queue, executes the next turn and adds
        it back to the other queue.

        During the episode's time between queues, its pi value will have been
        set by the NN actor.
        """

        t1 = time.time()
        ep = self.fromNNQueue.get()
        t2 = time.time()

        # print(f"Waiting on fromNNQueue for {t2 - t1} seconds")

        ep.mcts.searchPostNN()

        # Loop until a NN eval is needed
        while not ep.mcts.searchPreNN():
            ep.episodeStep += 1

            if not self.arena:
                ep.temp = int(ep.episodeStep < self.args.tempThreshold)
            else:
                ep.temp = 0

            if self.checkMoveandMake(ep):
                return

        if self.checkMoveandMake(ep):
            return

        ep.episodeStep += 1
        # Once the while loop is exited, it means a NN evaluation is needed
        self.toNNQueue.put(ep)

    # async def executeEpisode(self):
    #     """
    #     This function executes one episode of self-play, starting with player 1.
    #     As the game is played, each turn is added as a training example to
    #     trainExamples. The game is played till the game ends. After the game
    #     ends, the outcome of the game is used to assign values to each example
    #     in trainExamples.
    #
    #     It uses a temp=1 if episodeStep < tempThreshold, and thereafter
    #     uses temp=0.
    #
    #     If arena is set, it will play out with temp=0 and return the game result (1 for player 1, -1 for player 2, 0 for draw)
    #     The function assumes that the weights will be shifted in between predictions.
    #
    #     Returns:
    #         trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
    #                        pi is the MCTS informed policy vector, v is +1 if
    #                        the player eventually won the game, else -1.
    #     """
    #     trainExamples = []
    #     board = self.game.getInitBoard()
    #     curPlayer = 1
    #     episodeStep = 0
    #
    #     mcts = MCTS(self, self.game, self.nnet_actor, self.args)
    #
    #     while True:
    #         episodeStep += 1
    #         canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
    #
    #         if not self.arena:
    #             temp = int(episodeStep < self.args.tempThreshold)
    #         else:
    #             temp = 0
    #
    #         pi = await mcts.getActionProb(canonicalBoard, temp=temp)
    #
    #         if not self.arena:
    #             sym = self.game.getSymmetries(canonicalBoard, pi)
    #             for b, p in sym:
    #                 trainExamples.append([b, curPlayer, p, None])
    #
    #         action = np.random.choice(len(pi), p=pi)
    #         board, curPlayer = self.game.getNextState(board, curPlayer, action)
    #
    #         r = self.game.getGameEnded(board, curPlayer)
    #
    #         if r != 0:
    #             self.batch_size -= (
    #                 1  # No longer wait for this game to be present in batch
    #             )
    #             print(f"GAME COMPLETE - {self.batch_size} remaining")
    #
    #             if self.arena:
    #                 if r == 1:
    #                     return curPlayer
    #                 elif r == -1:
    #                     return -1 * curPlayer
    #                 else:
    #                     return 0
    #
    #             return [
    #                 (x[0], x[2], r * ((-1) ** (x[1] != curPlayer)))
    #                 for x in trainExamples
    #             ]
    #


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

                # for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                # Create the actors to run the episodes
                # workers = []
                # for _ in range(self.args.numCPUForMCTS):
                #     workers.append(
                #         ExecuteEpisodeActor.remote(self.game, self.nnet, self.args)
                #     )
                #
                # pool = ray.util.ActorPool(workers)

                workers = []
                for _ in range(self.args.numCPUForMCTS):
                    workers.append(
                        ExecuteEpisodeActor.remote(
                            self.game,
                            self.nnet,
                            self.args,
                            self.toNNQueue,
                            self.fromNNQueue,
                            self.resultsQueue,
                        )
                    )

                log.info("Starting games....")
                # Start all of the games
                for _ in range(self.args.numEps):
                    ray.get(workers[0].startEpisode.remote())

                log.info("Done starting games")

                worker_tasks = []
                for worker in workers:
                    worker_tasks.append(worker.loopExecuteEpisodeFromQueue.remote())

                # Wait until all of the results are available
                while self.resultsQueue.size() < self.args.numEps:
                    time.sleep(0.5)

                log.info("Games complete, stopping workers")

                # Release the workers
                del worker_tasks

                # Drain the results queue
                while not self.resultsQueue.empty():
                    iterationTrainExamples.extend(self.resultsQueue.get())

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
