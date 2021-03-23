import logging
import time

import coloredlogs
import numpy as np
import ray

from MCTS import MCTS


class NNetEvalTicket:
    """
    Contains the minimum amount of information the NNet needs
    to give an evaluation.
    """

    def __init__(self, workerID, gameID, canonicalBoard, weightsSet=None):
        self.canonicalBoard = canonicalBoard
        self.pi = None
        self.v = None
        self.workerID = workerID
        self.gameID = gameID
        self.weightsSet = weightsSet


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

        self.ticket = None


@ray.remote
class ExecuteEpisodeActor:
    def __init__(
        self,
        id_number,
        game,
        args,
        batchMaker,
        fromWOQueue,
        resultsQueue,
        arena=False,
    ):
        self.id = id_number

        self.game = game
        self.args = args

        self.batchMaker = batchMaker
        self.fromWOQueue = fromWOQueue
        self.resultsQueue = resultsQueue

        # Variables to keep track of batch processing of boards
        self.batch_size = self.args.CPUBatchSize

        self.log = logging.getLogger(self.__class__.__name__)

        self.arena = arena
        self.current_weight_set = 1

        self.ongoingGames = []

        coloredlogs.install(level="INFO", logger=self.log)

    def startMultipleEpisodes(self, numEpisodes):
        """
        Calls self.startEpisode() the given number of times
        """

        for _ in range(numEpisodes):
            self.startEpisode()

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

        # Add it to the ongoing games and create a ticket for the queue
        self.ongoingGames.append(new_episode)

        # Game ID is created from the episode index in self.ongoingGames
        ticket = NNetEvalTicket(
            self.id, len(self.ongoingGames) - 1, new_episode.mcts.canonicalBoard
        )

        # TODO: This function assumes that the NNet will be needed always for the first move, is this correct?

        # Add it to the queue for NN evaluation
        self.batchMaker.addTicket.remote(ticket)

    def checkMoveandMake(self, ep):
        """
        Return true if game is over at move
        :param ep:
        :return:
        """
        if ep.episodeStep >= self.args.numMCTSSims:
            # Get final results and make a move
            self.log.debug("Making a move")
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
                self.log.info(f"{self.resultsQueue.size() + 1} games are complete")
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
                # Game not complete - make move, start MCTS search again

                ep.mcts.moveStartBoard2Action(action)
                ep.episodeStep = 0
                ep.mcts.searchPreNN()
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

        # t1 = time.time()
        ticket = self.fromWOQueue.get()
        # t2 = time.time()

        # print(f"Waiting on WO for {t2 - t1} seconds")

        ep = self.ongoingGames[ticket.gameID]

        # Transfer the results that come back in the ticket to the episode
        ep.mcts.pi = ticket.pi
        ep.mcts.v = ticket.v

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

        # TODO:
        if self.checkMoveandMake(ep):
            return

        ep.episodeStep += 1
        # Once the while loop is exited, it means a NN evaluation is needed

        # Update the ticket and send it back
        ticket.canonicalBoard = ep.mcts.canonicalBoard
        self.batchMaker.addTicket.remote(ticket)
