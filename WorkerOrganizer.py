import asyncio
import time

import ray
from ray.util.queue import Queue

from Coach import ExecuteEpisodeActor
from args import args


class WorkerOrganizer:
    def __init__(
        self,
        game,
        numWorkers,
        toNNQueue,
        fromNNQueue,
        resultsQueue,
        nnetActor,
    ):
        self.toNNQueue = toNNQueue
        self.fromNNQueue = fromNNQueue
        self.resultsQueue = resultsQueue

        self.workers = []

        self.toWOQueue = Queue()
        self.fromWOQueues = {}

        self.nnetActor = nnetActor

        self.batchMaker = BatchMaker.remote(
            self.toWOQueue, self.toNNQueue, self.resultsQueue
        )

        for id_num in range(numWorkers):

            # Give a unique queue to each worker, but all incoming NNet tickets come from the same queue
            self.fromWOQueues[id_num] = Queue()
            self.workers.append(
                ExecuteEpisodeActor.remote(
                    id_num,
                    game,
                    args,
                    self.batchMaker,
                    self.fromWOQueues[id_num],
                    resultsQueue,
                )
            )

    def runWorkers(self):
        """
        Continually runs and checks on all the worker processes.
        """

        # Check that the number of episodes can be evenly divided amongst the CPU
        # TODO: Allow uneven distribution of episodes
        assert args.numEps % args.numCPUForMCTS == 0

        self.batchMaker.makeBatches.remote()

        # Start all the games
        episodesPerWorker = int(args.numEps / args.numCPUForMCTS)
        startingProcesses = []
        for worker in self.workers:
            startingProcesses.append(
                worker.startMultipleEpisodes.remote(episodesPerWorker)
            )

        # Wait until all the workers have started all their games
        ray.get(startingProcesses)

        for worker in self.workers:
            worker.loopExecuteEpisodeFromQueue.remote()

        # Loop until all the games are in the results queue
        while self.resultsQueue.size() < args.numEps:

            # Get the results and send them back
            # t1 = time.time()
            epTicket = self.fromNNQueue.get()
            # t2 = time.time()
            # print(f"Waiting on NNetEval for {t2 - t1} seconds")

            # Send the resulting ticket to the right worker
            self.fromWOQueues[epTicket.workerID].put(epTicket)

        # All games are done, kill the actors and get the results
        del self.batchMaker
        # TODO: Adjust workers to use and receive tickets
        # TODO: Tickets must identify the episode they came from
        # TODO: Workers need a list of episodes they are working on
        # TODO: Workers will first turn all unevaluateed boards into tickets, then loop continually though returned ticket queue
        # TODO: Coach needs to call WO; WO works until the games are all done

        # Step 1: Adjust ExecuteEpisodeActor to have a list of ongoing episodes
        # Produce tickets and recieve tickets and pair them back to the game (use dictionary)
        # Adjust NNet to use tickets (remote .mcts from elements)


@ray.remote
class BatchMaker:
    """
    Is responsible for putting batches together and passing them to the NN
    """

    def __init__(self, toWOQueue, toNNQueue, resultsQueue):
        self.toWOQueue = asyncio.Queue()
        self.toNNQueue = toNNQueue
        self.resultsQueue = resultsQueue

    async def addTicket(self, ticket):
        """
        Adds the given ticket to the queue for the next batch.
        """
        await self.toWOQueue.put(ticket)

    async def makeBatches(self):
        """
        Continually makes batches and sends them to the NN.
        """
        unevaluatedTickets = []

        # Loop forever
        while True:

            # Get the incoming evaluation tickets to fill a batch
            unevaluatedTickets.append(await self.toWOQueue.get())

            # If a full batch or not enough games left to fill a batch, send the batch
            length = len(unevaluatedTickets)
            if (
                length >= args.CPUBatchSize
                or args.CPUBatchSize > args.numEps - self.resultsQueue.size()
                # or True
                # or length >= args.numEps - self.resultsQueue.size()
            ):
                self.toNNQueue.put(unevaluatedTickets)
                unevaluatedTickets = []
