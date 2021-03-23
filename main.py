import logging
import os
from pickle import Unpickler
from random import shuffle

import coloredlogs
import ray
from ray.util.queue import Queue

from Coach import Coach

from UltimateTicTacToe import UltimateTicTacToeGame as UTicTacToe
from UltimateTicTacToe.keras.NNet import NNetWrapper as nn

from args import args


log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")  # Change this to DEBUG to see more info.


def main():
    ray.init()

    log.info("Loading %s...", UTicTacToe.TicTacToeGame.__name__)
    g = UTicTacToe.TicTacToeGame()

    log.info("Loading Neural Network (Ray actor)...")

    toNNQueue = Queue()
    fromNNQueue = Queue()
    resultsQueue = Queue()

    nnet = nn.remote(g, toNNQueue, fromNNQueue, resultsQueue, args)

    print("WATCHING QUEUE FOR EVER")
    nnet.forever_watch_queue.remote(args.CPUBatchSize)

    if args.load_model:
        log.info(
            'Loading checkpoint "%s/%s"...',
            args.load_folder_file[0],
            args.load_folder_file[1],
        )
        ray.get(
            nnet.load_checkpoint.remote(
                args.load_folder_file[0], args.load_folder_file[1]
            )
        )
    else:
        log.warning("Not loading a checkpoint!")

    log.info("Loading the Coach...")

    c = Coach(g, nnet, args, toNNQueue, fromNNQueue, resultsQueue)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info("Starting the learning process")
    c.learn()


def train_only():

    ray.init()

    log.info("Loading %s...", UTicTacToe.TicTacToeGame.__name__)
    g = UTicTacToe.TicTacToeGame()

    log.info("Loading Neural Network (Ray actor)...")
    nnet = nn.remote(g)

    # modelFile = os.path.join(args.load_folder_file[0], args.load_folder_file[1])
    examplesFile = "temp/checkpoint_0.pth.tar.examples"

    with open(examplesFile, "rb") as f:
        trainExamplesHistory = Unpickler(f).load()

    trainExamples = []
    for e in trainExamplesHistory:
        trainExamples.extend(e)
    shuffle(trainExamples)

    trainExamples = trainExamples[:30000]

    log.info(f"About to begin training with {len(trainExamples)} samples")

    ray.get(nnet.train.remote(trainExamples))


def model_illustrate():
    ray.init()

    g = UTicTacToe.TicTacToeGame()

    nnet = nn.remote(g)

    print(ray.get(nnet.model_summary.remote()))


if __name__ == "__main__":
    main()
