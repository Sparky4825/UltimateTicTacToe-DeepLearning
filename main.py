import logging
import os
from pickle import Unpickler
from random import shuffle

import coloredlogs
import ray

from Coach import Coach

from UltimateTicTacToe import UltimateTicTacToeGame as UTicTacToe
from UltimateTicTacToe.keras.NNet import NNetWrapper as nn

from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")  # Change this to DEBUG to see more info.

args = dotdict(
    {
        "numIters": 2,
        "numEps": 768,  # Number of complete self-play games to simulate during a new iteration.
        "tempThreshold": 15,  #
        "updateThreshold": 0.55,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "maxlenOfQueue": 500000,  # Number of game examples to train the neural networks.
        "numMCTSSims": 25,  # Number of games moves for MCTS to simulate.
        "arenaCompare": 60,  # Number of games to play during arena play to determine if new net will be accepted.
        "cpuct": 1,
        "checkpoint": "./temp/",
        # 'load_model': False,
        # 'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
        "load_model": False,
        "load_folder_file": ("./temp/", "best.pth.tar"),
        "numItersForTrainExamplesHistory": 20,
        "numCPUForMCTS": 12,  # The number of Ray actors to use to add boards to be predicted.
        "CPUBatchSize": 64,
    }
)


def main():
    ray.init()

    log.info("Loading %s...", UTicTacToe.TicTacToeGame.__name__)
    g = UTicTacToe.TicTacToeGame()

    log.info("Loading Neural Network (Ray actor)...")
    nnet = nn.remote(g)

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
    c = Coach(g, nnet, args)

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
    train_only()
