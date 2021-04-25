import logging
from pickle import Unpickler

import coloredlogs
import numpy as np

from Coach import Coach
from UltimateTicTacToe import UltimateTicTacToeGame as UTicTacToe
from UltimateTicTacToe.keras.NNet import NNetWrapper
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")  # Change this to DEBUG to see more info.

args = dotdict(
    {
        "numIters": 20,
        "numEps": 350,  # Number of complete self-play games to simulate during a new iteration.
        "tempThreshold": 16,  #
        "arenaTempThreshold": 5,  #
        # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "updateThreshold": 0.52,
        "maxlenOfQueue": 1000000,  # Number of game examples to train the neural networks.
        "pastTrainingIterations": 1,
        "numMCTSSims": 1000,  # Number of games moves for MCTS to simulate.
        "arenaCompare": 200,  # Number of games to play during arena play to determine if new net will be accepted.
        "cpuct": 2,
        "checkpoint": "./temp/",
        "load_model": True,
        "load_folder_file": ("./temp/", "best.ckpt"),
        "numItersForTrainExamplesHistory": 2,
        "numCPUForMCTS": 5,  # The number of Ray actors to use to add boards to be predicted.
        "CPUBatchSize": 375,
        "GPUBatchSize": 1,
        "skipFirstSelfPlay": True,
        "dir_a": 0.8,
        "dir_x": 0.5,
        "q_percent": 0.75,
    }
)


def model_illustrate():
    g = UTicTacToe.TicTacToeGame()

    nnet = NNetWrapper(g)

    print(nnet.model_summary())


def train_only():
    log.info("Loading %s...", UTicTacToe.TicTacToeGame.__name__)
    g = UTicTacToe.TicTacToeGame()

    log.info("Loading Neural Network (Ray actor)...")
    nnet = NNetWrapper(g)

    if args.load_model:
        log.info(
            'Loading checkpoint "%s/%s"...',
            args.load_folder_file[0],
            args.load_folder_file[1],
        )
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    examplesFile = "trainingData - Random Games.examples"

    with open(examplesFile, "rb") as f:
        inputs, pis, vs = Unpickler(f).load()

    validate_size = int(len(inputs) / 20)

    validate_boards = inputs[:validate_size]
    val_pis = pis[:validate_size]
    val_vs = vs[:validate_size]

    inputs = inputs[validate_size:]

    pis = pis[validate_size:]

    vs = vs[validate_size:]

    validate = (validate_boards, val_pis, val_vs)

    while True:
        rng_state = np.random.get_state()
        np.random.shuffle(inputs)
        np.random.set_state(rng_state)

        np.random.shuffle(pis)
        np.random.set_state(rng_state)

        np.random.shuffle(vs)

        log.info(f"About to begin training with {len(inputs)} samples")

        nnet.train(inputs, pis, vs, validation=validate, epochs=15)


def main():
    log.info("Loading %s...", UTicTacToe.TicTacToeGame.__name__)
    g = UTicTacToe.TicTacToeGame()

    log.info("Loading Neural Network (Ray actor)...")
    nnet = NNetWrapper(g)

    if args.load_model:
        log.info(
            'Loading checkpoint "%s/%s"...',
            args.load_folder_file[0],
            args.load_folder_file[1],
        )
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    else:
        log.warning("Not loading a checkpoint!")

    log.info("Loading the Coach...")
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info("Starting the learning process")
    c.learnIterations()


if __name__ == "__main__":
    main()
