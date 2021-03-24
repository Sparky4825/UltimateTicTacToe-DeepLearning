import logging

import coloredlogs
import ray

from Coach import Coach

from UltimateTicTacToe import UltimateTicTacToeGame as UTicTacToe
from UltimateTicTacToe.keras.NNet import NNetWrapper as nn

from utils import *

from timeit import default_timer as timer

import numpy as np

log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")  # Change this to DEBUG to see more info.

args = dotdict(
    {
        "numIters": 1000,
        "numEps": 200,  # Number of complete self-play games to simulate during a new iteration.
        "tempThreshold": 15,  #
        "updateThreshold": 0.55,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "maxlenOfQueue": 500000,  # Number of game examples to train the neural networks.
        "numMCTSSims": 25,  # Number of games moves for MCTS to simulate.
        "arenaCompare": 40,  # Number of games to play during arena play to determine if new net will be accepted.
        "cpuct": 1,
        "checkpoint": "./temp/",
        # 'load_model': False,
        # 'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
        "load_model": False,
        "load_folder_file": ("./temp/", "best.pth.tar"),
        "numItersForTrainExamplesHistory": 5,
        "numCPUForMCTS": 12,  # The number of Ray actors to use to add boards to be predicted.
        "CPUBatchSize": 64,
    }
)


def tflite_test():
    ray.init()

    log.info("Loading %s...", UTicTacToe.TicTacToeGame.__name__)
    g = UTicTacToe.TicTacToeGame()

    b = g.getInitBoard()

    b, _ = g.getNextState(b, 1, 3)

    a = g.getInitBoard()

    a, _ = g.getNextState(b, 1, 9)

    print(a)
    print(b)

    log.info("Loading Neural Network (Ray actor)...")
    nnet = nn(g)

    if args.load_model:
        log.info(
            'Loading checkpoint "%s/%s"...',
            args.load_folder_file[0],
            args.load_folder_file[1],
        )
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    else:
        log.warning("Not loading a checkpoint!")

    interpreter = nnet.convert_to_tflite()

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    pi_index = output_details[0]["index"]
    v_index = output_details[1]["index"]

    input_shape = input_details[0]["shape"]
    pi_shape = output_details[0]["shape"]
    v_shape = output_details[1]["shape"]

    input_dtype = input_details[0]["dtype"]
    pi_dtype = output_details[0]["dtype"]
    v_dtype = output_details[1]["dtype"]

    # Warm up
    interpreter.set_tensor(input_index, np.array([b], dtype=input_dtype))

    interpreter.invoke()

    pi = interpreter.get_tensor(pi_index)
    v = interpreter.get_tensor(v_index)

    start_time = timer()

    interpreter.set_tensor(input_index, np.array([a], dtype=input_dtype))
    interpreter.invoke()
    pi = interpreter.get_tensor(pi_index)
    v = interpreter.get_tensor(v_index)

    print(f"TF LITE TIME: {timer() - start_time:.6f} seconds")
    print(pi)
    print(v)


def main():
    ray.init()

    log.info("Loading %s...", UTicTacToe.TicTacToeGame.__name__)
    g = UTicTacToe.TicTacToeGame()

    log.info("Loading Neural Network (Ray actor)...")
    nnet = nn(g)

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
    c.learn()


if __name__ == "__main__":
    main()
