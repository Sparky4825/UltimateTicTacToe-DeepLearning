from utils import *

args = dotdict(
    {
        "numIters": 2,
        "numEps": 1025,  # Number of complete self-play games to simulate during a new iteration.
        "tempThreshold": 15,  #
        "updateThreshold": 0.55,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "maxlenOfQueue": 500000,  # Number of game examples to train the neural networks.
        "numMCTSSims": 10,  # Number of games moves for MCTS to simulate.
        "arenaCompare": 60,  # Number of games to play during arena play to determine if new net will be accepted.
        "cpuct": 1,
        "checkpoint": "./temp/",
        # 'load_model': False,
        # 'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
        "load_model": False,
        "load_folder_file": ("./temp/", "best.pth.tar"),
        "numItersForTrainExamplesHistory": 20,
        "numCPUForMCTS": 12,  # The number of Ray actors to use to add boards to be predicted.
        "CPUBatchSize": 128,
        # NNET ARGS
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 30,
        "batch_size": 2048,
        "cuda": True,
        "num_channels": 512,
    }
)
