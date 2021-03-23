import argparse
import logging
import os
import shutil
import time
import random

import coloredlogs
import numpy as np
import math
import sys

sys.path.append("..")
from utils import *
from NeuralNet import NeuralNet

import argparse
from .TicTacToeNNet import get_model

import time

import ray

import asyncio

"""
NeuralNet wrapper class for the TicTacToeNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""


args = dotdict(
    {
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 30,
        "batch_size": 1024,
        "cuda": True,
        "num_channels": 512,
    }
)


@ray.remote(num_gpus=1)
class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = get_model(game, args)
        self.args = args
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.pending_evaluations = np.empty((0, 3, 9, 10))
        self.run_evaluation = asyncio.Event()
        self.claim_evaluation = asyncio.Event()

        self.prediction_results = None
        self.unclaimed_results = None

        self.last_prediction_time = time.time()
        self.prediction_timer_running = False

        # Prevent too many concurrent threads at once - should be bigger than batch size
        self.sem = asyncio.Semaphore(args.batch_size)

        self.log = logging.getLogger(self.__class__.__name__)

        coloredlogs.install(level="INFO", logger=self.log)

    def get_batch_size(self):
        return self.args.batch_size

    def get_weights(self):
        return self.nnet.get_weights()

    def set_weights(self, weights):
        self.nnet.set_weights(weights)

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.fit(
            x=input_boards,
            y=[target_pis, target_vs],
            batch_size=args.batch_size,
            epochs=args.epochs,
        )

    def predict(self, board):
        """
        Predict a single board
        """

        board = board[np.newaxis, :, :, :]

        # timing
        start = time.time()

        # run
        prediction_results = self.nnet.predict(board)

        self.log.debug(
            "SINGLE PREDICTION TIME TAKEN : {0:03f}".format(time.time() - start)
        )

        self.last_prediction_time = time.time()

        return [prediction_results[0][0], prediction_results[1][0]]

    def predict_batch(self, batch):
        """
        The Neural Network is more efficient running on a batch,
        so run a batch of predictions at once.
        """
        self.log.debug("Starting prediction batch")

        # timing
        start = time.time()

        # run
        prediction_results = self.nnet.predict(batch)

        self.log.debug("PREDICTION TIME TAKEN : {0:03f}".format(time.time() - start))
        self.log.debug(f"PREDICTIONS MADE: {len(batch)}")

        return prediction_results

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            self.log.info(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            os.mkdir(folder)
        else:
            self.log.info("Checkpoint Directory exists! ")
        self.nnet.save_weights(filepath)

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        # if not os.path.exists(filepath):
        #     raise FileNotFoundError("No model in path '{}'".format(filepath))
        self.nnet.load_weights(filepath)
