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
        "epochs": 10,
        "batch_size": 512,
        "cuda": True,
        "num_channels": 512,
    }
)


@ray.remote(num_gpus=1)
class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = get_model(game, args)
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

        asyncio.create_task(self.prediction_timer())

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

    async def request_prediction(self, board):
        """
        board: np array with board
        """

        self.log.debug("Prediction requested")
        self.log.debug(f"Board: {board}")

        # Limit number of concurrent threads
        async with self.sem:

            # Update the timer every time a new prediction is requested
            self.last_prediction_time = time.time()

            # TODO: ensure that all of the results have been returned before running another prediction, else they will be lost

            # If waiting for other processes to claim their results, it must wait
            if self.run_evaluation.is_set():
                self.log.info(
                    "Waiting for another process to claim results to add prediction"
                )
                self.log.debug(f"Unclaimed results: {self.unclaimed_results}")
                await self.claim_evaluation.wait()

            # preparing input
            # board = board[np.newaxis, :, :]

            # Add the board to the list of predictions to be made
            self.pending_evaluations = np.append(
                self.pending_evaluations, board[np.newaxis, :, :], axis=0
            )

            # Save the board index to remember which results go with this board after predictions are calculated
            board_index = len(self.pending_evaluations) - 1

            # Check if its time to run a batch
            if len(self.pending_evaluations) >= args.batch_size:
                # Run the batch
                self.predict()

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

    def predict(self):
        """
        Runs all of the pending predictions
        """

        self.log.info(
            f"Starting predictions - last prediction was {time.time() - self.last_prediction_time} secs ago"
        )

        # timing
        start = time.time()

        # run
        self.prediction_results = self.nnet.predict(self.pending_evaluations)

        self.log.info("PREDICTION TIME TAKEN : {0:03f}".format(time.time() - start))
        self.log.info(f"PREDICTIONS MADE: {len(self.pending_evaluations)}")

        # Clear out the old boards
        self.pending_evaluations = np.empty((0, 3, 9, 10))

        # Add the boards need to be claimed
        self.unclaimed_results = np.full(len(self.prediction_results[0]), 1)

        # Tell the awaiting functions that the batch has been processed and they need to claim results
        self.claim_evaluation.clear()
        self.run_evaluation.set()

        self.last_prediction_time = time.time()

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
                self.predict()

            else:
                self.log.debug(
                    f"Prediction is not needed - Pending evaluations: {self.pending_evaluations}"
                )
                await asyncio.sleep(2)

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
