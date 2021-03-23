import argparse
import datetime
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


@ray.remote(num_gpus=1)
class NNetWrapper(NeuralNet):
    def __init__(self, game, toNNQueue, fromNNQueue, resultsQueue, args):
        self.nnet = get_model(game, args)
        self.args = args
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.pending_evaluations = np.empty((0, 3, 9, 10))
        self.run_evaluation = asyncio.Event()
        self.claim_evaluation = asyncio.Event()

        self.prediction_results = None
        self.unclaimed_results = None

        self.toNNQueue = toNNQueue
        self.fromNNQueue = fromNNQueue
        self.resultsQueue = resultsQueue

        self.last_prediction_time = time.time()
        self.prediction_timer_running = False

        # Prevent too many concurrent threads at once - should be bigger than batch size
        self.sem = asyncio.Semaphore(args.batch_size)

        self.log = logging.getLogger(self.__class__.__name__)

        self.episodes_to_predict = []

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

        import tensorflow as tf

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )

        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.fit(
            x=input_boards,
            y=[target_pis, target_vs],
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            callbacks=[tensorboard_callback],
        )

    def forever_watch_queue(self, cpu_batch_size):
        while True:
            self.get_episodes_from_queue(cpu_batch_size)

    def get_episodes_from_queue(self, cpu_batch_size):
        """
        Gets episodes from the queue, and predicts them when there are enough.
        """
        # TODO: Once there are no longer enough predictions to fill a batch, a prediction run must be run anyway

        pending_evaluations = np.empty((0, 3, 9, 10))
        episodes = []

        totalQueueTime = 0

        episodesAppendTime = 0
        npAppendTime = 0

        t1 = time.time()
        # Loop until there is a full batch or there are not enough unfinished games to make a full batch
        while len(episodes) < cpu_batch_size and (
            self.args.numEps - self.resultsQueue.size() > self.args.CPUBatchSize
            or len(episodes) == 0
        ):
            q1 = time.time()
            new_episode = self.toNNQueue.get()
            q2 = time.time()

            totalQueueTime += q2 - q1

            q1 = time.time()

            episodes.append(new_episode)
            q2 = time.time()
            episodesAppendTime += q2 - q1

            q1 = time.time()

            pending_evaluations = np.append(
                pending_evaluations,
                new_episode.mcts.canonicalBoard[np.newaxis, :, :, :],
                axis=0,
            )
            q2 = time.time()
            npAppendTime += q2 - q1

        t2 = time.time()

        totTime = t2 - t1

        print(
            f"Batch evaluation with {len(episodes)} eps! - Consolidating took {t2 - t1} sec - Queue % {round(totalQueueTime / totTime, 2)} - List append {round(episodesAppendTime / totTime, 2)} - NP append {round(npAppendTime / totTime, 2)}"
        )
        t1 = time.time()
        results = self.predict_batch(pending_evaluations)
        t2 = time.time()

        print(f"Batch evaluation took {t2 - t1} seconds")
        for i in range(len(episodes)):
            ep = episodes[i]
            ep.mcts.pi, ep.mcts.v = results[0][i], results[1][i][0]
            self.fromNNQueue.put(ep)

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

    def model_summary(self):
        return self.nnet.summary(line_length=275)
