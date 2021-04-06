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

import tensorflow as tf

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
        "epochs": 5,
        "batch_size": 2000,
        "cuda": True,
        "num_channels": 512,
    }
)


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

    def train(self, inputs, targetPis, targetVs, epochs=None, validation=None):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )

        if epochs is None:
            epochs = self.args.epochs

        if validation is None:
            validate_size = int(len(inputs) / 20)

            validate_boards = inputs[:validate_size]
            val_pis = targetPis[:validate_size]
            val_vs = targetVs[:validate_size]

            inputs = inputs[validate_size:]

            target_pis = targetPis[validate_size:]

            target_vs = targetVs[validate_size:]
        else:
            validate_boards = validation[0]
            val_pis = validation[1]
            val_vs = validation[2]

            target_pis = targetPis
            target_vs = targetVs

        self.nnet.fit(
            x=inputs,
            y=[target_pis, target_vs],
            batch_size=args.batch_size,
            epochs=epochs,
            callbacks=[tensorboard_callback],
            validation_data=(validate_boards, [val_pis, val_vs]),
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
        # prediction_results = self.nnet(board, training=False)

        self.log.debug(
            "SINGLE PREDICTION TIME TAKEN : {0:03f}".format(time.time() - start)
        )

        self.last_prediction_time = time.time()

        return [prediction_results[0][0], prediction_results[1][0]]

    def predict_on_batch(self, batch):
        return self.nnet.predict_on_batch(batch)

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

        self.log.info("PREDICTION TIME TAKEN : {0:03f}".format(time.time() - start))
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

    def convert_to_tflite(self):
        """
        Converts the model to a TensorFlow Lite model to run single predictions
        more quickly.
        :return:
        """

        converter = tf.lite.TFLiteConverter.from_keras_model(self.nnet)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        return tflite_model

    def model_summary(self):
        return self.nnet.summary(line_length=225)
