import sys

sys.path.append("..")
from utils import *

import argparse

"""
NeuralNet for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloNNet by SourKream and Surag Nair.
"""


def get_model(game, args):
    from keras.models import Model, Input
    from keras.layers import (
        Flatten,
        Activation,
        Conv2D,
        Dropout,
        BatchNormalization,
        Dense,
        Reshape,
    )
    from keras.optimizers import Adam

    # game params
    board_x, board_y, board_z = game.getBoardSize()
    action_size = game.getActionSize()
    args = args

    # Neural Net
    input_boards = Input(shape=199)  # s: batch_size x board_x x board_y

    # First pass though 2 layers with only one node per spot (plus 1 for won boards)

    dense1 = Dense(500, activation="relu")(input_boards)

    dense2 = Dense(500, activation="relu")(dense1)

    resize = Flatten()(dense2)

    num_dense_layers = 8

    previous_layer = resize

    for i in range(num_dense_layers):
        previous_layer = Dense(650, activation="relu")(previous_layer)
        # previous_layer = BatchNormalization()(previous_layer)

        # Add dropout layer every other
        if i % 3 == 2:
            previous_layer = Dropout(args.dropout)(previous_layer)

    final_dense_layer = previous_layer

    # Give the pi output some unique layers to learn from
    pilayer1 = Dense(250, activation="relu")(final_dense_layer)
    pilayer2 = Dense(375, activation="relu")(pilayer1)

    vlayer = Dense(500, activation="relu")(final_dense_layer)

    pi = Dense(action_size, activation="softmax", name="pi")(
        pilayer2
    )  # batch_size x action_size
    v = Dense(1, activation="tanh", name="v")(vlayer)  # batch_size x 1

    model = Model(inputs=input_boards, outputs=[pi, v])
    model.compile(
        loss=["categorical_crossentropy", "mean_squared_error"],
        optimizer=Adam(args.lr),
        metrics=["accuracy"],
    )

    return model
