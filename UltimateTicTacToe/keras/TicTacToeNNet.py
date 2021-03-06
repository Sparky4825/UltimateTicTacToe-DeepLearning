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
    input_boards = Input(
        shape=(board_x, board_y, board_z)
    )  # s: batch_size x board_x x board_y

    # First pass though 2 layers with only one node per spot (plus 1 for won boards)

    dense1 = Dense(900, activation="relu")(input_boards)

    dense2 = Dense(900, activation="relu")(dense1)

    resize = Flatten()(dense2)

    num_dense_layers = 20

    previous_layer = resize

    for i in range(num_dense_layers):
        previous_layer = Dense(900, activation="relu")(previous_layer)
        previous_layer = BatchNormalization()(previous_layer)

        # Add dropout layer every other
        # if i % 2 == 0:
        #     previous_layer = Dropout(args.dropout)(previous_layer)

    final_dense_layer = previous_layer

    pi = Dense(action_size, activation="softmax", name="pi")(
        final_dense_layer
    )  # batch_size x action_size
    v = Dense(1, activation="tanh", name="v")(final_dense_layer)  # batch_size x 1

    model = Model(inputs=input_boards, outputs=[pi, v])
    model.compile(
        loss=["categorical_crossentropy", "mean_squared_error"], optimizer=Adam(args.lr)
    )

    return model
