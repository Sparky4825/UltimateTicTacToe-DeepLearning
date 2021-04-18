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


def get_model1(game, args):
    from keras.models import Model, Input
    from keras.layers import (
        Flatten,
        Activation,
        Conv2D,
        Dropout,
        BatchNormalization,
        Dense,
        Reshape,
        Cropping2D,
        Cropping1D,
        Conv1D,
    )
    from keras.optimizers import Adam

    input_boards = Input(shape=(199))  # s: batch_size x board_x x board_y

    x_image = Reshape((9, 22, 1))(input_boards)  # batch_size  x board_x x board_y x 1
    h_conv1 = Activation("relu")(
        BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding="same")(x_image)
        )
    )  # batch_size  x board_x x board_y x num_channels
    h_conv2 = Activation("relu")(
        BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding="same")(h_conv1)
        )
    )  # batch_size  x board_x x board_y x num_channels
    h_conv3 = Activation("relu")(
        BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding="same")(h_conv2)
        )
    )  # batch_size  x (board_x) x (board_y) x num_channels
    h_conv4 = Activation("relu")(
        BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding="valid")(h_conv3)
        )
    )  # batch_size  x (board_x-2) x (board_y-2) x num_channels
    h_conv4_flat = Flatten()(h_conv4)
    s_fc1 = Dropout(args.dropout)(
        Activation("relu")(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat)))
    )  # batch_size x 1024
    s_fc2 = Dropout(args.dropout)(
        Activation("relu")(BatchNormalization(axis=1)(Dense(512)(s_fc1)))
    )  # batch_size x 1024
    pi = Dense(81, activation="softmax", name="pi")(s_fc2)  # batch_size x action_size
    v = Dense(1, activation="tanh", name="v")(s_fc2)  # batch_size x 1

    model = Model(inputs=input_boards, outputs=[pi, v])
    model.compile(
        loss=["categorical_crossentropy", "mean_squared_error"],
        optimizer=Adam(args.lr),
        metrics=["accuracy"],
    )

    return model


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
        Cropping2D,
        Cropping1D,
        Conv1D,
        MaxPool1D,
        Multiply,
    )
    from keras.optimizers import Adam

    num_filters = 250

    # game params
    board_x, board_y, board_z = game.getBoardSize()
    action_size = game.getActionSize()
    args = args

    # Neural Net
    input_boards = Input(shape=(199))  # s: batch_size x board_x x board_y
    valid_moves = Input(shape=81)

    reshape = Reshape((199, 1))(input_boards)

    # First pass though 2 layers with only one node per spot (plus 1 for won boards)

    # Take out the last bit that is only used to denote game winners
    crop = Cropping1D(cropping=(0, 1))(reshape)

    conv1 = Activation("relu")(
        BatchNormalization(axis=2)(
            Conv1D(
                num_filters, 22, 22, "valid", "channels_last", kernel_regularizer="l2"
            )(crop)
        )
    )

    conv2 = Activation("relu")(
        # BatchNormalization(axis=2)(
        Conv1D(num_filters, 9, 9, "valid", "channels_last")(conv1)
        # )
    )

    reshape2 = Flatten()(conv1)

    dense1 = Dense(500, activation="relu")(reshape2)

    dense2 = Dense(500, activation="relu")(dense1)

    # resize = Flatten()(dense2)

    # num_dense_layers = 8
    #
    # previous_layer = dense2
    #
    # for i in range(num_dense_layers):
    #     previous_layer = Dense(650, activation="relu")(previous_layer)
    #     # previous_layer = BatchNormalization()(previous_layer)
    #
    #     # Add dropout layer every other
    #     if i % 3 == 2:
    #         previous_layer = Dropout(args.dropout)(previous_layer)
    #
    # final_dense_layer = previous_layer

    final_dense_layer = dense2

    # Give the pi output some unique layers to learn from
    pilayer1 = Dense(250, activation="relu")(final_dense_layer)
    pilayer2 = Dense(375, activation="relu")(pilayer1)

    vlayer1 = Dense(500, activation="relu")(final_dense_layer)
    vlayer2 = Dense(500, activation="relu")(vlayer1)

    pi = Dense(action_size, activation="relu")(pilayer2)  # batch_size x action_size
    final_pi = Multiply()([pi, valid_moves])
    final_pi = Activation("softmax", name="pi")(final_pi)
    v = Dense(1, activation="tanh", name="v")(vlayer2)  # batch_size x 1

    model = Model(inputs=[input_boards, valid_moves], outputs=[final_pi, v])
    model.compile(
        loss=["categorical_crossentropy", "mean_squared_error"],
        loss_weights=[0.45, 0.65],
        optimizer=Adam(args.lr),
        metrics=["accuracy"],
    )

    return model
