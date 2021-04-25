# distutils: language = c++

from Minimax cimport Node

import tensorflow as tf

cimport numpy as np

from cython.operator cimport dereference as deref

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libc.string cimport memset, memcpy

import numpy as np
import coloredlogs
import logging

from random import randint

import time
import datetime

def runSelfPlayEpisodes(modelPath, int sims=50, float cpuct=1, double dirA=0.8, double dirX=0.5, float percentQ=1, int tempThreshold=8):
    cdef MCTS tree = MCTS(cpuct, dirA, dirX)

    a = np.zeros((1, 1), dtype=np.float32)
    cdef float[:, :] aView = a

    tree.startNewSearch(GameState())

    cdef int actionsTaken = 0;

    interpreter = tf.lite.Interpreter(modelPath)

    interpreter.allocate_tensors()

    board = interpreter.tensor(interpreter.get_input_details()[0]["index"])
    valid = interpreter.tensor(interpreter.get_input_details()[1]["index"])

    policy = interpreter.tensor(interpreter.get_output_details()[0]["index"])
    value  = interpreter.tensor(interpreter.get_output_details()[1]["index"])

    cdef float[:, :] boardView
    cdef float* boardPtr

    cdef float[:, :] validView
    cdef float* validPtr

    cdef float[:, :] policyView
    cdef float* policyPtr

    cdef float[:, :] valueView
    cdef float* valuePtr

    cdef GameState game
 
    cdef float[:] aview

    cdef int action, numActions, i, status

    cdef vector[double] dir

    cdef float prob


    while True:

        # Perform all the MCTS simulations
        for _ in range(sims):
            if tree.searchPreNNTFLite():

                # Get pointers to where the input data needs to be written
                boardView = board()
                boardPtr = &boardView[0, 0]

                validView = valid()
                validPtr = &validView[0, 0]

                
                game.writeCanonicalBoard(boardPtr)
                game.writeValidMoves(validPtr)

                # Release references to the data
                boardView = aView
                validView = aView

                # Make prediction
                interpreter.invoke()

                # Get pointers to output
                policyView = policy()
                policyPtr = &policyView[0, 0]

                valueView = value()
                valuePtr = &valueView[0, 0]

                tree.searchPostNNTFLite(policyPtr, valuePtr)

                # Release referrences to the data
                valueView = aView
                policyView = aView

        actionsTaken += 1

        tree.saveTrainingExample()

        print(tree.gameToString().decode("ascii"))


        if actionsTaken < tempThreshold:
            # Add dirichlet noise
            numActions = tree.rootNode.children.size();

            dir = tree.dir(dirA, numActions)

            i = 0
            for prob in tree.getActionProb():
                if prob != 0:
                    prob = dirX * prob + (1 - dirX) * dir[i]
                    i += 1

            action = RandomActionWeighted(tree.getActionProb())

        else:
            action = MaxAction(tree.getActionProb())

        tree.takeAction(action)

        status = tree.getStatus()

        if status != 0:
            if status == 3:
                status = 0
            elif status == 2:
                status = -1
            else:
                status = 1


            return