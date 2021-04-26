# distutils: language = c++

from Minimax cimport Node

import tensorflow as tf

cimport numpy as np

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement

from libcpp.vector cimport vector
from libcpp.list cimport list as cpplist
from libcpp.unordered_map cimport unordered_map
from libc.string cimport memset, memcpy

import numpy as np
import coloredlogs
import logging

from random import randint

import time
import datetime

def runSelfPlayEpisodesBatch(evaluate, int batchSize = 500, int sims=50, float cpuct=1, double dirA=0.8, double dirX=0.5, float percentQ=1, int tempThreshold=20):
    cdef cpplist[MCTS] episodes

    for i in range(batchSize):
        episodes.push_back(MCTS(cpuct, dirA, percentQ))
        episodes.back().startNewSearch(GameState())

    # The iterator used to loop over the list
    cdef cpplist[MCTS].iterator it

    cdef int nnSize = 0
    cdef int index, action, actionsTaken, numActions, status

    cdef np.ndarray[float, ndim=2] boards
    cdef np.ndarray[float, ndim=2] valids

    cdef float[:, :] boardsView, validsView, piView, vView

    cdef vector[float] probs
    cdef vector[double] dir

    cdef vector[trainingExampleVector] results

    actionsTaken = 0

    # Create numpy arrays for boards to be written to
    boards = np.zeros((nnSize, 199), dtype=np.float32)
    valids = np.zeros((nnSize, 81), dtype=np.float32)

    boardsView = boards
    validsView = valids

    while episodes.size() > 0:
        
        # All of the MCTS sims
        for _ in range(sims):
            it = episodes.begin()
            nnSize = 0

            # Prepare batch
            while it != episodes.end():

                # Check if NN is needed
                if deref(it).searchPreNNTFLite():
                    deref(deref(it).currentNode).board.writeCanonicalBoard(&boardsView[nnSize, 0])
                    deref(deref(it).currentNode).board.writeValidMoves(&validsView[nnSize, 0])
                    nnSize += 1

                # Advance to the next episode
                preincrement(it)

            # Perform NN evaluation
            pi, v = evaluate((boards[:nnSize], valids[:nnSize]))

            # Get view to results
            piView = pi
            vView = v

            it = episodes.begin()
            index = 0

            while it != episodes.end():

                # Check if NN is needed
                if deref(it).evaluationNeeded:
                    deref(it).searchPostNNTFLite(&piView[index, 0], &vView[index, 0])
                    index += 1

                # Advance to the next episode
                preincrement(it)

        actionsTaken += 1

        # Make moves
        print(f"Taking action {actionsTaken}")

        it = episodes.begin()
        while it != episodes.end():
            probs = deref(it).getActionProb()

            # Save probability before adding noise
            deref(it).saveTrainingExample()


            if actionsTaken < tempThreshold:
                # Add dirichlet noise
                numActions = deref(it).rootNode.children.size()
                dir = deref(it).dir(dirA, numActions)
                index = 0

                for i in range(81):
                    if probs[i] != 0:
                        probs[i] = dirX * probs[i] + (1 - dirX) * dir[index]
                        index += 1

                action = RandomActionWeighted(probs)
            
            else:
                action = MaxAction(probs)

            deref(it).takeAction(action)

            # Display game
            # print(deref(it).rootNode.board.gameToString().decode("ascii"))

            status = deref(it).getStatus()

            # If game over
            if status != 0:

                # Replace status with result value
                if status == 1:
                    status = 1
                elif status == 3:
                    status = 0
                elif status == 2:
                    status = -1

                # Save all the examples
                for ex in deref(it).getTrainingExamplesVector(status):
                    results.push_back(ex)

                # Remove from the list
                it = episodes.erase(it)
            else:
                # Advance to the next episode
                preincrement(it)


def runSelfPlayEpisodes(modelPath, int sims=50, float cpuct=1, double dirA=0.8, double dirX=0.5, float percentQ=1, int tempThreshold=8):
    cdef MCTS tree = MCTS(cpuct, dirA, dirX)

    a = np.zeros((1, 1), dtype=np.float32)
    cdef float[:, :] aView = a

    tree.startNewSearch(GameState())

    cdef int actionsTaken = 0

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

                
                deref(tree.currentNode).board.writeCanonicalBoard(boardPtr)
                deref(tree.currentNode).board.writeValidMoves(validPtr)

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
            numActions = tree.rootNode.children.size()

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