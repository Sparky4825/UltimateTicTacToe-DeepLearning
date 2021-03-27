# distutils: language = c++

from Minimax cimport Node

cimport numpy as np

from cython.operator cimport dereference

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map

import numpy as np
import coloredlogs
import logging

from random import randint

cdef extern from "math.h":
    double sqrt(double m)

cdef class PyGameState:
    cdef GameState c_gamestate


    def __cinit__(self):
        self.c_gamestate = GameState()

    def get_copy(self):
        new_copy = PyGameState()
        new_copy.c_gamestate = self.c_gamestate.getCopy()

        return new_copy

    def move(self, board, piece):
        self.c_gamestate.move(board, piece)

    def get_status(self):
        return self.c_gamestate.getStatus()

    def get_board_status(self, board):
        return self.c_gamestate.getBoardStatus(board)

    def get_position(self, board, piece):
        return self.c_gamestate.getPosition(board, piece)

    def get_required_board(self):
        return self.c_gamestate.getRequiredBoard()

    def get_to_move(self):
        return self.c_gamestate.getToMove()

    def is_valid_move(self, board, piece):
        return bool(self.c_gamestate.isValidMove(board, piece))


cdef class MCTS:
    cdef float cpuct
    cdef int numMCTSSims 
    cdef object evaluate, log

    def __cinit__(self, args, evaluate):
        self.evaluate = evaluate

        self.cpuct = args.cpuct
        self.numMCTSSims = args.numMCTSSims

        self.log = logging.getLogger(self.__class__.__name__)

        coloredlogs.install(level="INFO", logger=self.log)



    def getActionProb(self, PyGameState position, int temp=1):
        """
        Performs the given number of simulations of MCTS starting from the given
        position.
        """

        cdef Node start = Node(position.c_gamestate, 0)

        cdef int[81] counts
        cdef int maxCount = 0
        cdef int action
        cdef vector[int] maxCountAction
        cdef int totalCount


        # Run the simulations
        for i in range(self.numMCTSSims):
            self.search(start)

        # Get the counts of every child
        for child in start.children:
            action = child.board.previousMove.board * 9 + child.board.previousMove.piece
            counts[action] = child.n

            if temp == 0:
                if child.n > maxCount:
                    # New best move found, remove all others
                    maxCountAction.clear()
                    maxCount = child.n

                elif child.n == maxCount:
                    # Equally good move found, add it to list
                    maxCountAction.push_back(action)

            else:
                counts[action] = child.n
                totalCount += child.n


        # Select only the best move (at random if multiple best moves)
        if temp == 0:
            action = randint(0, maxCountAction.size() - 1)
            counts[action] = 1

        else:
            # Normalize to probabilities
            for i in range(81):
                counts[i] /= totalCount

        return list(counts)


    cdef float search(self, Node &node) except *:

        cdef int result = node.board.getStatus()
        cdef int currentPlayer = node.board.getToMove()

        cdef float value
        cdef float totalValidMoves = 0
        cdef int numValidMoves = 0
        cdef vector[float] policy

        node.n += 1

        if currentPlayer == 2:
            currentPlayer = -1

        # Check if the game is over
        if result == 1:
            return -1 * currentPlayer

        elif result == 2:
            return currentPlayer

        # Draw has very little value
        elif result == 3:
            return 0.0001

        if node.evaluationPerformed == 0:
            # Perform NN evaluation
            policy, value = self.evaluate( np.asarray(node.board.getCanonicalBoard(), dtype=np.uint8) )

            # Flip value if player 2 is to move
            if node.board.getToMove() == 2:
                value *= -1

            # Save the policy for all valid moves to the child nodes
            node.addChildren()


            for child in node.children:
                validAction = child.board.previousMove.board * 9 + child.board.previousMove.piece
                totalValidMoves += policy[validAction]
                numValidMoves += 1

            if totalValidMoves > 0:
                # Renormalize the values of all valid moves            
                for child in node.children:
                    validAction = child.board.previousMove.board * 9 + child.board.previousMove.piece
                    child.p = policy[validAction] / totalValidMoves
            else:
                # All valid moves masked, doing a workaround
                self.log.warning("All valid moves masked, doing a workaround")

                for child in node.children:
                    validAction = child.board.previousMove.board * 9 + child.board.previousMove.piece
                    child.p = 1 / numValidMoves

            node.evaluationPerformed = 1

            # Return negative because its for the perspective of the other player
            return -1 * value

        cdef float bestUCB = -1 * FLT_MAX

        cdef float u, q, v

        cdef Node *bestAction

        # Pick the action with the highest upper confidence bound
        for child in node.children:
            if child.n > 0:
                u = (child.w / child.n) + self.cpuct * child.p * ( sqrt(node.n) / (1 + child.n)  )
            else:
                # Q = 0 if child node has not been expored yet?
                u = self.cpuct * child.p * ( sqrt(node.n) / (1 + child.n)  )

            if u > bestUCB:
                bestUCB = u
                bestAction = &child

        # Start a recursive search on the best action
        v = self.search(dereference(bestAction))

        dereference(bestAction).w += v

        return -1 * v

        
