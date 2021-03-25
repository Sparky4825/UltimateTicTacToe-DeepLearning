# distutils: language = c++

from Minimax cimport Node

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

cdef float minimax(Node& node, int depth, float alpha, float beta, evaluate):
    cdef float bestEval, newEval

    # When a forced loss position is evaluated before a forced win position
    cdef int resetInfDepth = 0


    # Check if depth reached or game over
    if (depth <= 0 or node.board.getStatus() != 0):
        bestEval = evaluate(node.getCanonicalBoard())

        # Check if infinite (ie greater than evaluate bounds)
        if (bestEval > 1 or bestEval < -1):
            node.infDepth = node.depth

        # Return evaluation from opponent's perspective
        return bestEval * -1

    # Init to worst outcome
    bestEval = -100

    node.addChildren()

    cdef Node i
    for i in node.children:
        newEval = minimax(i, depth - 1, alpha, beta, evaluate)

        if newEval > bestEval:
            bestEval = newEval

        # If the position is lost, the best option is the one furthest from game over
        if bestEval < -1 and newEval < -1:
            if node.infDepth == -1 or node.infDepth < i.infDepth:
                node.infDepth = i.infDepth

        # If the position is won, best option is closest to game over
        elif bestEval > 1 and newEval > 1:
            if node.infDepth == -1 or node.infDepth > i.infDepth:
                node.infDepth = i.infDepth

        if alpha < newEval:
            alpha = newEval

        # Prune the position
        if beta <= alpha:
            break

    # Return evaluation from opponent's perspective
    return bestEval * -1
    

def minimaxSearch(PyGameState position, int depth):
    cdef Node start = Node(position.c_gamestate, 0)

    start.addChildren()

    # Board evals should always be between -1 and 1
    # So anything with abs > 1 is functionally the same as infinity
    cdef float bestEval = -1000
    cdef float newEval
    cdef Node bestMove = start.children[0]
    cdef int evalMultiplier

    if start.board.getToMove() == 1:
        evalMultiplier = 1
    else:
        evalMultiplier = -1

    cdef int shortestWinDepth, longestLoseDepth
    cdef Node winNode, loseNode
    # Max number of moves is 81, so 100 is always larger
    shortestWinDepth = 100
    longestLoseDepth = 0

    cdef Node i
    for i in start.children:
        newEval = minimax(i, depth - 1, -1000, 1000, 0)

        # Get best eval
        if newEval > bestEval:
            bestEval = newEval
            bestMove = i

        # Keep track of forced positions
        if newEval > 1 and i.infDepth < shortestWinDepth:
            shortestWinDepth = i.infDepth
            winNode = i

        elif newEval < -1 and bestEval < -1 and i.infDepth > longestLoseDepth:
            longestLoseDepth = i.infDepth
            loseNode = i

    cdef PyGameState finalState = PyGameState()

    # Return forced positions if necessary
    if bestEval > 1:
        finalState.c_gamestate = winNode.board

    elif bestEval < -1:
        finalState.c_gamestate = loseNode.board

    else:
        finalState.c_gamestate = bestMove.board

    return finalState

    

cdef class PyNode:
    cdef Node c_node

    def __cinit__(self, PyGameState game, int currentDepth):
        self.c_node = Node(game.c_gamestate, currentDepth)

        self.children = None

    def addChildren(self):
        self.c_node.addChildren()        
    

    def getCanonicalBoard(self):
        cdef vector[int] canonical = self.c_node.board.getCanonicalBoard()
        
        cdef np.ndarray[np.int8_t, ndim=1] result = np.zeros(198, dtype=np.int8)

        cdef int i
        for i in range(198):
            result[i] = canonical[i]

        return result



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

    def minimax_search_move(self, depth, playAsX):
        cdef boardCoords nextMove
        
        nextMove = minimaxSearchMove(self.c_gamestate, depth, playAsX)

        return [nextMove.board, nextMove.piece]

    def minimax_search_move_time(self, time, playAsX):
        cdef boardCoords nextMove

        nextMove = minimaxSearchTimeMove(self.c_gamestate, time, playAsX)
        return [nextMove.board, nextMove.piece]


    def get_required_board(self):
        return self.c_gamestate.getRequiredBoard()

    def get_to_move(self):
        return self.c_gamestate.getToMove()

    def is_valid_move(self, board, piece):
        return bool(self.c_gamestate.isValidMove(board, piece))
