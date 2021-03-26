# distutils: language = c++

from Minimax cimport Node

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

cdef class UTicTacToeGame():
    def __cinit__(self):
        self.n = 9

    def getInitBoard(self):
        cdef GameState initial

        return np.ndarray(initial.getBoardBitset(), dtype=np.uint8)

    def getBoardSize(self):
        return (199)

    def getActionSize(self):
        return 81

    def getNextState(self, np.ndarray board, int player, int action):
        cdef int board_index, piece_index
        board_index = int(action / 9)
        piece_index = action % 9

        cdef np.ndarray b = np.copy(board, dtype=np.uint8)

        cdef int[:] bmem = b

        if player == 1:
            bmem[board_index * 22 + piece_index * 2] = 1
        else:
            bmem[board_index * 22 + piece_index * 2 + 1] = 1


cdef float minimax(Node& node, int depth, float alpha, float beta, evaluate):
    cdef float bestEval, newEval

    # When a forced loss position is evaluated before a forced win position
    cdef int resetInfDepth = 0

    cdef vector[int] canonical
    cdef np.ndarray[np.int8_t, ndim=1] nparray

    # Check if depth reached or game over
    if (depth <= 0 or node.board.getStatus() != 0):
        canonical = node.board.getCanonicalBoard()
        nparray = np.asarray(canonical, dtype=np.int8)

        bestEval = evaluate(nparray)

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
    

def minimaxSearch(PyGameState position, int depth, evaluate):
    cdef Node start = Node(position.c_gamestate, 0)

    start.addChildren()

    # Board evals should always be between -1 and 1
    # So anything with abs > 1 is functionally the same as infinity
    cdef float bestEval = -1000
    cdef float newEval
    cdef Node bestMove = start.children[0]

    cdef int shortestWinDepth, longestLoseDepth
    cdef Node winNode, loseNode
    # Max number of moves is 81, so 100 is always larger
    shortestWinDepth = 100
    longestLoseDepth = 0

    cdef Node i
    for i in start.children:
        newEval = minimax(i, depth - 1, -100, 100, evaluate)

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


def getActionProbabilities(PyGameState position, int depth, evaluate, int temp=1):
    cdef Node start = Node(position.c_gamestate, 0)

    start.addChildren()
    
    # Store move evaluations
    cdef np.ndarray[np.float, ndim=1] moves = np.zeros(81, dtype=np.float)
    cdef float [:] movesMem = moves

    cdef float newEval, bestEval, total, minimum
    bestEval = -1000

    cdef int shortestWinDepth, longestLoseDepth, index
    cdef Node winNode, loseNode
    # Max number of moves is 81, so 100 is always larger
    shortestWinDepth = 100
    longestLoseDepth = 0

    cdef Node i
    for i in start.children:
        newEval = minimax(i, depth - 1, -100, 100, evaluate)

        index = i.board.previousMove.board * 9 + i.board.previousMove.piece

        movesMem[index] = newEval

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

    # Return forced positions if necessary
    if bestEval > 1:
        moves *= 0
        index = winNode.board.previousMove.board * 9 + winNode.board.previousMove.piece
        moves[index] = 1

    elif bestEval < -1:
        moves *= 0
        index = loseNode.board.previousMove.board * 9 + loseNode.board.previousMove.piece
        moves[index] = 1

    else:
        # Convert everything to percent

        if temp == 0:
            bestAs = np.array(np.argwhere(moves == np.max(moves))).flatten()
            bestA = np.random.choice(bestAs)
            probs = np.zeros(len(moves))
            probs[bestA] = 1
            return probs

        # Make all evaluations non-negative
        minimum = np.minimum(moves)
        if minimum < 0:
            moves += minimum

        total = np.sum(moves)

        moves /= total

    return moves


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
