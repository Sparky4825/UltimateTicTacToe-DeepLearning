# distutils: language = c++

from Minimax cimport Node

cimport numpy as np

from cython.operator cimport dereference as deref

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map

import numpy as np
import coloredlogs
import logging

from random import randint

import time

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

def displayAsMove(a):
    for j in range(81):
        board = int(j / 9)
        piece = j % 9
        if a[j] != 0:
            print(f"[{board}][{piece}] {a[j]}")

cdef class PyMCTS:
    cdef MCTS mcts
    def __cinit__(self, cpuct, dirichlet=1, percent_q=0.5):
        self.mcts = MCTS(cpuct, dirichlet, percent_q)

    def startNewSearch(self, PyGameState position):
        self.mcts.startNewSearch(position.c_gamestate)

    def searchPreNN(self):
        return self.mcts.searchPreNN()

    def searchPostNN(self, vector[float] policy, float v):
        return self.mcts.searchPostNN(policy, v)
    
    @property
    def evaluationNeeded(self):
        return self.mcts.evaluationNeeded

    def getActionProb(self, display_info=False):

        if display_info:
            print("=" * 10)
            print(self.gameToString())
            print("Pi - Final prob")
            displayAsMove(self.mcts.getActionProb())
            print("Q - Values after sims")
            displayAsMove(self.mcts.getQProb())
            print("P - Initial policy")
            displayAsMove(self.mcts.getPProb())
            print("V")
            displayAsMove(self.mcts.getVProb())


            print("\n\n")

        return self.mcts.getActionProb()

    def takeAction(self, int action):
        self.mcts.takeAction(action)

    def getStatus(self):
        return self.mcts.getStatus()

    def displayGame(self):
        self.mcts.displayGame()

    def gameToString(self):
        return self.mcts.gameToString().decode('UTF-8')

    def saveTrainingExample(self, vector[float] policy, float q):
        self.mcts.saveTrainingExample(policy, q)

    def purgeTrainingExamples(self):
        self.mcts.purgeTrainingExamples()

    def getTrainingExamples(self, int result):
        """
        Returns in the format (inputs, targetPi, targetV)
        """
        inputs = np.ndarray((0, 199), dtype=np.int)
        targetV = np.ndarray((0), dtype=np.float)
        targetPi = np.ndarray((0, 81), dtype=np.float)

        cdef trainingExampleVector currentExample
        cdef vector[trainingExampleVector] trainingExamples = self.mcts.getTrainingExamplesVector(result)
        cdef int i

        for i in range(trainingExamples.size()):
            currentExample = trainingExamples[i]

            inputs = np.concatenate((inputs, [currentExample.canonicalBoard]))

            targetV = np.concatenate((targetV, [currentExample.result]))

            targetPi = np.concatenate((targetPi, [currentExample.pi]))

        return inputs, targetPi, targetV


def runSelfPlayEpisodes(evaluate, int batchSize=512, int numThreads=1, int sims=850, int pastIterations=2, float cpuct=1, double dir_a=0.8, double dir_x=0.5, float percent_q=0.5):
    cdef BatchManager m = BatchManager(batchSize, numThreads, cpuct, sims, dir_a, dir_x, percent_q)

    print("Starting search...")

    m.createMCTSThreads()

    print("Threads made")

    while m.getOngoingGames() == 0:
        print("Waiting for games to start")
        time.sleep(0.02)

    cdef int batchesEvaled = 0
    cdef int boardsEvaled = 0
    cdef int i, j

    cdef vector[float] newPis


    cdef batch start

    cdef int *boardArray

    while m.getOngoingGames() > 0:

        if m.getBatchSize() > 0:
            batchesEvaled += 1;

            if batchesEvaled % 1000 == 0:
                # 81 is max possible number of game moves
                print(f"{batchesEvaled} batches evaluated / {numThreads * sims * 81}")

            start = m.getBatch()


            boardsEvaled += start.canonicalBoards.size()


            # boards = np.asarray(<np.int[:start.canonicalBoards.size(), 199:]> &(start.canonicalBoards))

            boards = boardToNp(start)
            validMoves = validToNp(start)

            # TODO: Convert batch to numpy array
            pi, v = evaluate((boards, validMoves))


            for i in range(start.canonicalBoards.size()):
                start.evaluations.push_back(v[i])

                start.pis.push_back(newPis)
                for j in range(81):
                    start.pis[i].push_back(pi[i][j])


            # TODO: Convert result back to batch
            m.putBatch(start)
    m.saveTrainingExampleHistory()
    allTrainingExamples = m.getTrainingExamples(pastIterations)

    return c_compileExamples(allTrainingExamples)
  

def prepareBatch(trees):
    """
    Takes a python list of MCTS objects and returns a numpy array that needs to be 
    evaluated by the NN.
    """

    boards = np.zeros((len(trees), 199), dtype="int")
    moves = np.zeros((len(trees), 81), dtype="int")

    cdef int [:, :] boardsView = boards
    cdef int [:, :] movesView = moves

    cdef Node *evalNode 
    cdef int i
    cdef vector[int] canBoard, movesVector

    cdef PyMCTS pymcts

    # Loop by reference
    for i in range(len(trees)):
        pymcts = trees[i]
        canBoard = pymcts.mcts.searchPreNN()

        if pymcts.evaluationNeeded:
            movesVector = pymcts.mcts.getAllPossibleMovesVector()
            for j in range(canBoard.size()):
                boardsView[i][j] = canBoard[j]

            for j in range(81):
                movesView[i][j] = movesVector[j]

    return [boards, moves]


def batchResults(trees, pi, v):
    """
    Takes a python list of MCTS objects and the numpy array of results and puts all
    of the results into the tree objects that need them.
    """
    cdef int index = 0
    cdef PyMCTS t


    for t in trees:
        if t.evaluationNeeded:
            t.searchPostNN(pi[index], v[index])
        index += 1

cdef np.ndarray boardToNp(batch b):
    cdef int i, j

    result = np.ndarray((b.canonicalBoards.size(), 199), dtype=np.int)

    cdef int [:, :] resultView = result

    for i in range(b.canonicalBoards.size()):
        for j in range(199):
            resultView[i][j] = b.canonicalBoards[i][j]

    return result

cdef np.ndarray validToNp(batch b):
    cdef int i, j

    result = np.ndarray((b.validMoves.size(), 81), dtype=np.int)

    cdef int [:, :] resultView = result

    for i in range(b.validMoves.size()):
        for j in range(81):
            resultView[i][j] = b.validMoves[i][j]

    return result

cdef testByReference(Node *n, int depth):
    cdef Node *startNode = n

    deref(startNode).addChildren()

    cdef int i 

    if depth == 0:
        for child in deref(startNode).children:
            child.board.displayGame()

    else:
        for i in range(deref(startNode).children.size()):
            testByReference(&deref(startNode).children[i], depth - 1)

def basicTest(PyGameState position, int depth):
    cdef Node start = Node(position.c_gamestate, 0)

    testByReference(&start, depth)


def compileExamples(trainingExamples):
    cdef int numExs = sum(i[0].shape[0] for i in trainingExamples)
    cdef int i, boardsAdded, j

    boards = np.ndarray((numExs, 199), dtype=np.int)
    pis = np.ndarray((numExs, 81), dtype=np.float)
    vs = np.ndarray((numExs), dtype=np.float)

    cdef int [:, :] boardsView = boards
    cdef double [:, :] pisView = pis
    cdef double [:] vsView = vs

    boardsAdded = 0

    for ex in trainingExamples:

        for i in range(ex[0].shape[0]):
            for j in range(199):
                boardsView[boardsAdded, j] = ex[0][i][j]
            for j in range(81):
                pisView[boardsAdded, j] = ex[1][i][j]
            vsView[boardsAdded] = ex[2][i]

            boardsAdded += 1

    return boards, pis, vs

cdef c_compileExamples(vector[trainingExampleVector] trainingExamples):
    cdef int numExs = trainingExamples.size()
    cdef int exIndex, i, j

    boards = np.ndarray((numExs, 199), dtype=np.int)
    valids = np.ndarray((numExs, 81), dtype=np.int)
    pis = np.ndarray((numExs, 81), dtype=np.float)
    vs = np.ndarray((numExs), dtype=np.float)

    cdef int [:, :] boardsView = boards
    cdef double [:, :] pisView = pis
    cdef double [:] vsView = vs
    cdef int [:, :] validsView = valids



    for exIndex in range(numExs):

        for j in range(199):
            boardsView[exIndex, j] = trainingExamples[exIndex].canonicalBoard[j]
        for j in range(81):
            pisView[exIndex, j] = trainingExamples[exIndex].pi[j]
            validsView[exIndex, j] = trainingExamples[exIndex].validMoves[j]
        vsView[exIndex] = trainingExamples[exIndex].result

    return [boards, valids], pis, vs
