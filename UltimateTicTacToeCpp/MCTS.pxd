from libcpp.vector cimport vector
from libcpp cimport bool as boolean
from libcpp.string cimport string
from Minimax cimport Node

cdef extern from "src/GameState.cpp":
    pass

cdef extern from "src/MonteCarlo.cpp":
    pass

cdef extern from "limits.h":
    cdef float FLT_MAX


cdef extern from "include/GameState.h":
    cdef struct boardCoords:
        int board, piece

    cdef cppclass GameState:
        GameState() except +
        void move(int, int)
        GameState getCopy()
        int getStatus()
        int getBoardStatus(int)
        int getBoard

        int getRequiredBoard()
        int getToMove()

        int getPosition(int, int)

        int isValidMove(int, int)


        vector[int] getCanonicalBoard()
        vector[int] getBoardBitset()

        boardCoords previousMove

cdef extern from "include/MonteCarlo.h":
    cdef struct trainingExampleVector:
        vector[int] canonicalBoard
        int result
        vector[float] pi

    cdef cppclass MCTS:
        MCTS()
        MCTS(float _cpuct)

        void startNewSearch(GameState position)
        void backpropagate(Node *finalNode, float result)

        vector[int] searchPreNN()

        void searchPostNN(vector[float] policy, float v)

        boolean evaluationNeeded

        vector[float] getActionProb()
        void takeAction(int actionIndex)
        int getStatus()
        void displayGame()
        string gameToString()

        void saveTrainingExample(vector[float] pi)
        vector[trainingExampleVector] getTrainingExamplesVector(int result) except +
        void purgeTrainingExamples()

