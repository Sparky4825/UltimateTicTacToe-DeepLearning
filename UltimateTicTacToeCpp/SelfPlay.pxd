from libcpp.vector cimport vector
from libcpp cimport bool as boolean
from libcpp.string cimport string
from Minimax cimport Node

cdef extern from "src/GameState.cpp":
    pass

cdef extern from "src/MonteCarlo.cpp":
    pass

cdef extern from "src/BatchManager.cpp":
    pass


cdef extern from "limits.h":
    cdef float FLT_MAX

cdef vector[int].iterator iter  #iter is declared as being of type vector<int>::iterator


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

        string gameToString()

        void writeCanonicalBoard(float *output);
        void writeValidMoves(float *output);


        vector[int] getCanonicalBoard()
        vector[int] getBoardBitset()

        boardCoords previousMove

cdef extern from "include/MonteCarlo.h":
    cdef struct trainingExampleVector:
        vector[int] canonicalBoard
        float result
        vector[float] pi
        vector[int] validMoves

    cdef cppclass MCTS:
        MCTS()
        MCTS(float _cpuct, double dirichlet_a, float percent_q)

        Node rootNode

        Node *currentNode

        vector[double] dir(double, int)

        void startNewSearch(GameState position)
        void backpropagate(Node *finalNode, float result)

        vector[int] searchPreNN()
        vector[int] getAllPossibleMovesVector()

        void searchPostNN(vector[float] policy, float v)

        boolean evaluationNeeded

        vector[float] getActionProb()
        vector[float] getQProb()
        vector[float] getPProb()
        vector[float] getVProb()
        void takeAction(int actionIndex)
        int getStatus()
        void displayGame()
        string gameToString()

        boolean searchPreNNTFLite()
        void searchPostNNTFLite(float*, float*)

        void saveTrainingExample()
        void saveTrainingExample(vector[float] pi, float w)
        vector[trainingExampleVector] getTrainingExamplesVector(int result) except +
        void purgeTrainingExamples()

cdef extern from "include/BatchManager.h":
    int RandomActionWeighted(vector[float] weights);

    int MaxAction(vector[float] weights);


    cdef struct batch:
        boolean batchRetrieved
        vector[vector[int]] canonicalBoards
        int workerID
        vector[float] evaluations
        vector[vector[float]] pis
        vector[vector[int]] validMoves

    cdef cppclass BatchManager:
        BatchManager()
        BatchManager(int _batchSize, int _numThreads, float _cpuct, int _numSims, double _dirichlet_a, float _dirichlet_x, float _percent_q)

        int batchSize, numSims, numThreads
        float cpuct, percent_q

        void createMCTSThreads()
        void stopMCTSThreads()

        batch getBatch()
        int getBatchSize()

        void putBatch(batch evaluation)

        int getOngoingGames()

        vector[trainingExampleVector] getTrainingExamples(int pastIterations)
        vector[trainingExampleVector] getTrainingExamples()

        void purgeHistory(int iterationsToSave)

        void saveTrainingExampleHistory()


    cdef void saveTrainingDataToFile(string filename)
    cdef void loadTrainingDataFromFile(string filename)
    