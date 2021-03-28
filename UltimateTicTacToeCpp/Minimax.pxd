from libcpp.vector cimport vector
from libcpp cimport bool as boolean

cdef extern from "src/Minimax.cpp":
    pass

cdef extern from "src/GameState.cpp":
    pass

cdef extern from "include/GameState.h":
    cdef struct boardCoords:
        char board, piece

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

        void displayGame()

        boardCoords previousMove

    cdef GameState boardVector2GameState(vector[int] board)




cdef extern from "include/Minimax.h":
    cdef cppclass Node:
        Node() except +
        Node(GameState, int) except +
        int infDepth, depth
        GameState board

        void addChildren()
        vector[Node] children
        Node *parent

        int evaluationPerformed, n
        float w, p

        boolean hasChildren


    cdef boardCoords minimaxSearchMove(GameState, int, bool)
    cdef boardCoords minimaxSearchTimeMove(GameState, int, bool)


