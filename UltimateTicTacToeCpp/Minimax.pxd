from libcpp.vector cimport vector

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

        boardCoords previousMove




cdef extern from "include/Minimax.h":
    cdef cppclass Node:
        Node() except +
        Node(GameState, int) except +
        int infDepth, depth
        GameState board

        void addChildren()
        vector[Node] children
        vector[int] getCanonicalBoard()

    cdef boardCoords minimaxSearchMove(GameState, int, bool)
    cdef boardCoords minimaxSearchTimeMove(GameState, int, bool)


