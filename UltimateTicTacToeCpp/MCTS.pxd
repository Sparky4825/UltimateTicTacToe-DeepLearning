from libcpp.vector cimport vector
from libcpp cimport bool as boolean

cdef extern from "src/GameState.cpp":
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
