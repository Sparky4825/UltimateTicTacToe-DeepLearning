#pragma once
using namespace std;

#include <bitset>
#include <vector>
#include <iostream>

const bitset<20> winningPosX[] = {
    0b00000000000001010100,
    0b00000001010100000000,
    0b01010100000000000000,
    0b00000100000100000100,
    0b00010000010000010000,
    0b01000001000001000000,
    0b01000000010000000100,
    0b00000100010001000000,
};

const bitset<20> winningPosO[] = {
    0b00000000000010101000,
    0b00000010101000000000,
    0b10101000000000000000,
    0b00001000001000001000,
    0b00100000100000100000,
    0b10000010000010000000,
    0b10000000100000001000,
    0b00001000100010000000,
};

struct boardCoords {
    int board, piece;
};

int checkMiniboardResultsWithTie(bitset<20> miniboard);
int checkMiniboardResults(bitset<20> miniboard);

int getMiniboardResults(bitset<20> miniboard);


class GameState {
    private:
        /**
         * Info - stores information about the GameState that is not stored on the board
         * Bits:
         * 0-3: Required board to move on
         * 4: Is there a required board (1=yes, 0=no)
         * 5: Player to move (1=X, 0=O)
         */ 

    public:
    int info;
    boardCoords previousMove;

    bool isValidMove(int board, int piece);

    bitset<20> board[9];

    GameState();

    void setToMove(int m);
    int getToMove();

    void setRequiredBoard(int requiredBoard);
    int getRequiredBoard();

    void setPosition(int boardLocation, int pieceLocation, int piece);
    int getPosition(int boardLocation, int pieceLocation);

    void move(int boardLocation, int pieceLocation);

    void updateMiniboardStatus();

    void updateSignleMiniboardStatus(int boardIndex);

    int getBoardStatus(int boardLocation);

    int getStatus();

    GameState getCopy();

    vector<GameState> allPossibleMoves();

    boardCoords absoluteIndexToBoardAndPiece(int i);

    void displayGame();
    string gameToString();

    vector<int> getCanonicalBoard();
    vector<int> getBoardBitset();

    bitset<199> getCanonicalBoardBitset();
    
};

GameState boardVector2GameState(vector<int> board);
