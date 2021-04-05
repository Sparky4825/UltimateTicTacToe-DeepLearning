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

struct board2D {
    vector<vector<int>> board = vector<vector<int>>(99, vector<int>(2));
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


    /**
     * Gets the current board in the Canonical form for
     * input to the NN.
     * 
     * The board is stored in the form 9 boards of 22 bits.
     * 
     * Bits 0-17 store the state of each of the 9 spots. If 
     * the first bit is set, the player to move has this 
     * spot, if the second bit is set, the opposing player
     * has the spot.
     * 
     * Bit 18 is set if the player to move has won the board
     * Bit 19 is set if the opposing player has won the board
     * Bit 20 is set if the board is tied
     * 
     * Bit 21 is set if the player is allowed to move on the
     * board
     * 
     * Bit 199 is unused; it is leftover from saving toMove
     * on a full board.
     * 
     * Board is then converted to a vector<int> for output
     */
    vector<int> getCanonicalBoard();


    vector<int> getBoardBitset();

    /**
     * Gets the current board in a canonical form with
     * two channels, one for each player. This is for
     * training the neural network and allowing it to
     * use the same filters for both players.
     * 
     * Returns a shape of (99, 2)
     * 
     * Each of the 9 boards are made up of 11 bits.
     * 
     * Bits 0-8 are set if the player has the spot.
     * Bit 9 is set if the player is allowed to move on the
     * board.
     * Bit 10 is set if the board is tied or the opposing player has won
     * 
     * (Bit 10 represents 'this board can never be won')
     * 
     * All spot bits will be set on boards that are won.
     * No spot bits will be set on boards that are lost.
     * No spot bits will be set on boards that are tied.
     */
    board2D get2DCanonicalBoard();

    bitset<199> getCanonicalBoardBitset();
    
};

GameState boardVector2GameState(vector<int> board);
