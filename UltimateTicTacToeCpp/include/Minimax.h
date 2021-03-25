#pragma once
using namespace std;

#include <GameState.h>
#include <bitset>
#include <vector>
#include <iostream>
#include <unordered_map>

const int c1 = 2, c2 = 1, cw = 10, cl = 0, ct = 0;

struct constants {
    int c1 = 2, c2 = 1, cw = 10, cl = 0, ct = 0;
};

struct dualEvals {
    float x, o;
};

float evaluate(GameState board, constants c);

float miniboardEvalOneSide(bitset<20> miniboard, int side, constants c);

const int winningPossibilities[9][4][2] = {
    {{1, 2}, {4, 8}, {3, 6}, {-1, -1}},
    {{4, 7}, {0, 2}, {-1, -1}, {-1, -1}},
    {{4, 6}, {0, 1}, {5, 8}, {-1, -1}},
    {{0, 6}, {4, 5}, {-1, -1}, {-1, -1}},
    {{0, 8}, {1, 7}, {2, 6}, {3, 5}},
    {{3, 4}, {2, 8}, {-1, -1}, {-1, -1}},
    {{0, 3}, {7, 8}, {4, 2}, {-1, -1}},
    {{6, 8}, {1, 4}, {-1, -1}, {-1, -1}},
    {{0, 4}, {6, 7}, {2, 5}, {-1, -1}},
};

/**
 * Calculated locations of all winning possibilities for x. Add 1 for o
 */
const int winningPossibilitiesLocations[9][4][2] =  {
    {{4, 6}, {10, 18}, {8, 14}, {-1, -1}},
    {{10, 16}, {2, 6}, {-1, -1}, {-1, -1}},
    {{10, 14}, {2, 4}, {12, 18}, {-1, -1}},
    {{2, 14}, {10, 12}, {-1, -1}, {-1, -1}},
    {{2, 18}, {4, 16}, {6, 14}, {8, 12}},
    {{8, 10}, {6, 18}, {-1, -1}, {-1, -1}},
    {{2, 8}, {16, 18}, {10, 6}, {-1, -1}},
    {{14, 18}, {4, 10}, {-1, -1}, {-1, -1}},
    {{2, 10}, {14, 16}, {6, 12}, {-1, -1}},
};

struct significances {
    float sigsX[9];
    float sigsO[9];
};

significances calcSignificances(bitset<20> fullBoard[9], float evaluationsX[9], float evaluationsY[9]);

const int wonSig = 10, lostSig = 0, tieSig = 0;

class Node{
    private:
        float eval;
        bool hasChildren = false;


    public:
        Node(GameState currentBoard, int currentDepth);
        Node();
        GameState board;

        int infDepth = -1;
        int depth;
        bool pruned = false;

        vector<Node> children;
        void addChildren();

        float getEval();

        

};

struct nodeAndEval {
    Node n;
    float e;
};

bool compareEval(nodeAndEval a, nodeAndEval b);


struct timeLimitedSearchResult {
    bool complete = false;
    float result;
};



float minimax(Node (&node), int depth, float alpha, float beta, bool maximizingPlayer, constants c);
timeLimitedSearchResult minimaxTimeLimited(Node (&node), int depth, float alpha, float beta, bool maximizingPlayer, int time, constants c);

GameState minimaxSearch(GameState position, int depth, bool playAsX);
GameState minimaxSearch(GameState position, int depth, bool playAsX, constants c);

boardCoords minimaxSearchMove(GameState position, int depth, bool playAsX);
boardCoords minimaxSearchMove(GameState position, int depth, bool playAsX, constants c);

GameState minimaxSearchTime(GameState position, int time, bool playAsX);
GameState minimaxSearchTime(GameState position, int time, bool playAsX, constants c);

boardCoords minimaxSearchTimeMove(GameState position, int time, bool playAsX);
boardCoords minimaxSearchTimeMove(GameState position, int time, bool playAsX, constants c);

int computerVcomputer(int depth1, constants c1, int depth2, constants c2, bool displayGames);
