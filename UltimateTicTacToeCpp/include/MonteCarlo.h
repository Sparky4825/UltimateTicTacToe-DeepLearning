#pragma once
using namespace std;

#include <bitset>
#include <vector>
#include <iostream>
#include <GameState.h>
#include <Minimax.h>
#include <limits>

class MCTS {
    float cpuct;
    Node rootNode;
    Node *currentNode;

    public:
        MCTS();
        MCTS(float _cupct);

        void startNewSearch(GameState position);

        void backpropagate(Node *finalNode, float result);

        vector<int> searchPreNN();
        void searchPostNN(vector<float> policy, float v);

        bool evaluationNeeded;

        vector<float> getActionProb();
        void takeAction(int actionIndex);
        int getStatus();
        void displayGame();
        string gameToString();
};
