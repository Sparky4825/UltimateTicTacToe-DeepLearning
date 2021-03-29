#pragma once
using namespace std;

#include <bitset>
#include <vector>
#include <iostream>
#include <GameState.h>
#include <Minimax.h>
#include <limits>

struct trainingExample {
    bitset<199> canonicalBoard;
    int result;
    float pi[81];
};

struct trainingExampleVector {
    vector<int> canonicalBoard;
    int result;
    vector<float> pi;
};

class MCTS {
    float cpuct;
    Node rootNode;
    Node *currentNode;

    vector<trainingExample> trainingPositions;


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

        void saveTrainingExample(vector<float> pi);
        vector<trainingExample> getTrainingExamples(int result);
        vector<trainingExampleVector> getTrainingExamplesVector(int result);
        void purgeTrainingExamples();

};
