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
    float result;
    vector<float> pi;
    int timesSeen = 1;
};

class MCTS {
    float cpuct;

    vector<trainingExample> trainingPositions;


    public:
    Node rootNode;
    Node *currentNode;


        MCTS();
        MCTS(float _cupct);

        bool gameOver = false;


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

vector<vector<int>> getSymmetriesBoard(vector<int> board);
vector<vector<float>> getSymmetriesPi(vector<float> pi);
vector<trainingExampleVector> getSymmetries(trainingExampleVector position);


int findCanonicalRotation(vector<int> board);

vector<int> getCanonicalBoardRotation(vector<int> board);
trainingExampleVector getCanonicalTrainingExampleRotation(trainingExampleVector ex);
