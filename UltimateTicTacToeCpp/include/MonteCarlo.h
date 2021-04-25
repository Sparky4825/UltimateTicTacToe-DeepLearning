#pragma once
using namespace std;

#include <bitset>
#include <vector>
#include <iostream>
#include <GameState.h>
#include <Minimax.h>
#include <limits>
#include <dirichlet.h>


struct trainingExample {
    bitset<199> canonicalBoard;
    float result;
    float pi[81];
    float q;

    vector<int> validMoves;
};

struct trainingExampleVector {
    vector<int> canonicalBoard;
    float result;
    float q;
    vector<float> pi;
    int timesSeen = 1;

    vector<int> validMoves;
};



class MCTS {
    float cpuct = 1;
    double dirichlet_a = 0.8;
    float percent_q = 0.5;

    vector<trainingExample> trainingPositions;

    mt19937 gen;
    dirichlet_distribution<mt19937> dirichlet;

    public:
    Node rootNode;
    Node *currentNode;


        MCTS();
        MCTS(float _cupct, double _dirichlet, float _percent_q);

        bool gameOver = false;


        void startNewSearch(GameState position);

        void backpropagate(Node *finalNode, float result);

        vector<int> searchPreNN();
        vector<int> getAllPossibleMovesVector();
        void searchPostNN(vector<float> policy, float v);

        /**
         * Start an iteration of the MCTS. Return true if a NN eval
         * and searchPostNNTfLite() is needed.
         */
        bool searchPreNNTFLite();

        /**
         * Finish the MCTS iteration after a NN eval. Only call if
         * necessary.
         */
        void searchPostNNTFLite(float *policyOutput, float *value);

        bool evaluationNeeded;

        vector<float> getActionProb();
        vector<float> getQProb();
        vector<float> getPProb();
        vector<float> getVProb();

        int maxActionProb();

        void takeAction(int actionIndex);
        int getStatus();
        void displayGame();
        string gameToString();

        void saveTrainingExample(vector<float> pi, float q);
        void saveTrainingExample();
        void saveTrainingExample(float *pi, float *q);

        vector<trainingExample> getTrainingExamples(int result);
        vector<trainingExampleVector> getTrainingExamplesVector(int result);
        void purgeTrainingExamples();

        vector<double> dir(double a, int dim);

};

vector<vector<int>> getSymmetriesBoard(vector<int> board);
vector<vector<float>> getSymmetriesPi(vector<float> pi);
vector<vector<int>> getSymmetriesMoves(vector<int> moves);
vector<trainingExampleVector> getSymmetries(trainingExampleVector position);


int findCanonicalRotation(vector<int> board);

vector<int> getCanonicalBoardRotation(vector<int> board);
trainingExampleVector getCanonicalTrainingExampleRotation(trainingExampleVector ex);
