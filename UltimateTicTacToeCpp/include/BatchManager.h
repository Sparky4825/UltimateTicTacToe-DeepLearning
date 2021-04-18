#pragma once
using namespace std;

#include <vector>
#include <thread>
#include <mutex>
#include <MonteCarlo.h>
#include <random>
#include <fstream>

#define CPUCT_DEFAULT               4

#define MAX_THREADS                 4
#define BATCH_SIZE_DEFAULT          300
#define SIMS_DEFAULT                800
#define DIRICHLET_DEFAULT_A         0.8
#define DIRICHLET_DEFAULT_X         0.5
#define PERCENT_Q_IN_TRAINING       0.5
#define TEMP_THRESHOLD              15


struct batch {
    bool batchRetrieved = true;
    vector<vector<int>> canonicalBoards;
    int workerID;
    vector<float> evaluations;
    vector<vector<float>> pis;
    vector<vector<int>> validMoves;
};

class BatchManager {
private:

    bool working = true;

public:
    BatchManager();
    BatchManager(int _batchSize, int _numThreads, float _cpuct, int _numSims, double _dirichlet_a, float _dirichlet_x, float _percent_q);

    int batchSize = BATCH_SIZE_DEFAULT, numSims = SIMS_DEFAULT;
    float cpuct = CPUCT_DEFAULT;
    int numThreads = MAX_THREADS;
    double dirichlet_a = DIRICHLET_DEFAULT_A;
    float percent_q = PERCENT_Q_IN_TRAINING;

    // Higher values favor the original value more
    double dirichlet_x = DIRICHLET_DEFAULT_X;


    /**
     * Starts the given number MCTS worker threads.
     *
     * @param num The number of threads to start
     */
    void createMCTSThreads();

    /**
     * Stops all of the working MCTS threads.
     */
    void stopMCTSThreads();

    /**
     * Gets the next batch to be evaluated.
     *
     * Throws ______ if needsEvaluation.size() < 1
     */
    batch getBatch();

    /**
     * Gets the current number of batches that need evaluation.
     */
    int getBatchSize();


    void putBatch(batch evaluation);

    int getOngoingGames();

    /**
     * Compile the training data from the given number of iterations.
     * Data will be combined with equal weight, and all duplicates will be combined.
     * If the number of iterations requested is greater than the number saved, only
     * the available examples will be used.
     * 
     * If past iterations is not specified, all available data will be compiled.
     */
    vector<trainingExampleVector> getTrainingExamples(int pastIterations);
    vector<trainingExampleVector> getTrainingExamples();

    /**
     * Deletes all of the historical training iterations older than the given value.
     */
    void purgeHistory(int iterationsToSave);

    /**
     * Saves all of the training examples to the history and clears the current results list.
     */
    void saveTrainingExampleHistory();

};

void mctsWorker(int workerID, BatchManager* parent);

void simple();

float RandomFloat(float a, float b);
int RandomActionWeighted(vector<float> weights);

int MaxAction(vector<float> weights);

/**
 * Add the newEx into the running average for Pi and Result for exisiting.
 */
void addExampleToTrainingVector(trainingExampleVector *existing, trainingExampleVector *newEx);


