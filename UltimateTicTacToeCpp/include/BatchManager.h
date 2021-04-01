#pragma once
using namespace std;

#include <vector>
#include <thread>
#include <mutex>
#include <MonteCarlo.h>


#define CPUCT_DEFAULT           1

#define NUM_THREADS             2
#define BATCH_SIZE_DEFAULT      512
#define SIMS_DEFAULT            750

struct batch {
    bool batchRetrieved = true;
    vector<vector<int>> canonicalBoards;
    int workerID;
    vector<float> evaluations;
    vector<vector<float>> pis;
};

class BatchManager {
private:

    bool working = true;

public:
    BatchManager();
    BatchManager(int _batchSize, float _cpuct, int _numSims);

    int batchSize = BATCH_SIZE_DEFAULT, numSims = SIMS_DEFAULT;
    float cpuct = CPUCT_DEFAULT;
    int numThreads = NUM_THREADS;


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

    vector<trainingExampleVector> getTrainingExamples();

};

void mctsWorker(int workerID, BatchManager* parent);

void simple();

float RandomFloat(float a, float b);
int RandomActionWeighted(vector<float> weights);
