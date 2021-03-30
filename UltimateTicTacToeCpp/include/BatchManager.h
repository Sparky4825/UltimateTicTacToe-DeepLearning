#pragma once
using namespace std;

#include <vector>
#include <thread>
#include <mutex>
#include <MonteCarlo.h>

struct batch {
    bool batchRetrieved = true;
    vector<int> canonicalBoard;
    int workerID;
    float evaluation;
    vector<float> pi;
};

class BatchManager {
    private:
    vector<thread> mctsThreads;
    vector<trainingExampleVector> results;
    vector<batch> needsEvaluation;
    vector<vector<batch>> completedEvaluation;

    mutex mtx;
    bool working = true;
    int ongoingGames;

    void mctsWorker(int workerID);

    
    public:
    BatchManager();

    /**
     * Starts the given number MCTS worker threads.
     * 
     * @param num The number of threads to start
     */
    void createMCTSThreads(int num);

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

};
