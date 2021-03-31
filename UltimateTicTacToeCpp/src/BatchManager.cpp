using namespace std;

#include <vector>
#include <utility>
#include <thread>
#include <mutex>
#include <BatchManager.h>
#include <exception>
#include <vector>
#include <GameState.h>
#include <Minimax.h>
#include <chrono>

#define QUEUE_CHECK_DELAY       20ms

float RandomFloat(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

int RandomActionWeighted(vector<float> weights) {
    int action = 0;

    float weightSum = 0;
    // Pick a random move based on the weights
    for (float f : weights) {
        weightSum += f;
    }

    float rnd = RandomFloat(0, weightSum);

    for (float f : weights) {
        // Stop at the right action
        if (rnd <= f) {
            return action;
        }

        action++;
        rnd -= f;
    }

    // Throw exception because execution should always return before this

}

BatchManager::BatchManager() {

}

BatchManager::BatchManager(int _batchSize, float _cpuct, int _numSims) {
    batchSize = _batchSize;
    cpuct = _cpuct;
    numSims = _numSims;
}

void mctsWorker(int workerID, BatchManager *parent) {
    vector<MCTS> episodes;

    // Start all of the episodes
    for (int i = 0; i < parent->batchSize; i++) {
        episodes.push_back(MCTS(parent->cpuct));

        episodes.back().startNewSearch(GameState());
    }

    mtx.lock();
    ongoingGames += parent->batchSize;
    mtx.unlock();

    int actionsTaken = 0;
    int remainingGames = episodes.size();

    while(remainingGames > 0) {

        // Run all MCTS sims
        for (int i = 0; i < parent->numSims; i++) {
            batch needsEval;
            needsEval.workerID = workerID;

            // Prepare Batch
            for (MCTS ep : episodes) {
                if (ep.gameOver) {
                    continue;
                }
                // TODO: Reduce copying that is performed here
                vector<int> newEval = ep.searchPreNN();

                if (ep.evaluationNeeded) {
                    needsEval.canonicalBoards.push_back(newEval);
                }
            }

            // Posting evaluation requires the lock
            mtx.lock();
            needsEvaluation.push_back(needsEval);
            mtx.unlock();

            // Wait for the result
            while (true) {

                // If the result is available, get it
                fromNNmtx[workerID].lock();
                if (fromNN[workerID].size() > 0) {
                    needsEval = fromNN[workerID].back();
                    fromNN[workerID].pop_back();
                    fromNNmtx[workerID].unlock();
                    break;
                }

                fromNNmtx[workerID].unlock();

                // Wait until checking again
                this_thread::sleep_for(QUEUE_CHECK_DELAY);
            }

            // Batch results
            int resultsIndex = 0;
            for (int epIndex = 0; epIndex < episodes.size(); epIndex++) {
                MCTS ep = episodes[epIndex];

                if (ep.gameOver) {
                    continue;
                }

                // Skip if no evaluation was needed
                if (!ep.evaluationNeeded) {
                    continue;
                }

                ep.searchPostNN(needsEval.pis[resultsIndex], needsEval.evaluations[resultsIndex]);

                // Move to the next set of results only if an evaluation was needed
                resultsIndex += 1;
            }

        }

        // Make moves
        for (MCTS ep : episodes) {
            if (ep.gameOver) {
                continue;
            }
            vector<float> probs = ep.getActionProb();
 
            int action = RandomActionWeighted(probs);
            ep.takeAction(action);
            ep.saveTrainingExample(probs);

            int status = ep.getStatus();

            // Save the examples if the game is over
            if (status != 0) {

                // Replace status with the result value {1, -1, 0}
                if (status == 3) {
                    status = 0;
                } else {
                    status = status == 1 ? 1 : -1;
                }

                // Save all the examples
                resultsMTX.lock();
                for (trainingExampleVector ex : ep.getTrainingExamplesVector(status)) {
                    results.push_back(ex);
                }
                resultsMTX.unlock();

                ep.gameOver = true;
                remainingGames--;
                mtx.lock();
                ongoingGames--;
                mtx.unlock();
            }
        }

    }
}

void BatchManager::createMCTSThreads(int num) {
    for (int i = 0; i < num; i++) {
        vector<batch> fromNNVector;

        // Vector for sending results back
        fromNN.push_back(fromNNVector);

        // Mutex lock for sending completed evaluations
        fromNNmtx.push_back(mutex());

        // Thread the do the work
        mctsThreads.push_back(thread(mctsWorker, i, this));
    }
}


int BatchManager::getBatchSize() {
    mtx.lock();
    int result = needsEvaluation.size();
    mtx.unlock();
    return result;
}

batch BatchManager::getBatch() {
    mtx.lock();
    if (needsEvaluation.size() > 1) {
        batch newBatch = needsEvaluation.back();
        needsEvaluation.pop_back();
        mtx.unlock();
        return newBatch;
    }

    mtx.unlock();
    // TODO: Throw exception if the size is < 1 
}

void BatchManager::putBatch(batch evaluation) {
    fromNNmtx[evaluation.workerID].lock();
    fromNN[evaluation.workerID].push_back(evaluation);
    fromNNmtx[evaluation.workerID].unlock();
}

int BatchManager::getOngoingGames() {
    mtx.lock();
    int result = ongoingGames;
    mtx.unlock();
    return result;
}

vector<trainingExampleVector> BatchManager::getTrainingExamples() {
    resultsMTX.lock();
    vector<trainingExampleVector> allExamples = results;
    resultsMTX.unlock();
    return allExamples;
}
