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
#include <iostream>

#define QUEUE_CHECK_DELAY       1ms

int ongoingGames;


mutex mtx;
vector<batch> needsEvaluation;
vector<vector<batch>> fromNN;
mutex fromNNmtx[NUM_THREADS];

thread *mctsThreads[NUM_THREADS];


mutex resultsMTX;
vector<trainingExampleVector> results;

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

void simple() {
    // this_thread::sleep_for(1000ms);
    for (int i = 0 ; i < 4; i++) {
    // cout << "Working\n";

    this_thread::sleep_for(100ms);

    }
}

void mctsWorker(int workerID, BatchManager *parent) {
    cout << "Thread created!";

    vector<MCTS> episodes;

    // Start all of the episodes
    for (int i = 0; i < parent->batchSize; i++) {
        episodes.push_back(MCTS(parent->cpuct));

    }

    for (MCTS &ep : episodes) {
        ep.startNewSearch(GameState());
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
            for (MCTS &ep : episodes) {
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
                    // cout << "GOT RESULTS BACK\n";
                    break;
                }

                // cout << "Results are not back\n";

                fromNNmtx[workerID].unlock();

                // Wait until checking again
                this_thread::sleep_for(QUEUE_CHECK_DELAY);
            }

            // cout << "Broken out of loop EP SIZE " << episodes.size() << '\n';



            // Batch results
            int resultsIndex = 0;
            for (int epIndex = 0; epIndex < episodes.size(); epIndex++) {
                MCTS ep = episodes[epIndex];

                if (ep.gameOver) {
                    cout << "Game over!\n";
                    continue;
                }

                // Skip if no evaluation was needed
                if (!ep.evaluationNeeded) {
                    // cout << "Evaluation was not needed\n";
                    continue;
                }

                // cout << "Starting search post nn\n";

                // ep.currentNode->board.displayGame();

                // cout << "Game diplsayed\n";

                ep.searchPostNN(needsEval.pis[resultsIndex], needsEval.evaluations[resultsIndex]);

                // cout << "Finishing search post nn\n";

                // Move to the next set of results only if an evaluation was needed
                resultsIndex += 1;
            }

        }

        cout << "Taking action!\n";

        // Make moves
        for (MCTS &ep : episodes) {
            if (ep.gameOver) {
                continue;
            }
            vector<float> probs = ep.getActionProb();
 
            int action = RandomActionWeighted(probs);
            ep.takeAction(action);
            ep.saveTrainingExample(probs);

            ep.displayGame();

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

void BatchManager::createMCTSThreads() {
    for (int i = 0; i < numThreads; i++) {
        cout << "Creating threads\n";

        vector<batch> fromNNVector;

        // Vector for sending results back
        fromNN.push_back(fromNNVector);
        // Thread the do the work
        mctsThreads[i] = new thread(mctsWorker, i, this);
        // mctsThreads[i] = new thread(simple);

        // mctsThreads[0].join();


        cout << "Done\n";

        // mctsThreads[i].join();

    }
}

void BatchManager::stopMCTSThreads() {

    for (int i = 0; i < NUM_THREADS; i++) {
        mctsThreads[i]->join();
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
    if (needsEvaluation.size() > 0) {
        batch newBatch = needsEvaluation.back();
        needsEvaluation.pop_back();
        mtx.unlock();
        return newBatch;
    }

    mtx.unlock();

    cout << "Warning :: Empty batch returned!\n";
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
