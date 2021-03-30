using namespace std;

#include <vector>
#include <thread>
#include <mutex>
#include <BatchManager.h>
#include <exception>

BatchManager::BatchManager() {

}

void BatchManager::createMCTSThreads(int num) {
    for (int i = 0; i < num; i++) {
        // Thread the do the work
        mctsThreads.push_back(thread(mctsWorker, i));

        // Vector to give results to
        vector<batch> completedEvaluationVector;
        completedEvaluation.push_back(completedEvaluationVector);
    }
}

void BatchManager::stopMCTSThreads() {
    // Tell the threads to stop working
    mtx.lock();
    working = false;
    mtx.unlock();

    // Wait for them all to join
    for (int i = 0; i < mctsThreads.size(); i++) {
        mctsThreads[i].join();
    }

    mctsThreads.clear();
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
    mtx.lock();
    completedEvaluation[evaluation.workerID].push_back(evaluation);
    mtx.unlock();
}

int BatchManager::getOngoingGames() {
    mtx.lock();
    int result = ongoingGames;
    mtx.unlock();
    return result;
}

void BatchManager::mctsWorker(int workerID) {
    // TODO: Implement here
}
