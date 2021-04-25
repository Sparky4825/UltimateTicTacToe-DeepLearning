#pragma once

#include "MonteCarlo.h"
#include <vector>
#include <string>

struct trainingParams {
    int sims;
    float cpuct;
    double dirA, dirX;
    float percentQ;
};

class SelfPlayManager {
    private:
    trainingParams params;

    public:
    SelfPlayManager(trainingParams _params);
};

std::vector<trainingExample> RunSelfPlayEpisodes(std::string modelPath, trainingParams params);

void tfLiteTest(std::string modelPath);

