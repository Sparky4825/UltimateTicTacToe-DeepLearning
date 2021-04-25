#pragma once

#include "MonteCarlo.h"
#include <vector>
#include <string>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

struct trainingParams {
    int sims;
    float cpuct;
    double dirA, dirX;
    float percentQ;
    int tempThreshold;
};

class SelfPlayManager {
    private:
    trainingParams params;

    public:
    SelfPlayManager(trainingParams _params);
};

std::vector<trainingExample> RunSelfPlayEpisodes(std::unique_ptr<tflite::FlatBufferModel>& model, trainingParams params);

void tfLiteTest(std::string modelPath);

