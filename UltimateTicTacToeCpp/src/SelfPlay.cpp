#include <cstdio>
#include "SelfPlay.hpp"

#include <vector>
#include <bitset>
#include <chrono>

#include <iostream>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "GameState.h"
#include "MonteCarlo.h"
#include "BatchManager.h"

std::vector<trainingExample> RunSelfPlayEpisodes(std::unique_ptr<tflite::FlatBufferModel>& model, trainingParams params) {

    // Start the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;

    builder(&interpreter);

    interpreter->AllocateTensors();

    // Start the tree search
    MCTS tree = MCTS(params.cpuct, params.dirA, params.percentQ);

    tree.startNewSearch(GameState());

    int actionsTaken = 0;

    while (true) {  

        // Perform enough MCTS simulations
        for (int i = 0; i < params.sims; i++) {
            // First half of MCTS search
            if (tree.searchPreNNTFLite()) {
                // NN eval and second half if necessary

                float *boardInput = interpreter->typed_input_tensor<float>(0);
                float *validInput = interpreter->typed_input_tensor<float>(1);


                tree.currentNode->board.writeCanonicalBoard(boardInput);
                tree.currentNode->board.writeValidMoves(validInput);

                auto start = std::chrono::high_resolution_clock::now();

                interpreter->Invoke();

                auto stop = std::chrono::high_resolution_clock::now();

                float *policyOutput = interpreter->typed_output_tensor<float>(0);
                float *valueOutput  = interpreter->typed_output_tensor<float>(1);


                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  
    cout << "The invoke taken by function: "
         << duration.count() << " milliseconds" << endl;

                tree.searchPostNNTFLite(policyOutput, valueOutput);
            }
        }

        actionsTaken++;

        std::vector<float> probs = tree.getActionProb();

        tree.saveTrainingExample(probs, tree.rootNode.w / tree.rootNode.n);

        int action;

        if (actionsTaken < params.tempThreshold) {
            // Add dirichlet noise
            int numActions = tree.rootNode.children.size();
            vector<double> dir = tree.dir(params.dirA, numActions);
            int i = 0;
            for (float &prob : probs) {
                if (prob != 0) {
                    prob = params.dirX * prob + (1 - params.dirX) * dir[i];
                    i++;
                }
            }

            action = RandomActionWeighted(probs);
        } else {
            action = MaxAction(probs);
        }

        tree.takeAction(action);

        cout << tree.rootNode.board.gameToString();

        int status = tree.getStatus();

        // Save the examples if the game is over
        if (status != 0) {

            // Replace status with the result value {1, -1, 0}
            if (status == 3) {
                status = 0;
            } else {
                status = status == 1 ? 1 : -1;
            }

            return tree.getTrainingExamples(status);

        }

    }


    // printf("=== Pre-invoke Interpreter State ===\n");
    // tflite::PrintInterpreterState(interpreter.get());

    // GameState game;

    // game.move(0, 0);

    
    // std::bitset<199> canBoard = game.getCanonicalBoardBitset();
    // std::vector<int> valid = game.getAllPossibleMovesVector();

    // float* boardInput = interpreter->typed_input_tensor<float>(0);
    // float* validInput = interpreter->typed_input_tensor<float>(1);

    // game.writeCanonicalBoard(boardInput);
    // game.writeValidMoves(validInput);

    // // for (int i = 0; i < 81; i++) {
    // //     validInput[i] = valid[i];
    // // }

    // interpreter->Invoke();

    // for (auto i : interpreter->outputs()) {
    //     std::cout << i << "\n";
    // }

    // std::cout <<"=================\n\n\n";


    // for (int i = 0 ; i < 81; i++) {
    //     std::cout << policyOutput[i] << '\n';
    // }


    // std::cout << "\n\nVALUE: " << *valueOutput << '\n';


    // return vector<trainingExample>();
}

