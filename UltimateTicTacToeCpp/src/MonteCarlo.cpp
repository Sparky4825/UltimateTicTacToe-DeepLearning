#include <MonteCarlo.h>
#include <limits>

#include <iostream>
using namespace std;

// These arrays are auto-generated using createGetSymmetries.py

int symmetriesMappingSingleBoard[8][9] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8},
    {2, 1, 0, 5, 4, 3, 8, 7, 6},
    {2, 5, 8, 1, 4, 7, 0, 3, 6},
    {8, 5, 2, 7, 4, 1, 6, 3, 0},
    {8, 7, 6, 5, 4, 3, 2, 1, 0},
    {6, 7, 8, 3, 4, 5, 0, 1, 2},
    {6, 3, 0, 7, 4, 1, 8, 5, 2},
    {0, 3, 6, 1, 4, 7, 2, 5, 8}
};

int symmetriesMapping[8][199] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198}, 
    {48, 49, 46, 47, 44, 45, 54, 55, 52, 53, 50, 51, 60, 61, 58, 59, 56, 57, 62, 63, 64, 65, 26, 27, 24, 25, 22, 23, 32, 33, 30, 31, 28, 29, 38, 39, 36, 37, 34, 35, 40, 41, 42, 43, 4, 5, 2, 3, 0, 1, 10, 11, 8, 9, 6, 7, 16, 17, 14, 15, 12, 13, 18, 19, 20, 21, 114, 115, 112, 113, 110, 111, 120, 121, 118, 119, 116, 117, 126, 127, 124, 125, 122, 123, 128, 129, 130, 131, 92, 93, 90, 91, 88, 89, 98, 99, 96, 97, 94, 95, 104, 105, 102, 103, 100, 101, 106, 107, 108, 109, 70, 71, 68, 69, 66, 67, 76, 77, 74, 75, 72, 73, 82, 83, 80, 81, 78, 79, 84, 85, 86, 87, 180, 181, 178, 179, 176, 177, 186, 187, 184, 185, 182, 183, 192, 193, 190, 191, 188, 189, 194, 195, 196, 197, 158, 159, 156, 157, 154, 155, 164, 165, 162, 163, 160, 161, 170, 171, 168, 169, 166, 167, 172, 173, 174, 175, 136, 137, 134, 135, 132, 133, 142, 143, 140, 141, 138, 139, 148, 149, 146, 147, 144, 145, 150, 151, 152, 153, 198}, 
    {48, 49, 54, 55, 60, 61, 46, 47, 52, 53, 58, 59, 44, 45, 50, 51, 56, 57, 62, 63, 64, 65, 114, 115, 120, 121, 126, 127, 112, 113, 118, 119, 124, 125, 110, 111, 116, 117, 122, 123, 128, 129, 130, 131, 180, 181, 186, 187, 192, 193, 178, 179, 184, 185, 190, 191, 176, 177, 182, 183, 188, 189, 194, 195, 196, 197, 26, 27, 32, 33, 38, 39, 24, 25, 30, 31, 36, 37, 22, 23, 28, 29, 34, 35, 40, 41, 42, 43, 92, 93, 98, 99, 104, 105, 90, 91, 96, 97, 102, 103, 88, 89, 94, 95, 100, 101, 106, 107, 108, 109, 158, 159, 164, 165, 170, 171, 156, 157, 162, 163, 168, 169, 154, 155, 160, 161, 166, 167, 172, 173, 174, 175, 4, 5, 10, 11, 16, 17, 2, 3, 8, 9, 14, 15, 0, 1, 6, 7, 12, 13, 18, 19, 20, 21, 70, 71, 76, 77, 82, 83, 68, 69, 74, 75, 80, 81, 66, 67, 72, 73, 78, 79, 84, 85, 86, 87, 136, 137, 142, 143, 148, 149, 134, 135, 140, 141, 146, 147, 132, 133, 138, 139, 144, 145, 150, 151, 152, 153, 198}, 
    {192, 193, 186, 187, 180, 181, 190, 191, 184, 185, 178, 179, 188, 189, 182, 183, 176, 177, 194, 195, 196, 197, 126, 127, 120, 121, 114, 115, 124, 125, 118, 119, 112, 113, 122, 123, 116, 117, 110, 111, 128, 129, 130, 131, 60, 61, 54, 55, 48, 49, 58, 59, 52, 53, 46, 47, 56, 57, 50, 51, 44, 45, 62, 63, 64, 65, 170, 171, 164, 165, 158, 159, 168, 169, 162, 163, 156, 157, 166, 167, 160, 161, 154, 155, 172, 173, 174, 175, 104, 105, 98, 99, 92, 93, 102, 103, 96, 97, 90, 91, 100, 101, 94, 95, 88, 89, 106, 107, 108, 109, 38, 39, 32, 33, 26, 27, 36, 37, 30, 31, 24, 25, 34, 35, 28, 29, 22, 23, 40, 41, 42, 43, 148, 149, 142, 143, 136, 137, 146, 147, 140, 141, 134, 135, 144, 145, 138, 139, 132, 133, 150, 151, 152, 153, 82, 83, 76, 77, 70, 71, 80, 81, 74, 75, 68, 69, 78, 79, 72, 73, 66, 67, 84, 85, 86, 87, 16, 17, 10, 11, 4, 5, 14, 15, 8, 9, 2, 3, 12, 13, 6, 7, 0, 1, 18, 19, 20, 21, 198}, 
    {192, 193, 190, 191, 188, 189, 186, 187, 184, 185, 182, 183, 180, 181, 178, 179, 176, 177, 194, 195, 196, 197, 170, 171, 168, 169, 166, 167, 164, 165, 162, 163, 160, 161, 158, 159, 156, 157, 154, 155, 172, 173, 174, 175, 148, 149, 146, 147, 144, 145, 142, 143, 140, 141, 138, 139, 136, 137, 134, 135, 132, 133, 150, 151, 152, 153, 126, 127, 124, 125, 122, 123, 120, 121, 118, 119, 116, 117, 114, 115, 112, 113, 110, 111, 128, 129, 130, 131, 104, 105, 102, 103, 100, 101, 98, 99, 96, 97, 94, 95, 92, 93, 90, 91, 88, 89, 106, 107, 108, 109, 82, 83, 80, 81, 78, 79, 76, 77, 74, 75, 72, 73, 70, 71, 68, 69, 66, 67, 84, 85, 86, 87, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51, 48, 49, 46, 47, 44, 45, 62, 63, 64, 65, 38, 39, 36, 37, 34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 40, 41, 42, 43, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 18, 19, 20, 21, 198}, 
    {144, 145, 146, 147, 148, 149, 138, 139, 140, 141, 142, 143, 132, 133, 134, 135, 136, 137, 150, 151, 152, 153, 166, 167, 168, 169, 170, 171, 160, 161, 162, 163, 164, 165, 154, 155, 156, 157, 158, 159, 172, 173, 174, 175, 188, 189, 190, 191, 192, 193, 182, 183, 184, 185, 186, 187, 176, 177, 178, 179, 180, 181, 194, 195, 196, 197, 78, 79, 80, 81, 82, 83, 72, 73, 74, 75, 76, 77, 66, 67, 68, 69, 70, 71, 84, 85, 86, 87, 100, 101, 102, 103, 104, 105, 94, 95, 96, 97, 98, 99, 88, 89, 90, 91, 92, 93, 106, 107, 108, 109, 122, 123, 124, 125, 126, 127, 116, 117, 118, 119, 120, 121, 110, 111, 112, 113, 114, 115, 128, 129, 130, 131, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 34, 35, 36, 37, 38, 39, 28, 29, 30, 31, 32, 33, 22, 23, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59, 60, 61, 50, 51, 52, 53, 54, 55, 44, 45, 46, 47, 48, 49, 62, 63, 64, 65, 198}, 
    {144, 145, 138, 139, 132, 133, 146, 147, 140, 141, 134, 135, 148, 149, 142, 143, 136, 137, 150, 151, 152, 153, 78, 79, 72, 73, 66, 67, 80, 81, 74, 75, 68, 69, 82, 83, 76, 77, 70, 71, 84, 85, 86, 87, 12, 13, 6, 7, 0, 1, 14, 15, 8, 9, 2, 3, 16, 17, 10, 11, 4, 5, 18, 19, 20, 21, 166, 167, 160, 161, 154, 155, 168, 169, 162, 163, 156, 157, 170, 171, 164, 165, 158, 159, 172, 173, 174, 175, 100, 101, 94, 95, 88, 89, 102, 103, 96, 97, 90, 91, 104, 105, 98, 99, 92, 93, 106, 107, 108, 109, 34, 35, 28, 29, 22, 23, 36, 37, 30, 31, 24, 25, 38, 39, 32, 33, 26, 27, 40, 41, 42, 43, 188, 189, 182, 183, 176, 177, 190, 191, 184, 185, 178, 179, 192, 193, 186, 187, 180, 181, 194, 195, 196, 197, 122, 123, 116, 117, 110, 111, 124, 125, 118, 119, 112, 113, 126, 127, 120, 121, 114, 115, 128, 129, 130, 131, 56, 57, 50, 51, 44, 45, 58, 59, 52, 53, 46, 47, 60, 61, 54, 55, 48, 49, 62, 63, 64, 65, 198}, 
    {0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 16, 17, 18, 19, 20, 21, 66, 67, 72, 73, 78, 79, 68, 69, 74, 75, 80, 81, 70, 71, 76, 77, 82, 83, 84, 85, 86, 87, 132, 133, 138, 139, 144, 145, 134, 135, 140, 141, 146, 147, 136, 137, 142, 143, 148, 149, 150, 151, 152, 153, 22, 23, 28, 29, 34, 35, 24, 25, 30, 31, 36, 37, 26, 27, 32, 33, 38, 39, 40, 41, 42, 43, 88, 89, 94, 95, 100, 101, 90, 91, 96, 97, 102, 103, 92, 93, 98, 99, 104, 105, 106, 107, 108, 109, 154, 155, 160, 161, 166, 167, 156, 157, 162, 163, 168, 169, 158, 159, 164, 165, 170, 171, 172, 173, 174, 175, 44, 45, 50, 51, 56, 57, 46, 47, 52, 53, 58, 59, 48, 49, 54, 55, 60, 61, 62, 63, 64, 65, 110, 111, 116, 117, 122, 123, 112, 113, 118, 119, 124, 125, 114, 115, 120, 121, 126, 127, 128, 129, 130, 131, 176, 177, 182, 183, 188, 189, 178, 179, 184, 185, 190, 191, 180, 181, 186, 187, 192, 193, 194, 195, 196, 197, 198}
};



MCTS::MCTS() {
    cpuct = 1;
}

MCTS::MCTS(float _cpuct) {
    cpuct = _cpuct;
}

void MCTS::startNewSearch(GameState position) {
    rootNode = Node(position, 0);
    rootNode.addChildren();
}

void MCTS::backpropagate(Node *finalNode, float result) {
    currentNode = finalNode->parent;

    while (currentNode->parent != NULL) {
        if (currentNode->board.getToMove() == 1) {
            currentNode->w += result;
        }

        else {
            currentNode->w += result * -1;
        }

        currentNode = currentNode->parent;
    }
}

vector<int> MCTS::searchPreNN() {
    // Select a node
    currentNode = &rootNode;
    Node *bestAction;
    Node *child;

    currentNode->addChildren();

    float bestUCB = -1 * numeric_limits<float>::max();
    float u, q, v;

    int status, i;

    vector<int> empty;

    // Search until an unexplored node is found
    while (currentNode->hasChildren) {
        currentNode->n += 1;

        // Pick the action with the highest upper confidence bound
        bestUCB = -1 * numeric_limits<float>::max();
        for (Node &child : currentNode->children) {
            if (child.n > 0) {
                u = (child.w / child.n) + cpuct * child.p * (sqrt(currentNode->n) / (1 + child.n));
            }
            else {
                // Always explore an unexplored node
                u = numeric_limits<float>::max();
            }

            if (u > bestUCB) {
                bestUCB = u;
                bestAction = &child;
            }
        }

        currentNode = bestAction;

        status = currentNode->board.getStatus();

        // If the game has ended, backpropagate the results and mark the board as visited
        if (status != 0) {
            currentNode->n++;
            if (status == 1) {
                backpropagate(currentNode, 1);
            }

            else if (status == 2) {
                backpropagate(currentNode, -1);
            }

            else {
                backpropagate(currentNode, 0);
            }

            evaluationNeeded = false;
            return empty;
        }

    }

    // A neural network evaluation is needed
    currentNode->addChildren();

    evaluationNeeded = true;
    return currentNode->board.getCanonicalBoard();
}

void MCTS::searchPostNN(vector<float> policy, float v) {
    int validAction, index, i;
    float totalValidMoves = 0;
    int numValidMoves = 0;
    Node *child;

    // Save policy value
    // Normalize policy values based on which moves are valid
    for (Node &child : currentNode->children) {
        validAction = child.board.previousMove.board * 9 + child.board.previousMove.piece;

        totalValidMoves += policy[validAction];
        numValidMoves++;
    }

    if (totalValidMoves > 0) {
        // Renormalize the values of all valid moves
        for (Node &child: currentNode->children) {
            validAction = child.board.previousMove.board * 9 + child.board.previousMove.piece;
            child.p = policy[validAction] / totalValidMoves;
        }
    } else {
        // All valid moves were masked, doing a workaround
        for (Node &child: currentNode->children) {
            validAction = child.board.previousMove.board * 9 + child.board.previousMove.piece;
            child.p = 1 / numValidMoves;
            cout << "Warning :: All valid moves masked, all valued equal.\n";
        }
    }

    backpropagate(currentNode, v);
}

vector<float> MCTS::getActionProb() {
    vector<float> result;

    float totalActionValue = 0;
    int numValidActions = 0;
    int maxActionValue = 0;
    int maxActionIndex = 0;

    for (int i = 0; i < 81; i++) {
        result.push_back(0);
    }
    
    for (Node &action : rootNode.children) {
        int actionIndex = action.board.previousMove.board * 9 + action.board.previousMove.piece;
        numValidActions++;
        totalActionValue += action.n;
    }

    for (Node &action : rootNode.children) {
        float actionValue = action.n / totalActionValue;
        int actionIndex = action.board.previousMove.board * 9 + action.board.previousMove.piece;
        result[actionIndex] = actionValue;

        if (actionValue > maxActionValue) {
            maxActionValue = actionValue;
            maxActionIndex = actionIndex;
        }
    }

    // Correct slight rounding error if necessary to ensure sum(result) = 1
    // if (totalActionValue != 1) {
    //     result[maxActionIndex] += 1 - totalActionValue;
    // }


    return result;
}

void MCTS::takeAction(int actionIndex) {
    rootNode.addChildren();
    for (Node &action : rootNode.children) {
        
        int newActionIndex = action.board.previousMove.board * 9 + action.board.previousMove.piece;
        if (actionIndex == newActionIndex) {
            rootNode = Node(action.board, 0);
            // TODO: Save tree search in between sims
            return;
        }
    }

    cout << "Warning :: No valid action was found with index " << actionIndex << '\n';
}

int MCTS::getStatus() {
    return rootNode.board.getStatus();
}

void MCTS::displayGame() {
    rootNode.board.displayGame();
}

string MCTS::gameToString() {
    return rootNode.board.gameToString();
}

void MCTS::saveTrainingExample(vector<float> pi) {
    /**
     * Saves the position in the root node to the list of training examples.
     */

    trainingExample newPosition;
    newPosition.canonicalBoard = rootNode.board.getCanonicalBoardBitset();

    // Save the pi values to the new position
    for (int i = 0; i < 81; i++) {
        newPosition.pi[i] = pi[i];
    }

    // Save which player is to move; This will later be multiplied by the result of the game
    // to get the result for the current player
    if (rootNode.board.getToMove() == 1) {
        newPosition.result = 1;
    } else {
        newPosition.result = -1;
    }

    trainingPositions.push_back(newPosition);
}

vector<trainingExample> MCTS::getTrainingExamples(int result) {
    // Update the result across all positions
    for (int i = 0; i < trainingPositions.size(); i++) {
        trainingPositions[i].result *= result;
    }

    return trainingPositions;
}

vector<trainingExampleVector> MCTS::getTrainingExamplesVector(int result) {
    for (int i = 0; i < trainingPositions.size(); i++) {
        trainingPositions[i].result *= result;
    }

    vector<trainingExampleVector> examplesVector;

    for (trainingExample example : trainingPositions) {
        trainingExampleVector newPosition;
        newPosition.result = example.result;

        // Copy the pi and board to the new training example
        for (int i = 0; i < 199; i++) {
            newPosition.canonicalBoard.push_back(example.canonicalBoard[i]);
        }

        for (int i = 0; i < 81; i++) {
            newPosition.pi.push_back(example.pi[i]);
        }

        // Add all symmetries
        for (trainingExampleVector symmetry : getSymmetries(newPosition)) {
            examplesVector.push_back(symmetry);

        }

    }

    return examplesVector;
}

void MCTS::purgeTrainingExamples() {
    /**
     * Clears the training positions from the memory.
     */
    trainingPositions.clear();
}

vector<trainingExampleVector> getSymmetries(trainingExampleVector position) {
    vector<trainingExampleVector> result;

    vector<vector<int>> boards = getSymmetriesBoard(position.canonicalBoard);
    vector<vector<float>> pis = getSymmetriesPi(position.pi);

    for (int i = 0; i < 8; i++) {
        trainingExampleVector newPosition;
        newPosition.result = position.result;
        newPosition.canonicalBoard = boards[i];
        newPosition.pi = pis[i];

        result.push_back(newPosition);
    }

    return result;
}

vector<vector<int>> getSymmetriesBoard(vector<int> board) {
    /**
     * Gets all of the equivilant position to the given board.
     */
    

    vector<vector<int>> result;


    for (int i = 0; i < 8; i++) {
        vector<int> temp;
        for (int j = 0; j < 199; j++) {
            
            temp.push_back(board[symmetriesMapping[i][j]]);
        }

        result.push_back(temp);
    }

    return result;
}

vector<vector<float>> getSymmetriesPi(vector<float> pi) {
    /**
     * Gets all of the equivilant position to the given board.
     */
    
    // This code is auto-generated using createGetSymmetries.py

    vector<vector<float>> result;


    for (int i = 0; i < 8; i++) {
        vector<float> temp;
        for (int j = 0; j < 81; j++) {
            temp.push_back(pi[symmetriesMappingSingleBoard[i][j]]);
        }

        result.push_back(temp);
    }

    return result;
}
