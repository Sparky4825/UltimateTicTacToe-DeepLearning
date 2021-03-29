#include <MonteCarlo.h>
#include <limits>

using namespace std;

MCTS::MCTS() {
    cpuct = 1;
}

MCTS::MCTS(float _cpuct) {
    cpuct = _cpuct;
}

void MCTS::startNewSearch(GameState position) {
    rootNode = Node(position, 0);
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

        examplesVector.push_back(newPosition);

    }

    return examplesVector;
}

void MCTS::purgeTrainingExamples() {
    /**
     * Clears the training positions from the memory.
     */
    trainingPositions.clear();
}
