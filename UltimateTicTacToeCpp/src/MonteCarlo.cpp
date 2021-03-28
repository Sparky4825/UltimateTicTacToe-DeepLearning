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
        currentNode->n += 1;

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

        // If the game has ended, backpropagate the results
        if (status != 0) {
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
            cout << "Warning :: All valid moves masked, all valued equal.";
        }
    }

    backpropagate(currentNode, v);
}