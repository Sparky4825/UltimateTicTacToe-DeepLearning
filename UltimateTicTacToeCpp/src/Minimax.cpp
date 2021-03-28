#include "Minimax.h"
#include "GameState.h"
#include <bitset>
#include <math.h>
#include <algorithm>
#include <ctime>

using namespace std;


unordered_map<bitset<20>, dualEvals> evaluationMap;


float evaluate(GameState board, constants c) {
    /**
     * Evaluates the given position.
     * @return The evaluation, Positive indicates advantage to X, negative indicates advantage to O
     */
    float miniboardEvalsX[9], miniboardEvalsO[9];
    bitset<20> (&position)[9] = board.board;

    float finalEval = 0;

    int status = board.getStatus();
    if (status == 1) { // X wins
        return numeric_limits<float>::infinity();
    } else if (status == 2) { // O wins
        return -1 * numeric_limits<float>::infinity();
    } else if (status == 3) { // Tie game
        return 0;
    }

    for (int i = 0; i < 9; i++) {
        // Check if the position is saved in the database with an evaluation
        bitset<20> currentMiniboard = position[i];
        auto matchingPattern = evaluationMap.find(currentMiniboard);

        // If the position is saved, use the saved evaluations
        if (matchingPattern != evaluationMap.end()) {
            miniboardEvalsX[i] = matchingPattern->second.x;
            miniboardEvalsO[i] = matchingPattern->second.o;
        } else {
            // Calculate the evaluations and save them
            miniboardEvalsX[i] = miniboardEvalOneSide(position[i], 1, c);
            miniboardEvalsO[i] = miniboardEvalOneSide(position[i], 2, c);

            dualEvals resultEvaluation;
            resultEvaluation.x = miniboardEvalsX[i];
            resultEvaluation.o = miniboardEvalsO[i];

            const bitset<20> positionConst = position[i]; 
            evaluationMap[positionConst] = resultEvaluation;
        }
    }

    significances sigs;
    sigs = calcSignificances(position, miniboardEvalsX, miniboardEvalsO);
    
    for (int i = 0; i < 9; i++)
    {
        finalEval += (miniboardEvalsX[i] * sigs.sigsX[i]) - (miniboardEvalsO[i] * sigs.sigsO[i]);
    }
    

    return finalEval;
}


float miniboardEvalOneSide(bitset<20> miniboard, int side, constants c) {
    /**
     * Evaluates a single miniboard for one side.
     * 
     * eval = (c1 * (w ^ 0.5)) + c2 * r)
     * 
     * Where:
     * c1, c2 - constants
     * w - spaces that are winning if taken
     * r - rows with only one spot taken (ie win in two moves)
     * 
     * cw - the value of an already won board
     * cl - the value of a lost board (should be <=0)
     * ct - the value of a tied board
     * 
     * @param side 1 to evaluate for X; 2 to evaluate for O
     */

    int r = 0, w = 0;


    // Check for win/loss

    int result = getMiniboardResults(miniboard);
    if (result == side) {
        return c.cw;
    // Check if the other side won
    } else if (result == 2 / side) {
        return c.cl;
    } else if (result == 3) {
        return c.ct;
    }

    // Calculate w and r
    int index = 0;

    // The amount to move from location to get to the corresponding bit for the other side
    int sideOffset = 0, posOffset = 0;
    if (side == 1) {
        sideOffset = 1;
    } else {
        sideOffset = 0;
        posOffset = 1;
    }

    for (int location = 1 + side; location < 20; location += 2) {
        
        if (!miniboard[location]) {
            // Check each winning possibility
            for (int i = 0; i < 4; i++) {
                if (winningPossibilitiesLocations[index][i][0] == -1) {
                    break;
                }

                if (miniboard[winningPossibilitiesLocations[index][i][0] + posOffset] && miniboard[winningPossibilitiesLocations[index][i][1] + posOffset]) {
                    w++;
                    break;
                }
            }
        } else {
            for (int i = 0; i < 4; i++) {
                // If this spot is taken by us and the other two are empty, this is a win-in-two index
                if (winningPossibilitiesLocations[index][i][0] == -1) {
                    break;
                }

                if (!miniboard[winningPossibilitiesLocations[index][i][0] + posOffset] &&
                        !miniboard[winningPossibilitiesLocations[index][i][0] + sideOffset] &&
                        !miniboard[winningPossibilitiesLocations[index][i][1] + posOffset] &&
                        !miniboard[winningPossibilitiesLocations[index][i][1] + sideOffset]) {
                    r++;
                }
            }
        }

        index += 1;
    }

    return c.c1 * sqrt(w) + c.c2 * r;
}

significances calcSignificances(bitset<20> fullBoard[9], float evaluationsX[9], float evaluationsY[9]) {
    /**
     * Calculate the significances of each miniboard based on the evaluations given.
     */
    significances result;

    int result1, result2;
    int winCoords1, winCoords2;
    int isUsableX, isUsableO;



    // Loop through each miniboard
    for (int i = 0; i < 9; i++)
    {
        
        if (fullBoard[i][0] && fullBoard[i][1]) {

            result.sigsO[i] = tieSig;
            result.sigsX[i] = tieSig;
        } else if (fullBoard[i][0]) {

            result.sigsX[i] = wonSig;
            result.sigsO[i] = lostSig;
        } else if (fullBoard[i][1]) {

            result.sigsO[i] = wonSig;
            result.sigsX[i] = lostSig;

        } else {
            // Boards that cannot be used in a win have a sig of 0
            result.sigsX[i] = 1;
            result.sigsO[i] = 1;

            isUsableO = 0;
            isUsableX = 0;
            // Loop through each winning possibility
            for (int winIndex = 0; winIndex < 4; winIndex++) {
                // Break when end of winningPossibilities is reached
                winCoords1 = winningPossibilities[i][winIndex][0];
                winCoords2 = winningPossibilities[i][winIndex][1];

                if (winCoords1 == -1) {
                    break;
                }

                // If either board is already won for the other side (or tied), sig is zero
                if (!fullBoard[winCoords1][1] && !fullBoard[winCoords2][1]) {
                    result1 = evaluationsX[winCoords1];
                    result2 = evaluationsX[winCoords2];
                    result.sigsX[i] += result1 + result2;



                    isUsableX += 1;
                }

                if (!fullBoard[winCoords1][0] && !fullBoard[winCoords2][0]) {
                    result1 = evaluationsY[winCoords1];
                    result2 = evaluationsY[winCoords2];
                    result.sigsO[i] += result1 + result2;

                    if(!isUsableO)
                        isUsableO = 1;

                }

            }

            if (!isUsableX) {
                result.sigsX[i] = 0;
            }

            if (!isUsableO) {
                result.sigsO[i] = 0;
            }
        }
    }

    return result;
};

Node::Node (GameState currentBoard, int currentDepth){
    board = currentBoard;
    depth = currentDepth;
    parent = nullptr;

}

Node::Node (GameState currentBoard, int currentDepth, Node *nodeParent) {
    board = currentBoard;
    depth = currentDepth;
    parent = nodeParent;
}

Node::Node() {
    parent = nullptr;
    depth = 0;
};

void Node::addChildren() {
    /**
     * Add all possible moves as children. Checks if children have already been added and will not add again.
     */
    if (!hasChildren) {
        for (GameState i : board.allPossibleMoves()) {
            children.push_back(Node(i, depth + 1, this));
        }
        hasChildren = true;
    }
}

bool compareEval(nodeAndEval a, nodeAndEval b) {
    // Return true if the first element should be before the second
    return a.e > b.e;
}

float minimax(Node (&node), int depth, float alpha, float beta, bool maximizingPlayer, constants c) {
    return minimaxTimeLimited(node, depth, alpha, beta, maximizingPlayer, 0, c).result;
};

timeLimitedSearchResult minimaxTimeLimited(Node (&node), int depth, float alpha, float beta, bool maximizingPlayer, int time, constants c) {
    /**
     * Calculates the evaluation of the given board to the given depth.
     * Note that updateMiniboardStatus() or updateSignleMiniboardStatus() must be called before this function.
     */

    timeLimitedSearchResult result;


    // Check if the search time has expired
    if (time > 0 && std::time(nullptr) > time) {
        result.complete = false;
        result.result = 0;
        return result;
    }

    float bestEval, newEval;

    const float inf = numeric_limits<float>::infinity();


    // When a forced loss position is evaluated before a forced win position,
    // node.infDepth is set. When a win position is evaluated for the first time,
    // node.infDepth must be reset to get an accurate reading for the forced win
    bool resetInfDepth = false;

    // Check if depth is reached or game is over
    if (depth <= 0 || node.board.getStatus() != 0) {
        bestEval = evaluate(node.board, c);

        if (isinf(bestEval)) {
            // Save the depth if the evaluation is infinite
            node.infDepth = node.depth;
        }

        result.result = bestEval;
        result.complete = true;

        return result;
    }

    // Init with worst outcome, so anything else is always better
    if (maximizingPlayer)
        bestEval = -1 * numeric_limits<float>::infinity();
    else
        bestEval = numeric_limits<float>::infinity();

    node.addChildren();

    
    for (Node i : node.children) {
        newEval = minimax(i, depth - 1, alpha, beta, !maximizingPlayer, c);

        if (maximizingPlayer) {
            // Get the highest evaluation
            bestEval = (newEval > bestEval) ? newEval : bestEval;

            // If the position is lost, the best option is the one furthest from game over
            if (newEval == -1 * numeric_limits<float>::infinity() && bestEval == newEval) {
                if (node.infDepth == -1 || node.infDepth < i.infDepth) {
                    resetInfDepth = true;
                    node.infDepth = i.infDepth;
                }
            }

            // If the position is won, the best option is the one closest from game over
            else if (newEval == numeric_limits<float>::infinity()) {
                // if (resetInfDepth) {
                //     resetInfDepth = false;
                //     cout << "Resetting infdepth " << node.infDepth <<"\n";
                //     node.infDepth = -1;
                // }

                if (node.infDepth == -1 || node.infDepth > i.infDepth) {
                    node.infDepth = i.infDepth;
                }
            }

            alpha = (alpha > newEval) ? alpha : newEval;

            // Prune the position
            if (beta <= alpha) {
                break;
            }
        } else {
            // Get the lowest evaluation
            bestEval = (newEval < bestEval) ? newEval : bestEval;

            // If the position is lost, the best option is the one furthest from game over
            if (newEval == numeric_limits<float>::infinity() && bestEval == newEval) {
                if (node.infDepth == -1 || node.infDepth < i.infDepth) {
                    resetInfDepth = true;
                    node.infDepth = i.infDepth;
                }
            }

            // If the position is won, the best option is the one closest from game over
            else if (newEval == -1 * numeric_limits<float>::infinity()) {
                // if (resetInfDepth) {
                //     resetInfDepth = false;
                //     node.infDepth = -1;
                // }

                if (node.infDepth == -1 || node.infDepth > i.infDepth) {
                    node.infDepth = i.infDepth;
                }
            }

            beta = (beta < newEval) ? beta : newEval;

            // Prune the position
            if (beta <= alpha) {
                break;
            }
        }
    }

    result.complete = true;
    result.result = bestEval;

    return result;
};

GameState minimaxSearch(GameState position, int depth, bool playAsX) {
    constants c;
    
    return minimaxSearch(position, depth, playAsX, c);
}

GameState minimaxSearch(GameState position, int depth, bool playAsX, constants c) {
    Node start = Node(position, 0);

    start.addChildren();

    const float inf = numeric_limits<float>::infinity();

    float bestEval = inf * -1;
    float newEval;
    Node bestMove = start.children[0];
    int evalMultiplier = (playAsX) ? 1 : -1;

    int shortestWinDepth = numeric_limits<int>::max();
    Node winNode, loseNode;
    int longestLoseDepth = numeric_limits<int>::min();


    for (Node i : start.children) {
        newEval = minimax(i, depth - 1, -1 * inf, inf, !playAsX, c);

        newEval *= evalMultiplier;

        if (newEval > bestEval) {
            bestEval = newEval;
            bestMove = i;
        }

        // If a forced win is possible, find the shortest path
        if (newEval == inf && i.infDepth < shortestWinDepth) {
            shortestWinDepth = i.infDepth;
            winNode = i;
        }
        
        // If a forced loss is possible, find the longest path
        else if (newEval == -1 * inf && i.infDepth > longestLoseDepth) {
            longestLoseDepth = i.infDepth;
            loseNode = i;
        }

    }

    // Return the correct forced result board if necessary
    if (bestEval == inf) {
        cout << "Forced win: " << shortestWinDepth << '\n';
        return winNode.board;
    }
    
    if (bestEval == -1 * inf) {
        cout << "Forced loss: " << longestLoseDepth << '\n';
        return loseNode.board;
    }

    return bestMove.board;
};

boardCoords minimaxSearchMove(GameState position, int depth, bool playAsX) {
    constants c;

    return minimaxSearchMove(position, depth, playAsX, c);
}

boardCoords minimaxSearchMove(GameState position, int depth, bool playAsX, constants c) {
    return minimaxSearch(position, depth, playAsX, c).previousMove;
};

GameState minimaxSearchTime(GameState position, int time, bool playAsX)  {
    constants c;
    return minimaxSearchTime(position, time, playAsX, c);
}

GameState minimaxSearchTime(GameState position, int time, bool playAsX, constants c) {

    int endTime = std::time(nullptr) + time;

    Node start = Node(position, 0);

    start.addChildren();

    // Keep track of the children and their evaluations
    int numChildren = start.children.size();
    int depth = 2;
    vector<nodeAndEval> childEvals;
    

    const float inf = numeric_limits<float>::infinity();

    float bestEval = inf * -1;
    float newEval;
    Node bestMove = start.children[0];
    Node bestFullySearchedMove = start.children[0];
    int evalMultiplier = (playAsX) ? 1 : -1;

    int shortestWinDepth = numeric_limits<int>::max();
    Node winNode, loseNode;
    int longestLoseDepth = numeric_limits<int>::min();

    for (Node i : start.children) {
        nodeAndEval childAndEval;
        childAndEval.n = i;
        childAndEval.e = evaluate(i.board, c) * evalMultiplier;
        childEvals.push_back(childAndEval);
    }



    // Loop until time expires or forced outcome is found
    while (true) {
        depth++;
        std::sort (childEvals.begin(), childEvals.end(), compareEval);

        bestFullySearchedMove = childEvals.at(0).n;

        for (nodeAndEval &j : childEvals) {
            Node i = j.n;
            timeLimitedSearchResult result;
            result = minimaxTimeLimited(i, depth - 1, -1 * inf, inf, !playAsX, endTime, c);

            // If time has expired, return the best fully searched move
            if (!result.complete) {
                return bestFullySearchedMove.board;
            }

            newEval = result.result;

            newEval *= evalMultiplier;

            // Update the stored eval value of the child
            j.e = newEval;

            if (newEval > bestEval) {
                bestEval = newEval;
                bestMove = i;
            }

            // If a forced win is possible, find the shortest path
            if (newEval == inf && i.infDepth < shortestWinDepth) {
                shortestWinDepth = i.infDepth;
                winNode = i;
            }
            
            // If a forced loss is possible, find the longest path
            else if (newEval == -1 * inf && i.infDepth > longestLoseDepth) {
                longestLoseDepth = i.infDepth;
                loseNode = i;
            }

        }

        // Return the correct forced result board if necessary
        if (bestEval == inf) {
            cout << "Forced win: " << shortestWinDepth << '\n';
            return winNode.board;
        }
        
        if (bestEval == -1 * inf) {
            cout << "Forced loss: " << longestLoseDepth << '\n';
            return loseNode.board;
        }

    }
};

boardCoords minimaxSearchTimeMove(GameState position, int time, bool playAsX) {
    constants c;
    return minimaxSearchTimeMove(position, time, playAsX, c);
}

boardCoords minimaxSearchTimeMove(GameState position, int time, bool playAsX, constants c) {
    return minimaxSearchTime(position, time, playAsX, c).previousMove;
};

int computerVcomputer(int depth1, constants c1, int depth2, constants c2, bool displayGames) {
    GameState game;
    boardCoords move;

    game.move(4, 0);
    game.move(0, 0);

    while (game.getStatus() == 0) {
        move = minimaxSearchMove(game, depth1, true, c1);

        game.move(move.board, move.piece);

        if (displayGames)
            game.displayGame();

        if (game.getStatus() != 0)
            break;

        move = minimaxSearchMove(game, depth2, false, c2);

        game.move(move.board, move.piece);

        if (displayGames)
            game.displayGame();
    }

    return game.getStatus();
}
