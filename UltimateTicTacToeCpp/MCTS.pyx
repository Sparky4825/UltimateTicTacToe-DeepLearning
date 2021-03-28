# distutils: language = c++

from Minimax cimport Node

cimport numpy as np

from cython.operator cimport dereference as deref

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map

import numpy as np
import coloredlogs
import logging

from random import randint

cdef extern from "math.h":
    double sqrt(double m)

cdef class PyGameState:
    cdef GameState c_gamestate


    def __cinit__(self):
        self.c_gamestate = GameState()

    def get_copy(self):
        new_copy = PyGameState()
        new_copy.c_gamestate = self.c_gamestate.getCopy()

        return new_copy

    def move(self, board, piece):
        self.c_gamestate.move(board, piece)

    def get_status(self):
        return self.c_gamestate.getStatus()

    def get_board_status(self, board):
        return self.c_gamestate.getBoardStatus(board)

    def get_position(self, board, piece):
        return self.c_gamestate.getPosition(board, piece)

    def get_required_board(self):
        return self.c_gamestate.getRequiredBoard()

    def get_to_move(self):
        return self.c_gamestate.getToMove()

    def is_valid_move(self, board, piece):
        return bool(self.c_gamestate.isValidMove(board, piece))


# cdef cppclass MCTS:
#     float cpuct
#     int numMCTSSims 
#     object evaluate, log
#     Node rootNode
#     Node *currentNode
#     boolean evaluationNeeded

#     MCTS():
#         cpuct = 1 


#     # def __cinit__(self, args, evaluate):
#     #     self.evaluate = evaluate

#     #     self.cpuct = args.cpuct
#     #     self.numMCTSSims = args.numMCTSSims

#     #     self.log = logging.getLogger(self.__class__.__name__)

#     #     coloredlogs.install(level="INFO", logger=self.log)
#     #     self.evaluationNeeded = False

#     void startNewSearch (GameState position):
#         self.rootNode = Node(position, 0)

#     # def getActionProb(self, PyGameState position, int temp=1):
#     #     """
#     #     Performs the given number of simulations of MCTS starting from the given
#     #     position.
#     #     """

#     #     cdef Node start = Node(position.c_gamestate, 0)

#     #     cdef int[81] counts
#     #     cdef int maxCount = 0
#     #     cdef int action
#     #     cdef vector[int] maxCountAction
#     #     cdef int totalCount

#     #     print("HERE")

#     #     # Run the simulations
#     #     for i in range(self.numMCTSSims):
#     #         self.search(start)

#     #     print("done searchign")


#     #     # Get the counts of every child
#     #     for child in start.children:
#     #         action = child.board.previousMove.board * 9 + child.board.previousMove.piece
#     #         counts[action] = child.n

#     #         if temp == 0:
#     #             if child.n > maxCount:
#     #                 # New best move found, remove all others
#     #                 maxCountAction.clear()
#     #                 maxCountAction.push_back(action)
#     #                 maxCount = child.n

#     #             elif child.n == maxCount:
#     #                 # Equally good move found, add it to list
#     #                 maxCountAction.push_back(action)

#     #         else:
#     #             counts[action] = child.n
#     #             totalCount += child.n

#     #     return list(counts)

#     #     # Select only the best move (at random if multiple best moves)
#     #     if temp == 0:
#     #         action = randint(0, maxCountAction.size() - 1)
#     #         counts[action] = 1

#     #     else:
#     #         # Normalize to probabilities
#     #         for i in range(81):
#     #             counts[i] /= totalCount

#     #     return list(counts)

#     void backpropagate( Node *finalNode, float result) except *:
#         cdef Node *currentNode = deref(finalNode).parent

#         while deref(currentNode).parent is not NULL:
#             deref(currentNode).n += 1

#             if deref(currentNode).board.getToMove() == 1:
#                 deref(currentNode).w += result
#             else:
#                 deref(currentNode).w += result * -1

#             currentNode = deref(currentNode).parent

#     vector[int] searchPreNN():
#         """
#         Performs the first half of a MCTS simulation.
        
#         Returns the pointer to a node to evaluate if a NN evaluation is needed. Returns
#         a null pointer if a NN evaluation is not needed (ie terminal node is reached).
#         """
#         # Select a node
#         cdef Node *currentNode = &rootNode
#         cdef Node *bestAction
#         cdef Node *child

#         deref(currentNode).addChildren()

#         cdef float bestUCB = -1 * FLT_MAX

#         cdef float u, q, v

#         cdef int status, i

#         cdef vector[int] empty

#         # Search until an unexplored node is found
#         while deref(currentNode).hasChildren:
#             deref(currentNode).n += 1

#             # Pick the action with the highest upper confidence bound
#             bestUCB = -1 * FLT_MAX
#             for i in range(deref(currentNode).children.size()):
#                 child = &deref(currentNode).children[i]
#                 if deref(child).n > 0:
#                     u = (deref(child).w / deref(child).n) + cpuct * deref(child).p * ( sqrt(deref(currentNode).n) / (1 + deref(child).n)  )
#                 else:
#                     # Always expore an unexplored node
#                     u = FLT_MAX

#                 if u > bestUCB:
#                     bestUCB = u
#                     bestAction = child

#             currentNode = bestAction

#             status = deref(currentNode).board.getStatus()
#             # If the game is over, backpropagate results, NN eval is not needed

#             # Check if the game is over
#             if status == 1:
#                 backpropagate(currentNode, 1)
#                 evaluationNeeded = False
#                 return empty

#             elif status == 2:
#                 backpropagate(currentNode, -1)
#                 evaluationNeeded = False
#                 return empty

#             # Draw has very little value
#             elif status == 3:
#                 backpropagate(currentNode, -0.00001)
#                 evaluationNeeded = False
#                 return empty


#         # A neural network evaluation is needed
#         deref(currentNode).addChildren()

#         evaluationNeeded = True
#         currentNode = currentNode
#         return deref(currentNode).board.getCanonicalBoard()

#     void searchPostNN(policy, float v):
#         cdef int validAction, index, i
#         cdef float totalValidMoves = 0
#         cdef int numValidMoves = 0
#         cdef Node *child

#         # Save policy value
#         # Normalize policy values based on which moves are valid
#         for i in range(deref(currentNode).children.size()):
#             child = &deref(currentNode).children[i]

#             validAction = deref(child).board.previousMove.board * 9 + deref(child).board.previousMove.piece
            
#             totalValidMoves += policy[validAction]
#             numValidMoves += 1

#         if totalValidMoves > 0:
#             # Renormalize the values of all valid moves            
#             for i in range(deref(currentNode).children.size()):
#                 child = &deref(currentNode).children[i]
#                 validAction = deref(child).board.previousMove.board * 9 + deref(child).board.previousMove.piece
#                 deref(child).p = policy[validAction] / totalValidMoves
#         else:
#             # All valid moves masked, doing a workaround
#             # self.log.warning("All valid moves masked, doing a workaround")

#             # Be careful, for loops create copies, not references
#             for i in range(deref(currentNode).children.size()):
#                 child = &deref(currentNode).children[i]
#                 validAction = deref(child).board.previousMove.board * 9 + deref(child).board.previousMove.piece
#                 deref(child).p = 1 / numValidMoves

#         backpropagate(currentNode, v)


#     # float search(Node &node) except *:

#     #     cdef int result = node.board.getStatus()
#     #     cdef int currentPlayer = node.board.getToMove()

#     #     cdef float value
#     #     cdef float totalValidMoves = 0
#     #     cdef int numValidMoves = 0
#     #     cdef vector[float] policy
#     #     cdef int index

#     #     print("ENTERING A SEaRCH")
#     #     print(node.n)
#     #     print(node.depth)
#     #     node.n += 1

#     #     print("n adjusted A SEaRCH")

#     #     if currentPlayer == 2:
#     #         currentPlayer = -1

#     #     # Check if the game is over
#     #     if result == 1:
#     #         return -1 * currentPlayer

#     #     elif result == 2:
#     #         return currentPlayer

#     #     # Draw has very little value
#     #     elif result == 3:
#     #         return 0.0001

#     #     if node.evaluationPerformed == 0:
#     #         # Perform NN evaluation
#     #         policy, value = evaluate( np.asarray(node.board.getCanonicalBoard(), dtype=np.uint8) )

#     #         # Flip value if player 2 is to move
#     #         if node.board.getToMove() == 2:
#     #             value *= -1

#     #         # Save the policy for all valid moves to the child nodes
#     #         node.addChildren()


#     #         for child in node.children:
#     #             validAction = child.board.previousMove.board * 9 + child.board.previousMove.piece
#     #             totalValidMoves += policy[validAction]
#     #             numValidMoves += 1

#     #         if totalValidMoves > 0:
#     #             # Renormalize the values of all valid moves            
#     #             index = -1
#     #             for child in node.children:
#     #                 index += 1
#     #                 validAction = child.board.previousMove.board * 9 + child.board.previousMove.piece
#     #                 node.children[index].p = policy[validAction] / totalValidMoves
#     #         else:
#     #             # All valid moves masked, doing a workaround
#     #             self.log.warning("All valid moves masked, doing a workaround")

#     #             index = -1
#     #             # Be careful, for loops create copies, not references
#     #             for child in node.children:
#     #                 index += 1
#     #                 validAction = child.board.previousMove.board * 9 + child.board.previousMove.piece
#     #                 node.children[index].p = 1 / numValidMoves

#     #         node.evaluationPerformed = 1

#     #         # Return negative because its for the perspective of the other player
#     #         return -1 * value

#     #     cdef float bestUCB = -1 * FLT_MAX

#     #     cdef float u, q, v

#     #     cdef Node *bestAction

#     #     # Pick the action with the highest upper confidence bound
#     #     for child in node.children:
#     #         if child.n > 0:
#     #             u = (child.w / child.n) + self.cpuct * child.p * ( sqrt(node.n) / (1 + child.n)  )
#     #         else:
#     #             # Always expore an unexplored node
#     #             bestUCB = FLT_MAX
#     #             u = FLT_MAX

#     #         if u > bestUCB:
#     #             bestUCB = u
#     #             bestAction = &child

#     #     # Start a recursive search on the best action
#     #     v = self.search(deref(bestAction))

#     #     deref(bestAction).w += v

#     #     return -1 * v
        


cdef class PyMCTS:
    cdef MCTS mcts
    def __cinit__(self, args):
        self.mcts = MCTS(args.cpuct)

    def startNewSearch(self, PyGameState position):
        self.mcts.startNewSearch(position.c_gamestate)

    def searchPreNN(self):
        return self.mcts.searchPreNN()

    def searchPostNN(self, vector[float] policy, float v):
        return self.mcts.searchPostNN(policy, v)
    
    @property
    def evaluationNeeded(self):
        return self.mcts.evaluationNeeded



def prepareBatch(trees):
    """
    Takes a python list of MCTS objects and returns a numpy array that needs to be 
    evaluated by the NN.
    """

    boards = np.zeros((len(trees), 199), dtype="int")

    cdef int [:, :] boardsView = boards

    cdef Node *evalNode 
    cdef int i
    cdef vector[int] canBoard

    cdef PyMCTS pymcts

    # Loop by reference
    for i in range(len(trees)):
        pymcts = trees[i]
        canBoard = pymcts.mcts.searchPreNN()

        for j in range(canBoard.size()):
            boardsView[i][j] = canBoard[j]

    return boards


def batchResults(trees, evaluations):
    """
    Takes a python list of MCTS objects and the numpy array of results and puts all
    of the results into the tree objects that need them.
    """
    cdef int index = 0
    cdef PyMCTS t


    for t in trees:
        if t.evaluationNeeded:
            t.searchPostNN(evaluations[0][index], evaluations[1][index])
            index += 1


cdef testByReference(Node *n, int depth):
    cdef Node *startNode = n

    deref(startNode).addChildren()

    cdef int i 

    if depth == 0:
        for child in deref(startNode).children:
            child.board.displayGame()

    else:
        for i in range(deref(startNode).children.size()):
            testByReference(&deref(startNode).children[i], depth - 1)

def basicTest(PyGameState position, int depth):
    cdef Node start = Node(position.c_gamestate, 0)

    testByReference(&start, depth)
