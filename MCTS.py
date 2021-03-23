import logging
import math
import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class Node:
    """
    Represents a single node in the MCTS tree.
    """

    def __init__(self, board, curPlayer, game, a, parent, p):
        self.board = board
        self.curPlayer = curPlayer
        self.parent = parent

        self.game = game

        self.children = []

        # The action that was used to get to this board
        self.a = a

        self.W = 0
        self.N = 0

        # Probability of taking this move according to the NN
        self.P = p

        # TODO: Replace this with an arg version
        self.cpuct = 1

    @property
    def Q(self):
        if self.N == 0:
            return 0

        return self.W / self.N

    @property
    def U(self):
        """
        Calculate the upper confidence bound (U) based on the current node and its parent.

        U is defined as

        U = Q + cpuct * P * [sqrt(N of parent) / (1 + N of child)]

        :return:
        """

        return self.Q + self.cpuct * self.P * (math.sqrt(self.parent.N) / (1 + self.N))

    def addChildren(self):
        """
        Add all possible moves as children to this Node.
        Raises RuntimeError if called more than once on the same Node.
        """
        if len(self.children) > 0:
            raise RuntimeError(
                f"Attempted to add children to a node that already has {len(self.children)} children"
            )

        valid_moves = self.game.getValidMoves(self.board, self.curPlayer)
        for i in range(self.game.getActionSize()):
            if valid_moves[i]:
                new_board, new_player = self.game.getNextState(
                    self.board, self.curPlayer, i
                )

                self.children.append(
                    Node(new_board, new_player, self.game, i, self, None)
                )


class MCTS:
    def __init__(self, game, start_node=None):
        self.start_node = start_node
        self.game = game

        # Store the board that will need evaluation by the NN
        if start_node is None:
            self.start_node = Node(
                self.game.getInitBoard(), 1, self.game, None, None, None
            )
        self.current_node = start_node

        # Save the board that the NN will need to evaluate
        self.canonicalBoard = None

        # These will be set by the NN
        self.pi = None
        self.v = None

    def searchPreNN(self):
        """
        Preforms one iteration of MCTS search and return when
        a NN evaluation is needed. Call searchPostNN() after
        the evaluation is done.

        Return True is NN eval and searchPostNN is needed,
        false if not (ie terminal node reached)
        """

        self.current_node = self.start_node

        # Step 1: Traverse to find a leaf node
        while True:

            # If a leaf node is found, pass it to NN for evaluation

            game_result = self.game.getGameEnded(
                self.current_node.board, self.current_node.curPlayer
            )
            if game_result != 0:
                # Terminal node reached, back propagate result
                while self.current_node is not None:
                    self.current_node.N += 1
                    self.current_node.W += self.v * self.current_node.curPlayer

                    # Flip value for other player
                    self.v *= -1

                    self.current_node = self.current_node.parent

                return False

            if len(self.current_node.children) == 0:
                # NN will set the values of self.
                self.canonicalBoard = self.game.getCanonicalForm(
                    self.current_node.board, self.current_node.curPlayer
                )
                return True

            # Pick node with maximum U value
            # TODO: V needs to be negative to account for the other player?
            self.current_node = max(self.current_node.children, key=lambda x: x.U)

    def searchPostNN(self):
        # Add possible moves as nodes
        self.current_node.addChildren()

        # TODO: Optimize pi loops

        # Set the P value for each child based on what the NN said
        sum_pi = 0
        for child in self.current_node.children:
            sum_pi += self.pi[child.a]

        # Renormalize so all pi values add to 1
        for child in self.current_node.children:
            child.P = self.pi[child.a] / sum_pi

        # Backpropagate

        while self.current_node is not None:
            self.current_node.N += 1
            self.current_node.W += self.v * self.current_node.curPlayer

            # Flip value for other player
            self.v *= -1

            self.current_node = self.current_node.parent

    def getActionProb(self, temp=1):
        """
        Get the probability for each possible move

        Must be called after searchPreNN and searchPostNN.

        temp controls how much the engine explores. Set to 0 for competitive
        play.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        counts = np.zeros(self.game.getActionSize())

        for action in self.start_node.children:
            if temp == 0:
                counts[action.a] = action.N

            else:
                counts[action.a] = action.N ** (1.0 / temp)

        if temp == 0:
            bestActions = np.array(np.argwhere(counts == np.max(counts))).flatten()

            randomBestAction = np.ranom.choice(bestActions)
            probs = [0] * len(counts)

            probs[randomBestAction] = 1
            return probs

        # Renormalize
        counts_sum = float(np.sum(counts))
        counts /= counts_sum

        return list(counts)

    def moveStartBoard2Action(self, action):
        new_board, new_player = self.game.getNextState(
            self.start_node.board, self.start_node.curPlayer, action
        )

        # Search for the node that corresponds to the given action
        for i in self.start_node.children:

            if np.array_equal(i.board, new_board):
                self.start_node = i
                return
