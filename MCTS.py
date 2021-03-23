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


#
# class OLDMCTS:
#     """
#     This class handles the MCTS tree.
#     """
#
#     def __init__(
#         self,
#         episode_actor,
#         game,
#         nnet,
#         args,
#         use_async=True,
#     ):
#         self.episode_actor = episode_actor
#         self.game = game
#         self.nnet = nnet  # Reference to ray actor responsible for NN
#         self.args = args
#
#         self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
#         self.Nsa = {}  # stores #times edge s,a was visited
#         self.Ns = {}  # stores #times board s was visited
#         self.Ps = {}  # stores initial policy (returned by neural net)
#
#         self.Es = {}  # stores game.getGameEnded ended for board s
#         self.Vs = {}  # stores game.getValidMoves for board s
#
#         self.canonicalBoard = None  # Stores the board that is being evaluated between preNNSearch and postNNSearch
#
#         self.batch_size = ray.get(self.nnet.get_batch_size.remote())
#         self.use_async = use_async
#
#         self.log = logging.getLogger(self.__class__.__name__)
#
#         coloredlogs.install(level="INFO", logger=self.log)
#
#     async def getActionProb(self, canonicalBoard, temp=1):
#         """
#         This function performs numMCTSSims simulations of MCTS starting from
#         canonicalBoard.
#
#         Returns:
#             probs: a policy vector where the probability of the ith action is
#                    proportional to Nsa[(s,a)]**(1./temp)
#         """
#         for i in range(self.args.numMCTSSims):
#             await self.search(canonicalBoard)
#
#         s = self.game.stringRepresentation(canonicalBoard)
#         counts = [
#             self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
#             for a in range(self.game.getActionSize())
#         ]
#
#         if temp == 0:
#             # Get array of all the best move action values
#             bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
#
#             # Pick a random best move
#             bestA = np.random.choice(bestAs)
#             probs = [0] * len(counts)
#
#             # 100% chance of picking the best move
#             probs[bestA] = 1
#             return probs
#
#         counts = [x ** (1.0 / temp) for x in counts]
#         counts_sum = float(sum(counts))
#         probs = [x / counts_sum for x in counts]
#         return probs
#
#     def searchPreNN(self, canonicalBoard):
#         """
#         This function and its second part searchPostNN perform the same function
#         as search. The functions needs to be called before and after being
#         passed througth the NN queue. The NN queue will set variables in the MCTS
#         object that were previously awaited.
#         """
#
#         self.canonicalBoard = canonicalBoard
#
#         s = self.game.stringRepresentation(self.canonicalBoard)
#
#         if s not in self.Es:
#             self.Es[s] = self.game.getGameEnded(self.canonicalBoard, 1)
#         if self.Es[s] != 0:
#             # terminal node
#             return -self.Es[s]
#
#         if s not in self.Ps:
#             # leaf node
#             # Wait until the prediction is made, allowing other trees to be searched in the meantime
#
#             # THIS IS WHERER POST NN SERACH WILL START
#
#             # tHIS IS WHERE POST NN SEARCH ENDS
#
#         valids = self.Vs[s]
#         cur_best = -float("inf")
#         best_act = -1
#
#         # pick the action with the highest upper confidence bound
#         for a in range(self.game.getActionSize()):
#             if valids[a]:
#                 if (s, a) in self.Qsa:
#                     u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(
#                         self.Ns[s]
#                     ) / (1 + self.Nsa[(s, a)])
#                 else:
#                     u = (
#                         self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
#                     )  # Q = 0 ?
#
#                 if u > cur_best:
#                     cur_best = u
#                     best_act = a
#
#         a = best_act
#         next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
#         next_s = self.game.getCanonicalForm(next_s, next_player)
#
#         v = await self.searchPreNN(next_s)
#
#
#
#     def postNNsearch(self):
#         s = self.game.stringRepresentation(self.canonicalBoard)
#
#         self.Ps[s], v = await self.episode_actor.request_prediction(self.canonicalBoard)
#
#         valids = self.game.getValidMoves(self.canonicalBoard, 1)
#         self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
#         sum_Ps_s = np.sum(self.Ps[s])
#         if sum_Ps_s > 0:
#             self.Ps[s] /= sum_Ps_s  # renormalize
#         else:
#             # if all valid moves were masked make all valid moves equally probable
#
#             # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
#             # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
#             log.error("All valid moves were masked, doing a workaround.")
#             self.Ps[s] = self.Ps[s] + valids
#             self.Ps[s] /= np.sum(self.Ps[s])
#
#         self.Vs[s] = valids
#         self.Ns[s] = 0
#
#         if (s, a) in self.Qsa:
#             self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
#                 self.Nsa[(s, a)] + 1
#             )
#             self.Nsa[(s, a)] += 1
#
#         else:
#             self.Qsa[(s, a)] = v
#             self.Nsa[(s, a)] = 1
#
#         self.Ns[s] += 1
#         return -v
#
#     async def search(self, canonicalBoard):
#         """
#         This function performs one iteration of MCTS. It is recursively called
#         till a leaf node is found. The action chosen at each node is one that
#         has the maximum upper confidence bound as in the paper.
#
#         Once a leaf node is found, the neural network is called to return an
#         initial policy P and a value v for the state. This value is propagated
#         up the search path. In case the leaf node is a terminal state, the
#         outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
#         updated.
#
#         NOTE: the return values are the negative of the value of the current
#         state. This is done since v is in [-1,1] and if v is the value of a
#         state for the current player, then its value is -v for the other player.
#
#         Returns:
#             v: the negative of the value of the current canonicalBoard
#         """
#
#         s = self.game.stringRepresentation(canonicalBoard)
#
#         if s not in self.Es:
#             self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
#         if self.Es[s] != 0:
#             # terminal node
#             return -self.Es[s]
#
#         if s not in self.Ps:
#             # leaf node
#             # Wait until the prediction is made, allowing other trees to be searched in the meantime
#             if self.use_async and self.episode_actor is not None:
#                 self.Ps[s], v = await self.episode_actor.request_prediction(
#                     canonicalBoard
#                 )
#             else:
#                 self.Ps[s], v = ray.get(self.nnet.predict.remote(canonicalBoard))
#
#             valids = self.game.getValidMoves(canonicalBoard, 1)
#             self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
#             sum_Ps_s = np.sum(self.Ps[s])
#             if sum_Ps_s > 0:
#                 self.Ps[s] /= sum_Ps_s  # renormalize
#             else:
#                 # if all valid moves were masked make all valid moves equally probable
#
#                 # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
#                 # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
#                 log.error("All valid moves were masked, doing a workaround.")
#                 self.Ps[s] = self.Ps[s] + valids
#                 self.Ps[s] /= np.sum(self.Ps[s])
#
#             self.Vs[s] = valids
#             self.Ns[s] = 0
#             return -v
#
#         valids = self.Vs[s]
#         cur_best = -float("inf")
#         best_act = -1
#
#         # pick the action with the highest upper confidence bound
#         for a in range(self.game.getActionSize()):
#             if valids[a]:
#                 if (s, a) in self.Qsa:
#                     u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(
#                         self.Ns[s]
#                     ) / (1 + self.Nsa[(s, a)])
#                 else:
#                     u = (
#                         self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
#                     )  # Q = 0 ?
#
#                 if u > cur_best:
#                     cur_best = u
#                     best_act = a
#
#         a = best_act
#         next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
#         next_s = self.game.getCanonicalForm(next_s, next_player)
#
#         v = await self.search(next_s)
#
#         if (s, a) in self.Qsa:
#             self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
#                 self.Nsa[(s, a)] + 1
#             )
#             self.Nsa[(s, a)] += 1
#
#         else:
#             self.Qsa[(s, a)] = v
#             self.Nsa[(s, a)] = 1
#
#         self.Ns[s] += 1
#         return -v
