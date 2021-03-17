from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .TicTacToeLogic import Board
import numpy as np

winning_positions = np.array([
[0, 0, 0, 0, 0, 0, 1, 1, 1],
[0, 0, 0, 1, 1, 1, 0, 0, 0],
[1, 1, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 1, 0, 0, 1],
[0, 1, 0, 0, 1, 0, 0, 1, 0],
[1, 0, 0, 1, 0, 0, 1, 0, 0],
[1, 0, 0, 0, 1, 0, 0, 0, 1],
[0, 0, 1, 0, 1, 0, 1, 0, 0],
])

# Match a winning mask against every miniboard simultaneously
winning_positions_large = np.zeros((8, 9, 9))
for i in range(8):
    winning_positions_large[i] = np.full((9, 9), winning_positions[i])

def check_mask(mask, a2):
    """
    Check if the every bit set in mask is set in a2
    """
    if np.array_equal(mask, np.bitwise_and(mask, a2):
        return 1
    else:
        return 0


def check_miniboard(miniboard):
    """
    Checks a miniboard for a win.
    1 = player 1 wins
    2 = player 2 wins
    3 = draw
    0 = ongoing game

    Miniboard should be a numpy array of size (2, 9)
    """

    result = np.zeros((2, 9))
    for player_index in range(1, 3):
        for wp_index in range(8):
        
            # If the board is winning, return the player that won
            if check_mask(winning_positions[wp_index], miniboard[player_index - 1]):
                return player_index

    # If there are no found wins, check if there are empty spaces
    if 0 in np.bitwise_or(miniboard[0], miniboard[1]):
        # Ongoing game
        return 0
    # No empty spaces, draw
    return 3


def check_miniboard_with_ties(miniboard):
    """
    Checks a miniboard for a win, including the possibility of a square being tied (ie not usable by either side for a win but unclaimable).
    1 = player 1 wins
    2 = player 2 wins
    3 = draw
    0 = ongoing game

    Miniboard should be a numpy array of size (2, 9)
    """

    result = np.zeros((2, 9))
    for player_index in range(1, 3):
        for wp_index in range(8):
        
            # If the board is winning and the other player has no claims, return the player that won
            if check_mask(winning_positions[wp_index], miniboard[player_index - 1]) and not np.bitwise_and(winning_positions[wp_index], miniboard[(2 / player_index) - 1]).any():
                return player_index

    # If there are no found wins, check if there are empty spaces
    if 0 in np.bitwise_or(miniboard[0], miniboard[1]):
        # Ongoing game
        return 0
    # No empty spaces, draw
    return 3

class TicTacToeGame(Game):
    def __init__(self):
        self.n = 9

    def getInitBoard(self):
        """
        Board is stored in the structure of a 3d array 9 x 9 x 3
        Where the first depth is the board with 1 marked as pieces taken by the player to move
        Second depth is the board with 1 marked as pieces taken by the other player
        Third depth is marked as 1 in any position the player is allowed to move in (required board)

        An action is a number in [0, 80] inclusive in which to place a move
        """
        # return initial board (numpy board)
        # Whole board is empty at the start
        b = numpy.zeros((3, 9, 9))

        # First player can move anywhere
        b[2] = numpy.full((9, 9), 1)

        return b

    def getBoardSize(self):
        # (a,b) tuple
        # Board has three dimensions
        return (3, self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n

    def getNextState(self, board, player, action):
        """
        Board is a numpy array of size (3, 9, 9)
        Player is 1 or -1
        Action is a number in [0, 80] for the player to play a move
        """

        # if player takes action on board, return next (board,player)
        # Action must be a valid move
        b = np.copy(board)
        board_index = int(action / 9)
        piece_index = action % 9
        assert board[0][board_index][piece_index] == 0
        assert board[1][board_index][piece_index] == 0

        if player == 1:
            b[0][board_index][piece_index] = 1
        else:
            b[1][board_index][piece_index] = 1

        # Check if the required_board is won/tied and there is no required board
        if check_miniboard([board[0][board_index], board[1][board_index]]) == 0:
            b[2][board_index] = np.full((9), 1)
        else:
            b[2] = np.full((9, 9), 1)



        return (b, -player)

    def getValidMoves(self, board, player):
        """
        Returns a numpy array of size (81) where 0 indicates that action is not allowed in that square and 1 indicates action is allowed in that square
        """
        # For a move to be valid, b[0] and b[1] at that move must be 0 and b[2] at that move must be 1
        result = np.invert(np.bitwise_or(board[0], board[1]))
        result = np.bitwise_and(result, board[2])

        # Convert to a 1d array
        return result.flatten()

    def getGameEnded(self, board, player):
        result = np.zeros((2, 9))
        # Step 1: Find if each miniboard is winning
        # Any tied boards will be represented with bits set in both dimensions of result for the board

        for i in range(9):
            miniboard_status = check_miniboard([board[0][i], board[1][i]])
            if miniboard_status == 1:
                result[0][i] = 1
            elif miniboard_status == 2:
                result[1][i] = 1
            elif miniboard_status == 3:
                result[0][i] = 1
                result[1][i] = 1
            # Else draw and leave value at 0



        # Step 2: Check for a win overall
        overall_status = check_miniboard_with_ties(result)
        if overall_status == 1:
            return 1
        elif overall_status == 2:
            return -1


        # If there are any valid moves the game is ongoing
        if np.any(self.getValidMoves(board, player)):
            return 0

        # If no wins and no legal moves, game is a draw
        # Draw has very little value 
        return 1e-4

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        if player == 1:
            return board
        else:
            b = np.copy(board)
            # Flip first two layers
            b[0], b[1] = board[1], board[0]
            return b

    def getSymmetries(self, board, pi):
        """
        Returns a list of tuples in the form (board, pi) where each board and pi have have the exact same rotational transformations appplied
        """
        # mirror, rotational
        assert(len(pi) == 81)  # 81 possible moves
        # Change into 3 rows of 3 boards of 3 rows of 3 spaces
        pi_board = np.reshape(pi, (3, 3, 3, 3))

        # Change into 3 layers of 3 rows of 3 boards of 3 rows of 3 spaces
        b = np.reshape(board, (3, 3, 3, 3, 3))
        l = []

        # Four possible degrees of rotation
        for i in range(1, 5):
            # Each rotation may be either reflected or not
            for j in [True, False]:

                # Each miniboard must be rotated in place, and then the larger board rotated
                newB = np.rot90(board, i, axes=(3, 4))
                newB = np.rot90(newB, i, axes=(1, 2))

                # Same as above, but axes are all 1 less b/c there is only one layer and not three
                newPi = np.rot90(pi_board, i, axes=(2, 3))
                newPi = np.rot90(newPi, i, axes=(0, 1))

                if j:
                    for flip_row_index in range(3):
                        for flip_miniboard_index in range(3):

                            # Flip the pieces in each miniboard
                            newB[0][flip_row_index][flip_miniboard_index] = np.fliplr(newB[0][flip_row_index][flip_miniboard_index])
                            newB[1][flip_row_index][flip_miniboard_index] = np.fliplr(newB[1][flip_row_index][flip_miniboard_index])
                            newB[2][flip_row_index][flip_miniboard_index] = np.fliplr(newB[1][flip_row_index][flip_miniboard_index])

                    # Flip each miniboard in whole board
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                
                # Add the transformed board to the result
                l += [(numpy.reshape(newB, (3, 9, 9)), list(newPi.ravel()))]
        return l

    def stringRepresentation(self, board):
        # TODO: Make stringRepresentation(board) more space and time effecient
        # 8x8 numpy array (canonical board)
        return np.array2string(board)

    @staticmethod
    def display(board):
        # TODO: Implement display(board) function
        pass
