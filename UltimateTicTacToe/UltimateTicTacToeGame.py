from __future__ import print_function
import sys

sys.path.append("..")
from Game import Game
import numpy as np

winning_positions = np.array(
    [
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    ]
)

# Match a winning mask against every miniboard simultaneously
winning_positions_large = np.zeros((8, 9, 10))
for i in range(8):
    winning_positions_large[i] = np.full((9, 10), winning_positions[i])


def check_mask(mask, a2):
    """
    Check if the every bit set in mask is set in a2
    """
    if np.array_equal(mask, np.logical_and(mask, a2)):
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

    result = np.zeros((2, 10))
    for player_index in range(1, 3):
        for wp_index in range(8):

            # If the board is winning, return the player that won
            if check_mask(winning_positions[wp_index], miniboard[player_index - 1]):
                return player_index

    # If there are no found wins, check if there are empty spaces
    if 0 in np.logical_or(miniboard[0], miniboard[1]):
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

    Miniboard should be a numpy array of size (2, 10)
    """

    result = np.zeros((2, 10))
    for player_index in range(1, 3):
        for wp_index in range(8):

            # If the board is winning and the other player has no claims, return the player that won
            if (
                check_mask(
                    winning_positions[wp_index], miniboard[int(player_index - 1)]
                )
                and not np.logical_and(
                    winning_positions[wp_index], miniboard[int((2 / player_index) - 1)]
                ).any()
            ):
                return player_index

    # If there are no found wins, check if there are empty spaces
    if 0 in np.logical_or(miniboard[0], miniboard[1]):
        # Ongoing game
        return 0
    # No empty spaces, draw
    return 3


def absolute_index_to_board_and_piece(index):
    """
    Gets an absolute piece index to a board and piece
    :param index: The index to be found
    :return: (board_index, piece_index)
    """

    # i = index
    # gx = global x value (independent of board)
    # gy = same
    #
    # lx = local x value (within board)
    # ly = same
    #
    # bx = x value of the whole board
    # by = same
    #
    # pi = piece index
    # bi = board index

    i = index

    gx = i % 9
    gy = int(i / 9)

    lx = gx % 3
    ly = gy % 3

    bx = int((i % 9) / 3)
    by = int(i / 27)

    pi = ly * 3 + lx
    bi = by * 3 + bx

    return bi, pi


class TicTacToeGame(Game):
    def __init__(self):
        self.n = 9

    def getInitBoard(self):
        """
        Board is stored in the structure of a 3d array size 3 layers x 9 miniboards x 10 spots (last spot is used to record a won board)
        Where the first depth is the board with 1 marked as pieces taken by the player to move
        Second depth is the board with 1 marked as pieces taken by the other player
        Third depth is marked as 1 in any position the player is allowed to move in (required board)

        An action is a number in [0, 80] inclusive in which to place a move
        """
        # TODO: Have numpy work in int instead of float for increased performance
        # return initial board (numpy board)
        # Whole board is empty at the start
        b = np.zeros((3, 9, 10))

        # First player can move anywhere
        b[2] = np.full((9, 10), 1)
        return b

    def getBoardSize(self):
        # (a,b) tuple
        # Board has three dimensions
        return (3, 9, 10)

    def getActionSize(self):
        # return number of actions
        return self.n * self.n

    def getNextState(self, board, player, action):
        """
        Board is a numpy array of size (3, 9, 10)
        Player is 1 or -1
        Action is a number in [0, 80] for the player to play a move
        """

        # TODO: Add a tie marker to make things easier on the NN?

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

        board_result = check_miniboard([b[0][board_index], b[1][board_index]])
        if board_result == 1:
            b[0][board_index][9] = 1
        elif board_result == 2:
            b[1][board_index][9] = 1

        elif board_result == 3:
            b[0][board_index][9] = 1
            b[1][board_index][9] = 1

        # Check if the required_board is not won/tied and there is a required board
        if b[0][piece_index][9] == 0 and b[1][piece_index][9] == 0:
            b[2] = np.zeros((9, 10))
            b[2][piece_index] = np.full((10), 1)

        # Else there is not a required board
        else:
            b[2] = np.full((9, 10), 1)

            # Even if there is no required board, moves cannot be played on boards that are won/lost/tied
            for miniboard_index in range(9):
                if b[0][miniboard_index][9] == 1 or b[1][miniboard_index][9] == 1:
                    b[2][miniboard_index] = np.zeros((10))

        return (b, -player)

    def getValidMoves(self, board, player):
        """
        Returns a numpy array of size (81) where 0 indicates that action is not allowed in that square and 1 indicates action is allowed in that square
        """
        # For a move to be valid, b[0] and b[1] at that move must be 0 and b[2] at that move must be 1
        # Cut off the won/lost/tie board markers
        board = board[:, :, :-1]
        result = np.invert(np.logical_or(board[0], board[1]))
        result = np.logical_and(result, board[2])

        # Convert to a 1d array
        return result.flatten().astype("int")

    def getGameEnded(self, board, player):
        result = np.zeros((2, 10))
        # Step 1: Find if each miniboard is winning
        # Any tied boards will be represented with bits set in both dimensions of result for the board

        for i in range(9):
            # Miniboards are checked for win/loss/tie on move creation. Calling check_miniboard again is unnecessary
            if board[0][i][9] == 1 and board[1][i][9] == 0:
                result[0][i] = 1
            elif board[0][i][9] == 0 and board[1][i][9] == 1:
                result[1][i] = 1
            elif board[0][i][9] == 1 and board[1][i][9] == 1:
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
        assert len(pi) == 81  # 81 possible moves
        # Change into 3 rows of 3 boards of 3 rows of 3 spaces
        pi_board = np.reshape(pi, (3, 3, 3, 3))

        # Change into 3 layers of 3 rows of 3 boards of 3 rows of 3 spaces
        # Remove the last element from board b/c it is used to record won/lost boards and should not be rotated
        b = np.reshape(board[:, :, :-1], (3, 3, 3, 3, 3))

        # Create an array of won/lost markers to be rotated
        w = np.reshape(board[:, :, -1:], (3, 3, 3, 1))

        # Store all symmetries
        l = []

        # Four possible degrees of rotation
        for i in range(1, 5):
            # Each rotation may be either reflected or not
            for j in [True, False]:

                # Each miniboard must be rotated in place, and then the larger board rotated
                newB = np.rot90(b, i, axes=(3, 4))
                newB = np.rot90(newB, i, axes=(1, 2))

                # Same as above, but axes are all 1 less b/c there is only one layer and not three
                newPi = np.rot90(pi_board, i, axes=(2, 3))
                newPi = np.rot90(newPi, i, axes=(0, 1))

                newW = np.rot90(w, i, axes=(1, 2))

                if j:
                    for flip_row_index in range(3):
                        for flip_miniboard_index in range(3):

                            # Flip the pieces in each miniboard
                            newB[0][flip_row_index][flip_miniboard_index] = np.fliplr(
                                newB[0][flip_row_index][flip_miniboard_index]
                            )
                            newB[1][flip_row_index][flip_miniboard_index] = np.fliplr(
                                newB[1][flip_row_index][flip_miniboard_index]
                            )
                            newB[2][flip_row_index][flip_miniboard_index] = np.fliplr(
                                newB[1][flip_row_index][flip_miniboard_index]
                            )

                    # Flip each miniboard in whole board
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                    newW = np.fliplr(newW)

                newB = np.reshape(newB, (3, 9, 9))
                newW = np.reshape(w, (3, 9, 1))

                # Add the won/lost markers in
                newB = np.append(newB, newW, axis=2)

                # Add the transformed board to the result
                l += [(newB, list(newPi.ravel()))]
        return l

    def stringRepresentation(self, board):
        # TODO: Make stringRepresentation(board) more space and time efficient
        # 8x8 numpy array (canonical board)
        return np.array2string(board)

    @staticmethod
    def display(board):
        result = ""
        for row in range(9):
            for board_row in range(3):
                for col in range(3):
                    absolute_piece_index = (row * 9) + (board_row * 3) + col

                    board_index, piece_index = absolute_index_to_board_and_piece(
                        absolute_piece_index
                    )

                    piece_char = " "
                    if board[0][board_index][piece_index] == 1:
                        piece_char = "X"
                    elif board[1][board_index][piece_index] == 1:
                        piece_char = "O"

                    result += piece_char
                    result += " | "
                result = result[:-3] + "\\\\ "
            if (row + 1) % 3 != 0:
                result += f"\n{'---------   ' * 3}\n"
            else:
                result += "\n=================================\n"

        print(result)
