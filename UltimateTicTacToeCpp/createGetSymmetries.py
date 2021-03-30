import numpy as np


def getSymmetries(board, pi):
    """
    Returns a list of tuples in the form (board, pi) where each board and pi have have the exact same rotational transformations appplied
    """

    # mirror, rotational
    # assert len(pi) == 81  # 81 possible moves
    # Change into 3 rows of 3 boards of 3 rows of 3 spaces
    # pi_board = np.reshape(pi, (3, 3, 3, 3))

    # Change into 3 layers of 3 rows of 3 boards of 3 rows of 3 spaces
    # Remove the last element from board b/c it is used to record won/lost boards and should not be rotated
    b = np.reshape(board, (3, 3, 3, 3))
    w = np.arange(9).reshape(3, 3)
    # Store all symmetries
    l = []

    # Four possible degrees of rotation
    for i in range(4):
        # Each rotation may be either reflected or not
        for j in [False, True]:

            # Each miniboard must be rotated in place, and then the larger board rotated
            newB = np.copy(np.rot90(b, i, axes=(2, 3)))
            newB = np.rot90(newB, i, axes=(0, 1))

            # Same as above, but axes are all 1 less b/c there is only one layer and not three
            # newPi = np.copy(np.rot90(pi_board, i, axes=(2, 3)))
            # newPi = np.rot90(newPi, i, axes=(0, 1))

            newW = np.copy(np.rot90(w, i, axes=(0, 1)))

            if j:
                for flip_row_index in range(3):
                    for flip_miniboard_index in range(3):

                        # Flip the pieces in each miniboard
                        newB[flip_row_index][flip_miniboard_index] = np.fliplr(
                            newB[flip_row_index][flip_miniboard_index]
                        )
                        # newB[1][flip_row_index][flip_miniboard_index] = np.fliplr(
                        #     newB[1][flip_row_index][flip_miniboard_index]
                        # )
                        # newB[2][flip_row_index][flip_miniboard_index] = np.fliplr(
                        #     newB[2][flip_row_index][flip_miniboard_index]
                        # )

                    # Flip each miniboard in whole board
                    newB = np.flip(newB, axis=1)
                    # newPi = np.flip(newPi, axis=1)
                    newW = np.flip(newW, axis=1)

            newB = np.reshape(newB, (9, 9))
            newW = np.reshape(newW, (3, 3))

            # Add the won/lost markers in
            # newB = np.append(newB, newW, axis=2)

            # Add the transformed board to the result
            l += [(newB, newW)]
    return l


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


def display(board):
    board = np.reshape(board, (9, 9))
    result = ""
    for row in range(9):
        for board_row in range(3):
            for col in range(3):
                absolute_piece_index = (row * 9) + (board_row * 3) + col

                board_index, piece_index = absolute_index_to_board_and_piece(
                    absolute_piece_index
                )

                piece_char = board[board_index][piece_index]

                result += str(piece_char).zfill(2)
                result += " | "
            result = result[:-3] + " \\\\ "
        if (row + 1) % 3 != 0:
            result += f"\n{'-------------   ' * 3}\n"
        else:
            result += "\n=================================\n"

    print(result)


def pieceIndex2index(piece):
    boardIndex = int(piece / 9)
    pieceIndex = piece % 9

    return 22 * boardIndex + pieceIndex * 2


code1 = "temp.push_back(board[{}]);\n"


if __name__ == "__main__":

    board = np.arange(81).reshape((3, 3, 3, 3))

    # board = np.zeros(81).reshape((3, 3, 3, 3))

    # board[0][1][2][1] = 1
    # board[0][1][2][0] = 1
    # display(board)

    print()
    print()
    print()
    print()

    boards = getSymmetries(board, np.zeros(81))

    allMapping = []
    miniboardMapping = []

    for both in boards:
        i = both[0]
        mini = both[1].ravel()
        print(mini)
        mapping = np.reshape(i, 81)

        allMapping.append([])
        miniboardMapping.append([])
        for j in range(81):
            # print(code1.format(mapping[j]), end='')
            allMapping[-1].append(mapping[j])
        for j in range(9):
            miniboardMapping[-1].append(mini[j])
    print(miniboardMapping)
    print(allMapping)

    allTogether = []

    index = 0
    for symmetryIndex in range(8):
        fullMap = []

        for miniboardIndex in range(9):

            for pieceIndex in range(9):
                fullMap.append(
                    pieceIndex2index(
                        allMapping[symmetryIndex][miniboardIndex * 9 + pieceIndex]
                    )
                )
                fullMap.append(
                    pieceIndex2index(
                        allMapping[symmetryIndex][miniboardIndex * 9 + pieceIndex]
                    )
                    + 1
                )

            newMiniboard = miniboardMapping[symmetryIndex][miniboardIndex]
            boardOffset = 22 * newMiniboard + 18

            fullMap.append(boardOffset)
            fullMap.append(boardOffset + 1)
            fullMap.append(boardOffset + 2)
            fullMap.append(boardOffset + 3)
        fullMap.append(198)

        allTogether.append(fullMap)
    print(allTogether)
