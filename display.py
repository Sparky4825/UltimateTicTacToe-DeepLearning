from pickle import Unpickler


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
    result = ""
    for row in range(9):
        for board_row in range(3):
            for col in range(3):
                absolute_piece_index = (row * 9) + (board_row * 3) + col

                board_index, piece_index = absolute_index_to_board_and_piece(
                    absolute_piece_index
                )

                piece_char = " "
                if board[board_index * 22 + piece_index * 2] == 1:
                    piece_char = "X"
                elif board[board_index * 22 + piece_index * 2 + 1] == 1:
                    piece_char = "O"

                result += piece_char
                result += " | "
            result = result[:-3] + "\\\\ "
        if (row + 1) % 3 != 0:
            result += f"\n{'---------   ' * 3}\n"
        else:
            result += "\n=================================\n"

    print(result)


def display2D(board):
    result = ""
    for row in range(9):
        for board_row in range(3):
            for col in range(3):
                absolute_piece_index = (row * 9) + (board_row * 3) + col

                board_index, piece_index = absolute_index_to_board_and_piece(
                    absolute_piece_index
                )

                piece_char = " "
                if board[board_index * 11 + piece_index][1] == 1:
                    piece_char = "O"
                elif board[board_index * 11 + piece_index][0] == 1:
                    piece_char = "X"

                result += piece_char
                result += " | "
            result = result[:-3] + "\\\\ "
        if (row + 1) % 3 != 0:
            result += f"\n{'---------   ' * 3}\n"
        else:
            result += "\n=================================\n"

    print(result)


def countEmptySpaces(board):
    e = 81
    result = ""
    for row in range(9):
        for board_row in range(3):
            for col in range(3):
                absolute_piece_index = (row * 9) + (board_row * 3) + col

                board_index, piece_index = absolute_index_to_board_and_piece(
                    absolute_piece_index
                )

                piece_char = " "
                if board[board_index * 22 + piece_index * 2] == 1:
                    piece_char = "X"
                    e -= 1
                elif board[board_index * 22 + piece_index * 2 + 1] == 1:
                    piece_char = "O"
                    e -= 1

                result += piece_char
                result += " | "
            result = result[:-3] + "\\\\ "
        if (row + 1) % 3 != 0:
            result += f"\n{'---------   ' * 3}\n"
        else:
            result += "\n=================================\n"

    return e


if __name__ == "__main__":

    display(
    
    (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
)

    exit()

    with open("trainingData.examples", "rb") as f:
        a = Unpickler(f).load()

    count = 0

    input(a[0][0].shape)

    for i in range(0, len(a[0][0]), 8):
        count += 1
        print("=====================")
        display(a[0][0][i])
        print(a[0][1][i])
        # print(a[0][i])
        for j in range(81):
            board = int(j / 9)
            piece = j % 9
            if a[1][i][j] != 0:
                print(f"[{board}][{piece}] {a[1][i][j]}")
        print(a[2][i])

    print(f"{count} boards displayed!")
