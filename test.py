import unittest
import numpy as np

from UltimateTicTacToe import UltimateTicTacToeGame


class TestUltimateTicTacToeGame(unittest.TestCase):
    def setUp(self):
        self.g = UltimateTicTacToeGame.TicTacToeGame()

    def test_getInitBoard(self):
        start = np.zeros((3, 9, 10))
        start[2] = np.full((9, 10), 1)

        self.assertTrue(np.array_equal(start, self.g.getInitBoard()))

    def test_getBoardSize(self):
        self.assertEqual((3, 9, 10), self.g.getBoardSize())

    def test_getActionSize(self):
        self.assertEqual(81, self.g.getActionSize())

    def test_getNextState(self):
        next_state = self.g.getInitBoard()

        next_state[0][2][2] = 1
        next_state[2] = np.zeros((9, 10))
        next_state[2][2] = np.full(10, 1)

        result = self.g.getNextState(self.g.getInitBoard(), 1, 20)

        self.assertTrue(np.array_equal(next_state, result[0]))
        self.assertEqual(result[1], -1)

    def test_getSymmetries(self):
        b = np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ]
        )

        b = self.g.getInitBoard()
        b, _ = self.g.getNextState(b, 1, 1)
        b, _ = self.g.getNextState(b, -1, 12)

        self.g.display(b)
        for sym, pi in self.g.getSymmetries(b, [0] * 81):
            self.g.display(sym)

    def test_check_miniboard(self):
        test_value = np.zeros((2, 10))

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[0][0] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[0][1] = 1
        test_value[0][2] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 1)

        test_value = np.zeros((2, 10))

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[1][0] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[1][1] = 1
        test_value[1][2] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 2)

        test_value = np.zeros((2, 10))

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[0][1] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[0][4] = 1
        test_value[0][7] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 1)

        test_value = np.zeros((2, 10))

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[1][1] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[1][4] = 1
        test_value[1][7] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 2)

        test_value = np.zeros((2, 10))

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[0][1] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[0][4] = 1
        test_value[0][7] = 1

        test_value[1][2] = 1
        test_value[1][6] = 1
        test_value[1][5] = 1
        test_value[1][0] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 1)

        # Check ties
        test_value = np.zeros((2, 10))

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[0][1] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[1][0] = 1
        test_value[1][2] = 1
        test_value[1][4] = 1
        test_value[1][7] = 1

        test_value[1][3] = 1
        test_value[0][6] = 1
        test_value[0][5] = 1
        test_value[0][8] = 1

        # print(test_value)
        #
        # self.g.display(test_value)

        # self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 1)

    # def test_getValidMoves(self):
    #     inital = self.g.getInitBoard()
    #
    #     for i in range(81):
    #         state = self.g.getNextState(inital, 1, i)
    #
    #         moves = np.zeros(81)
    #
    #         for j in range(9):
    #             if int(i / 9) * 9 + j != i:
    #                 moves[int(i / 9) * 9 + j] = 1
    #
    #         print(moves)
    #         print(self.g.getValidMoves(state[0], state[1]))
    #
    #         self.assertTrue(
    #             np.array_equal(moves, self.g.getValidMoves(state[0], state[1]))
    #         )


if __name__ == "__main__":
    unittest.main()
