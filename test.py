import unittest
import numpy as np

from UltimateTicTacToe import UltimateTicTacToeGame

class TestUltimateTicTacToeGame(unittest.TestCase):
    def setUp(self):
        self.g = UltimateTicTacToeGame.TicTacToeGame()

    def test_getInitBoard(self):
        start = np.zeros((3, 9, 9))
        start[2] = np.full((9, 9), 1)

        self.assertTrue(np.array_equal(start, self.g.getInitBoard()))

    def test_getBoardSize(self):
        self.assertEqual((3, 9, 9), self.g.getBoardSize())
    
    def test_getActionSize(self):
        self.assertEqual(81, self.g.getActionSize())

    def test_getNextState(self):
        next_state = self.g.getInitBoard()

        next_state[0][2][2] = 1
        next_state[2] = np.zeros((9, 9))
        next_state[2][2] = np.full(9, 1)

        result = self.g.getNextState(self.g.getInitBoard(), 1, 20)


        self.assertTrue(np.array_equal( next_state, result[0] ))
        self.assertEqual(result[1], -1)

    def test_check_miniboard(self):
        test_value = np.zeros((2, 9))

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[0][0] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[0][1] = 1
        test_value[0][2] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 1)


        test_value = np.zeros((2, 9))

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[1][0] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[1][1] = 1
        test_value[1][2] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 2)


        test_value = np.zeros((2, 9))

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[0][1] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[0][4] = 1
        test_value[0][7] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 1)

        test_value = np.zeros((2, 9))

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[1][1] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[1][4] = 1
        test_value[1][7] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 2)


        test_value = np.zeros((2, 9))

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
        test_value = np.zeros((2, 9))

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[0][1] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 0)

        test_value[1][0] = 1
        test_value[1][2] = 1
        test_value[1][4] = 1
        test_value[1][7] = 1

        test_value[0][3] = 1
        test_value[0][6] = 1
        test_value[0][5] = 1
        test_value[0][8] = 1

        self.assertEqual(UltimateTicTacToeGame.check_miniboard(test_value), 3)

    def test_getValidMoves(self):
        state = self.g.getInitBoard()

        state = self.g.getNextState(state, 1, 4)

        moves = np.zeros(81)

        for i in range(9):
            moves[i + (4 * 9)] = 1

        self.assertTrue(np.array_equal(moves, self.g.getValidMoves(state[0], state[1])))
        


if __name__ == "__main__":
    unittest.main()