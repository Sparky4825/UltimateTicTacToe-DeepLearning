#include <bitset>
#include <vector>
#include <iostream>
#include <GameState.h>
#include <Minimax.h>
#include <chrono>

using namespace std;

int main() {
    GameState myboard;


    // myboard.setPosition(0, 0, 1);
    // myboard.setPosition(0, 3, 1);
    // myboard.setPosition(0, 2, 2);

    // cout << "EVAL BOARD ONE SIDE " << miniboardEvalOneSide(myboard.board[0], 1) << '\n';
    // cout << "EVAL BOARD ONE SIDE " << miniboardEvalOneSide(myboard.board[0], 2) << '\n';

    // cout << "OVERALL " << evaluate(myboard) << '\n';

    // return 0;



    // myboard.move(4, 5);
    // myboard.move(5, 0);
    // myboard.move(0, 1);
    // myboard.move(1, 1);
    // myboard.move(1, 8);
    // myboard.move(8, 4);
    // myboard.move(4, 3);
    // myboard.move(3, 0);
    // myboard.move(0, 0);
    // myboard.move(0, 6);
    // myboard.move(6, 3);
    // myboard.move(3, 1);
    // myboard.move(1, 2);

    // myboard.move(4, 4);

    
    // constants c1, c2;

    // // c1.c1 = 6;
    // // c2.c1 = 6;

    // std::cout << computerVcomputer(6, c1, 8, c2, true);

    // return 0;

    // myboard.move(4, 4);
    // myboard.move(8, 2);

    myboard.displayGame();


    int board, piece;

    auto start = chrono::high_resolution_clock::now();



    // // vector<GameState> allMoves = myboard.allPossibleMoves();
    // // cout << "IN BETWEEN \n";
    myboard.move(4, 0);

    myboard = minimaxSearch(myboard, 10, false);

    // boardCoords test = minimaxSearchTimeMove(myboard, 4, true);


    auto stop = chrono::high_resolution_clock::now();


    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    cout << "TIME TAKEN ";
    cout << duration.count() << " milliseconds\n\n\n";

    myboard.displayGame();

    return 0;

    // constants c1, c2;

    // c1.c1 = 6;
    // c2.c1 = 6;

    // std::cout << computerVcomputer(11, c1,  2, c2, true);

    // return 0;

    while (true) {

        cout << "\n\n";
        myboard = minimaxSearchTime(myboard, 5, true);

        myboard.displayGame();

        if (myboard.getStatus() == 1) {
            cout << "X WINS!!!\n";
            return 0;
        } else if (myboard.getStatus() == 2) {
            cout << "O WINS!!!\n";
            return 0;
        } else if (myboard.getStatus() == 3) {
            cout << "TIE GAME!!!\n";
            return 0;
        }
        cout << "\n\n";

        myboard = minimaxSearch(myboard, 6, false);


        myboard.displayGame();

        if (myboard.getStatus() == 1) {
            cout << "X WINS!!!\n";
            return 0;
        } else if (myboard.getStatus() == 2) {
            cout << "O WINS!!!\n";
            return 0;
        } else if (myboard.getStatus() == 3) {
            cout << "TIE GAME!!!\n";
            return 0;
        }

    }


    while (true) {

        if (myboard.getRequiredBoard() != -1) {
            cout << "BOARD: " << myboard.getRequiredBoard();
            board = myboard.getRequiredBoard();
        } else {
            cout << "BOARD: ";
            cin >> board;
        }
        
        cout << "\nPIECE: ";
        cin >> piece;
        cout << "\n";

        myboard.move(board, piece);

        myboard.displayGame();

        auto start = chrono::high_resolution_clock::now();



        // vector<GameState> allMoves = myboard.allPossibleMoves();
        // cout << "IN BETWEEN \n";

        myboard = minimaxSearch(myboard, 6, true);


        auto stop = chrono::high_resolution_clock::now();


        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

        cout << "TIME TAKEN ";
        cout << duration.count() << " milliseconds\n\n\n";

        myboard.displayGame();

    }
    return 0;
}
