cmake_minimum_required(VERSION 3.19)

# project(UltimateTicTacToe)

# project(MCTS)

project(SelfPlay C CXX)



add_subdirectory(
  "C:/Users/Mason/Desktop/tensorflow_src/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

set(CMAKE_CXX_STANDARD 11)


add_executable(SelfPlay
src/BatchManager.cpp
src/GameState.cpp
src/MonteCarlo.cpp
src/Minimax.cpp
src/SelfPlay.cpp

    src/test.cpp
)

target_link_libraries(SelfPlay
    tensorflow-lite
    ${CMAKE_DL_LIBS}
)


# file(GLOB UTTT_SRC
#      "src/*.cpp"
# )

LINK_DIRECTORIES(C:/Users/Mason/anaconda3/envs/tensorflow-gpu/libs)

# add_executable(UltimateTicTacToe
#     src/BatchManager.cpp
#     src/GameState.cpp
#     src/MonteCarlo.cpp
#     src/test.cpp
#     src/Minimax.cpp
#     src/SelfPlay.cpp
#     )


# add_library(MCTS SHARED
#     MCTS.cpp
# )

# target_include_directories(UltimateTicTacToe PUBLIC "${PROJECT_SOURCE_DIR}/include")

target_include_directories(SelfPlay PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(SelfPlay PUBLIC "C:/Users/Mason/Desktop/tensorflow_src")


# target_include_directories(MCTS PUBLIC "${PROJECT_SOURCE_DIR}/include")
# target_include_directories(MCTS PUBLIC C:/Users/Mason/anaconda3/envs/tensorflow-gpu/include)
# target_include_directories(MCTS PUBLIC C:/Users/Mason/anaconda3/envs/tensorflow-gpu/Lib/site-packages/numpy/core/include)

# TARGET_LINK_LIBRARIES(MCTS python38)
