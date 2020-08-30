# Introduction

Project made by GENTY Laurent and CHATAIGNER Johan

# Algorithms

At first, we only implemented MinMax algorithm. But in GO, the size of the search tree is very huge. Indeed, compared to Chess, the combinatorics are enormous so we must use a better algorithm. Thats why during the majority of the project, we used Alpha Beta algorithm. In fact, thanks to alpha-beta cuts, we are reducing a lot the number of "useless" nodes. With the time we are saving, we are able to increase the depth of the research during the game : in the early game, there are too much possibilities and we have a depth of 2. But during the mid game, 20-40 turns, we want to establish strategies and we want to see the further we can, so we deepen the research (depth 3) and in very late game we go even deeper because the combinatorics are plunging and it is not a lot of time consuming. But the problem with this method: we do not have any time management, which is not a good idea in duels where we only have 5 minutes maximum for each game and. Here in GO, where the combinatorics are huge and increasing by 1 the depth can explodes the time we consume during AlphaBeta.

Thanks to Iterative Deepening method, we are able to handle this issue. We approximate a 9*9 board GO game with 60 moves for each players (with suicides and captures, we are able to do more than 9*9 moves). It means we have 5 seconds maximum for each move. Like that, we can adjust our depth depending on the progression of the game. We think that is the best compromise because we can see adjust depth depending on when we are during the game but by taking only 5 seconds during the whole game for each moves.

# Heuristics

In order to use algorithms, we must know **how to evaluate a specific board**. At first we dove into the evaluation of the board:
- global positions
- overall positions
- good patterns
- threats
- ...

But the more we deepen into our heuristic, the more it was difficult to handle every situation. It is known that GO has no good heuristic found yet, so we think it is not the good idea to evaluate the board like that.

So we took the decision to use the **CNN model** developed during the previous lab work: from a specific board, it computes the probability for blacks and whites to win. We found more interesting to evaluate a board like that because basically the model does not compute the positions of blacks and whites pawns, it only predicts the probability of winning based on the training set, where for each data (each board), we played 100 games and both players did the best moves resulting a number of wins for each players. Which means, during our AlphaBeta (or Iterative Deepening) we are able to evaluate a board and to know if this is a good thing to make this move.

But we had to do more. Indeed, our neural network takes our board inputs in a specific format. Which means we have to format our data at each evaluation of our board in order to make predictions. But formatting the data is in O(n²) with n the board size in length. As we previously said, the combinatorics are very huge which means doing the board formatting is very heavy and time consuming (which is not a good idea for Iterative Deepening). That is the reason why we also used a variable corresponding to the neural network board where we push and pop new pawns on it. Thanks to that, we are able to make predictions much faster than before.

# Future improvements

In order to improve our player, here is a non exhaustive list of possible things to do:
- dynamic time per round based on how many time left: if we arrive at very late game, we must manage our time preciously, so if we have 30 seconds left but approximately 10 moves to play, we must reduce our time per round. We could also compute time dynamically with the speed of the algorithm (nodes / second) at a given moment (preivous move, mean, ...)
- improve heuristic: even though our heuristic works, we should probably handle by hand few threats and dangerous patterns by hand in order to avoid serious situations for our player
- clean our code (!!)
- try a Monte-Carlo heuristic for our player
