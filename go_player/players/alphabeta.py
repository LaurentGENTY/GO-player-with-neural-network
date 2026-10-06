"""The 2020 Alpha-Beta + iterative deepening player, with its bugs fixed."""

import math
import random
import time

from go_player.goban import Board
from go_player.nn import ValueNet
from go_player.opening import OpeningBook
from go_player.players.base import PlayerInterface, is_winning

WIN_SCORE = 1000


class AlphaBetaPlayer(PlayerInterface):
    def __init__(
        self,
        value_net: ValueNet,
        time_budget: float = 5.0,
        seed: int | None = None,
        opening_book: bool = True,
        komi: float = 0.0,
        max_depth: int = 10,
        name: str = "Alpha-Beta",
    ):
        self._net = value_net
        self._time_budget = time_budget
        self._rng = random.Random(seed)
        self._book = OpeningBook.load(self._rng) if opening_book else None
        self._board = Board(komi)
        self._max_depth = max_depth
        self._name = name
        self._me = Board._BLACK
        self._deadline = 0.0
        self._timed_out = False
        self.last_depth = 0

    def getPlayerName(self) -> str:
        return self._name

    def newGame(self, color: int) -> None:
        self._me = color

    def playOpponentMove(self, move: str) -> None:
        self._board.push(Board.name_to_flat(move))

    def getPlayerMove(self) -> str:
        board = self._board
        if board.is_game_over():
            return "PASS"
        self._me = board.next_player()
        move = self._choose()
        board.push(move)
        return Board.flat_to_name(move)

    def _choose(self) -> int:
        if self._board._lastPlayerHasPassed and is_winning(self._board, self._me):
            return -1
        if self._book is not None:
            move = self._book.next_move(self._board)
            if move is not None:
                return move
        return self.iterative_deepening()

    def evaluate(self) -> float:
        return float(self._net.evaluate(self._board)[self._me - 1]) * 100

    def _terminal_score(self) -> float:
        result = self._board.result()
        if result == "1/2-1/2":
            return 0.0
        winner = Board._WHITE if result == "1-0" else Board._BLACK
        return WIN_SCORE if winner == self._me else -WIN_SCORE

    def _out_of_time(self) -> bool:
        if time.perf_counter() >= self._deadline:
            self._timed_out = True
        return self._timed_out

    def iterative_deepening(self) -> int:
        self._deadline = time.perf_counter() + 0.9 * self._time_budget
        best, self.last_depth = None, 0
        for depth in range(1, self._max_depth + 1):
            move, completed = self._root(depth)
            if completed:
                best, self.last_depth = move, depth
            elif best is None:
                best = move  # a partial depth-1 search beats a random move
            if not completed or time.perf_counter() >= self._deadline:
                break
        if best is None:
            legal = [m for m in self._board.legal_moves() if m != -1]
            best = self._rng.choice(legal) if legal else -1
        return best

    def _root(self, depth: int) -> tuple[int | None, bool]:
        board = self._board
        self._timed_out = False
        best_moves, best_value = [], None
        for move in board.weak_legal_moves():
            if self._out_of_time():
                break
            if board.push(move):
                value = self._min_value(depth - 1, -math.inf, math.inf)
                if best_value is None or value > best_value:
                    best_moves, best_value = [move], value
                elif value == best_value:
                    best_moves.append(move)
            board.pop()
        move = self._rng.choice(best_moves) if best_moves else None
        return move, not self._timed_out

    def _max_value(self, depth: int, alpha: float, beta: float) -> float:
        board = self._board
        if board.is_game_over():
            return self._terminal_score()
        if depth <= 0:
            return self.evaluate()
        value = -math.inf
        for move in board.weak_legal_moves():
            if self._out_of_time():
                break
            if board.push(move):
                value = max(value, self._min_value(depth - 1, alpha, beta))
            board.pop()
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def _min_value(self, depth: int, alpha: float, beta: float) -> float:
        board = self._board
        if board.is_game_over():
            return self._terminal_score()
        if depth <= 0:
            return self.evaluate()
        value = math.inf
        for move in board.weak_legal_moves():
            if self._out_of_time():
                break
            if board.push(move):
                value = min(value, self._max_value(depth - 1, alpha, beta))
            board.pop()
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value
