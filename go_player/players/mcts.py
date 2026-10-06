"""UCT Monte-Carlo tree search with batched leaf evaluation.

The tree is walked with push/pop on the player's own board (Board has no cheap
copy). Leaves are collected in batches with a virtual loss (a visit counted
before its value is known), so the CNN evaluates many leaves per call.
"""

import math
import random
import time

import numpy as np

from go_player.goban import Board
from go_player.nn import ValueNet, encode
from go_player.opening import OpeningBook
from go_player.players.base import PlayerInterface, candidate_moves, is_winning


def terminal_value(board: Board) -> np.ndarray:
    result = board.result()
    if result == "0-1":
        return np.array([1.0, 0.0])
    if result == "1-0":
        return np.array([0.0, 1.0])
    return np.array([0.5, 0.5])


class NNEvaluator:
    def __init__(self, net: ValueNet):
        self._net = net

    def prepare(self, board: Board) -> np.ndarray:
        return encode(board)

    def evaluate(self, prepared: list) -> np.ndarray:
        return self._net.predict(np.stack(prepared))


class RolloutEvaluator:
    def __init__(self, rng: random.Random, max_moves: int = 200):
        self._rng = rng
        self._max_moves = max_moves

    def prepare(self, board: Board) -> np.ndarray:
        pushed = 0
        while not board.is_game_over() and pushed < self._max_moves:
            moves = candidate_moves(board)
            self._rng.shuffle(moves)
            for move in moves + [-1]:
                pushed += 1
                if board.push(move):
                    break
                board.pop()  # superko: undo and try the next move
                pushed -= 1
        if board.is_game_over():
            value = terminal_value(board)
        else:
            black, white = board.compute_score()
            value = np.array([1.0, 0.0]) if black > white else np.array([0.0, 1.0]) if white > black else np.array([0.5, 0.5])
        for _ in range(pushed):
            board.pop()
        return value

    def evaluate(self, prepared: list) -> np.ndarray:
        return np.stack(prepared)


class Node:
    __slots__ = ("move", "player", "children", "untried", "n", "w")

    def __init__(self, move: int | None, player: int):
        self.move = move
        self.player = player  # color that played `move`; w is from this color's point of view
        self.children: list["Node"] = []
        self.untried: list[int] | None = None
        self.n = 0
        self.w = 0.0


class MCTSPlayer(PlayerInterface):
    def __init__(
        self,
        evaluator,
        time_budget: float = 5.0,
        batch_size: int = 16,
        c: float = 1.4,
        seed: int | None = None,
        opening_book: bool = True,
        max_simulations: int | None = None,
        komi: float = 0.0,
        name: str = "MCTS",
    ):
        self._evaluator = evaluator
        self._time_budget = time_budget
        self._batch_size = batch_size
        self._c = c
        self._rng = random.Random(seed)
        self._book = OpeningBook.load(self._rng) if opening_book else None
        self._max_simulations = max_simulations
        self._board = Board(komi)
        self._name = name
        self.last_simulations = 0

    def getPlayerName(self) -> str:
        return self._name

    def playOpponentMove(self, move: str) -> None:
        self._board.push(Board.name_to_flat(move))

    def getPlayerMove(self) -> str:
        self.last_simulations = 0
        if self._board.is_game_over():
            return "PASS"
        move = self._choose()
        self._board.push(move)
        return Board.flat_to_name(move)

    def _choose(self) -> int:
        board = self._board
        if board._lastPlayerHasPassed and is_winning(board, board.next_player()):
            return -1
        if self._book is not None:
            move = self._book.next_move(board)
            if move is not None:
                return move
        return self.search()

    def search(self) -> int:
        board = self._board
        root = Node(None, Board.flip(board.next_player()))
        deadline = time.perf_counter() + 0.9 * self._time_budget
        sims = 0
        while True:
            if self._max_simulations is not None:
                remaining = self._max_simulations - sims
                if remaining <= 0:
                    break
            elif time.perf_counter() >= deadline:
                break
            else:
                remaining = self._batch_size
            batch = [self._select_and_expand(root) for _ in range(min(self._batch_size, remaining))]
            sims += len(batch)
            self._backup(batch)
        self.last_simulations = sims
        if not root.children:
            legal = [m for m in board.legal_moves() if m != -1]
            return self._rng.choice(legal) if legal else -1
        return max(root.children, key=lambda child: child.n).move

    def _uct(self, child: Node, parent_n: int) -> float:
        if child.n == 0:
            return math.inf
        return child.w / child.n + self._c * math.sqrt(math.log(parent_n) / child.n)

    def _select_and_expand(self, root: Node):
        board = self._board
        node, path, pushed = root, [root], 0
        root.n += 1  # virtual loss: the visit counts now, its value is added in _backup
        while True:
            if board.is_game_over():
                leaf = ("terminal", terminal_value(board))
                break
            if node.untried is None:
                node.untried = candidate_moves(board)
                self._rng.shuffle(node.untried)
            child = None
            while node.untried:
                move = node.untried.pop()
                if board.push(move):
                    pushed += 1
                    child = Node(move, Board.flip(board.next_player()))
                    node.children.append(child)
                    break
                board.pop()  # superko
            if child is not None:
                child.n += 1
                path.append(child)
                leaf = ("prepared", self._evaluator.prepare(board))
                break
            if not node.children:
                node.untried = [-1]  # every move was superko: only passing is left
                continue
            node = max(node.children, key=lambda ch: self._uct(ch, node.n))
            board.push(node.move)
            pushed += 1
            node.n += 1
            path.append(node)
        for _ in range(pushed):
            board.pop()
        return path, leaf

    def _backup(self, batch) -> None:
        prepared = [payload for _, (kind, payload) in batch if kind == "prepared"]
        values = iter(self._evaluator.evaluate(prepared)) if prepared else iter(())
        for path, (kind, payload) in batch:
            value = payload if kind == "terminal" else next(values)
            for node in path:
                node.w += value[node.player - 1]
