"""Tactical puzzles used both as tests and as showcase clips.

Each puzzle is a static position with Black to play. The known limits are kept
on purpose: the 2020 value network never learned life and death, and the
showcase says so instead of hiding it.
"""

from dataclasses import dataclass

from go_player.goban import Board
from go_player.nn import ValueNet
from go_player.players.mcts import MCTSPlayer, NNEvaluator

# A low exploration constant concentrates visits on the best moves, which is
# what makes the heatmaps readable (c=1.4 spreads them almost uniformly).
PUZZLE_C = 0.3
PUZZLE_SIMULATIONS = 3000


@dataclass(frozen=True)
class Puzzle:
    name: str
    title: str
    black: tuple[str, ...]
    white: tuple[str, ...]
    answers: frozenset[str]
    known_limit: bool = False

    def setup_moves(self) -> list[str]:
        # Stones are placed by alternating moves; the side with fewer stones passes,
        # and White always moves last so that Black is to play and nobody just passed.
        black, white, moves = list(self.black), list(self.white), []
        while len(black) > len(white):
            moves += [black.pop(0), "PASS"]
        while len(white) > len(black):
            moves += ["PASS", white.pop(0)]
        for b, w in zip(black, white):
            moves += [b, w]
        return moves

    def board(self) -> Board:
        board = Board()
        for name in self.setup_moves():
            assert board.push(Board.name_to_flat(name)), f"{self.name}: illegal setup move {name}"
        return board


PUZZLES = [
    Puzzle("capture", "Capture the two stones",
           ("D5", "D4", "F5", "F4", "E3", "C7"), ("E5", "E4", "G7", "G3"), frozenset({"E6"})),
    Puzzle("bigger-capture", "Take the bigger capture",
           ("D5", "D4", "D3", "F5", "F4", "F3", "E6", "A8", "C8", "B9"),
           ("E5", "E4", "E3", "B8", "H7", "H2", "G8", "J4"), frozenset({"E2"})),
    Puzzle("connect", "Connect before White cuts",
           ("C5", "D5", "F5", "G5", "C3"), ("E6", "E4", "G7", "C7"), frozenset({"E5"})),
    Puzzle("escape", "Escape from atari",
           ("E5", "E4", "G3", "C7"), ("D5", "D4", "F5", "F4", "E3", "G7"), frozenset({"E6"}), known_limit=True),
    Puzzle("live", "Make two eyes",
           ("A2", "B2", "C2", "D2", "D1", "F5"), ("A3", "B3", "C3", "D3", "E2", "E1", "E5"),
           frozenset({"B1"}), known_limit=True),
    Puzzle("kill", "Kill the group",
           ("A3", "B3", "C3", "D3", "E2", "E1", "E5"), ("A2", "B2", "C2", "D2", "D1", "F5"),
           frozenset({"B1"}), known_limit=True),
    Puzzle("net", "Trap the stone in a net",
           ("D5", "E4", "C3"), ("E5", "G7"), frozenset({"F6"}), known_limit=True),
]


@dataclass
class Solution:
    move: str
    visits: dict[int, int]
    simulations: int
    correct: bool


def solve(puzzle: Puzzle, net: ValueNet, simulations: int = PUZZLE_SIMULATIONS, seed: int = 0) -> Solution:
    player = MCTSPlayer(NNEvaluator(net), max_simulations=simulations, c=PUZZLE_C, seed=seed, opening_book=False)
    player.newGame(Board._BLACK)
    for name in puzzle.setup_moves():
        player.playOpponentMove(name)
    move = player.getPlayerMove()
    return Solution(move, player.last_root_visits, player.last_simulations, move in puzzle.answers)
