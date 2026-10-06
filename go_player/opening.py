import json
import random
from importlib.resources import files

from go_player.goban import Board

MAX_LINE_ATTEMPTS = 10


class OpeningBook:
    """2020 opening strategy: follow one side of a random recorded game, skipping unplayable moves."""

    def __init__(self, games: list[list[str]], rng: random.Random, length: int = 5):
        self._games = games
        self._rng = rng
        self._length = length
        self._played = 0
        self._line: list[str] | None = None
        self._index = 0

    @classmethod
    def load(cls, rng: random.Random, length: int = 5) -> "OpeningBook":
        data = json.loads((files("go_player") / "assets" / "games.json").read_text())
        return cls([game["moves"] for game in data], rng, length)

    def _pick_line(self) -> list[str]:
        game = self._rng.choice(self._games)
        return game[self._rng.randint(0, 1)::2]

    def next_move(self, board: Board) -> int | None:
        if self._played >= self._length:
            return None
        for _ in range(MAX_LINE_ATTEMPTS):
            if self._line is None:
                self._line, self._index = self._pick_line(), 0
            while self._index < len(self._line):
                name = self._line[self._index]
                self._index += 1
                if name == "PASS":
                    continue
                move = Board.name_to_flat(name)
                if move not in board.weak_legal_moves():
                    continue
                legal = board.push(move)  # push() is the only superko check
                board.pop()
                if legal:
                    self._played += 1
                    return move
            self._line = None
        return None
