import random

from go_player.goban import Board
from go_player.players.base import PlayerInterface


class RandomPlayer(PlayerInterface):
    def __init__(self, seed: int | None = None, komi: float = 0.0, name: str = "Random"):
        self._board = Board(komi)
        self._rng = random.Random(seed)
        self._name = name

    def getPlayerName(self) -> str:
        return self._name

    def getPlayerMove(self) -> str:
        if self._board.is_game_over():
            return "PASS"
        move = self._rng.choice(self._board.legal_moves())
        self._board.push(move)
        return Board.flat_to_name(move)

    def playOpponentMove(self, move: str) -> None:
        self._board.push(Board.name_to_flat(move))
