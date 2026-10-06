from go_player.goban import Board


class PlayerInterface:
    """The 2020 tournament contract. Moves are strings: "A1" ... "J9" or "PASS"."""

    def getPlayerName(self) -> str:
        return "Not Defined"

    def getPlayerMove(self) -> str:
        return "PASS"

    def playOpponentMove(self, move: str) -> None:
        pass

    def newGame(self, color: int) -> None:
        pass

    def endGame(self, winner: int) -> None:
        pass


def candidate_moves(board) -> list[int]:
    # Passing early is never useful for a search; keep it only when nothing else is legal.
    moves = [m for m in board.weak_legal_moves() if m != -1]
    return moves or [-1]


def is_winning(board, color: int) -> bool:
    black, white = board.compute_score()
    return black > white if color == Board._BLACK else white > black
