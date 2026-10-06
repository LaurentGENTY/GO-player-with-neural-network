import shutil
import subprocess

from go_player.goban import Board
from go_player.players.base import PlayerInterface


class GnuGoNotFound(Exception):
    pass


class GnuGoError(Exception):
    pass


def normalize_gtp_move(move: str) -> str:
    move = move.strip().upper()
    return "PASS" if move in ("PASS", "RESIGN") else move


def find_gnugo() -> str:
    exe = shutil.which("gnugo")
    if exe is None:
        raise GnuGoNotFound("gnugo not found in PATH. Install it with: brew install gnu-go")
    return exe


class GnuGoPlayer(PlayerInterface):
    def __init__(self, level: int = 1, komi: float = 0.0, name: str | None = None):
        exe = find_gnugo()
        self._proc = subprocess.Popen(
            [exe, "--mode", "gtp", "--boardsize", str(Board._BOARDSIZE), "--chinese-rules",
             "--capture-all-dead", "--never-resign", "--level", str(level), "--komi", str(komi)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        self._name = name or f"GnuGo-L{level}"
        self._color = Board._BLACK

    def _query(self, command: str) -> str:
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()
        lines = []
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                raise GnuGoError(f"gnugo exited while running {command!r}")
            line = line.rstrip("\n")
            if not line:
                if lines:
                    break  # GTP responses end with an empty line
                continue
            lines.append(line)
        if lines[0].startswith("?"):
            raise GnuGoError(f"gnugo rejected {command!r}: {lines[0][1:].strip()}")
        return " ".join([lines[0][1:].strip(), *lines[1:]]).strip()

    def getPlayerName(self) -> str:
        return self._name

    def newGame(self, color: int) -> None:
        self._color = color
        self._query("clear_board")

    def getPlayerMove(self) -> str:
        return normalize_gtp_move(self._query(f"genmove {Board.player_name(self._color)}"))

    def playOpponentMove(self, move: str) -> None:
        self._query(f"play {Board.player_name(Board.flip(self._color))} {move}")

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.stdin.write("quit\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=5)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
