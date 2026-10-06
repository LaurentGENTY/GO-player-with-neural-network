import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Callable

from go_player.goban import Board
from go_player.players.base import PlayerInterface

PlayerFactory = Callable[[int], PlayerInterface]


@dataclass
class GameRecord:
    black: str
    white: str
    moves: list[str] = field(default_factory=list)
    winner: int | None = None
    score: str = ""
    illegal_by: int | None = None
    move_times: dict[int, list[float]] = field(default_factory=lambda: {Board._BLACK: [], Board._WHITE: []})
    simulations: dict[int, list[int]] = field(default_factory=lambda: {Board._BLACK: [], Board._WHITE: []})


def _to_flat(move: str) -> int | None:
    try:
        return Board.name_to_flat(move)
    except (KeyError, ValueError, IndexError, TypeError):
        return None


def play_game(black: PlayerInterface, white: PlayerInterface, komi: float = 0.0, max_moves: int = 200) -> GameRecord:
    board = Board(komi)
    players = {Board._BLACK: black, Board._WHITE: white}
    black.newGame(Board._BLACK)
    white.newGame(Board._WHITE)
    record = GameRecord(black=black.getPlayerName(), white=white.getPlayerName())
    color = Board._BLACK
    while not board.is_game_over() and len(record.moves) < max_moves:
        player = players[color]
        start = time.perf_counter()
        move = player.getPlayerMove()
        record.move_times[color].append(time.perf_counter() - start)
        sims = getattr(player, "last_simulations", 0)
        if sims:
            record.simulations[color].append(sims)
        flat = _to_flat(move)
        if flat is None or flat not in board.legal_moves():
            record.illegal_by = color
            break
        board.push(flat)
        record.moves.append(move)
        players[Board.flip(color)].playOpponentMove(move)
        color = Board.flip(color)

    if record.illegal_by is not None:
        record.winner, record.score = Board.flip(record.illegal_by), "illegal move"
    elif not board.is_game_over():
        record.winner, record.score = None, "move cap"
    else:
        result = board.result()
        record.winner = {"1-0": Board._WHITE, "0-1": Board._BLACK}.get(result)
        record.score = board.final_go_score()
    for player in players.values():
        player.endGame(record.winner or 0)
    return record


def _avg(values: list[float]) -> float | None:
    return mean(values) if values else None


@dataclass
class MatchResult:
    a: str
    b: str
    games: list[GameRecord] = field(default_factory=list)
    a_colors: list[int] = field(default_factory=list)

    def summary(self) -> dict:
        pairs = list(zip(self.games, self.a_colors))
        a_times = [t for g, c in pairs for t in g.move_times[c]]
        b_times = [t for g, c in pairs for t in g.move_times[Board.flip(c)]]
        a_sims = [s for g, c in pairs for s in g.simulations[c]]
        b_sims = [s for g, c in pairs for s in g.simulations[Board.flip(c)]]
        return {
            "games": len(self.games),
            "a_wins": sum(g.winner == c for g, c in pairs),
            "b_wins": sum(g.winner == Board.flip(c) for g, c in pairs),
            "draws": sum(g.winner is None for g in self.games),
            "a_wins_as_black": sum(g.winner == c == Board._BLACK for g, c in pairs),
            "a_wins_as_white": sum(g.winner == c == Board._WHITE for g, c in pairs),
            "illegal_moves": sum(g.illegal_by is not None for g in self.games),
            "a_avg_move_s": _avg(a_times),
            "b_avg_move_s": _avg(b_times),
            "a_avg_sims": _avg(a_sims),
            "b_avg_sims": _avg(b_sims),
        }


def run_match(
    factory_a: PlayerFactory,
    factory_b: PlayerFactory,
    games: int = 20,
    seed: int = 0,
    komi: float = 0.0,
    max_moves: int = 200,
    on_game: Callable[[int, GameRecord], None] | None = None,
) -> MatchResult:
    result: MatchResult | None = None
    for i in range(games):
        game_seed = seed * 1000 + 2 * i
        a, b = factory_a(game_seed), factory_b(game_seed + 1)
        if result is None:
            result = MatchResult(a=a.getPlayerName(), b=b.getPlayerName())
        # Alternate colors so that the first-move advantage (no komi by default) cancels out.
        a_color = Board._BLACK if i % 2 == 0 else Board._WHITE
        black, white = (a, b) if a_color == Board._BLACK else (b, a)
        record = play_game(black, white, komi=komi, max_moves=max_moves)
        result.games.append(record)
        result.a_colors.append(a_color)
        if on_game is not None:
            on_game(i, record)
    assert result is not None, "games must be >= 1"
    return result


def _fmt(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def write_report(results: list[MatchResult], out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Match (A vs B) | Games | A wins | B wins | Draws | A wins as Black/White | A s/move | B s/move | A sims/move |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    matches = []
    for r in results:
        s = r.summary()
        lines.append(
            f"| {r.a} vs {r.b} | {s['games']} | {s['a_wins']} | {s['b_wins']} | {s['draws']} | "
            f"{s['a_wins_as_black']}/{s['a_wins_as_white']} | {_fmt(s['a_avg_move_s'], 2)} | "
            f"{_fmt(s['b_avg_move_s'], 2)} | {_fmt(s['a_avg_sims'], 0)} |"
        )
        matches.append({"a": r.a, "b": r.b, "summary": s, "games": [asdict(g) for g in r.games]})
    md = out_dir / "arena.md"
    md.write_text("\n".join(lines) + "\n")
    js = out_dir / "arena.json"
    js.write_text(json.dumps({"matches": matches}, indent=2))
    return md, js
