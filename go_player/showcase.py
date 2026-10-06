"""Puzzle clips and the portfolio reel: position, MCTS visit heatmap, move played."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from go_player.goban import Board
from go_player.nn import ValueNet
from go_player.players.alphabeta import AlphaBetaPlayer
from go_player.puzzles import PUZZLE_SIMULATIONS, Puzzle, Solution, solve
from go_player.record import HEIGHT, LINE, WIDTH, _font, render_frame, render_title_card, write_media

TITLE_MS, POSITION_MS, HEAT_MS, MOVE_MS = 1500, 1500, 2500, 2000
LIMIT_HEAT_MS, LIMIT_MOVE_MS, CARD_MS, SCORECARD_MS = 2000, 2000, 2500, 4000
GREEN, RED = (39, 174, 96), (192, 57, 43)


@dataclass
class PuzzleResult:
    puzzle: Puzzle
    solution: Solution
    alphabeta_move: str

    @property
    def alphabeta_correct(self) -> bool:
        return self.alphabeta_move in self.puzzle.answers


def _alphabeta_move(puzzle: Puzzle, net: ValueNet, time_budget: float) -> str:
    player = AlphaBetaPlayer(net, time_budget=time_budget, seed=0, opening_book=False)
    player.newGame(Board._BLACK)
    for name in puzzle.setup_moves():
        player.playOpponentMove(name)
    return player.getPlayerMove()


def _verdict(result: PuzzleResult) -> str:
    solution = result.solution
    if solution.correct:
        return f"MCTS plays {solution.move}: solved"
    return f"MCTS plays {solution.move}: missed (answer {', '.join(sorted(result.puzzle.answers))})"


def _heat_and_move_frames(result: PuzzleResult, heat_ms: int, move_ms: int) -> list[tuple[Image.Image, int]]:
    puzzle, solution = result.puzzle, result.solution
    board = puzzle.board()
    total = sum(solution.visits.values()) or 1
    heat = {move: n / total for move, n in solution.visits.items()}
    thinking = render_frame(board, puzzle.title, f"What MCTS considers ({solution.simulations} simulations)",
                            None, None, heat=heat)
    move = Board.name_to_flat(solution.move)
    board.push(move)
    return [(thinking, heat_ms), (render_frame(board, puzzle.title, _verdict(result), move, None), move_ms)]


def puzzle_clip(result: PuzzleResult) -> list[tuple[Image.Image, int]]:
    puzzle = result.puzzle
    return [
        (render_title_card(puzzle.title, "Black to play"), TITLE_MS),
        (render_frame(puzzle.board(), puzzle.title, "Black to play", None, None), POSITION_MS),
        *_heat_and_move_frames(result, HEAT_MS, MOVE_MS),
    ]


DEFAULT_SCORECARD_TITLE = "MCTS + 2020 CNN on Go puzzles"
DEFAULT_FOOTNOTE = "Known limit: the 2020 network never learned life and death"


def render_scorecard(
    results: list[PuzzleResult], title: str = DEFAULT_SCORECARD_TITLE, footnote: str = DEFAULT_FOOTNOTE,
) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    solved = sum(r.solution.correct for r in results)
    draw.text((WIDTH // 2, 48), title, anchor="mm", fill="black", font=_font(20))
    draw.text((WIDTH // 2, 84), f"{solved}/{len(results)} solved", anchor="mm", fill=LINE, font=_font(18))
    for i, r in enumerate(results):
        y = 136 + i * 40
        draw.text((32, y), r.puzzle.title, anchor="lm", fill="black", font=_font(16))
        ok = r.solution.correct
        draw.text((WIDTH - 32, y), "SOLVED" if ok else "MISSED", anchor="rm", fill=GREEN if ok else RED, font=_font(16))
    draw.text((WIDTH // 2, HEIGHT - 40), footnote, anchor="mm", fill=LINE, font=_font(12))
    return image


def _write(frames: list[tuple[Image.Image, int]], out_stem: Path) -> None:
    write_media([f for f, _ in frames], [ms for _, ms in frames], out_stem)


def _mark(move: str, ok: bool) -> str:
    return f"{move} {'✅' if ok else '❌'}"


def _results_table(results: list[PuzzleResult]) -> str:
    lines = [
        "| Puzzle | Answer | MCTS 2026 | AlphaBeta 2020 |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r.puzzle.title} | {', '.join(sorted(r.puzzle.answers))} | "
                     f"{_mark(r.solution.move, r.solution.correct)} | {_mark(r.alphabeta_move, r.alphabeta_correct)} |")
    solved = sum(r.solution.correct for r in results)
    ab_solved = sum(r.alphabeta_correct for r in results)
    lines.append("")
    lines.append(f"MCTS solved {solved}/{len(results)}, AlphaBeta solved {ab_solved}/{len(results)}.")
    return "\n".join(lines) + "\n"


def render_puzzles(
    puzzles: list[Puzzle],
    net: ValueNet,
    out_dir: Path,
    simulations: int = PUZZLE_SIMULATIONS,
    alphabeta_time: float = 2.0,
) -> list[PuzzleResult]:
    out_dir = Path(out_dir)
    results = [
        PuzzleResult(p, solve(p, net, simulations=simulations), _alphabeta_move(p, net, alphabeta_time))
        for p in puzzles
    ]
    for r in results:
        _write(puzzle_clip(r), out_dir / r.puzzle.name)

    reel = [frame for r in results if r.solution.correct for frame in puzzle_clip(r)]
    missed = [r for r in results if not r.solution.correct]
    if missed:
        reel.append((render_title_card("Known limits", "where the 2020 network goes wrong"), CARD_MS))
        for r in missed:
            reel += _heat_and_move_frames(r, LIMIT_HEAT_MS, LIMIT_MOVE_MS)
    reel.append((render_scorecard(results), SCORECARD_MS))
    _write(reel, out_dir / "reel")

    (out_dir / "results.md").write_text(_results_table(results))
    return results
