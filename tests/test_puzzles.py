import pytest

from go_player.goban import Board
from go_player.puzzles import PUZZLES, solve

SOLVED = [p for p in PUZZLES if not p.known_limit]
LIMITS = [p for p in PUZZLES if p.known_limit]


@pytest.mark.parametrize("puzzle", PUZZLES, ids=lambda p: p.name)
def test_puzzle_position_is_legal_with_black_to_play(puzzle):
    board = puzzle.board()
    assert board.next_player() == Board._BLACK
    assert not board._lastPlayerHasPassed  # otherwise the "pass when winning" rule kicks in
    for name in puzzle.black:
        assert board[Board.name_to_flat(name)] == Board._BLACK
    for name in puzzle.white:
        assert board[Board.name_to_flat(name)] == Board._WHITE
    assert all(Board.name_to_flat(a) in board.legal_moves() for a in puzzle.answers)


@pytest.mark.parametrize("puzzle", SOLVED, ids=lambda p: p.name)
def test_mcts_solves_puzzle(net, puzzle):
    solution = solve(puzzle, net)
    assert solution.correct, f"played {solution.move}, expected one of {sorted(puzzle.answers)}"
    assert sum(solution.visits.values()) == solution.simulations


@pytest.mark.parametrize("puzzle", LIMITS, ids=lambda p: p.name)
@pytest.mark.xfail(strict=True, reason="known limit of the 2020 value network")
def test_known_limits_stay_unsolved(net, puzzle):
    assert solve(puzzle, net).correct


def test_showcase_has_solved_puzzles_and_known_limits():
    assert len(SOLVED) >= 3 and len(LIMITS) >= 1
    assert len({p.name for p in PUZZLES}) == len(PUZZLES)
