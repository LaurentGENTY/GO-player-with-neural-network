import random

from go_player.goban import Board
from go_player.opening import OpeningBook
from go_player.players.base import candidate_moves, is_winning
from go_player.players.random_player import RandomPlayer
from positions import BLACK_LOSING_AFTER_PASS, BLACK_WINNING_AFTER_PASS, board_with


def test_candidate_moves_exclude_pass():
    moves = candidate_moves(Board())
    assert len(moves) == 81
    assert -1 not in moves


def test_candidate_moves_fall_back_to_pass():
    class OnlyPass:
        def weak_legal_moves(self):
            return [-1]

    assert candidate_moves(OnlyPass()) == [-1]


def test_is_winning():
    assert is_winning(board_with(BLACK_WINNING_AFTER_PASS), Board._BLACK)
    assert not is_winning(board_with(BLACK_LOSING_AFTER_PASS), Board._BLACK)
    assert is_winning(board_with(BLACK_LOSING_AFTER_PASS), Board._WHITE)


def test_random_player_is_legal_and_seeded():
    first, second = RandomPlayer(seed=4), RandomPlayer(seed=4)
    first.newGame(Board._BLACK)
    second.newGame(Board._BLACK)
    move = first.getPlayerMove()
    assert move == second.getPlayerMove()
    assert Board.name_to_flat(move) in Board().legal_moves()


def test_opening_book_loads_2020_games():
    book = OpeningBook.load(random.Random(0))
    move = book.next_move(Board())
    assert move in Board().weak_legal_moves()


def test_opening_book_stops_after_length():
    book = OpeningBook([["E5", "E6", "D5", "D6", "C5", "C6"]], random.Random(0), length=2)
    board = Board()
    played = []
    while (move := book.next_move(board)) is not None:
        played.append(move)
        board.push(move)
        board.push(-1)  # opponent passes
    assert len(played) == 2


def test_opening_book_skips_occupied_points_and_gives_up():
    book = OpeningBook([["E5", "D5", "E5", "D5"]], random.Random(0), length=5)
    board = board_with(["E5", "D5"])
    assert book.next_move(board) is None
