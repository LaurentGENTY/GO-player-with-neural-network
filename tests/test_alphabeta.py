import pytest

from go_player.arena import play_game
from go_player.goban import Board
from go_player.players.alphabeta import AlphaBetaPlayer
from go_player.players.random_player import RandomPlayer
from positions import BLACK_LOSING_AFTER_PASS, BLACK_WINNING_AFTER_PASS, CAPTURE_SETUP, feed


def alphabeta(net, **kwargs):
    kwargs.setdefault("opening_book", False)
    kwargs.setdefault("seed", 0)
    return AlphaBetaPlayer(net, **kwargs)


def test_finds_the_obvious_capture(net):
    player = alphabeta(net, time_budget=2.0)
    player.newGame(Board._BLACK)
    feed(player, CAPTURE_SETUP)
    assert player.getPlayerMove() == "E6"
    assert player.last_depth >= 1


def test_evaluation_sees_captures(net):
    player = alphabeta(net)
    player.newGame(Board._WHITE)
    feed(player, CAPTURE_SETUP + ["E6"])  # White to move, E5-E4 were captured
    player._me = Board._WHITE
    assert player.evaluate() == pytest.approx(float(net.evaluate(player._board)[1]) * 100)


def test_plays_only_legal_moves_against_random(net):
    record = play_game(alphabeta(net, time_budget=0.3), RandomPlayer(seed=5), max_moves=20)
    assert record.illegal_by is None


def test_alphabeta_passes_when_winning_after_opponent_pass(net):
    player = alphabeta(net, time_budget=0.5)
    player.newGame(Board._BLACK)
    feed(player, BLACK_WINNING_AFTER_PASS)
    assert player.getPlayerMove() == "PASS"


def test_alphabeta_does_not_pass_when_losing(net):
    player = alphabeta(net, time_budget=0.5)
    player.newGame(Board._BLACK)
    feed(player, BLACK_LOSING_AFTER_PASS)
    assert player.getPlayerMove() != "PASS"
