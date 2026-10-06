import random
import time

import numpy as np

from go_player.arena import play_game
from go_player.goban import Board
from go_player.players.mcts import MCTSPlayer, NNEvaluator, RolloutEvaluator
from go_player.players.random_player import RandomPlayer
from positions import BLACK_LOSING_AFTER_PASS, BLACK_WINNING_AFTER_PASS, CAPTURE_SETUP, board_with, feed


def mcts(net, **kwargs):
    kwargs.setdefault("opening_book", False)
    kwargs.setdefault("seed", 0)
    return MCTSPlayer(NNEvaluator(net), **kwargs)


def test_finds_the_obvious_capture(net):
    player = mcts(net, max_simulations=600)
    player.newGame(Board._BLACK)
    feed(player, CAPTURE_SETUP)
    assert player.getPlayerMove() == "E6"
    assert player.last_simulations == 600


def test_plays_only_legal_moves_against_random(net):
    record = play_game(mcts(net, max_simulations=16), RandomPlayer(seed=2), max_moves=40)
    assert record.illegal_by is None
    assert len(record.moves) == 40 or record.winner is not None


def test_respects_time_budget(net):
    player = mcts(net, time_budget=0.5)
    player.newGame(Board._BLACK)
    start = time.perf_counter()
    player.getPlayerMove()
    assert time.perf_counter() - start <= 0.55
    assert player.last_simulations > 0


def test_is_reproducible_with_a_seed(net):
    moves = []
    for _ in range(2):
        player = mcts(net, max_simulations=64, seed=7)
        player.newGame(Board._BLACK)
        feed(player, CAPTURE_SETUP[:6])
        moves.append(player.getPlayerMove())
    assert moves[0] == moves[1]


def test_mcts_passes_when_winning_after_opponent_pass(net):
    player = mcts(net, max_simulations=32)
    player.newGame(Board._BLACK)
    feed(player, BLACK_WINNING_AFTER_PASS)
    assert player.getPlayerMove() == "PASS"


def test_mcts_does_not_pass_when_losing(net):
    player = mcts(net, max_simulations=32)
    player.newGame(Board._BLACK)
    feed(player, BLACK_LOSING_AFTER_PASS)
    assert player.getPlayerMove() != "PASS"


def test_rollout_evaluator_leaves_board_unchanged():
    board = board_with(CAPTURE_SETUP)
    cells, trail = board._board.copy(), len(board._trailMoves)
    value = RolloutEvaluator(random.Random(0), max_moves=60).prepare(board)
    assert np.array_equal(board._board, cells) and len(board._trailMoves) == trail
    assert value.sum() == 1.0


def test_rollout_mcts_returns_a_legal_move():
    player = MCTSPlayer(RolloutEvaluator(random.Random(0), max_moves=60), max_simulations=8, opening_book=False, seed=0)
    player.newGame(Board._BLACK)
    move = player.getPlayerMove()
    assert Board.name_to_flat(move) in Board().legal_moves()


def test_root_visits_are_exposed_after_search(net):
    player = mcts(net, max_simulations=200)
    player.newGame(Board._BLACK)
    feed(player, CAPTURE_SETUP)
    move = player.getPlayerMove()
    visits = player.last_root_visits
    assert sum(visits.values()) == 200
    assert max(visits, key=visits.get) == Board.name_to_flat(move)


def test_root_visits_are_empty_for_book_moves(net):
    player = mcts(net, max_simulations=50, opening_book=True)
    player.newGame(Board._BLACK)
    player.getPlayerMove()
    assert player.last_root_visits == {}
