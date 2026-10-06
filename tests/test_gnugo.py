import shutil

import pytest

from go_player.arena import play_game
from go_player.goban import Board
from go_player.players import gnugo
from go_player.players.gnugo import GnuGoNotFound, GnuGoPlayer, normalize_gtp_move
from go_player.players.random_player import RandomPlayer

needs_gnugo = pytest.mark.skipif(shutil.which("gnugo") is None, reason="gnugo not installed")


def test_normalize_gtp_move():
    assert normalize_gtp_move(" e5\n") == "E5"
    assert normalize_gtp_move("pass") == "PASS"
    assert normalize_gtp_move("resign") == "PASS"


def test_missing_binary_has_install_hint(monkeypatch):
    monkeypatch.setattr(gnugo.shutil, "which", lambda name: None)
    with pytest.raises(GnuGoNotFound, match="brew install gnu-go"):
        GnuGoPlayer()


@pytest.mark.gnugo
@needs_gnugo
def test_gnugo_plays_a_game_against_random():
    player = GnuGoPlayer(level=1)
    try:
        record = play_game(player, RandomPlayer(seed=3), max_moves=30)
    finally:
        player.close()
    assert record.black == "GnuGo-L1"
    assert record.illegal_by != Board._WHITE
    assert len(record.moves) > 0
