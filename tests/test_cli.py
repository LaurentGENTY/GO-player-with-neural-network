import pytest

from go_player.cli import main
from go_player.players import make_player


def test_make_player_rejects_unknown_kind():
    with pytest.raises(ValueError, match="foo"):
        make_player("foo", seed=0, time_budget=1.0)


def test_play_random_game(capsys):
    assert main(["play", "--black", "random", "--white", "random", "--max-moves", "10"]) == 0
    assert "Result:" in capsys.readouterr().out


def test_arena_writes_report(tmp_path):
    assert main(["arena", "--match", "random:random", "--games", "2", "--max-moves", "20", "--out", str(tmp_path)]) == 0
    assert "Random vs Random" in (tmp_path / "arena.md").read_text()


def test_record_writes_media(tmp_path):
    assert main(["record", "--black", "random", "--white", "random", "--max-moves", "4", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "random-vs-random.gif").exists()
    assert (tmp_path / "random-vs-random.mp4").exists()


def test_unknown_player_kind_exits_2():
    with pytest.raises(SystemExit) as exc:
        main(["play", "--black", "foo", "--white", "random"])
    assert exc.value.code == 2


def test_bad_match_spec_exits_2():
    with pytest.raises(SystemExit) as exc:
        main(["arena", "--match", "random-random"])
    assert exc.value.code == 2


def test_missing_gnugo_fails_before_any_game(monkeypatch, capsys):
    from go_player.nn import ValueNet
    from go_player.players import gnugo

    monkeypatch.setattr(gnugo.shutil, "which", lambda name: None)
    monkeypatch.setattr(ValueNet, "load", classmethod(lambda cls, *a: pytest.fail("network loaded before gnugo check")))
    assert main(["arena", "--match", "mcts:random", "--match", "random:gnugo", "--games", "1"]) == 1
    assert "brew install gnu-go" in capsys.readouterr().err


def test_puzzles_command_writes_reel(tmp_path):
    assert main(["puzzles", "--only", "capture", "--simulations", "300", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "reel.gif").exists() and (tmp_path / "results.md").exists()


def test_puzzles_command_rejects_unknown_puzzle():
    with pytest.raises(SystemExit) as exc:
        main(["puzzles", "--only", "nope"])
    assert exc.value.code == 2
