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
