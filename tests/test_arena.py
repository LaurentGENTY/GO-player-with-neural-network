import json

from go_player.arena import play_game, run_match, write_report
from go_player.goban import Board
from go_player.players.base import PlayerInterface
from go_player.players.random_player import RandomPlayer


class FixedMovesPlayer(PlayerInterface):
    def __init__(self, moves):
        self._moves = list(moves)

    def getPlayerName(self):
        return "Fixed"

    def getPlayerMove(self):
        return self._moves.pop(0)


def test_random_game_is_complete_and_consistent():
    record = play_game(RandomPlayer(seed=1, name="R1"), RandomPlayer(seed=2, name="R2"))
    assert record.black == "R1" and record.white == "R2"
    assert record.illegal_by is None
    assert record.winner in (Board._BLACK, Board._WHITE, None)
    replay = Board()
    for name in record.moves:
        assert replay.push(Board.name_to_flat(name))
    assert len(record.move_times[Board._BLACK]) >= len(record.moves) // 2


def test_illegal_move_loses_and_is_recorded():
    record = play_game(FixedMovesPlayer(["E5", "D4"]), FixedMovesPlayer(["E5"]))
    assert record.illegal_by == Board._WHITE
    assert record.winner == Board._BLACK
    assert record.moves == ["E5"]
    assert record.score == "illegal move"


def test_malformed_move_loses():
    record = play_game(FixedMovesPlayer(["Z9"]), RandomPlayer(seed=1))
    assert record.illegal_by == Board._BLACK
    assert record.winner == Board._WHITE


def test_move_cap_is_a_draw():
    record = play_game(RandomPlayer(seed=1), RandomPlayer(seed=2), max_moves=10)
    assert len(record.moves) == 10
    assert record.winner is None
    assert record.score == "move cap"


def test_run_match_alternates_colors():
    result = run_match(
        lambda seed: RandomPlayer(seed=seed, name="A"),
        lambda seed: RandomPlayer(seed=seed, name="B"),
        games=2, max_moves=20,
    )
    assert [g.black for g in result.games] == ["A", "B"]
    assert result.a_colors == [Board._BLACK, Board._WHITE]
    summary = result.summary()
    assert summary["games"] == 2
    assert summary["a_wins"] + summary["b_wins"] + summary["draws"] == 2


def test_write_report(tmp_path):
    result = run_match(
        lambda seed: RandomPlayer(seed=seed, name="A"),
        lambda seed: RandomPlayer(seed=seed, name="B"),
        games=2, max_moves=20,
    )
    md, js = write_report([result], tmp_path)
    assert "| A vs B | 2 |" in md.read_text()
    data = json.loads(js.read_text())
    assert data["matches"][0]["summary"]["games"] == 2
