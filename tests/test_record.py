from PIL import Image

from go_player.goban import Board
from go_player.players.random_player import RandomPlayer
from go_player.record import _center, record_game, render_frame
from positions import CAPTURE_SETUP, board_with


def test_render_frame_size_and_stones():
    board = board_with(CAPTURE_SETUP)
    image = render_frame(board, "A vs B", "Move 10", Board.name_to_flat("J7"), 0.75)
    assert image.size == (448, 528)
    # E5 (col 4, lin 4) holds a white stone: its center pixel is white.
    assert image.getpixel(_center(4, 4)) == (255, 255, 255)
    # A1 (col 0, lin 0) is empty: wood color.
    assert image.getpixel((_center(0, 0)[0] + 10, _center(0, 0)[1] - 10)) == (220, 179, 92)


def test_record_game_at_move_cap(tmp_path):
    gif, mp4, record = record_game(
        RandomPlayer(seed=1, name="R1"), RandomPlayer(seed=2, name="R2"), tmp_path / "r1-vs-r2", max_moves=6,
    )
    assert record.score == "move cap"
    assert Image.open(gif).n_frames == 7  # empty board + 6 moves
    assert mp4.stat().st_size > 0
