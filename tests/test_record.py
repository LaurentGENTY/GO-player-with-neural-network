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


def test_heatmap_marks_the_most_visited_point():
    from go_player.record import render_frame

    board = board_with(CAPTURE_SETUP)
    e6 = Board.name_to_flat("E6")
    image = render_frame(board, "A vs B", "thinking", None, None, heat={e6: 0.6, Board.name_to_flat("A1"): 0.05})
    x, y = _center(4, 5)
    r, g, b = image.getpixel((x, y - 12))  # above the percentage label
    assert r > 150 and g < 120  # strongly red where most visits went
    assert image.getpixel((x + 60, y + 60)) == (220, 179, 92)  # empty wood stays untouched


def test_title_card_size():
    from go_player.record import render_title_card

    assert render_title_card("Capture the two stones", "Black to play").size == (448, 528)


def test_write_media(tmp_path):
    from go_player.record import write_media

    frames = [render_frame(Board(), "t", f"frame {i}", None, None) for i in range(3)]  # identical GIF frames get merged
    gif, mp4 = write_media(frames, [600, 600, 3000], tmp_path / "clip")
    assert Image.open(gif).n_frames == 3
    assert mp4.stat().st_size > 0


def test_heatmap_hides_rarely_visited_points():
    board = board_with(CAPTURE_SETUP)
    a1 = Board.name_to_flat("A1")
    image = render_frame(board, "A vs B", "thinking", None, None, heat={Board.name_to_flat("E6"): 0.6, a1: 0.01})
    x, y = _center(0, 0)
    assert image.getpixel((x + 5, y - 5)) == (220, 179, 92)  # 1 % of visits: not drawn
