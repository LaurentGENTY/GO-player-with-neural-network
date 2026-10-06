"""Renders a game to GIF and MP4 with Pillow (no system cairo dependency)."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from go_player.arena import GameRecord, play_game
from go_player.goban import Board

# Both sides are multiples of 16 so libx264 does not resize the frames.
WIDTH, HEADER, BOARD_H, BAR = 448, 64, 432, 32
HEIGHT = HEADER + BOARD_H + BAR
STEP = 42
GX, GY = (WIDTH - 8 * STEP) // 2, HEADER + 44  # position of A9, grid centered horizontally
MARGIN = 36  # wood around the grid, room for coordinates
STONE_R = 19
WOOD, LINE, RED = (220, 179, 92), (60, 40, 20), (192, 57, 43)
VIDEO_FPS = 5
STAR_POINTS = [(2, 2), (6, 2), (4, 4), (2, 6), (6, 6)]


def _font(size: int):
    return ImageFont.load_default(size=size)


def _center(col: int, lin: int) -> tuple[int, int]:
    return GX + STEP * col, GY + STEP * (Board._BOARDSIZE - 1 - lin)


def render_frame(board: Board, title: str, caption: str, last_move: int | None, p_black: float | None) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 10), title, fill="black", font=_font(20))
    draw.text((16, 38), caption, fill=(80, 80, 80), font=_font(16))

    size = Board._BOARDSIZE - 1
    draw.rectangle((GX - MARGIN, GY - MARGIN, GX + STEP * size + MARGIN, GY + STEP * size + MARGIN), fill=WOOD)
    for i in range(Board._BOARDSIZE):
        draw.line((GX + STEP * i, GY, GX + STEP * i, GY + STEP * size), fill=LINE, width=2)
        draw.line((GX, GY + STEP * i, GX + STEP * size, GY + STEP * i), fill=LINE, width=2)
        draw.text((GX + STEP * i - 4, GY - 34), "ABCDEFGHJ"[i], fill=LINE, font=_font(12))
        draw.text((GX - 32, GY + STEP * i - 7), str(Board._BOARDSIZE - i), fill=LINE, font=_font(12))
    for col, lin in STAR_POINTS:
        x, y = _center(col, lin)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=LINE)

    for flat in range(Board._BOARDSIZE ** 2):
        color = board[flat]
        if color == Board._EMPTY:
            continue
        x, y = _center(*Board.unflatten(flat))
        fill = "black" if color == Board._BLACK else "white"
        draw.ellipse((x - STONE_R, y - STONE_R, x + STONE_R, y + STONE_R), fill=fill, outline=(51, 51, 51), width=2)
    if last_move is not None and last_move != -1:
        x, y = _center(*Board.unflatten(last_move))
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=RED, width=3)

    if p_black is not None:
        top = HEADER + BOARD_H
        split = int(WIDTH * p_black)
        draw.rectangle((0, top, split, HEIGHT), fill="black")
        draw.rectangle((split, top, WIDTH, HEIGHT), fill=(235, 235, 235))
        draw.text((WIDTH // 2 - 70, top + 8), f"P(black wins) = {p_black:.0%}", fill="white", font=_font(14),
                  stroke_width=2, stroke_fill="black")
    return image


def _result_text(record: GameRecord) -> str:
    if record.winner is None:
        return f"Draw ({record.score})"
    winner = record.black if record.winner == Board._BLACK else record.white
    return f"{winner} wins ({record.score})"


def record_game(
    black,
    white,
    out_stem: Path,
    value_net=None,
    komi: float = 0.0,
    max_moves: int = 200,
    frame_ms: int = 600,
    final_ms: int = 3000,
) -> tuple[Path, Path, GameRecord]:
    record = play_game(black, white, komi=komi, max_moves=max_moves)
    title = f"{record.black} (B)  vs  {record.white} (W)"

    def p_black(board: Board) -> float | None:
        return None if value_net is None else float(value_net.evaluate(board)[0])

    board = Board(komi)
    frames = [render_frame(board, title, "Move 0", None, p_black(board))]
    for number, name in enumerate(record.moves, start=1):
        move = Board.name_to_flat(name)
        board.push(move)
        caption = f"Move {number}: {name}"
        if number == len(record.moves):
            caption += f"  -  {_result_text(record)}"
        frames.append(render_frame(board, title, caption, move, p_black(board)))

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    gif = out_stem.with_suffix(".gif")
    frames[0].save(
        gif, save_all=True, append_images=frames[1:], loop=0, optimize=True,
        duration=[frame_ms] * (len(frames) - 1) + [final_ms],
    )
    mp4 = out_stem.with_suffix(".mp4")
    repeat = max(1, round(frame_ms * VIDEO_FPS / 1000))
    final_repeat = max(1, round(final_ms * VIDEO_FPS / 1000))
    with imageio.get_writer(mp4, fps=VIDEO_FPS, codec="libx264", macro_block_size=16) as writer:
        for i, frame in enumerate(frames):
            array = np.asarray(frame)
            for _ in range(final_repeat if i == len(frames) - 1 else repeat):
                writer.append_data(array)
    return gif, mp4, record
