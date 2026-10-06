"""Builds the 1024x640 portfolio video (.webm) and poster (.webp).

Usage: uv run python scripts/portfolio_video.py <out_dir>
Inputs: media/puzzles/*.gif (run `go-player puzzles` first). The full game is recorded here with neutral
player names (the repository media keep the 2020/2026 story; the portfolio card does not).
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageSequence

from go_player.goban import Board
from go_player.nn import ValueNet
from go_player.players import make_player
from go_player.puzzles import PUZZLES, solve
from go_player.record import record_game
from go_player.showcase import PuzzleResult, render_scorecard

W, H = 1024, 640
BG, FG, MUTED, ACCENT = (18, 18, 20), (240, 240, 240), (150, 150, 155), (231, 76, 60)
BOARD_SCALE = 1.15
FPS = 30
TARGET_BITRATE = "650k"  # about 3 MB for ~35 s, like the other portfolio videos
GAME_TIME_PER_MOVE = 3.0
GAME_ATTEMPTS = 4  # the search is time-based, so retry seeds until MCTS (White) wins

SEGMENTS = {
    "game": ("Full game", "MCTS (White) beats Alpha-Beta, both guided by the same neural network"),
    "puzzles": ("Puzzles", "Red discs: where the tree search spends its simulations"),
    "score": ("Honest scorecard", "3/7 solved: life and death is still out of reach"),
}


def font(size: int):
    return ImageFont.load_default(size=size)


def frames_of(gif: Path) -> list[tuple[Image.Image, int]]:
    im = Image.open(gif)
    return [(f.convert("RGB"), f.info.get("duration", 600)) for f in ImageSequence.Iterator(im)]


def wrap(draw: ImageDraw.ImageDraw, text: str, width: int, size: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=font(size)) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + [line]


def compose(board_frame: Image.Image, segment: str) -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    board = board_frame.resize((int(448 * BOARD_SCALE), int(528 * BOARD_SCALE)), Image.LANCZOS)
    canvas.paste(board, (24, (H - board.height) // 2))
    draw = ImageDraw.Draw(canvas)
    x = 24 + board.width + 40
    draw.text((x, 70), "9x9 Go AI", fill=FG, font=font(40))
    draw.text((x, 124), "Neural network + tree search", fill=MUTED, font=font(18))
    label, text = SEGMENTS[segment]
    draw.text((x, 230), label.upper(), fill=ACCENT, font=font(16))
    for i, line in enumerate(wrap(draw, text, W - x - 32, 22)):
        draw.text((x, 262 + i * 32), line, fill=FG, font=font(22))
    draw.text((x, H - 58), "Python · Keras CNN · Monte-Carlo tree search", fill=MUTED, font=font(14))
    return canvas


def record_showcase_game(net: ValueNet, work_dir: Path) -> Path:
    for seed in range(GAME_ATTEMPTS):
        black = make_player("alphabeta", seed=seed, time_budget=GAME_TIME_PER_MOVE, value_net=net, name="Alpha-Beta")
        white = make_player("mcts", seed=seed + 1, time_budget=GAME_TIME_PER_MOVE, value_net=net, name="MCTS")
        gif, _, record = record_game(black, white, work_dir / "game", value_net=net)
        print(f"seed {seed}: {record.score}", flush=True)
        if record.winner == Board._WHITE:
            return gif
    raise SystemExit(f"MCTS did not win as White in {GAME_ATTEMPTS} games")


def build_sequence(media: Path, net: ValueNet, work_dir: Path) -> list[tuple[Image.Image, int]]:
    seq = []
    game = frames_of(record_showcase_game(net, work_dir))
    for i, (frame, _) in enumerate(game):
        seq.append((compose(frame, "game"), 2500 if i == len(game) - 1 else 110))
    for name in ("capture", "bigger-capture", "connect"):
        _, position, heat, move = [f for f, _ in frames_of(media / "puzzles" / f"{name}.gif")][:4]
        seq += [(compose(position, "puzzles"), 1000), (compose(heat, "puzzles"), 2200), (compose(move, "puzzles"), 1400)]
    results = [PuzzleResult(p, solve(p, net), "") for p in PUZZLES]
    scorecard = render_scorecard(results, title="MCTS on Go puzzles", footnote="Known limit: life and death")
    seq.append((compose(scorecard, "score"), 3500))
    return seq


def encode(seq: list[tuple[Image.Image, int]], webm: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp:
        lines = []
        for i, (img, ms) in enumerate(seq):
            path = Path(tmp) / f"{i:05d}.png"
            img.save(path)
            lines += [f"file '{path}'", f"duration {ms / 1000:.3f}"]
        lines.append(f"file '{Path(tmp) / f'{len(seq) - 1:05d}.png'}'")  # concat demuxer ignores the last duration otherwise
        listing = Path(tmp) / "list.txt"
        listing.write_text("\n".join(lines))
        for pass_no in (1, 2):
            out = str(webm) if pass_no == 2 else "/dev/null"
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                 "-r", str(FPS), "-c:v", "libvpx-vp9", "-b:v", TARGET_BITRATE, "-pix_fmt", "yuv420p",
                 "-pass", str(pass_no), "-passlogfile", str(Path(tmp) / "vp9"), "-an",
                 *(["-f", "webm"] if pass_no == 1 else []), out],
                check=True,
            )


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as work_dir:
        seq = build_sequence(Path("media"), ValueNet.load(), Path(work_dir))
    encode(seq, out_dir / "go-ai.webm")
    poster = next(img for img, ms in seq if ms == 2200)  # first puzzle heatmap
    poster.save(out_dir / "go-ai.webp", quality=85)
    print(out_dir / "go-ai.webm", out_dir / "go-ai.webp", f"{sum(ms for _, ms in seq) / 1000:.1f}s")


if __name__ == "__main__":
    main()
