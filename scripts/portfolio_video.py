"""Builds the 1024x640 portfolio video (.webm) and poster (.webp) from the generated media.

Usage: uv run python scripts/portfolio_video.py <out_dir>
Inputs: media/alphabeta-vs-mcts.gif and media/puzzles/*.gif (run `go-player record` and `go-player puzzles` first).
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageSequence

W, H = 1024, 640
BG, FG, MUTED, ACCENT = (18, 18, 20), (240, 240, 240), (150, 150, 155), (231, 76, 60)
BOARD_SCALE = 1.15
FPS = 30
TARGET_BITRATE = "650k"  # about 3 MB for ~35 s, like the other portfolio videos

SEGMENTS = {
    "game": ("Full game", "MCTS 2026 (White) beats the 2020 Alpha-Beta, same neural network"),
    "puzzles": ("Puzzles", "Red discs: where the tree search spends its simulations"),
    "score": ("Honest scorecard", "3/7 solved: the 2020 network never learned life and death"),
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
    draw.text((x, 124), "Alpha-Beta 2020  vs  MCTS 2026", fill=MUTED, font=font(18))
    label, text = SEGMENTS[segment]
    draw.text((x, 230), label.upper(), fill=ACCENT, font=font(16))
    for i, line in enumerate(wrap(draw, text, W - x - 32, 22)):
        draw.text((x, 262 + i * 32), line, fill=FG, font=font(22))
    draw.text((x, H - 70), "Python · Keras CNN · Monte-Carlo tree search", fill=MUTED, font=font(14))
    draw.text((x, H - 46), "ENSEIRB-MATMECA 2020 · revived 2026", fill=MUTED, font=font(14))
    return canvas


def build_sequence(media: Path) -> list[tuple[Image.Image, int]]:
    seq = []
    game = frames_of(media / "alphabeta-vs-mcts.gif")
    for i, (frame, _) in enumerate(game):
        seq.append((compose(frame, "game"), 2500 if i == len(game) - 1 else 110))
    for name in ("capture", "bigger-capture", "connect"):
        _, position, heat, move = [f for f, _ in frames_of(media / "puzzles" / f"{name}.gif")][:4]
        seq += [(compose(position, "puzzles"), 1000), (compose(heat, "puzzles"), 2200), (compose(move, "puzzles"), 1400)]
    scorecard = frames_of(media / "puzzles" / "reel.gif")[-1][0]
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
    seq = build_sequence(Path("media"))
    encode(seq, out_dir / "go-ai.webm")
    poster = next(img for img, ms in seq if ms == 2200)  # first puzzle heatmap
    poster.save(out_dir / "go-ai.webp", quality=85)
    print(out_dir / "go-ai.webm", out_dir / "go-ai.webp", f"{sum(ms for _, ms in seq) / 1000:.1f}s")


if __name__ == "__main__":
    main()
