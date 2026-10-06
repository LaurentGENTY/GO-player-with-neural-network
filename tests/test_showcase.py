from PIL import Image

from go_player.puzzles import PUZZLES
from go_player.showcase import render_puzzles


def test_render_puzzles_writes_clips_reel_and_results(net, tmp_path):
    solved = next(p for p in PUZZLES if not p.known_limit)
    limit = next(p for p in PUZZLES if p.known_limit)
    results = render_puzzles([solved, limit], net, tmp_path, simulations=3000)
    assert [r.puzzle.name for r in results] == [solved.name, limit.name]
    for name in (solved.name, limit.name, "reel"):
        assert Image.open(tmp_path / f"{name}.gif").n_frames >= 3
        assert (tmp_path / f"{name}.mp4").stat().st_size > 0
    table = (tmp_path / "results.md").read_text()
    assert f"| {solved.title} |" in table and "1/2" in table
