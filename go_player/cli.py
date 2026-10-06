import argparse
import sys
from pathlib import Path

from go_player.goban import Board
from go_player.players import DEFAULT_MATCHES, NET_KINDS, PLAYER_KINDS, make_player


def _match(text: str) -> tuple[str, str]:
    a, sep, b = text.partition(":")
    if not sep or a not in PLAYER_KINDS or b not in PLAYER_KINDS:
        raise argparse.ArgumentTypeError(f"expected A:B with A and B in {', '.join(PLAYER_KINDS)}, got {text!r}")
    return a, b


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="go-player", description="9x9 Go: 2020 Alpha-Beta vs 2026 MCTS")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--time", type=float, default=5.0, help="seconds per move (default: 5)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--komi", type=float, default=0.0)
        p.add_argument("--max-moves", type=int, default=200)

    play = sub.add_parser("play", help="play one game in the terminal")
    play.add_argument("--black", choices=PLAYER_KINDS, required=True)
    play.add_argument("--white", choices=PLAYER_KINDS, required=True)
    common(play)

    arena = sub.add_parser("arena", help="play matches and write media/arena.md + arena.json")
    arena.add_argument("--match", type=_match, action="append", help="A:B, repeatable (default: the showcase set)")
    arena.add_argument("--games", type=int, default=20)
    arena.add_argument("--out", type=Path, default=Path("media"))
    common(arena)

    record = sub.add_parser("record", help="play one game and save it as GIF + MP4")
    record.add_argument("--black", choices=PLAYER_KINDS, required=True)
    record.add_argument("--white", choices=PLAYER_KINDS, required=True)
    record.add_argument("--out", type=Path, default=Path("media"))
    common(record)

    from go_player.puzzles import PUZZLE_SIMULATIONS, PUZZLES
    puzzles = sub.add_parser("puzzles", help="solve the tactical puzzles and render clips, reel and results.md")
    puzzles.add_argument("--only", choices=[p.name for p in PUZZLES], action="append", help="repeatable")
    puzzles.add_argument("--simulations", type=int, default=PUZZLE_SIMULATIONS)
    puzzles.add_argument("--out", type=Path, default=Path("media/puzzles"))
    return parser


def _load_net(kinds, always: bool = False):
    if not always and not NET_KINDS.intersection(kinds):
        return None
    from go_player.nn import ValueNet
    return ValueNet.load()


def _run(args) -> int:
    if args.command == "puzzles":
        from go_player.nn import ValueNet
        from go_player.puzzles import PUZZLES
        from go_player.showcase import render_puzzles
        selected = [p for p in PUZZLES if not args.only or p.name in args.only]
        results = render_puzzles(selected, ValueNet.load(), args.out, simulations=args.simulations)
        for r in results:
            print(f"{r.puzzle.name}: MCTS {r.solution.move} {'solved' if r.solution.correct else 'missed'}")
        print(f"Reel: {args.out / 'reel.gif'}")
        return 0

    kinds = [k for pair in args.match or DEFAULT_MATCHES for k in pair] if args.command == "arena" \
        else [args.black, args.white]
    if "gnugo" in kinds:
        from go_player.players.gnugo import find_gnugo
        find_gnugo()  # fail before loading the network or playing hours of earlier matches

    def factory(kind, net):
        return lambda seed: make_player(kind, seed=seed, time_budget=args.time, komi=args.komi, value_net=net)

    if args.command == "play":
        from go_player.arena import play_game
        net = _load_net([args.black, args.white])
        record = play_game(factory(args.black, net)(args.seed), factory(args.white, net)(args.seed + 1),
                           komi=args.komi, max_moves=args.max_moves)
        board = Board(args.komi)
        for move in record.moves:
            board.push(Board.name_to_flat(move))
        board.prettyPrint()
        print("Moves:", " ".join(record.moves))
        print("Result:", record.score, "| winner:", Board.player_name(record.winner) if record.winner else "none")
        return 0

    if args.command == "arena":
        from go_player.arena import run_match, write_report
        matches = args.match or DEFAULT_MATCHES
        net = _load_net([k for pair in matches for k in pair])
        results = []
        for a, b in matches:
            print(f"== {a} vs {b} ({args.games} games, {args.time}s/move)")
            results.append(run_match(
                factory(a, net), factory(b, net), games=args.games, seed=args.seed, komi=args.komi,
                max_moves=args.max_moves,
                on_game=lambda i, r: print(f"  game {i + 1}: {r.black} (B) vs {r.white} (W) -> {r.score}", flush=True),
            ))
            write_report(results, args.out)  # rewrite after each match so a long run keeps partial results
        print(f"Report: {args.out / 'arena.md'}")
        return 0

    from go_player.record import record_game
    net = _load_net([args.black, args.white], always=True)
    gif, mp4, record = record_game(
        factory(args.black, net)(args.seed), factory(args.white, net)(args.seed + 1),
        args.out / f"{args.black}-vs-{args.white}", value_net=net, komi=args.komi, max_moves=args.max_moves,
    )
    print(f"{record.black} vs {record.white}: {record.score}\n{gif}\n{mp4}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from go_player.nn import ModelLoadError
    from go_player.players.gnugo import GnuGoNotFound
    try:
        return _run(args)
    except (ModelLoadError, GnuGoNotFound) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
