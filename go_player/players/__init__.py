import random

from go_player.players.base import PlayerInterface

PLAYER_KINDS = ("random", "gnugo", "alphabeta", "mcts", "mcts-rollout")
NET_KINDS = {"alphabeta", "mcts"}
DEFAULT_MATCHES = [("mcts", "alphabeta"), ("mcts", "mcts-rollout"), ("mcts", "gnugo"), ("alphabeta", "gnugo")]


def make_player(
    kind: str, *, seed: int | None, time_budget: float, komi: float = 0.0, value_net=None, name: str | None = None,
) -> PlayerInterface:
    player = _make_player(kind, seed=seed, time_budget=time_budget, komi=komi, value_net=value_net)
    if name is not None:
        player._name = name  # every player stores its display name in _name
    return player


def _make_player(kind: str, *, seed: int | None, time_budget: float, komi: float, value_net) -> PlayerInterface:
    # Imports are local so that `random`/`gnugo` games never import TensorFlow.
    if kind == "random":
        from go_player.players.random_player import RandomPlayer
        return RandomPlayer(seed=seed, komi=komi)
    if kind == "gnugo":
        from go_player.players.gnugo import GnuGoPlayer
        return GnuGoPlayer(level=1, komi=komi)
    if kind == "alphabeta":
        from go_player.players.alphabeta import AlphaBetaPlayer
        return AlphaBetaPlayer(value_net, time_budget=time_budget, seed=seed, komi=komi)
    if kind == "mcts":
        from go_player.players.mcts import MCTSPlayer, NNEvaluator
        return MCTSPlayer(NNEvaluator(value_net), time_budget=time_budget, seed=seed, komi=komi, name="MCTS-NN")
    if kind == "mcts-rollout":
        from go_player.players.mcts import MCTSPlayer, RolloutEvaluator
        return MCTSPlayer(RolloutEvaluator(random.Random(seed)), time_budget=time_budget, seed=seed, komi=komi,
                          name="MCTS-rollout")
    raise ValueError(f"Unknown player kind {kind!r}; choose one of {', '.join(PLAYER_KINDS)}")
