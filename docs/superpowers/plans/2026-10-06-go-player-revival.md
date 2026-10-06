# Go Player Revival Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revive the 2020 9x9 Go player as a modern Python package, add an MCTS player that reuses the 2020 CNN as its value function, and generate reproducible showcase media (GIF/MP4 + arena table) for the README and portfolio.

**Architecture:** A `go_player/` package (managed by `uv`) wraps the 2020 `Goban` rules engine (plus komi), a Keras 3 rebuild of the 2020 CNN (`ValueNet`), players behind the unchanged 2020 `PlayerInterface` (random, GnuGo, cleaned Alpha-Beta, MCTS), an arena/referee, a Pillow-based recorder and a CLI. `GO/` and `ML/` stay untouched as the 2020 archive.

**Tech Stack:** Python 3.12, uv, numpy, TensorFlow/Keras 3, h5py, Pillow, imageio[ffmpeg], pytest, GnuGo (`brew install gnu-go`).

**Spec:** `docs/superpowers/specs/2026-10-06-go-player-revival-design.md`

## Global Constraints

- Python pinned to 3.12 via `.python-version`; `requires-python = ">=3.11,<3.14"`. Run everything through `uv run ...`.
- `GO/` and `ML/` are never modified (2020 archive).
- Code, comments, docstrings, commit messages and README are in English. Comment only the *why*.
- Every commit message ends with the trailer line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (pass it as a second `-m`).
- Work on branch `claude/silly-williams-f00013`. Never `git push` without Laurent's explicit confirmation.
- Default komi is `0.0` (2020 behavior); default time budget is `5.0` s per move; searches stop at `0.9 × budget`.
- Network encoding: `(9, 9, 2)` float32, plane 0 = Black, plane 1 = White, indexed `[col][lin]` (as `Board.unflatten`). Network output: `[P(Black wins), P(White wins)]`.
- No silent fallback to the 2020 "stone count" heuristic when the model cannot load: raise `ModelLoadError`.
- Media files: `media/<black-kind>-vs-<white-kind>.gif` and `.mp4`, frames 448×528 px.
- `uv run pytest -m "not gnugo"` passes offline in under 60 s.

## Review Focus

1. **The opponent passes while we are losing**: the AI must keep playing and must not pass. Tests: `test_mcts_does_not_pass_when_losing` (Task 5) and `test_alphabeta_does_not_pass_when_losing` (Task 6).
2. **An opponent's engine plays a move our `Goban` rejects** (GnuGo's superko differs, or a malformed string): the referee records `illegal_by` and the game ends without crashing. Tests: `test_illegal_move_loses_and_is_recorded` and `test_malformed_move_loses` (Task 4).
3. **GTP move strings that differ from our format** (`pass`, `resign`, lowercase): they are normalized to `PASS` or `E5`. Test: `test_normalize_gtp_move` (Task 7).
4. **Bad CLI input** (unknown player kind, malformed `--match`): argparse error with exit code 2, no traceback. Tests: `test_unknown_player_kind_exits_2` and `test_bad_match_spec_exits_2` (Task 9).
5. **A recorded game that ends on the move cap or on an illegal move**: it still yields a valid GIF and MP4 containing the moves actually played. Test: `test_record_game_at_move_cap` (Task 8).

---

### Task 1: Package scaffold + Goban with komi

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`
- Create: `go_player/__init__.py`, `go_player/goban.py` (copy of `GO/Goban.py` + komi)
- Create: `go_player/assets/model.h5`, `go_player/assets/games.json` (copies)
- Create: `tests/positions.py`, `tests/test_goban.py`

**Interfaces:**
- Produces: `go_player.goban.Board(komi: float = 0.0)`, with the full 2020 API (`push`, `pop`, `weak_legal_moves`, `legal_moves`, `is_game_over`, `result`, `compute_score -> (black, white_with_komi)`, `final_go_score`, `next_player`, `flatten`, `unflatten`, `name_to_flat`, `flat_to_name`, `flip`, `player_name`, `prettyPrint`, `_board`, `_lastPlayerHasPassed`).
- Produces: `tests/positions.py` with `CAPTURE_SETUP`, `KO_SETUP`, `BLACK_WINNING_AFTER_PASS`, `BLACK_LOSING_AFTER_PASS`, `board_with(moves, komi=0.0) -> Board` and `feed(player, moves) -> None`.

- [ ] **Step 1: Create the project files**

`pyproject.toml`:
```toml
[project]
name = "go-player"
version = "2.0.0"
description = "9x9 Go player: the 2020 Alpha-Beta + CNN value network, revived with MCTS"
readme = "README.md"
requires-python = ">=3.11,<3.14"
authors = [{ name = "Laurent Genty" }, { name = "Johan Chataigner" }]
dependencies = [
    "numpy>=1.26",
    "tensorflow>=2.16",
    "h5py>=3.10",
    "pillow>=10.1",
    "imageio[ffmpeg]>=2.34",
]

[project.scripts]
go-player = "go_player.cli:main"

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["go_player"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]
markers = ["gnugo: needs the gnugo binary in PATH (brew install gnu-go)"]
```

`.python-version`:
```
3.12
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

`go_player/__init__.py`:
```python
"""9x9 Go player: 2020 Alpha-Beta + CNN value network, revived with MCTS."""
```

Copy the 2020 files:
```bash
mkdir -p go_player/assets tests
cp GO/Goban.py go_player/goban.py
cp GO/model.h5 go_player/assets/model.h5
cp GO/games.json go_player/assets/games.json
uv sync
```
Expected: `uv sync` creates `.venv` and installs the dependencies (TensorFlow takes about 1 minute).

- [ ] **Step 2: Write the test helpers and the failing rules tests**

`tests/positions.py`:
```python
from go_player.goban import Board

# Black to move; E6 captures the two white stones E5-E4 (single liberty).
CAPTURE_SETUP = ["D5", "E5", "D4", "E4", "F5", "J9", "F4", "J8", "E3", "J7"]
# Black E5 just captured D5; White retaking at D5 would repeat a position (superko).
KO_SETUP = ["D6", "E6", "C5", "D5", "D4", "E4", "J9", "F5", "E5"]
# White just passed; Black (to move) has 2 stones vs 1: Black is winning.
BLACK_WINNING_AFTER_PASS = ["A1", "E5", "A2", "PASS"]
# White just passed; Black (to move) has 2 stones vs 3: Black is losing.
BLACK_LOSING_AFTER_PASS = ["A1", "E5", "PASS", "D5", "PASS", "F5", "A2", "PASS"]


def board_with(moves, komi=0.0):
    board = Board(komi)
    for name in moves:
        assert board.push(Board.name_to_flat(name)), name
    return board


def feed(player, moves):
    """Replays moves on a player's private board through the opponent-move hook."""
    for name in moves:
        player.playOpponentMove(name)
```

`tests/test_goban.py`:
```python
from go_player.goban import Board
from positions import CAPTURE_SETUP, KO_SETUP, board_with


def at(board, name):
    return board[Board.name_to_flat(name)]


def test_capture_removes_stones():
    board = board_with(CAPTURE_SETUP + ["E6"])
    assert at(board, "E5") == Board._EMPTY
    assert at(board, "E4") == Board._EMPTY
    assert at(board, "E6") == Board._BLACK


def test_suicide_is_not_a_legal_move():
    board = board_with(["J9", "A2", "J8", "B1"])  # Black to move, A1 has no liberty
    assert Board.name_to_flat("A1") not in board.weak_legal_moves()


def test_superko_retake_is_rejected():
    board = board_with(KO_SETUP)
    retake = Board.name_to_flat("D5")
    assert retake not in board.legal_moves()
    assert board.push(retake) is False
    board.pop()


def test_empty_board_without_komi_is_a_draw():
    board = board_with(["PASS", "PASS"])
    assert board.is_game_over()
    assert board.result() == "1/2-1/2"


def test_komi_goes_to_white():
    board = board_with(["PASS", "PASS"], komi=7.5)
    assert board.compute_score() == (0, 7.5)
    assert board.result() == "1-0"
    assert board.final_go_score() == "W+7.5"


def test_komi_can_be_overcome():
    board = board_with(["E5", "PASS", "PASS"], komi=7.5)
    assert board.compute_score() == (81, 7.5)
    assert board.result() == "0-1"


def test_reset_keeps_komi():
    board = Board(komi=6.5)
    board.reset()
    assert board.compute_score() == (0, 6.5)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_goban.py -v`
Expected: the komi tests FAIL with `TypeError: Board.__init__() got an unexpected keyword argument 'komi'`. The capture, suicide and superko tests already PASS (2020 behavior).

- [ ] **Step 4: Add komi to `go_player/goban.py`**

Replace:
```python
    def __init__(self):
      ''' Main constructor. Instantiate all non static variables.'''
      self._nbWHITE = 0
```
with:
```python
    def __init__(self, komi=0.0):
      ''' Main constructor. Instantiate all non static variables.
      komi: points added to WHITE's area score (0.0 keeps the 2020 behavior).'''
      self._komi = komi
      self._nbWHITE = 0
```

Replace (in `reset`):
```python
        self.__init__()
```
with:
```python
        self.__init__(self._komi)
```

Replace (in `result`):
```python
        score = self._count_areas()
        score_black = self._nbBLACK + score[0]
        score_white = self._nbWHITE + score[1]
        if score_white > score_black:
            return "1-0"
```
with:
```python
        score_black, score_white = self.compute_score()
        if score_white > score_black:
            return "1-0"
```

Replace (in `compute_score`):
```python
        ''' Computes the score (chinese rules) and return the scores for (blacks, whites) in this order'''
        score = self._count_areas()
        return (self._nbBLACK + score[0], self._nbWHITE + score[1])
```
with:
```python
        ''' Computes the score (chinese rules, komi added to whites) and return the scores for (blacks, whites)'''
        score = self._count_areas()
        return (self._nbBLACK + score[0], self._nbWHITE + score[1] + self._komi)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_goban.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version .gitignore uv.lock go_player tests
git commit -m "feat: scaffold go_player package with komi-aware Goban" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: ValueNet (the 2020 CNN under Keras 3)

**Files:**
- Create: `go_player/nn.py`
- Create: `tests/conftest.py`, `tests/test_nn.py`

**Interfaces:**
- Consumes: `Board` (Task 1), `CAPTURE_SETUP`, `board_with` (Task 1).
- Produces:
  - `go_player.nn.encode(board: Board) -> np.ndarray` with shape `(9, 9, 2)`, dtype float32;
  - `go_player.nn.ModelLoadError(Exception)`;
  - `go_player.nn.ValueNet.load(weights_path: Path | None = None) -> ValueNet`;
  - `ValueNet.predict(boards: np.ndarray) -> np.ndarray`, mapping `(N, 9, 9, 2)` to `(N, 2)`;
  - `ValueNet.evaluate(board: Board) -> np.ndarray`, shape `(2,)`;
  - pytest fixture `net` (session-scoped `ValueNet`).

- [ ] **Step 1: Write the failing tests**

`tests/conftest.py`:
```python
import pytest

from go_player.nn import ValueNet


@pytest.fixture(scope="session")
def net():
    return ValueNet.load()
```

`tests/test_nn.py`:
```python
import numpy as np
import pytest

from go_player.goban import Board
from go_player.nn import ModelLoadError, ValueNet, encode
from positions import CAPTURE_SETUP, board_with


def test_encode_uses_col_lin_planes():
    x = encode(board_with(["E5", "D4"]))
    assert x.shape == (9, 9, 2)
    assert x.dtype == np.float32
    assert x[4, 4, 0] == 1  # E5 black: col 4, lin 4
    assert x[3, 3, 1] == 1  # D4 white: col 3, lin 3
    assert x.sum() == 2


def test_encode_drops_captured_stones():
    x = encode(board_with(CAPTURE_SETUP + ["E6"]))
    assert x[4, 4, 1] == 0  # E5 captured
    assert x[4, 3, 1] == 0  # E4 captured
    assert x[4, 5, 0] == 1  # E6 black


def test_empty_board_value_matches_2020_model(net):
    p = net.predict(np.zeros((1, 9, 9, 2), dtype=np.float32))[0]
    assert p == pytest.approx([0.600, 0.400], abs=0.01)


def test_predict_is_batched(net):
    p = net.predict(np.zeros((5, 9, 9, 2), dtype=np.float32))
    assert p.shape == (5, 2)
    assert p.sum(axis=1) == pytest.approx(np.ones(5), abs=1e-5)


def test_value_is_roughly_rotation_invariant(net):
    x = encode(board_with(CAPTURE_SETUP))
    values = net.predict(np.stack([np.rot90(x, k) for k in range(4)]))[:, 0]
    assert values.max() - values.min() <= 0.1


def test_capture_is_the_best_one_ply_move(net):
    board = board_with(CAPTURE_SETUP)
    scores = {}
    for move in board.weak_legal_moves():
        if move != -1 and board.push(move):
            scores[Board.flat_to_name(move)] = float(net.evaluate(board)[0])
        board.pop()
    assert max(scores, key=scores.get) == "E6"


def test_missing_weights_raise_model_load_error(tmp_path):
    with pytest.raises(ModelLoadError, match="nope.h5"):
        ValueNet.load(tmp_path / "nope.h5")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_nn.py -v`
Expected: collection ERROR with `ModuleNotFoundError: No module named 'go_player.nn'`.

- [ ] **Step 3: Implement `go_player/nn.py`**

```python
"""The 2020 CNN value network, rebuilt for Keras 3.

The 2020 `model.json` (Keras 2.3) cannot be deserialized by Keras 3, so the
architecture is rebuilt in code and the `model.h5` weights are loaded layer
by layer with h5py.
"""

import os
from importlib.resources import files
from pathlib import Path

import numpy as np

from go_player.goban import Board

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

DEFAULT_WEIGHTS = Path(str(files("go_player") / "assets" / "model.h5"))


class ModelLoadError(Exception):
    pass


def encode(board: Board) -> np.ndarray:
    # Flat index is 9*lin + col, so the transpose gives the 2020 [col][lin] layout.
    cells = board._board.reshape(Board._BOARDSIZE, Board._BOARDSIZE).T
    return np.stack([cells == Board._BLACK, cells == Board._WHITE], axis=-1).astype(np.float32)


def _build_model():
    import keras
    from keras import layers

    conv_blocks = []
    for _ in range(3):
        conv_blocks += [
            layers.Conv2D(64, 5, padding="same", activation="relu"),
            layers.BatchNormalization(),
        ]
    return keras.Sequential([
        keras.Input((9, 9, 2)),
        *conv_blocks,
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(32, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ])


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _load_legacy_weights(model, path: Path) -> None:
    import h5py

    with h5py.File(path, "r") as f:
        names = [_text(n) for n in f.attrs["layer_names"]]
        for layer, name in zip(model.layers, names, strict=True):
            group = f[name]
            layer.set_weights([group[_text(w)][()] for w in group.attrs["weight_names"]])


class ValueNet:
    def __init__(self, model):
        self._model = model

    @classmethod
    def load(cls, weights_path: Path | None = None) -> "ValueNet":
        path = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS
        if not path.exists():
            raise ModelLoadError(f"Value network weights not found: {path}")
        try:
            model = _build_model()
            _load_legacy_weights(model, path)
        except (ImportError, OSError, KeyError, ValueError) as e:
            raise ModelLoadError(f"Cannot load the 2020 value network from {path}: {e}") from e
        net = cls(model)
        # The first call traces the graph; doing it here keeps it out of move time budgets.
        net.predict(np.zeros((1, 9, 9, 2), dtype=np.float32))
        return net

    def predict(self, boards: np.ndarray) -> np.ndarray:
        return self._model(boards.astype(np.float32), training=False).numpy()

    def evaluate(self, board: Board) -> np.ndarray:
        return self.predict(encode(board)[None])[0]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_nn.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add go_player/nn.py tests/conftest.py tests/test_nn.py
git commit -m "feat: load the 2020 CNN value network under Keras 3" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Player contract, random player, opening book

**Files:**
- Create: `go_player/players/__init__.py` (empty for now; filled in Task 9)
- Create: `go_player/players/base.py`, `go_player/players/random_player.py`, `go_player/opening.py`
- Create: `tests/test_players_base.py`

**Interfaces:**
- Consumes: `Board` (Task 1).
- Produces:
  - `PlayerInterface`, with methods `getPlayerName() -> str`, `getPlayerMove() -> str`, `playOpponentMove(move: str)`, `newGame(color: int)`, `endGame(winner: int)`;
  - `candidate_moves(board) -> list[int]`: non-pass weak-legal moves, or `[-1]` if there are none;
  - `is_winning(board, color: int) -> bool`;
  - `RandomPlayer(seed: int | None = None, komi: float = 0.0, name: str = "Random")`;
  - `OpeningBook(games: list[list[str]], rng: random.Random, length: int = 5)` and `OpeningBook.load(rng, length=5)`;
  - `OpeningBook.next_move(board) -> int | None`, which returns a flat move or `None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_players_base.py`:
```python
import random

from go_player.goban import Board
from go_player.opening import OpeningBook
from go_player.players.base import candidate_moves, is_winning
from go_player.players.random_player import RandomPlayer
from positions import BLACK_LOSING_AFTER_PASS, BLACK_WINNING_AFTER_PASS, board_with


def test_candidate_moves_exclude_pass():
    moves = candidate_moves(Board())
    assert len(moves) == 81
    assert -1 not in moves


def test_candidate_moves_fall_back_to_pass():
    class OnlyPass:
        def weak_legal_moves(self):
            return [-1]

    assert candidate_moves(OnlyPass()) == [-1]


def test_is_winning():
    assert is_winning(board_with(BLACK_WINNING_AFTER_PASS), Board._BLACK)
    assert not is_winning(board_with(BLACK_LOSING_AFTER_PASS), Board._BLACK)
    assert is_winning(board_with(BLACK_LOSING_AFTER_PASS), Board._WHITE)


def test_random_player_is_legal_and_seeded():
    first, second = RandomPlayer(seed=4), RandomPlayer(seed=4)
    first.newGame(Board._BLACK)
    second.newGame(Board._BLACK)
    move = first.getPlayerMove()
    assert move == second.getPlayerMove()
    assert Board.name_to_flat(move) in Board().legal_moves()


def test_opening_book_loads_2020_games():
    book = OpeningBook.load(random.Random(0))
    move = book.next_move(Board())
    assert move in Board().weak_legal_moves()


def test_opening_book_stops_after_length():
    book = OpeningBook([["E5", "E6", "D5", "D6", "C5", "C6"]], random.Random(0), length=2)
    board = Board()
    played = []
    while (move := book.next_move(board)) is not None:
        played.append(move)
        board.push(move)
        board.push(-1)  # opponent passes
    assert len(played) == 2


def test_opening_book_skips_occupied_points_and_gives_up():
    book = OpeningBook([["E5", "D5", "E5", "D5"]], random.Random(0), length=5)
    board = board_with(["E5", "D5"])
    assert book.next_move(board) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_players_base.py -v`
Expected: collection ERROR with `ModuleNotFoundError: No module named 'go_player.opening'`.

- [ ] **Step 3: Implement the modules**

`go_player/players/__init__.py`:
```python
```

`go_player/players/base.py`:
```python
from go_player.goban import Board


class PlayerInterface:
    """The 2020 tournament contract. Moves are strings: "A1" ... "J9" or "PASS"."""

    def getPlayerName(self) -> str:
        return "Not Defined"

    def getPlayerMove(self) -> str:
        return "PASS"

    def playOpponentMove(self, move: str) -> None:
        pass

    def newGame(self, color: int) -> None:
        pass

    def endGame(self, winner: int) -> None:
        pass


def candidate_moves(board) -> list[int]:
    # Passing early is never useful for a search; keep it only when nothing else is legal.
    moves = [m for m in board.weak_legal_moves() if m != -1]
    return moves or [-1]


def is_winning(board, color: int) -> bool:
    black, white = board.compute_score()
    return black > white if color == Board._BLACK else white > black
```

`go_player/players/random_player.py`:
```python
import random

from go_player.goban import Board
from go_player.players.base import PlayerInterface


class RandomPlayer(PlayerInterface):
    def __init__(self, seed: int | None = None, komi: float = 0.0, name: str = "Random"):
        self._board = Board(komi)
        self._rng = random.Random(seed)
        self._name = name

    def getPlayerName(self) -> str:
        return self._name

    def getPlayerMove(self) -> str:
        if self._board.is_game_over():
            return "PASS"
        move = self._rng.choice(self._board.legal_moves())
        self._board.push(move)
        return Board.flat_to_name(move)

    def playOpponentMove(self, move: str) -> None:
        self._board.push(Board.name_to_flat(move))
```

`go_player/opening.py`:
```python
import json
import random
from importlib.resources import files

from go_player.goban import Board

MAX_LINE_ATTEMPTS = 10


class OpeningBook:
    """2020 opening strategy: follow one side of a random recorded game, skipping unplayable moves."""

    def __init__(self, games: list[list[str]], rng: random.Random, length: int = 5):
        self._games = games
        self._rng = rng
        self._length = length
        self._played = 0
        self._line: list[str] | None = None
        self._index = 0

    @classmethod
    def load(cls, rng: random.Random, length: int = 5) -> "OpeningBook":
        data = json.loads((files("go_player") / "assets" / "games.json").read_text())
        return cls([game["moves"] for game in data], rng, length)

    def _pick_line(self) -> list[str]:
        game = self._rng.choice(self._games)
        return game[self._rng.randint(0, 1)::2]

    def next_move(self, board: Board) -> int | None:
        if self._played >= self._length:
            return None
        for _ in range(MAX_LINE_ATTEMPTS):
            if self._line is None:
                self._line, self._index = self._pick_line(), 0
            while self._index < len(self._line):
                name = self._line[self._index]
                self._index += 1
                if name == "PASS":
                    continue
                move = Board.name_to_flat(name)
                if move not in board.weak_legal_moves():
                    continue
                legal = board.push(move)  # push() is the only superko check
                board.pop()
                if legal:
                    self._played += 1
                    return move
            self._line = None
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_players_base.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add go_player/players go_player/opening.py tests/test_players_base.py
git commit -m "feat: add player contract, random player and 2020 opening book" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Arena (referee, matches, report)

**Files:**
- Create: `go_player/arena.py`
- Create: `tests/test_arena.py`

**Interfaces:**
- Consumes: `Board` (Task 1), `PlayerInterface`, `RandomPlayer` (Task 3).
- Produces:
  - `GameRecord` dataclass with fields `black: str`, `white: str`, `moves: list[str]`, `winner: int | None`, `score: str`, `illegal_by: int | None`, `move_times: dict[int, list[float]]`, `simulations: dict[int, list[int]]`;
  - `play_game(black, white, komi=0.0, max_moves=200) -> GameRecord`;
  - `MatchResult` dataclass with fields `a: str`, `b: str`, `games: list[GameRecord]`, `a_colors: list[int]`, and a method `summary() -> dict`;
  - `run_match(factory_a, factory_b, games=20, seed=0, komi=0.0, max_moves=200, on_game=None) -> MatchResult`. Factories have type `Callable[[int], PlayerInterface]` (they receive the seed); `on_game` is an optional `Callable[[int, GameRecord], None]`;
  - `write_report(results: list[MatchResult], out_dir: Path) -> tuple[Path, Path]`, which writes `arena.md` and `arena.json`.

- [ ] **Step 1: Write the failing tests**

`tests/test_arena.py`:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_arena.py -v`
Expected: collection ERROR with `ModuleNotFoundError: No module named 'go_player.arena'`.

- [ ] **Step 3: Implement `go_player/arena.py`**

```python
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Callable

from go_player.goban import Board
from go_player.players.base import PlayerInterface

PlayerFactory = Callable[[int], PlayerInterface]


@dataclass
class GameRecord:
    black: str
    white: str
    moves: list[str] = field(default_factory=list)
    winner: int | None = None
    score: str = ""
    illegal_by: int | None = None
    move_times: dict[int, list[float]] = field(default_factory=lambda: {Board._BLACK: [], Board._WHITE: []})
    simulations: dict[int, list[int]] = field(default_factory=lambda: {Board._BLACK: [], Board._WHITE: []})


def _to_flat(move: str) -> int | None:
    try:
        return Board.name_to_flat(move)
    except (KeyError, ValueError, IndexError, TypeError):
        return None


def play_game(black: PlayerInterface, white: PlayerInterface, komi: float = 0.0, max_moves: int = 200) -> GameRecord:
    board = Board(komi)
    players = {Board._BLACK: black, Board._WHITE: white}
    black.newGame(Board._BLACK)
    white.newGame(Board._WHITE)
    record = GameRecord(black=black.getPlayerName(), white=white.getPlayerName())
    color = Board._BLACK
    while not board.is_game_over() and len(record.moves) < max_moves:
        player = players[color]
        start = time.perf_counter()
        move = player.getPlayerMove()
        record.move_times[color].append(time.perf_counter() - start)
        sims = getattr(player, "last_simulations", 0)
        if sims:
            record.simulations[color].append(sims)
        flat = _to_flat(move)
        if flat is None or flat not in board.legal_moves():
            record.illegal_by = color
            break
        board.push(flat)
        record.moves.append(move)
        players[Board.flip(color)].playOpponentMove(move)
        color = Board.flip(color)

    if record.illegal_by is not None:
        record.winner, record.score = Board.flip(record.illegal_by), "illegal move"
    elif not board.is_game_over():
        record.winner, record.score = None, "move cap"
    else:
        result = board.result()
        record.winner = {"1-0": Board._WHITE, "0-1": Board._BLACK}.get(result)
        record.score = board.final_go_score()
    for player in players.values():
        player.endGame(record.winner or 0)
    return record


def _avg(values: list[float]) -> float | None:
    return mean(values) if values else None


@dataclass
class MatchResult:
    a: str
    b: str
    games: list[GameRecord] = field(default_factory=list)
    a_colors: list[int] = field(default_factory=list)

    def summary(self) -> dict:
        pairs = list(zip(self.games, self.a_colors))
        a_times = [t for g, c in pairs for t in g.move_times[c]]
        b_times = [t for g, c in pairs for t in g.move_times[Board.flip(c)]]
        a_sims = [s for g, c in pairs for s in g.simulations[c]]
        b_sims = [s for g, c in pairs for s in g.simulations[Board.flip(c)]]
        return {
            "games": len(self.games),
            "a_wins": sum(g.winner == c for g, c in pairs),
            "b_wins": sum(g.winner == Board.flip(c) for g, c in pairs),
            "draws": sum(g.winner is None for g in self.games),
            "a_wins_as_black": sum(g.winner == c == Board._BLACK for g, c in pairs),
            "a_wins_as_white": sum(g.winner == c == Board._WHITE for g, c in pairs),
            "illegal_moves": sum(g.illegal_by is not None for g in self.games),
            "a_avg_move_s": _avg(a_times),
            "b_avg_move_s": _avg(b_times),
            "a_avg_sims": _avg(a_sims),
            "b_avg_sims": _avg(b_sims),
        }


def run_match(
    factory_a: PlayerFactory,
    factory_b: PlayerFactory,
    games: int = 20,
    seed: int = 0,
    komi: float = 0.0,
    max_moves: int = 200,
    on_game: Callable[[int, GameRecord], None] | None = None,
) -> MatchResult:
    result: MatchResult | None = None
    for i in range(games):
        game_seed = seed * 1000 + 2 * i
        a, b = factory_a(game_seed), factory_b(game_seed + 1)
        if result is None:
            result = MatchResult(a=a.getPlayerName(), b=b.getPlayerName())
        # Alternate colors so that the first-move advantage (no komi by default) cancels out.
        a_color = Board._BLACK if i % 2 == 0 else Board._WHITE
        black, white = (a, b) if a_color == Board._BLACK else (b, a)
        record = play_game(black, white, komi=komi, max_moves=max_moves)
        result.games.append(record)
        result.a_colors.append(a_color)
        if on_game is not None:
            on_game(i, record)
    assert result is not None, "games must be >= 1"
    return result


def _fmt(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def write_report(results: list[MatchResult], out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Match (A vs B) | Games | A wins | B wins | Draws | A wins as Black/White | A s/move | B s/move | A sims/move |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    matches = []
    for r in results:
        s = r.summary()
        lines.append(
            f"| {r.a} vs {r.b} | {s['games']} | {s['a_wins']} | {s['b_wins']} | {s['draws']} | "
            f"{s['a_wins_as_black']}/{s['a_wins_as_white']} | {_fmt(s['a_avg_move_s'], 2)} | "
            f"{_fmt(s['b_avg_move_s'], 2)} | {_fmt(s['a_avg_sims'], 0)} |"
        )
        matches.append({"a": r.a, "b": r.b, "summary": s, "games": [asdict(g) for g in r.games]})
    md = out_dir / "arena.md"
    md.write_text("\n".join(lines) + "\n")
    js = out_dir / "arena.json"
    js.write_text(json.dumps({"matches": matches}, indent=2))
    return md, js
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_arena.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add go_player/arena.py tests/test_arena.py
git commit -m "feat: add arena referee, matches and markdown/json report" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: MCTS player (NN and rollout evaluators)

**Files:**
- Create: `go_player/players/mcts.py`
- Create: `tests/test_mcts.py`

**Interfaces:**
- Consumes: `Board`, `ValueNet`, `encode`, `PlayerInterface`, `candidate_moves`, `is_winning`, `OpeningBook`, `play_game`, `RandomPlayer`, test helpers `feed`, `CAPTURE_SETUP`, `BLACK_WINNING_AFTER_PASS`, `BLACK_LOSING_AFTER_PASS`.
- Produces:
  - `terminal_value(board) -> np.ndarray`, which returns `[1, 0]`, `[0, 1]` or `[0.5, 0.5]`;
  - `NNEvaluator(net: ValueNet)` and `RolloutEvaluator(rng: random.Random, max_moves: int = 200)`. Both expose `prepare(board) -> object` (the board is unchanged afterwards) and `evaluate(prepared: list) -> np.ndarray` of shape `(N, 2)`;
  - `MCTSPlayer(evaluator, time_budget=5.0, batch_size=16, c=1.4, seed=None, opening_book=True, max_simulations=None, komi=0.0, name="MCTS")`;
  - the attribute `MCTSPlayer.last_simulations: int` and the method `MCTSPlayer.search() -> int` (a flat move).

- [ ] **Step 1: Write the failing tests**

`tests/test_mcts.py`:
```python
import random
import time

import numpy as np

from go_player.arena import play_game
from go_player.goban import Board
from go_player.players.mcts import MCTSPlayer, NNEvaluator, RolloutEvaluator
from go_player.players.random_player import RandomPlayer
from positions import BLACK_LOSING_AFTER_PASS, BLACK_WINNING_AFTER_PASS, CAPTURE_SETUP, board_with, feed


def mcts(net, **kwargs):
    kwargs.setdefault("opening_book", False)
    kwargs.setdefault("seed", 0)
    return MCTSPlayer(NNEvaluator(net), **kwargs)


def test_finds_the_obvious_capture(net):
    player = mcts(net, max_simulations=600)
    player.newGame(Board._BLACK)
    feed(player, CAPTURE_SETUP)
    assert player.getPlayerMove() == "E6"
    assert player.last_simulations == 600


def test_plays_only_legal_moves_against_random(net):
    record = play_game(mcts(net, max_simulations=16), RandomPlayer(seed=2), max_moves=40)
    assert record.illegal_by is None
    assert len(record.moves) == 40 or record.winner is not None


def test_respects_time_budget(net):
    player = mcts(net, time_budget=0.5)
    player.newGame(Board._BLACK)
    start = time.perf_counter()
    player.getPlayerMove()
    assert time.perf_counter() - start <= 0.55
    assert player.last_simulations > 0


def test_is_reproducible_with_a_seed(net):
    moves = []
    for _ in range(2):
        player = mcts(net, max_simulations=64, seed=7)
        player.newGame(Board._BLACK)
        feed(player, CAPTURE_SETUP[:6])
        moves.append(player.getPlayerMove())
    assert moves[0] == moves[1]


def test_mcts_passes_when_winning_after_opponent_pass(net):
    player = mcts(net, max_simulations=32)
    player.newGame(Board._BLACK)
    feed(player, BLACK_WINNING_AFTER_PASS)
    assert player.getPlayerMove() == "PASS"


def test_mcts_does_not_pass_when_losing(net):
    player = mcts(net, max_simulations=32)
    player.newGame(Board._BLACK)
    feed(player, BLACK_LOSING_AFTER_PASS)
    assert player.getPlayerMove() != "PASS"


def test_rollout_evaluator_leaves_board_unchanged():
    board = board_with(CAPTURE_SETUP)
    cells, trail = board._board.copy(), len(board._trailMoves)
    value = RolloutEvaluator(random.Random(0), max_moves=60).prepare(board)
    assert np.array_equal(board._board, cells) and len(board._trailMoves) == trail
    assert value.sum() == 1.0


def test_rollout_mcts_returns_a_legal_move():
    player = MCTSPlayer(RolloutEvaluator(random.Random(0), max_moves=60), max_simulations=8, opening_book=False, seed=0)
    player.newGame(Board._BLACK)
    move = player.getPlayerMove()
    assert Board.name_to_flat(move) in Board().legal_moves()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_mcts.py -v`
Expected: collection ERROR with `ModuleNotFoundError: No module named 'go_player.players.mcts'`.

- [ ] **Step 3: Implement `go_player/players/mcts.py`**

```python
"""UCT Monte-Carlo tree search with batched leaf evaluation.

The tree is walked with push/pop on the player's own board (Board has no cheap
copy). Leaves are collected in batches with a virtual loss (a visit counted
before its value is known), so the CNN evaluates many leaves per call.
"""

import math
import random
import time

import numpy as np

from go_player.goban import Board
from go_player.nn import ValueNet, encode
from go_player.opening import OpeningBook
from go_player.players.base import PlayerInterface, candidate_moves, is_winning


def terminal_value(board: Board) -> np.ndarray:
    result = board.result()
    if result == "0-1":
        return np.array([1.0, 0.0])
    if result == "1-0":
        return np.array([0.0, 1.0])
    return np.array([0.5, 0.5])


class NNEvaluator:
    def __init__(self, net: ValueNet):
        self._net = net

    def prepare(self, board: Board) -> np.ndarray:
        return encode(board)

    def evaluate(self, prepared: list) -> np.ndarray:
        return self._net.predict(np.stack(prepared))


class RolloutEvaluator:
    def __init__(self, rng: random.Random, max_moves: int = 200):
        self._rng = rng
        self._max_moves = max_moves

    def prepare(self, board: Board) -> np.ndarray:
        pushed = 0
        while not board.is_game_over() and pushed < self._max_moves:
            moves = candidate_moves(board)
            self._rng.shuffle(moves)
            for move in moves + [-1]:
                pushed += 1
                if board.push(move):
                    break
                board.pop()  # superko: undo and try the next move
                pushed -= 1
        if board.is_game_over():
            value = terminal_value(board)
        else:
            black, white = board.compute_score()
            value = np.array([1.0, 0.0]) if black > white else np.array([0.0, 1.0]) if white > black else np.array([0.5, 0.5])
        for _ in range(pushed):
            board.pop()
        return value

    def evaluate(self, prepared: list) -> np.ndarray:
        return np.stack(prepared)


class Node:
    __slots__ = ("move", "player", "children", "untried", "n", "w")

    def __init__(self, move: int | None, player: int):
        self.move = move
        self.player = player  # color that played `move`; w is from this color's point of view
        self.children: list["Node"] = []
        self.untried: list[int] | None = None
        self.n = 0
        self.w = 0.0


class MCTSPlayer(PlayerInterface):
    def __init__(
        self,
        evaluator,
        time_budget: float = 5.0,
        batch_size: int = 16,
        c: float = 1.4,
        seed: int | None = None,
        opening_book: bool = True,
        max_simulations: int | None = None,
        komi: float = 0.0,
        name: str = "MCTS",
    ):
        self._evaluator = evaluator
        self._time_budget = time_budget
        self._batch_size = batch_size
        self._c = c
        self._rng = random.Random(seed)
        self._book = OpeningBook.load(self._rng) if opening_book else None
        self._max_simulations = max_simulations
        self._board = Board(komi)
        self._name = name
        self.last_simulations = 0

    def getPlayerName(self) -> str:
        return self._name

    def playOpponentMove(self, move: str) -> None:
        self._board.push(Board.name_to_flat(move))

    def getPlayerMove(self) -> str:
        self.last_simulations = 0
        if self._board.is_game_over():
            return "PASS"
        move = self._choose()
        self._board.push(move)
        return Board.flat_to_name(move)

    def _choose(self) -> int:
        board = self._board
        if board._lastPlayerHasPassed and is_winning(board, board.next_player()):
            return -1
        if self._book is not None:
            move = self._book.next_move(board)
            if move is not None:
                return move
        return self.search()

    def search(self) -> int:
        board = self._board
        root = Node(None, Board.flip(board.next_player()))
        deadline = time.perf_counter() + 0.9 * self._time_budget
        sims = 0
        while True:
            if self._max_simulations is not None:
                remaining = self._max_simulations - sims
                if remaining <= 0:
                    break
            elif time.perf_counter() >= deadline:
                break
            else:
                remaining = self._batch_size
            batch = [self._select_and_expand(root) for _ in range(min(self._batch_size, remaining))]
            sims += len(batch)
            self._backup(batch)
        self.last_simulations = sims
        if not root.children:
            legal = [m for m in board.legal_moves() if m != -1]
            return self._rng.choice(legal) if legal else -1
        return max(root.children, key=lambda child: child.n).move

    def _uct(self, child: Node, parent_n: int) -> float:
        if child.n == 0:
            return math.inf
        return child.w / child.n + self._c * math.sqrt(math.log(parent_n) / child.n)

    def _select_and_expand(self, root: Node):
        board = self._board
        node, path, pushed = root, [root], 0
        root.n += 1  # virtual loss: the visit counts now, its value is added in _backup
        while True:
            if board.is_game_over():
                leaf = ("terminal", terminal_value(board))
                break
            if node.untried is None:
                node.untried = candidate_moves(board)
                self._rng.shuffle(node.untried)
            child = None
            while node.untried:
                move = node.untried.pop()
                if board.push(move):
                    pushed += 1
                    child = Node(move, Board.flip(board.next_player()))
                    node.children.append(child)
                    break
                board.pop()  # superko
            if child is not None:
                child.n += 1
                path.append(child)
                leaf = ("prepared", self._evaluator.prepare(board))
                break
            if not node.children:
                node.untried = [-1]  # every move was superko: only passing is left
                continue
            node = max(node.children, key=lambda ch: self._uct(ch, node.n))
            board.push(node.move)
            pushed += 1
            node.n += 1
            path.append(node)
        for _ in range(pushed):
            board.pop()
        return path, leaf

    def _backup(self, batch) -> None:
        prepared = [payload for _, (kind, payload) in batch if kind == "prepared"]
        values = iter(self._evaluator.evaluate(prepared)) if prepared else iter(())
        for path, (kind, payload) in batch:
            value = payload if kind == "terminal" else next(values)
            for node in path:
                node.w += value[node.player - 1]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_mcts.py -v`
Expected: 8 passed. If `test_finds_the_obvious_capture` fails, print the visit counts of the root children (`sorted(((ch.n, Board.flat_to_name(ch.move)) for ch in root.children), reverse=True)[:5]`, by temporarily keeping `root` on `self`) and check the value sign in `_backup` before changing any constant. `test_capture_is_the_best_one_ply_move` (Task 2) proves the network ranks E6 first.

- [ ] **Step 5: Commit**

```bash
git add go_player/players/mcts.py tests/test_mcts.py
git commit -m "feat: add MCTS player with batched CNN and rollout evaluators" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Alpha-Beta 2020 player, cleaned up

**Files:**
- Create: `go_player/players/alphabeta.py`
- Create: `tests/test_alphabeta.py`

**Interfaces:**
- Consumes: `Board`, `ValueNet`, `PlayerInterface`, `is_winning`, `OpeningBook`, `play_game`, `RandomPlayer`, test helpers.
- Produces:
  - `AlphaBetaPlayer(value_net, time_budget=5.0, seed=None, opening_book=True, komi=0.0, max_depth=10, name="AlphaBeta-2020")`;
  - `AlphaBetaPlayer.evaluate() -> float`: P(our color wins) × 100;
  - `AlphaBetaPlayer.iterative_deepening() -> int`;
  - the attribute `AlphaBetaPlayer.last_depth: int`.

Fixes compared with `GO/myPlayer.py`:
1. `NNboard` is removed; positions are encoded from the board itself (it never removed captured stones).
2. Iterative deepening keeps the move of the **last completed depth**, instead of the best score across all depths.
3. There is no dead MinMax code, and no working-directory-relative paths.

- [ ] **Step 1: Write the failing tests**

`tests/test_alphabeta.py`:
```python
import pytest

from go_player.arena import play_game
from go_player.goban import Board
from go_player.players.alphabeta import AlphaBetaPlayer
from go_player.players.random_player import RandomPlayer
from positions import BLACK_LOSING_AFTER_PASS, BLACK_WINNING_AFTER_PASS, CAPTURE_SETUP, feed


def alphabeta(net, **kwargs):
    kwargs.setdefault("opening_book", False)
    kwargs.setdefault("seed", 0)
    return AlphaBetaPlayer(net, **kwargs)


def test_finds_the_obvious_capture(net):
    player = alphabeta(net, time_budget=2.0)
    player.newGame(Board._BLACK)
    feed(player, CAPTURE_SETUP)
    assert player.getPlayerMove() == "E6"
    assert player.last_depth >= 1


def test_evaluation_sees_captures(net):
    player = alphabeta(net)
    player.newGame(Board._WHITE)
    feed(player, CAPTURE_SETUP + ["E6"])  # White to move, E5-E4 were captured
    player._me = Board._WHITE
    assert player.evaluate() == pytest.approx(float(net.evaluate(player._board)[1]) * 100)


def test_plays_only_legal_moves_against_random(net):
    record = play_game(alphabeta(net, time_budget=0.3), RandomPlayer(seed=5), max_moves=20)
    assert record.illegal_by is None


def test_alphabeta_passes_when_winning_after_opponent_pass(net):
    player = alphabeta(net, time_budget=0.5)
    player.newGame(Board._BLACK)
    feed(player, BLACK_WINNING_AFTER_PASS)
    assert player.getPlayerMove() == "PASS"


def test_alphabeta_does_not_pass_when_losing(net):
    player = alphabeta(net, time_budget=0.5)
    player.newGame(Board._BLACK)
    feed(player, BLACK_LOSING_AFTER_PASS)
    assert player.getPlayerMove() != "PASS"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_alphabeta.py -v`
Expected: collection ERROR with `ModuleNotFoundError: No module named 'go_player.players.alphabeta'`.

- [ ] **Step 3: Implement `go_player/players/alphabeta.py`**

```python
"""The 2020 Alpha-Beta + iterative deepening player, with its bugs fixed."""

import math
import random
import time

from go_player.goban import Board
from go_player.nn import ValueNet
from go_player.opening import OpeningBook
from go_player.players.base import PlayerInterface, is_winning

WIN_SCORE = 1000


class AlphaBetaPlayer(PlayerInterface):
    def __init__(
        self,
        value_net: ValueNet,
        time_budget: float = 5.0,
        seed: int | None = None,
        opening_book: bool = True,
        komi: float = 0.0,
        max_depth: int = 10,
        name: str = "AlphaBeta-2020",
    ):
        self._net = value_net
        self._time_budget = time_budget
        self._rng = random.Random(seed)
        self._book = OpeningBook.load(self._rng) if opening_book else None
        self._board = Board(komi)
        self._max_depth = max_depth
        self._name = name
        self._me = Board._BLACK
        self._deadline = 0.0
        self._timed_out = False
        self.last_depth = 0

    def getPlayerName(self) -> str:
        return self._name

    def newGame(self, color: int) -> None:
        self._me = color

    def playOpponentMove(self, move: str) -> None:
        self._board.push(Board.name_to_flat(move))

    def getPlayerMove(self) -> str:
        board = self._board
        if board.is_game_over():
            return "PASS"
        self._me = board.next_player()
        move = self._choose()
        board.push(move)
        return Board.flat_to_name(move)

    def _choose(self) -> int:
        if self._board._lastPlayerHasPassed and is_winning(self._board, self._me):
            return -1
        if self._book is not None:
            move = self._book.next_move(self._board)
            if move is not None:
                return move
        return self.iterative_deepening()

    def evaluate(self) -> float:
        return float(self._net.evaluate(self._board)[self._me - 1]) * 100

    def _terminal_score(self) -> float:
        result = self._board.result()
        if result == "1/2-1/2":
            return 0.0
        winner = Board._WHITE if result == "1-0" else Board._BLACK
        return WIN_SCORE if winner == self._me else -WIN_SCORE

    def _out_of_time(self) -> bool:
        if time.perf_counter() >= self._deadline:
            self._timed_out = True
        return self._timed_out

    def iterative_deepening(self) -> int:
        self._deadline = time.perf_counter() + 0.9 * self._time_budget
        best, self.last_depth = None, 0
        for depth in range(1, self._max_depth + 1):
            move, completed = self._root(depth)
            if completed:
                best, self.last_depth = move, depth
            elif best is None:
                best = move  # a partial depth-1 search beats a random move
            if not completed or time.perf_counter() >= self._deadline:
                break
        if best is None:
            legal = [m for m in self._board.legal_moves() if m != -1]
            best = self._rng.choice(legal) if legal else -1
        return best

    def _root(self, depth: int) -> tuple[int | None, bool]:
        board = self._board
        self._timed_out = False
        best_moves, best_value = [], None
        for move in board.weak_legal_moves():
            if self._out_of_time():
                break
            if board.push(move):
                value = self._min_value(depth - 1, -math.inf, math.inf)
                if best_value is None or value > best_value:
                    best_moves, best_value = [move], value
                elif value == best_value:
                    best_moves.append(move)
            board.pop()
        move = self._rng.choice(best_moves) if best_moves else None
        return move, not self._timed_out

    def _max_value(self, depth: int, alpha: float, beta: float) -> float:
        board = self._board
        if board.is_game_over():
            return self._terminal_score()
        if depth <= 0:
            return self.evaluate()
        value = -math.inf
        for move in board.weak_legal_moves():
            if self._out_of_time():
                break
            if board.push(move):
                value = max(value, self._min_value(depth - 1, alpha, beta))
            board.pop()
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def _min_value(self, depth: int, alpha: float, beta: float) -> float:
        board = self._board
        if board.is_game_over():
            return self._terminal_score()
        if depth <= 0:
            return self.evaluate()
        value = math.inf
        for move in board.weak_legal_moves():
            if self._out_of_time():
                break
            if board.push(move):
                value = min(value, self._max_value(depth - 1, alpha, beta))
            board.pop()
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_alphabeta.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add go_player/players/alphabeta.py tests/test_alphabeta.py
git commit -m "feat: port the 2020 alpha-beta player with capture and ID fixes" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: GnuGo player (GTP)

**Files:**
- Create: `go_player/players/gnugo.py`
- Create: `tests/test_gnugo.py`

**Interfaces:**
- Consumes: `Board`, `PlayerInterface`, `play_game`, `RandomPlayer`.
- Produces:
  - the exceptions `GnuGoNotFound(Exception)` and `GnuGoError(Exception)`;
  - `normalize_gtp_move(move: str) -> str`;
  - `GnuGoPlayer(level: int = 1, komi: float = 0.0, name: str | None = None)`, with methods `close() -> None`, `getPlayerName()` (default `f"GnuGo-L{level}"`) and `getPlayerMove()`.

- [ ] **Step 1: Install GnuGo**

Run: `brew install gnu-go && gnugo --version`
Expected: `GNU Go 3.8`.

- [ ] **Step 2: Write the failing tests**

`tests/test_gnugo.py`:
```python
import shutil

import pytest

from go_player.arena import play_game
from go_player.goban import Board
from go_player.players import gnugo
from go_player.players.gnugo import GnuGoNotFound, GnuGoPlayer, normalize_gtp_move
from go_player.players.random_player import RandomPlayer

needs_gnugo = pytest.mark.skipif(shutil.which("gnugo") is None, reason="gnugo not installed")


def test_normalize_gtp_move():
    assert normalize_gtp_move(" e5\n") == "E5"
    assert normalize_gtp_move("pass") == "PASS"
    assert normalize_gtp_move("resign") == "PASS"


def test_missing_binary_has_install_hint(monkeypatch):
    monkeypatch.setattr(gnugo.shutil, "which", lambda name: None)
    with pytest.raises(GnuGoNotFound, match="brew install gnu-go"):
        GnuGoPlayer()


@pytest.mark.gnugo
@needs_gnugo
def test_gnugo_plays_a_game_against_random():
    player = GnuGoPlayer(level=1)
    try:
        record = play_game(player, RandomPlayer(seed=3), max_moves=30)
    finally:
        player.close()
    assert record.black == "GnuGo-L1"
    assert record.illegal_by != Board._WHITE
    assert len(record.moves) > 0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gnugo.py -v`
Expected: collection ERROR with `ModuleNotFoundError: No module named 'go_player.players.gnugo'`.

- [ ] **Step 4: Implement `go_player/players/gnugo.py`**

```python
import shutil
import subprocess

from go_player.goban import Board
from go_player.players.base import PlayerInterface


class GnuGoNotFound(Exception):
    pass


class GnuGoError(Exception):
    pass


def normalize_gtp_move(move: str) -> str:
    move = move.strip().upper()
    return "PASS" if move in ("PASS", "RESIGN") else move


class GnuGoPlayer(PlayerInterface):
    def __init__(self, level: int = 1, komi: float = 0.0, name: str | None = None):
        exe = shutil.which("gnugo")
        if exe is None:
            raise GnuGoNotFound("gnugo not found in PATH. Install it with: brew install gnu-go")
        self._proc = subprocess.Popen(
            [exe, "--mode", "gtp", "--boardsize", str(Board._BOARDSIZE), "--chinese-rules",
             "--capture-all-dead", "--never-resign", "--level", str(level), "--komi", str(komi)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        self._name = name or f"GnuGo-L{level}"
        self._color = Board._BLACK

    def _query(self, command: str) -> str:
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()
        lines = []
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                raise GnuGoError(f"gnugo exited while running {command!r}")
            line = line.rstrip("\n")
            if not line:
                if lines:
                    break  # GTP responses end with an empty line
                continue
            lines.append(line)
        if lines[0].startswith("?"):
            raise GnuGoError(f"gnugo rejected {command!r}: {lines[0][1:].strip()}")
        return " ".join([lines[0][1:].strip(), *lines[1:]]).strip()

    def getPlayerName(self) -> str:
        return self._name

    def newGame(self, color: int) -> None:
        self._color = color
        self._query("clear_board")

    def getPlayerMove(self) -> str:
        return normalize_gtp_move(self._query(f"genmove {Board.player_name(self._color)}"))

    def playOpponentMove(self, move: str) -> None:
        self._query(f"play {Board.player_name(Board.flip(self._color))} {move}")

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.stdin.write("quit\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=5)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gnugo.py -v`
Expected: 3 passed (or 2 passed and 1 skipped if gnugo is missing).

- [ ] **Step 6: Commit**

```bash
git add go_player/players/gnugo.py tests/test_gnugo.py
git commit -m "feat: add GnuGo GTP player" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Recorder (Pillow frames → GIF + MP4)

**Files:**
- Create: `go_player/record.py`
- Create: `tests/test_record.py`

**Interfaces:**
- Consumes: `Board`, `ValueNet`, `play_game`, `GameRecord`, `RandomPlayer`.
- Produces:
  - `render_frame(board, title: str, caption: str, last_move: int | None, p_black: float | None) -> PIL.Image.Image`, of size `(448, 528)`;
  - `record_game(black, white, out_stem: Path, value_net=None, komi=0.0, max_moves=200, frame_ms=600, final_ms=3000) -> tuple[Path, Path, GameRecord]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_record.py`:
```python
from PIL import Image

from go_player.goban import Board
from go_player.players.random_player import RandomPlayer
from go_player.record import record_game, render_frame
from positions import CAPTURE_SETUP, board_with


def test_render_frame_size_and_stones():
    board = board_with(CAPTURE_SETUP)
    image = render_frame(board, "A vs B", "Move 10", Board.name_to_flat("J7"), 0.75)
    assert image.size == (448, 528)
    # E5 (col 4, lin 4) holds a white stone: its center pixel is white.
    assert image.getpixel((64 + 44 * 4, 104 + 44 * 4)) == (255, 255, 255)


def test_record_game_at_move_cap(tmp_path):
    gif, mp4, record = record_game(
        RandomPlayer(seed=1, name="R1"), RandomPlayer(seed=2, name="R2"), tmp_path / "r1-vs-r2", max_moves=6,
    )
    assert record.score == "move cap"
    assert Image.open(gif).n_frames == 7  # empty board + 6 moves
    assert mp4.stat().st_size > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_record.py -v`
Expected: collection ERROR with `ModuleNotFoundError: No module named 'go_player.record'`.

- [ ] **Step 3: Implement `go_player/record.py`**

```python
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
STEP, GX, GY = 44, 64, HEADER + 40  # grid spacing and position of A9
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
    draw.rectangle((GX - 28, GY - 28, GX + STEP * size + 28, GY + STEP * size + 28), fill=WOOD)
    for i in range(Board._BOARDSIZE):
        draw.line((GX + STEP * i, GY, GX + STEP * i, GY + STEP * size), fill=LINE, width=2)
        draw.line((GX, GY + STEP * i, GX + STEP * size, GY + STEP * i), fill=LINE, width=2)
        draw.text((GX + STEP * i - 4, GY - 26), "ABCDEFGHJ"[i], fill=LINE, font=_font(12))
        draw.text((GX - 26, GY + STEP * i - 7), str(Board._BOARDSIZE - i), fill=LINE, font=_font(12))
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
        draw.text((WIDTH // 2 - 90, top + 8), f"P(black wins) = {p_black:.0%}", fill=RED, font=_font(14))
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_record.py -v`
Expected: 2 passed.

- [ ] **Step 5: Look at one frame**

Run: `PYTHONPATH=tests uv run python -c "from go_player.record import render_frame; from positions import CAPTURE_SETUP, board_with; from go_player.goban import Board; render_frame(board_with(CAPTURE_SETUP), 'MCTS-NN (B)  vs  GnuGo-L1 (W)', 'Move 10: J7', Board.name_to_flat('J7'), 0.62).save('/tmp/frame.png')"`, then open `/tmp/frame.png` with the Read tool.
Expected: the stones sit on the intersections, the labels A–J and 9–1 are readable, the red ring is on J7 and the bar is about 62 % black.

- [ ] **Step 6: Commit**

```bash
git add go_player/record.py tests/test_record.py
git commit -m "feat: record games as GIF and MP4 with value bar overlay" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Player registry + CLI

**Files:**
- Modify: `go_player/players/__init__.py`
- Create: `go_player/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: every player (Tasks 3, 5, 6, 7), `ValueNet`, `ModelLoadError`, `run_match`, `write_report`, `play_game`, `record_game`.
- Produces:
  - `PLAYER_KINDS = ("random", "gnugo", "alphabeta", "mcts", "mcts-rollout")` and `NET_KINDS = {"alphabeta", "mcts"}`;
  - `make_player(kind, *, seed, time_budget, komi=0.0, value_net=None) -> PlayerInterface`;
  - `DEFAULT_MATCHES = [("mcts", "alphabeta"), ("mcts", "mcts-rollout"), ("mcts", "gnugo"), ("alphabeta", "gnugo")]`;
  - `go_player.cli.main(argv: list[str] | None = None) -> int`, with the subcommands `play`, `arena` and `record`.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: collection ERROR with `ModuleNotFoundError: No module named 'go_player.cli'`.

- [ ] **Step 3: Implement the registry in `go_player/players/__init__.py`**

```python
import random

from go_player.players.base import PlayerInterface

PLAYER_KINDS = ("random", "gnugo", "alphabeta", "mcts", "mcts-rollout")
NET_KINDS = {"alphabeta", "mcts"}
DEFAULT_MATCHES = [("mcts", "alphabeta"), ("mcts", "mcts-rollout"), ("mcts", "gnugo"), ("alphabeta", "gnugo")]


def make_player(kind: str, *, seed: int | None, time_budget: float, komi: float = 0.0, value_net=None) -> PlayerInterface:
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
```

- [ ] **Step 4: Implement `go_player/cli.py`**

```python
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
    return parser


def _load_net(kinds, always: bool = False):
    if not always and not NET_KINDS.intersection(kinds):
        return None
    from go_player.nn import ValueNet
    return ValueNet.load()


def _run(args) -> int:
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 6 passed.

- [ ] **Step 6: Run the whole suite**

Run: `time uv run pytest -m "not gnugo"`
Expected: all tests pass, and the `real` time is under 60 s.

- [ ] **Step 7: Commit**

```bash
git add go_player/players/__init__.py go_player/cli.py tests/test_cli.py
git commit -m "feat: add player registry and go-player CLI (play, arena, record)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Showcase media + README

**Files:**
- Create: `media/mcts-vs-gnugo.gif`, `media/mcts-vs-gnugo.mp4`, `media/alphabeta-vs-mcts.gif`, `media/alphabeta-vs-mcts.mp4`, `media/arena.md`, `media/arena.json`
- Modify: `README.md` (full rewrite)

**Interfaces:**
- Consumes: the `go-player` CLI (Task 9).
- Produces: the README and media that the portfolio reuses as they are.

- [ ] **Step 1: Record the two showcase games**

```bash
uv run go-player record --black mcts --white gnugo --time 5 --seed 1
uv run go-player record --black alphabeta --white mcts --time 5 --seed 1
```
Expected: each command prints the result and two paths under `media/`. It takes about 5 to 10 minutes per game. Open each GIF and check that it plays to the end. If a game is very short or ends with an illegal move, re-run it with `--seed 2`.

- [ ] **Step 2: Run the arena in the background**

Run (in the background, about 2 to 3 hours): `uv run go-player arena --games 20 --time 1 --seed 0`
Expected: `media/arena.md` lists 4 matches. The report is rewritten after each match, so partial results survive an interruption. Record the MCTS-NN vs AlphaBeta-2020 win rate. The target is above 60 %, but it does not block delivery: report the real figure in the README either way.

- [ ] **Step 3: Rewrite `README.md`**

Replace the whole file with the text below. Paste the content of `media/arena.md` verbatim where `<!-- arena table -->` stands, and remove that marker line.

```markdown
# 9x9 Go player: Alpha-Beta (2020) → MCTS (2026)

![MCTS-NN (black) vs GnuGo level 1 (white)](media/mcts-vs-gnugo.gif)

A 9x9 Go engine in Python. In 2020, as an ENSEIRB-MATMECA school project, **Laurent Genty and Johan Chataigner**
built an Alpha-Beta / iterative-deepening player whose evaluation function is a **CNN trained to predict the winner
of a position** (Keras, notebook in [`ML/`](ML/tp_ml_note.ipynb)).

In 2026 the project was revived with the same network but a better search: **Monte-Carlo tree search** that
evaluates leaves with the 2020 CNN in batches.

## Same network, better search

| | 2020 | 2026 |
|---|---|---|
| Search | Alpha-Beta + iterative deepening | UCT Monte-Carlo tree search |
| Evaluation | CNN, one position per call | Same CNN, 16 leaves per call |
| Time per move | 5 s | 5 s |

Bugs fixed while porting the 2020 player: the incremental network board never removed captured stones, and
iterative deepening compared scores across depths instead of keeping the last completed depth.

## Arena

20 games per match, 1 s per move, colors alternated, no komi (as in 2020).

<!-- arena table -->

![AlphaBeta-2020 (black) vs MCTS-NN (white)](media/alphabeta-vs-mcts.gif)

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and, for the GnuGo opponent, `brew install gnu-go`.

```bash
uv sync
uv run go-player play --black mcts --white random --time 2
uv run go-player record --black mcts --white gnugo
uv run go-player arena --games 20 --time 1
uv run pytest
```

Players: `random`, `gnugo`, `alphabeta` (2020), `mcts` (CNN leaves), `mcts-rollout` (random playouts, baseline).

## Layout

- `go_player/`: the 2026 package (rules engine, value network, players, arena, recorder, CLI)
- `GO/`, `ML/`: the original 2020 code and training notebook, kept untouched
- `media/`: generated GIFs, videos and arena results

## Credits

Laurent Genty and Johan Chataigner (ENSEIRB-MATMECA, 2020). Rules engine `Goban.py` by Laurent Simon (MIT).
```

- [ ] **Step 4: Verify the README renders and links resolve**

Run: `ls media/ && grep -o 'media/[a-z-]*\.gif' README.md | xargs ls -la`
Expected: every GIF referenced in the README exists. Open `README.md` in the IDE preview and check that the GIF and the table render.

- [ ] **Step 5: Run the full test suite one last time**

Run: `uv run pytest`
Expected: all tests pass (including the `gnugo` test, since GnuGo is installed).

- [ ] **Step 6: Commit**

```bash
git add media README.md
git commit -m "docs: rewrite README with showcase media and arena results" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Ask Laurent before pushing**

Do not push. Report the branch name, the arena numbers and the media paths, then ask whether to push and open a PR to `master`.
