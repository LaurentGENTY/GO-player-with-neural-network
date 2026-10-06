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
