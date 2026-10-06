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
