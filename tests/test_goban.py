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
