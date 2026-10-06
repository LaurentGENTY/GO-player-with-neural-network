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
