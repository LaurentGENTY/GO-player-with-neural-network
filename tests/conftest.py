import pytest

from go_player.nn import ValueNet


@pytest.fixture(scope="session")
def net():
    return ValueNet.load()
