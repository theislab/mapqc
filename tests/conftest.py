from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the absolute path to the test data directory."""
    # Get the directory where this conftest.py file is located
    test_dir = Path(__file__).parent
    # Return the absolute path to the data directory
    return test_dir / "data"


@pytest.fixture(scope="session")
def input_data_dir(test_data_dir):
    """Return the absolute path to the input test data directory."""
    return test_data_dir / "input"


@pytest.fixture(scope="session")
def intermediate_data_dir(test_data_dir):
    """Return the absolute path to the intermediate test data directory."""
    return test_data_dir / "intermediate"


@pytest.fixture(scope="session")
def expected_data_dir(test_data_dir):
    """Return the absolute path to the expected test data directory."""
    return test_data_dir / "expected"
