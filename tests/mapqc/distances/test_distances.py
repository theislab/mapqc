import numpy as np
import pytest

from mapqc.distances.distances import distance_between_cell_sets


def test_pairwise_euclidean_simple():
    """Test pairwise euclidean distance with simple 2D points"""
    cell_set_1 = np.array(
        [
            [
                0,
                0,
            ],
            [np.sqrt(2), np.sqrt(2)],
            [np.sqrt(8), np.sqrt(8)],
        ]
    )
    cell_set_2 = np.array([[0, 0]])

    distance = distance_between_cell_sets(cell_set_1, cell_set_2, distance_metric="pairwise_euclidean")

    # The mean distance between all pairs should be:
    # (0 + √(2+2) + √(8+8)) / 3 = (0 + 2 + 4) / 3 = 2
    expected = 2
    np.testing.assert_almost_equal(distance, expected, decimal=6)


def test_energy_distance_simple():
    """Test energy distance with simple 2D points"""
    cell_set_1 = np.array(
        [
            [
                0,
                0,
            ],
            [np.sqrt(2), np.sqrt(2)],
            [np.sqrt(8), np.sqrt(8)],
        ]
    )
    cell_set_2 = np.array([[0, 0], [0, 0]])

    distance = distance_between_cell_sets(cell_set_1, cell_set_2, distance_metric="energy_distance")

    # Calculate expected components:
    # sigma: mean distance within each set, excluding distances to self
    # delta: mean distance between sets (same as pairwise_euclidean)
    sigma_1 = np.mean(
        [
            np.sqrt(2 + 2),  # cell 0 to 1
            np.sqrt(8 + 8),  # cell 0 to 2
            np.sqrt(2 * (np.sqrt(8) - np.sqrt(2)) ** 2),
        ]  # cell 1 to 2
    )
    sigma_2 = 0  # twice the same point
    delta = 2  # (see test_pairwise_euclidean_simple)

    expected = 2 * delta - sigma_1 - sigma_2
    np.testing.assert_almost_equal(distance, expected, decimal=6)


def test_with_precomputed_distances():
    """Test that using precomputed distances gives same result"""
    cell_set_1 = np.array(
        [
            [
                0,
                0,
            ],
            [np.sqrt(2), np.sqrt(2)],
            [np.sqrt(8), np.sqrt(8)],
        ]
    )
    cell_set_2 = np.array([[0, 0]])

    # Precompute distances
    precomputed = np.array([[0], [2], [4]])

    d1 = distance_between_cell_sets(cell_set_1, cell_set_2, distance_metric="pairwise_euclidean")

    d2 = distance_between_cell_sets(
        cell_set_1, cell_set_2, distance_metric="pairwise_euclidean", precomputed_distance_matrix=precomputed
    )

    np.testing.assert_almost_equal(d1, d2, decimal=6)


def test_invalid_precomputed_shape():
    """Test that wrong shape of precomputed distances raises error"""
    cell_set_1 = np.array([[0, 0], [1, 1]])
    cell_set_2 = np.array([[2, 2], [3, 3]])

    wrong_shape = np.zeros((3, 2))  # Wrong shape

    with pytest.raises(ValueError):
        distance_between_cell_sets(cell_set_1, cell_set_2, precomputed_distance_matrix=wrong_shape)
