from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances


def distance_between_cell_sets(
    cell_set_1: NDArray,
    cell_set_2: NDArray,
    distance_metric: Literal["energy_distance", "pairwise_euclidean"] = "energy_distance",
    precomputed_distance_matrix: NDArray | None = None,
) -> float:
    """
    Calculate the distance between two sets of cells.

    Parameters
    ----------
    cell_set_1: NDArray
        Array of shape (n_cells_1, n_features) for the first set of cells.
    cell_set_2: NDArray
        Array of shape (n_cells_2, n_features) for the second set of cells.
    distance_metric: Literal["energy_distance", "pairwise_euclidean"]
        The distance metric to use. Default is "energy_distance".
    precomputed_distance_matrix: Optional[NDArray]
        Precomputed distance matrix of shape (n_cells_1, n_cells_2). Distances
        might already have been pre-computed for calculating k nearest neighbors.

    Returns
    -------
    distance: float
        The distance between the two sets of cells.
    """
    n_cells_1 = cell_set_1.shape[0]
    n_cells_2 = cell_set_2.shape[0]
    if precomputed_distance_matrix is not None:
        # check that the dimensions are correct:
        if precomputed_distance_matrix.shape != (n_cells_1, n_cells_2):
            raise ValueError("Precomputed distance matrix should have shape (n_cells_1, n_cells_2).")
        pairwise_dists = precomputed_distance_matrix
    else:
        pairwise_dists = pairwise_distances(cell_set_1, cell_set_2, metric="euclidean")

    delta = pairwise_dists.mean()
    if distance_metric == "pairwise_euclidean":
        return delta
    elif distance_metric == "energy_distance":
        # NOTE: in scPerturb and pertpy, they use sqeuclidean. This is also what I had
        # in my code originally and what was used for the manuscript. However, the
        # correct distance metric here is euclidean, which is what we will use here.
        self_dists_1 = pairwise_distances(cell_set_1, cell_set_1, metric="euclidean")
        # ignore the 0 diagonal (distances of each cell to itself)
        mask_1 = ~np.eye(n_cells_1, dtype=bool)
        # compute mean of pairwise distances, ignoring diagonal:
        sigma_1 = self_dists_1[mask_1].mean()
        # same for second set of cells:
        self_dists_2 = pairwise_distances(cell_set_2, cell_set_2, metric="euclidean")
        mask_2 = ~np.eye(n_cells_2, dtype=bool)
        sigma_2 = self_dists_2[mask_2].mean()
        return 2 * delta - sigma_1 - sigma_2
