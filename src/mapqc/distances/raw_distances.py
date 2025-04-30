from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances

# from sklearn_ann.kneighbors.annoy import AnnoyTransformer


def pairwise_sample_distances(
    emb: np.ndarray,
    obs: pd.DataFrame,
    samples_r_all: list[str],
    samples_q_all: list[str],
    sample_key: str,
    min_n_cells: int,
    exclude_same_study: bool,
    sample_df: pd.DataFrame = None,
    study_key: str = None,
    distance_metric: Literal["energy_distance", "pairwise_euclidean"] = "energy_distance",
) -> np.ndarray:
    """Calculate pairwise distances between samples in the neighborhood.

    Parameters
    ----------
    emb: np.ndarray
        Array of shape (n_cells, n_features) for all and only the cells in the
        neighborhood.
    obs: pd.DataFrame
        Metadata (containing sample column) dataframe for all cells in emb, in the
        same order as emb.
    samples_r_all: list[str]
        List of all reference sample names, ordered consistently across neighborhoods,
        including also samples that are not in the neighborhood.
    samples_q_all: list[str]
        List of all query sample names, ordered consistently across neighborhoods,
        including also samples that are not in the neighborhood.
    sample_key: str
        Column name of the sample column in obs.
    min_n_cells: int
        Minimum number of cells per sample.
    exclude_same_study: bool
        Whether to exclude pairs of samples from the same study.
    sample_df: pd.DataFrame = None
        Metadata (containing study column) dataframe for all samples in the
        neighborhood, order does not matter. Only needed if exclude_same_study is True.
    study_key: str = None
        Column name of the study column in sample_df. Only needed if exclude_same_study is True.
    distance_metric: Literal["energy_distance", "pairwise_euclidean"] = "energy_distance"
        Distance metric to use to calculate distances between samples (i.e. between
        two sets of cells).

    Returns
    -------
    pw_dists: np.ndarray
        Array of shape (n_samples_r_all, n_samples_r_all + n_samples_q_all) containing pairwise distances
        between all samples in the neighborhood, for all pairs that passed filtering, otherwise nan.
        Filtering includes:
        - minimum number of cells per sample
        - exclusion of pairs of samples from the same study if exclude_same_study is True
        - lower triangle set to nan to save computation time (dist(i,j) = dist(j,i))
        - diagonal set to nan, as distance to self should be excluded (always 0 for e-distance)
    """
    # check which samples pass min_n_cells filter, we only want to include
    # those when calculating pairwise distances
    sample_cell_counts = obs.groupby(sample_key, observed=True).size()
    samples_with_enough_cells = sample_cell_counts[sample_cell_counts >= min_n_cells].index
    samples_r = [s for s in samples_r_all if s in samples_with_enough_cells]
    samples_q = [s for s in samples_q_all if s in samples_with_enough_cells]
    # create an empty matrix for *all* samples in adata, of size
    # n_all_samples_r x (n_all_samples_r + n_all_samples_q)
    pw_dists = np.full((len(samples_r_all), len(samples_r_all) + len(samples_q_all)), np.nan)
    # create a matching mask array, masking:
    # 1. samples not in the nhood
    rows_in_nhood = np.array([s in samples_r for s in samples_r_all])
    cols_in_nhood = np.array([s in samples_r + samples_q for s in samples_r_all + samples_q_all])
    mask_s_in_nhood = np.outer(rows_in_nhood, cols_in_nhood)  # boolean
    # 2. Only if pairs from same study excluded:
    # samples from same study (will also only apply to
    # the square part, i.e. to the ref samples, as query
    # samples are only compared to ref and therefore never
    # have same-study pairs)
    # Then combine masks into final mask
    if exclude_same_study:
        row_studies = np.array([sample_df.loc[s, study_key] if s in samples_r else None for s in samples_r_all])
        col_studies = [
            sample_df.loc[s, study_key] if s in samples_r + samples_q else None for s in samples_r_all + samples_q_all
        ]
        # Handling of None values (i.e. samples not in the neighborhood):
        # they are treated as all other values, so two samples both
        # missing will be set to False, others to True. But note that
        # missing samples are handled already in the s_in_nhood mask,
        # so does not matter.
        mask_same_study = np.array(row_studies[:, None] != col_studies)
        mask = mask_s_in_nhood * mask_same_study
    else:
        mask = mask_s_in_nhood
    # fill the distances in the pw_dists matrix
    # loop through sample pairs, only calculate distance if mask is True
    for i, s1 in enumerate(samples_r_all):
        if s1 in samples_r:
            # get cell embeddings for ref sample
            s1_cell_idc = np.where(obs[sample_key] == s1)[0]
            emb_s1 = emb[s1_cell_idc, :]
            for j, s2 in enumerate(samples_r_all + samples_q_all):
                # only calculate for upper triangle
                # i.e. don't calculate distance for (i,j) if already calculated for (j,i)
                if j > i:
                    # if this pair was not masked, i.e. if the mask is True:
                    if mask[i, j]:
                        # get cell embeddings for second sample
                        s2_cell_idc = np.where(obs[sample_key] == s2)[0]
                        emb_s2 = emb[s2_cell_idc]
                        # and calculate distance between samples
                        pw_dists[i, j] = _distance_between_cell_sets(emb_s1, emb_s2, distance_metric)
    return pw_dists


def _distance_between_cell_sets(
    cell_set_1: NDArray,
    cell_set_2: NDArray,
    distance_metric: Literal["energy_distance", "pairwise_euclidean"] = "energy_distance",
    precomputed_distance_matrix: NDArray | None = None,
) -> float:
    """Calculate the distance between two sets of cells.

    This function is specifically designed for calculating the overall distance
    (i.e. a single output scalar) between two sets of data points.

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
        self_dists_1 = pairwise_distances(cell_set_1, metric="euclidean")
        # ignore the 0 diagonal (distances of each cell to itself)
        mask_1 = ~np.eye(n_cells_1, dtype=bool)
        # compute mean of pairwise distances, ignoring diagonal:
        sigma_1 = self_dists_1[mask_1].mean()
        # same for second set of cells:
        self_dists_2 = pairwise_distances(cell_set_2, metric="euclidean")
        mask_2 = ~np.eye(n_cells_2, dtype=bool)
        sigma_2 = self_dists_2[mask_2].mean()
        return 2 * delta - sigma_1 - sigma_2


# def pairwise_distances_fast(X, k, distance_metric="euclidean"):
#     """
#     TO DO: WRITE DOCSTRING ONCE I'M SURE I WANT TO INCLUDE THIS
#     (I.E. ONCE I'VE DONE TESTING REGARDING SPEED)
#     """
#     annoy_transformer = AnnoyTransformer(
#         n_neighbors=k,  # Choose a reasonable number
#         metric=distance_metric,
#         n_trees=10
#     )
#     n_obs = X.shape[0]
#     annoy_transformer.fit(X)
#     indices = np.empty((n_obs, k), dtype=int)
#     distances = np.empty((n_obs, k))
#     for i, x in enumerate(X):
#         indices[i], distances[i] = annoy_transformer.annoy_.get_nns_by_vector(
#             x.tolist(), k, annoy_transformer.search_k, include_distances=True
#         )
#     return indices, distances
