import warnings

import numpy as np
import pandas as pd

from mapqc._params import _MapQCParams


def _calculate_mapqc_scores(
    params: _MapQCParams,
    sample_dist_to_ref_per_nhood: np.ndarray,
    nhood_info_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate cell-level mapqc scores using nhood-sample-specific dists to ref.

    Parameters
    ----------
    params: _MapQCParams
        MapQC parameters object.
    sample_dist_to_ref_per_nhood: np.ndarray
        Array of shape (n_samples_r + n_samples_q, n_nhoods, n_cells_q) containing
        the normalized distance of each sample to the reference for each neighborhood.
    nhood_info_df: pd.DataFrame
        DataFrame containing information about each neighborhood, taken from the output
        of process_nhood.

    Returns
    -------
    mapqc_scores: np.ndarray
        Array of shape (n_cells_q,) containing the mapqc scores for each query cell, in
        the order of the query cells in adata_obs.
    cell_filtering_info: np.ndarray
        Array of shape (n_cells_q,) containing the filtering info for each cell, same
        order as mapqc_scores array.
    """
    sample_dist_to_ref_per_nhood_query = sample_dist_to_ref_per_nhood[
        -len(params.samples_q) :, :
    ]  # first rows represent ref samples, last rows query samples
    # create mask to apply to sample_dist_to_ref_per_nhood, such
    # that for each cell, we only keep distances to the reference
    # for its sample, and for the neighborhoods in which it occurred.
    full_mask, nhood_mask, _ = _create_sample_and_nhood_based_cell_mask(
        params=params,
        nhood_info_df=nhood_info_df,
    )
    # apply mask to sample_dist_to_ref_per_nhood
    sample_dist_to_ref_per_nhood_masked = np.where(
        full_mask, sample_dist_to_ref_per_nhood_query[:, :, np.newaxis], np.nan
    )
    # now take the mean for each cell, across all the neighborhoods in which
    # it occurred and in which it passed filtering, to calculate the cell's
    # mean distance to the reference. These are the mapQC scores!
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        mapqc_scores = np.nanmean(sample_dist_to_ref_per_nhood_masked, axis=(0, 1))
    # store filtering information for cells without value (set rest to 'pass')
    cell_filtering_info = _get_per_cell_filtering_info(mapqc_scores, nhood_mask, nhood_info_df)
    # return mapqc scores and filtering info
    return mapqc_scores, cell_filtering_info


def _create_sample_and_nhood_based_cell_mask(
    params: _MapQCParams,
    nhood_info_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create mask set to 1 for each cell at the sample and neighborhood(s) it was part of.

    Returns
    -------
        mask: np.ndarray
            Mask set to 1 for each cell at the sample and neighborhood(s) it was part of.
            Shape: (n_samples_q, n_nhoods, n_cells_q)
        cell_nhood_mask: np.ndarray
            Mask set to 1 for each neighborhood that a cell was part of.
            Shape: (n_nhoods, n_cells_q)
        cell_sample_mask: np.ndarray
            Mask set to 1 for each sample that a cell was part of.
            Shape: (n_samples_q, n_cells_q)
    """
    n_nhoods = nhood_info_df.shape[0]
    n_samples_q = len(params.samples_q)
    query_cells = params.adata.obs.loc[params.adata.obs[params.ref_q_key] == params.q_cat, :].index.values
    n_query_cells = len(query_cells)
    # 1. CREATE NEIGHBORHOOD-BASED MASK
    # start nhood_mask with n columns (ncells) to full number of cells (instead of query only),
    # as we get index numbers of cells in each neighborhood based on indices of the full adata.
    # 1a: set neighborhood-cell pairs to 1 if cell occurred in neighborhood
    # now set values representing the neighborhood(s) that a cell was part of as 1
    # For each entry to set, get row and column indices. Row indices are
    # the neighborhood index:
    cell_nhood_mask_cell_in_nhood = np.zeros((n_nhoods, params.adata.shape[0]), dtype=int)
    row_indices = np.repeat(
        np.arange(n_nhoods),
        [0 if x is None else len(x) for x in nhood_info_df["knn_idc"]],
    )
    # Column indices are the cell indices (as specified in nhood_info_df for every neighborhood):
    col_indices = np.concatenate(
        [
            np.array(knn_idc, dtype=int) if knn_idc is not None else np.array([], dtype=int)
            for knn_idc in nhood_info_df["knn_idc"].values
        ]
    )
    # now set row,col pairs for each cell-neighborhood pair to 1
    cell_nhood_mask_cell_in_nhood[row_indices, col_indices] = 1
    # 1b: set cells to 1 if its sample passed filtering in that neighborhood:
    sample_array = np.array(params.adata.obs[params.sample_key].values, dtype=object)[:, np.newaxis]
    cell_nhood_mask_sample_passed_filter = np.array(
        [
            np.any(sample_array == np.array(sample_list, dtype=object), axis=1)
            for sample_list in nhood_info_df.samples_q.values
        ]
    )
    # 1b alternative, slower but possibly more robust:
    # cell_nhood_mask_sample_passed_filter = np.array([
    #     params.adata.obs[params.sample_key].isin(sample_list).values
    #     for sample_list in nhood_info_df.samples_q.values])
    # 1c: combine the two masks:
    cell_nhood_mask_full = cell_nhood_mask_cell_in_nhood * cell_nhood_mask_sample_passed_filter
    # 1d: subset to query cells only (i.e. exclude reference cells):
    cell_nhood_mask = cell_nhood_mask_full[:, params.adata.obs[params.ref_q_key] == params.q_cat]
    # 2. CREATE SAMPLE-BASED MASK
    # now calculate sample mask, specifying for each query cell from which sample it came
    cell_sample_mask = np.zeros((n_samples_q, n_query_cells), dtype=int)
    # check that samples are sorted alphabetically, otherwise the np.searchsorted function will
    # mess up the results completely:
    if not params.samples_q == sorted(params.samples_q):
        raise ValueError("List of samples samples_q should be in alphabetical order!")
    sample_idc_per_cell = np.searchsorted(params.samples_q, params.adata.obs.loc[query_cells, params.sample_key])
    cell_sample_mask[sample_idc_per_cell, np.arange(n_query_cells)] = 1
    # Now combine the two, creating a mask per per cell, such that only sample-neighborhood
    # pairs that belong to the cell are set to 1 (True), i.e. for each cell, set the values
    # matching with its sample AND the neighborhood(s) in which it occurred to 1. Add axes
    # to enable broadcasting.
    # We want everything to have shape (n_samples, n_nhoods, n_cells)
    mask = cell_sample_mask[:, np.newaxis, :] * cell_nhood_mask[np.newaxis, :, :]
    # return mask
    return mask, cell_nhood_mask, cell_sample_mask


def _get_per_cell_filtering_info(
    mapqc_scores: np.ndarray,
    cell_nhood_mask: np.ndarray,
    nhood_info_df: pd.DataFrame,
) -> np.ndarray:
    """Extract filtering info for each cell based on its neighborhood(s)."""
    cell_filtering_info = np.full_like(
        mapqc_scores, fill_value=np.nan
    )  # note that np.nan will be converted to string automatically below
    cell_filtering_info = np.where(cell_nhood_mask.sum(axis=0) == 0, "not sampled", cell_filtering_info)
    cell_filtering_info[~np.isnan(mapqc_scores)] = "pass"
    # collect filtering info for all of each cell's neighborhoods:
    # for all entries that have 1 at neighborhood, fill in the neighborhood
    # filtering info. Otherwise, fill in None
    per_cell_per_nhood_filter = pd.DataFrame(
        np.where(cell_nhood_mask, nhood_info_df.filter_info.values[:, np.newaxis], None)
    )  # work with dataframe here because numpy operations along axis create weird string artifacts
    # Collect the most prevalent filtering outcome for each cell:
    per_cell_filter_most_prevalent = per_cell_per_nhood_filter.mode().iloc[
        0
    ]  # note that in case of ties, it orders alphabetically
    # now get filtering info for cells that did not pass filtering, but were sampled:
    cell_filtering_info = np.where(
        cell_filtering_info == "nan",
        per_cell_filter_most_prevalent,
        cell_filtering_info,
    )
    return cell_filtering_info
