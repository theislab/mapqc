import numpy as np
import pandas as pd

from mapqc._params import _MapQCParams


def _filter_and_get_min_k_query(
    params: _MapQCParams,
    cell_df: pd.DataFrame,
) -> tuple[bool, int | None, str]:
    """Checks if neighborhood has enough query cells for any query sample.

    Parameters
    ----------
    params: _MapQCParams
        MapQC parameters object.
    cell_df : pd.DataFrame
        Dataframe with cells ordered by distance to the center cell, as well as information
        about sample, and reference/query category..

    Returns
    -------
    (filter_pass, min_k_out, filter_info) : tuple[bool, int | None, str]
        Tuple of: 1) boolean specifying whether the neighborhood passed the min_n_cells
        filter for at least one query sample, 2) an integer with the minimum number of neighbors
        if filter was passed, otherwise None, and 3) a string with the reason for the filter
        pass/fail.
    """
    # Set all reference cells to None
    samples_q = [
        s if r_q == params.q_cat else None
        for s, r_q in zip(cell_df[params.sample_key], cell_df[params.ref_q_key], strict=False)
    ]
    # Get nth occurrence of each query sample
    sample_nth_occ_idc = _get_idc_nth_instances(pd.Categorical(samples_q), params.min_n_cells)
    # if no sample has enough cells:
    if len(sample_nth_occ_idc) == 0:
        return (False, None, "not enough query cells")
    # Get lowest index (i.e. first query sample) that suffices the min_n_cells condition
    # Note that we add a +1 because of python indexing: 1st instance means
    # second observation, i.e. k=2 neighbors needed
    min_k_query = min(sample_nth_occ_idc.values) + 1  #
    if min_k_query <= params.k_min:
        return (True, params.k_min, "pass")
    else:
        # note that if we adapt the k, we add a margin to not limit the neighborhood
        # size EXACTLY at the point where we fulfill conditions, as I think this
        # could create some weird biases.
        min_k_incl_margin = min_k_query + min_k_query * params.adaptive_k_margin
        return (
            True,
            np.ceil(min_k_incl_margin).astype(int),
            "pass",
        )  # round up and make into integer


def _filter_and_get_min_k_ref(
    params: _MapQCParams,
    cell_df: pd.DataFrame,
    k_min_query: int,
    sample_df: pd.DataFrame = None,
) -> tuple[bool, int | None, str]:
    """Checks if neighborhood has enough reference samples with enough cells.

    Parameters
    ----------
    params: _MapQCParams
        MapQC parameters object.
    cell_df : pd.DataFrame
        Dataframe with cells ordered by distance to the center cell, as well as information
        about sample, and reference/query category. Note that cell_df should be of row size k_min
        if no adaptive k is used, and of size (max_k / (1 + adaptive_k_margin)) if adaptive k
        is used.
    k_min_query : int
        Minimum number of neighbors based on already performed query-based filtering.
    sample_df : pd.DataFrame, optional
        Dataframe with sample information of each sample's study. Order does not matter.
        Only needed if exclude_same_study is True.

    Returns
    -------
    (filter_pass, min_k_out, filter_info) : tuple[bool, int | None, str]
        Tuple of: 1) boolean specifying whether the neighborhood passed the min_n_cells
        and min_n_samples_r conditions, 2) an integer with the minimum number of neighbors
        if filter was passed, otherwise None, and 3) a string with the reason for the filter
        pass/fail.
    """
    # Get the sample for each cell, while setting query cells to None
    cell_samples_r = [
        s if r_q == params.r_cat else None
        for s, r_q in zip(cell_df[params.sample_key], cell_df[params.ref_q_key], strict=False)
    ]
    # Convert to categorical for easier processing
    cell_samples_r = pd.Categorical(cell_samples_r)
    # if no cells from the reference, filter is not passed
    if pd.isnull(cell_samples_r).all():
        return (False, None, "not enough reference samples")
    # get nth occurrence of each reference sample
    # (Samples are now ordered by closest to the center cell
    # (based on nth occurrence) to furthest, samples with
    # too few cells are excluded)
    sample_nth_occ_idc = _get_idc_nth_instances(pd.Categorical(cell_samples_r), params.min_n_cells)
    if not params.exclude_same_study:
        # if fewer than min_n_samples_r have at least min_n_cells, filter is not passed
        if len(sample_nth_occ_idc) < params.min_n_samples_r:
            return (False, None, "not enough reference samples")
        else:
            # subtract 1 from min_n_samples_r because of python indexing
            # (if we want 5 samples included, we index with [4])
            # add 1 also because of python indexing: if our [k-1]th
            # index is the index at which our condition is fulfilled,
            # we want k neighbors
            min_k_ref = sample_nth_occ_idc.values[params.min_n_samples_r - 1] + 1
            if min_k_ref <= k_min_query:
                return (True, k_min_query, "pass")
            else:
                min_k_incl_margin = min_k_ref + min_k_ref * params.adaptive_k_margin
                return (True, np.ceil(min_k_incl_margin).astype(int), "pass")
    else:
        # if we want to exclude same-study pairs
        # order the sample studies by the order at which their nth cell
        # occurs in the neighborhood (note that these are ordered,
        # and that samples with too few cells have already been excluded)
        samples_r = sample_nth_occ_idc.index
        # if the total of reference samples does not even satisfy min_n_samples_r,
        # we can already stop here:
        if len(samples_r) < params.min_n_samples_r:
            return (False, None, "not enough reference samples")
        # get the study for each reference sample
        sample_studies_r = sample_df.loc[samples_r, params.study_key]
        # a minimum number of reference samples can be seen as a minimum
        # number of pairwise comparisons. If min_n_samples_r is n, then
        # we expect n**2 comparisons, or (n**2) - n comparisons, if we
        # exclude comparisons-to-self (the diagonal). Moreover, if we
        # exclude same pairs (row x compared to column y, row y compared
        # to column x), we expect ((n**2) - n) / 2 pairs.
        # We can now calculate the minimum number of pairs, excluding pairs
        # from the same study, and see if we have enough pairs left
        # Note that if min_n_samples_r is 1, we set min_n_pairs to 1 manually,
        # as the calculation would result in a 0
        if params.min_n_samples_r == 1:
            min_n_pairs = 1
        else:
            min_n_pairs = ((params.min_n_samples_r**2) - params.min_n_samples_r) / 2
        # create a matrix specifying for our samples if they come
        # from the same study or not
        diff_study = _create_difference_matrix(sample_studies_r)
        # check, starting at the min_n_samples_r-th sample, how many
        # pairs are left after filtering out same-study pairs
        # use a mask to get only the upper triangular part of the matrix
        # (i.e. don't include pairs twice, the matrix is symmetric, and
        # don't include pairs of the same (i,i), i.e. the diagonal)
        mask = np.triu(np.ones_like(diff_study), k=1)
        # if we do not want to adapt k, we just check the total number of pairs:
        if not params.adapt_k:
            n_valid_pairs = (diff_study * mask).sum()
            if n_valid_pairs < min_n_pairs:
                return (
                    False,
                    None,
                    "not enough reference samples from different studies",
                )
            else:
                return (True, k_min_query, "pass")
        # if we do want to adapt k, we want to check at which point we have a large
        # enough neighborhood (i.e. at which number of samples)
        # We here calculate the number of valid pairs, increasing the number of
        # samples by one each time.
        n_valid_pair_cumsum_per_sample = np.cumsum((diff_study * mask).sum(axis=0))
        # check at which point (at how many samples) we have enough pairs:
        passing_samples_idc = np.where(n_valid_pair_cumsum_per_sample >= min_n_pairs)[0]
        # if for now sample we have enough pairs, the filter is not passed:
        if len(passing_samples_idc) == 0:
            return (False, None, "not enough reference samples from different studies")
        else:
            # get the minimum number of neighbors that satisfies the filter
            first_passing_sample = samples_r[passing_samples_idc[0]]
            # check which k comes with that sample (i.e. at which index that
            # specific sample has its min_n_cells-th occurrence)
            # we add one, as [k-1]th index means k cells
            min_k_ref = sample_nth_occ_idc.loc[first_passing_sample] + 1
            min_k_ref_incl_margin = min_k_ref + min_k_ref * params.adaptive_k_margin
            # if that k is smaller than k_min_query, we return k_min_query:
            if min_k_ref_incl_margin <= k_min_query:
                return (True, k_min_query, "pass")
            # otherwise, we take the needed k plus an added margin:
            else:
                return (True, np.ceil(min_k_ref_incl_margin).astype(int), "pass")


def _get_idc_nth_instances(seq: pd.Categorical, n: int) -> pd.Series:
    """Get indices of the nth occurrence of each category in a sequence.

    For each unique value (category) in a sequence, get the index at which
    the category occurs for the nth time. If a category has fewer than n
    instances, it is excluded from the output.

    Parameters
    ----------
    seq: pd.Categorical
        The sequence to get the indices from.
    n: int
        The nth instance to get the index of.

    Returns
    -------
    idc: pd.Series
        Sorted series (from small to large) with categories as index, indices (in
        the form of integers) of their nth occurrence as values. Categories with
        fewer than n occurrences are excluded.
    """
    cats = seq.categories
    idc = pd.Series(index=cats, data=np.nan)
    for cat in cats:
        cat_positions = np.where(seq == cat)[0]
        if len(cat_positions) >= n:
            idc.loc[cat] = int(cat_positions[n - 1])  # -1 because of python indexing: nth instance is at position n-1
    idc = idc.dropna().sort_values(ascending=True)
    return idc.astype(int)


def _create_difference_matrix(lst):
    """Create a 2D array specifying differentness for each combination of list elements.

    Parameters
    ----------
    lst : list
        List of elements.

    Returns
    -------
    np.ndarray
        A 2D array where the element at position (i, j) is 1 if the i-th
        and j-th elements are different, and 0 otherwise.
    """
    arr = np.array(lst)
    return (arr[:, None] != arr).astype(int)
