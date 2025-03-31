import numpy as np
import pandas as pd


def check_n_ref_samples(
    sample_set: list,
    sample_info: pd.DataFrame,
    r_q_key: str,
    r_cat: str,
    min_n_samples_r: int,
    exclude_same_study: bool,
    study_key: str = None,
) -> tuple[bool, str]:
    """Check if sufficient reference samples are present in a given sample set.

    Checks, for a given set of samples (present in a specific neighborhood with
    sufficient n_cells), if it includes enough reference samples given the
    pre-set filtering conditions. If samples from the same study are not allowed,
    these are excluded from the count.

    Parameters
    ----------
    sample_set : list
        List of sample IDs present in a specific neighborhood with sufficient n_cells.
    sample_info : pd.DataFrame
        DataFrame with sample information of each sample's study, and reference/query
        origin.
    r_q_key : str
        Key in sample_info that indicates the reference/query category of each sample.
    r_cat : str
        Reference category in r_q_key column.
    min_n_samples_r : int
        Minimum number of reference samples required per neighborhood.
    exclude_same_study : bool
        If True, exclude samples from the same study when counting the number of
        reference samples.
    study_key : str, optional
        Key in sample_info that indicates the study of each sample. Only needed if
        exclude_same_study is True.

    Returns
    -------
    tuple[bool, str]
        A tuple with a boolean indicating if the number of reference samples is sufficient,
        (i.e. if the neighborhood passed the filtering), and a string with the reason for
        the result.
    """
    # get study of each reference sample
    sample_set_r = [s for s in sample_set if sample_info.loc[s, r_q_key] == r_cat]
    if len(sample_set_r) < min_n_samples_r:
        return (False, "too few reference samples")
    elif not exclude_same_study:
        # if enough reference samples, and study exclusion not required:
        return (True, "pass")
    else:
        # if we want to exclude same-study pairs:
        sample_studies_r = sample_info.loc[sample_set_r, study_key]
        # a minimum number of reference samples can be seen as a minimum
        # number of pairwise comparisons. If min_n_samples_r is n, then
        # we expect n**2 comparisons, or (n**2) - n comparisons, if we
        # exclude comparisons-to-self (the diagonal). Moreover, if we
        # exclude same pairs (row x compared to column y, row y compared
        # to column x), we expect ((n**2) - n) / 2 pairs.
        # We can now calculate the minimum number of pairs, excluding pairs
        # from the same study, and see if we have enough pairs left
        min_n_pairs = ((min_n_samples_r**2) - min_n_samples_r) / 2
        # create a matrix specifying for our samples if they come
        # from the same study or not
        diff_study = _create_difference_matrix(sample_studies_r)
        n_valid_pairs = diff_study.sum() / 2  # divide by 2 because we count each pair twice
        if n_valid_pairs < min_n_pairs:
            return (False, "too few reference samples from different batches")
        else:
            return (True, "pass")


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
