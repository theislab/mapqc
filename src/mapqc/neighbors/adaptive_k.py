import numpy as np
import pandas as pd


def get_idc_nth_instances(seq: pd.Categorical, n: int) -> pd.Series:
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
