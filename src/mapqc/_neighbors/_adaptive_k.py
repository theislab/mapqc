import pandas as pd

from mapqc._neighbors._filter import _filter_and_get_min_k_query, _filter_and_get_min_k_ref
from mapqc._params import _MapQCParams


def _filter_and_get_adaptive_k(
    params: _MapQCParams,
    cell_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> tuple[bool, int | None, str]:
    """Check if neighborhood fulfills filtering conditions and return (adapted) k.

    Parameters
    ----------
    params: _MapQCParams
        MapQC parameters object.
    cell_df : pd.DataFrame
        Dataframe with cells ordered by distance to the center cell, as well as information
        about sample, and reference/query category.
    sample_df : pd.DataFrame
        Dataframe with sample information of each sample's study. Order does not matter.
        Should include ref_q column, and study column, the latter only if exclude_same_study is True.

    Returns
    -------
    (filter_pass_ref, min_k_out_ref, filter_info_ref) : tuple[bool, int | None, str]
        Tuple of: 1) boolean specifying whether the neighborhood passed all conditions
        for both the reference and query; 2) an integer with the minimum number of neighbors
        if filter was passed, otherwise None; 3) a string with the reason for the filter
        pass/fail.
    """
    filter_pass_query, min_k_out_query, filter_info_query = _filter_and_get_min_k_query(
        params=params,
        cell_df=cell_df,
    )

    if not filter_pass_query:
        return (filter_pass_query, min_k_out_query, filter_info_query)

    filter_pass_ref, min_k_out_ref, filter_info_ref = _filter_and_get_min_k_ref(
        params=params,
        cell_df=cell_df,
        k_min_query=min_k_out_query,
        sample_df=sample_df,
    )

    return (filter_pass_ref, min_k_out_ref, filter_info_ref)
