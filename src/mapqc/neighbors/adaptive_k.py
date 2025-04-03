import pandas as pd

from mapqc.neighbors.filter import filter_and_get_min_k_query, filter_and_get_min_k_ref


def filter_and_get_adaptive_k(
    cell_df: pd.DataFrame,
    ref_q_key: str,
    sample_key: str,
    sample_df: pd.DataFrame,
    q_cat: str,
    r_cat: str,
    min_n_cells: int,
    min_n_samples_r: int,
    k_min: int,
    adapt_k: bool,
    exclude_same_study: bool,
    adaptive_k_margin: float = None,
    study_key: str = None,
) -> tuple[bool, int | None, str]:
    """Check if neighborhood fulfills filtering conditions and return (adapted) k.

    Parameters
    ----------
    cell_df : pd.DataFrame
        Dataframe with cells ordered by distance to the center cell, as well as information
        about sample, and reference/query category.
    ref_q_key : str
        Key in cell_df that indicates the reference/query category of each cell.
    sample_key : str
        Key in cell_df that indicates each cell's sample identifier.
    sample_df : pd.DataFrame
        Dataframe with sample information of each sample's study. Order does not matter.
        Should include ref_q column, and study column, the latter only if exclude_same_study is True.
    q_cat : str
        Query category in cell_df's ref_q_key column.
    r_cat : str
        Reference category in cell_df's ref_q_key column.
    min_n_cells : int
        Minimum number of cells per sample.
    min_n_samples_r : int
        Minimum number of reference samples per neighborhood (counting only samples
        with at least min_n_cells).
    k_min : int
        Minimum number of neighbors for a neighborhood.
    adapt_k : bool
        Whether to adapt the number of neighbors if filtering conditions are not met.
    exclude_same_study : bool
        Whether to exclude same-study pairs when counting the number of reference
        samples (see code for details on how number of pairs is used to check
        if filtering conditions are met.)
    adaptive_k_margin : float, optional
        Margin to add (as fraction) to the minimum number of neighbors at which
        all filter conditions are met, if this number if larger than k_min. Note
        that this argument has to be set only if adapt_k is True.
    study_key : str, optional
        Key in cell_df that indicates the study of each cell's sample. Only needed if
        exclude_same_study is True.

    Returns
    -------
    (filter_pass_ref, min_k_out_ref, filter_info_ref) : tuple[bool, int | None, str]
        Tuple of: 1) boolean specifying whether the neighborhood passed all conditions
        for both the reference and query; 2) an integer with the minimum number of neighbors
        if filter was passed, otherwise None; 3) a string with the reason for the filter
        pass/fail.
    """
    # if exclude_same_study:
    #     sample_df = cell_df.groupby(sample_key).agg({ref_q_key: "first", study_key: "first"})
    # else:
    #     sample_df = cell_df.groupby(sample_key).agg({ref_q_key: "first"})
    filter_pass_query, min_k_out_query, filter_info_query = filter_and_get_min_k_query(
        cell_df=cell_df,
        ref_q_key=ref_q_key,
        sample_key=sample_key,
        q_cat=q_cat,
        k_min=k_min,
        adaptive_k_margin=adaptive_k_margin,
        min_n_cells=min_n_cells,
    )

    if not filter_pass_query:
        return (filter_pass_query, min_k_out_query, filter_info_query)

    filter_pass_ref, min_k_out_ref, filter_info_ref = filter_and_get_min_k_ref(
        cell_df=cell_df,
        ref_q_key=ref_q_key,
        sample_key=sample_key,
        r_cat=r_cat,
        k_min=min_k_out_query,
        min_n_cells=min_n_cells,
        min_n_samples_r=min_n_samples_r,
        adapt_k=adapt_k,
        exclude_same_study=exclude_same_study,
        sample_df=sample_df,
        adaptive_k_margin=adaptive_k_margin,
        study_key=study_key,
    )

    return (filter_pass_ref, min_k_out_ref, filter_info_ref)
