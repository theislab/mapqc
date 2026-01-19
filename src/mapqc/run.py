"""
Main entry point for the mapqc package.

This module provides the primary function that users will interact with.
"""

from typing import Literal

import anndata
import numpy as np
import pandas as pd
from tqdm import tqdm

from mapqc._center_cells._sampling import _sample_center_cells_by_group
from mapqc._distances._normalized_distances import _get_normalized_dists_to_ref
from mapqc._mapqc_scores import _calculate_mapqc_scores
from mapqc._params import _MapQCParams
from mapqc._process_nhood import _process_neighborhood
from mapqc._utils._validation import _validate_input_params, _validate_return_parameters


def run_mapqc(
    adata: anndata.AnnData,
    adata_emb_loc: str,
    ref_q_key: str,
    q_cat: str,
    r_cat: str,
    sample_key: str,
    n_nhoods: int,
    k_min: int,
    k_max: int,
    min_n_cells: int = 10,
    min_n_samples_r: int = 3,
    study_key: str | None = None,
    exclude_same_study: bool = True,
    grouping_key: str | None = None,
    distance_metric: Literal["energy_distance", "pairwise_euclidean"] = "energy_distance",
    seed: int | None = None,
    overwrite: bool = False,
    return_nhood_info_df: bool = False,
    return_sample_dists_to_ref_df: bool = False,
    verbose: bool = True,
) -> None | pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate mapQC scores.

    This function modifies the input AnnData object in-place by adding several new columns to adata.obs:

    * 'mapqc_score': Contains the mapqc scores for query cells (NaN for reference cells)
    * 'mapqc_filtering': Contains filtering information for query cells (None for reference cells)
    * 'mapqc_nhood_filtering': Contains filtering information for each neighborhood
    * 'mapqc_nhood_number': Contains the number of the neighborhood
    * 'mapqc_k': Contains the size of the neighborhood

    It also adds a dictionary including the input parameter values to adata.uns['mapqc_params']

    Finally, if return_nhood_info_df is True, the function will return a pandas DataFrame containing
    the neighborhood information, and if return_sample_dists_to_ref_df is True, the function will return
    a pandas DataFrame containing the sample distances to reference for each neighborhood.

    Parameters
    ----------
    adata
        The AnnData object including both the reference and the query cells.
        This object will be modified in-place. Important! The AnnData object should include
        *only* controls for the reference, and should include some controls for the query.
    adata_emb_loc
        The location of the embeddings in adata.obsm or "X" if the embedding is in adata.X
    ref_q_key
        Key in adata.obs that contains reference/query labels
    q_cat
        Category label for query samples
    r_cat
        Category label for reference samples
    sample_key
        Key in adata.obs that contains sample identifiers
    n_nhoods
        Number of neighborhoods to analyze
    k_min
        Minimum number of cells per neighborhood
    k_max
        Maximum number of cells per neighborhood, if the neighborhood of size k_min does not fulfill
        filtering criteria.
    min_n_cells
        Minimum number of cells required per sample, in a neighborhood. Default is 10.
    min_n_samples_r
        Minimum number of reference samples (with at least min_n_cells cells) required per neighborhood. Default is 3.
    exclude_same_study
        Whether to exclude samples from the same study when calculating distances
        between reference samples. To prevent bias in inter-sample distances within
        the reference, we recommend excluding inter-sample distances between samples
        from the same study, i.e. setting this argument to True. Default is True.
    study_key
        Key in adata.obs that contains study identifiers (needed if exclude_same_study is True)
    grouping_key
        Key in adata.obs that contains grouping information, which will be used to sample
        center cells (i.e. the centers of neighborhoods). If not provided, center cells will
        be sampled randomly from the query. If provided, center cells will be sampled based
        on query and reference cell proportions per group of the grouping key. This can be
        set to e.g. a clustering performed on the joint reference and query, or a (preliminary)
        cell type annotation of reference and query.
    distance_metric
        Distance metric to use to calculate distances between samples (i.e. between
        two sets of cells). Default is "energy_distance".
    seed
        Seed for random number generator. Set the seed to ensure reproducibility of results.
    overwrite
        Whether to overwrite existing mapqc_score and mapqc_filtering columns in adata.obs. Default is False.
    return_nhood_info_df
        Whether to return a pandas DataFrame containing detailed neighborhood information.
        This can be useful for debugging, or for getting a detailed understanding of your
        neighborhoods and the mapqc output. Default is False.
    return_sample_dists_to_ref_df
        Whether to return a pandas DataFrame containing the sample distances to reference for each neighborhood. Default is False.
    verbose
        Whether to print progress messages. Default is True.

    Returns
    -------
    None or pd.DataFrame or tuple
        This function modifies the input AnnData object in-place by adding:

        * 'mapqc_score'
        * 'mapqc_filtering'
        * 'mapqc_nhood_filtering'
        * 'mapqc_nhood_number'
        * 'mapqc_k'

        columns to adata.obs. It furthermore adds a dictionary including the input parameter
        values to adata.uns['mapqc_params'].

        The return value depends on the input parameters:

        * If return_nhood_info_df is True, returns a pandas DataFrame containing detailed neighborhood information.
        * If return_sample_dists_to_ref_df is True, returns a pandas DataFrame containing the sample distances to reference.
        * If both are True, returns a tuple of (nhood_info_df, sample_dists_to_ref_df).
        * If neither is True, returns None.
    """
    # Create parameter object for internal use
    params = _MapQCParams(
        adata=adata,
        adata_emb_loc=adata_emb_loc,
        ref_q_key=ref_q_key,
        q_cat=q_cat,
        r_cat=r_cat,
        sample_key=sample_key,
        n_nhoods=n_nhoods,
        k_min=k_min,
        k_max=k_max,
        min_n_cells=min_n_cells,
        min_n_samples_r=min_n_samples_r,
        exclude_same_study=exclude_same_study,
        study_key=study_key,
        grouping_key=grouping_key,
        adaptive_k_margin=0.1,  # default value
        distance_metric=distance_metric,
        seed=seed,
        overwrite=overwrite,
        verbose=verbose,
    )
    # validate input
    _validate_input_params(params)
    # validate return parameters:
    _validate_return_parameters(return_nhood_info_df, return_sample_dists_to_ref_df)
    # now prepare run
    if grouping_key is None:
        # randomly sample center cells:
        query_cells = adata.obs_names[adata.obs[ref_q_key] == q_cat]
        center_cells = list(np.random.choice(query_cells, size=params.n_nhoods, replace=False))
    else:
        center_cells = _sample_center_cells_by_group(
            params=params,
        )

    samples_r = sorted(
        params.adata.obs.loc[params.adata.obs[params.ref_q_key] == params.r_cat, params.sample_key].unique().tolist()
    )
    samples_q = sorted(
        params.adata.obs.loc[params.adata.obs[params.ref_q_key] == params.q_cat, params.sample_key].unique().tolist()
    )
    params.samples_r = samples_r
    params.samples_q = samples_q

    nhood_info = pd.DataFrame(columns=["nhood_number", "filter_info", "k", "knn_idc", "samples_q"])
    dists = np.full(
        shape=(len(samples_r), len(samples_r) + len(samples_q), len(center_cells)),
        fill_value=np.nan,
    )

    # Create progress iterator if verbose is True
    center_cells_iter = (
        tqdm(
            center_cells,
            desc="Processing neighborhoods",
            ncols=80,  # Total width of the progress bar
            bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt}",  # Shorter bar format
        )
        if verbose
        else center_cells
    )
    for i, cell in enumerate(center_cells_iter):
        nhood_info.loc[cell], dists[:, :, i] = _process_neighborhood(params=params, center_cell=cell)
        nhood_info.loc[cell, "nhood_number"] = i

    dists_to_ref = _get_normalized_dists_to_ref(params, dists)
    mapqc_scores, filtering_info_per_cell = _calculate_mapqc_scores(
        params=params,
        sample_dist_to_ref_per_nhood=dists_to_ref,
        nhood_info_df=nhood_info,
    )
    # modify input adata object:
    # 1. add mapqc scores and filtering info to adata.obs:
    params.adata.obs["mapqc_score"] = np.nan
    params.adata.obs.loc[params.adata.obs[params.ref_q_key] == params.q_cat, "mapqc_score"] = mapqc_scores
    params.adata.obs["mapqc_filtering"] = None
    params.adata.obs.loc[params.adata.obs[params.ref_q_key] == params.q_cat, "mapqc_filtering"] = (
        filtering_info_per_cell
    )
    # 2. add neighborhood info to adata.obs:
    nhood_info_cols_to_copy = ["nhood_number", "filter_info", "k"]
    for col in nhood_info_cols_to_copy:
        if col == "filter_info":
            col_name = "mapqc_nhood_filtering"
            params.adata.obs[col_name] = None
            params.adata.obs.loc[nhood_info.index, col_name] = nhood_info[col]
        else:
            col_name = f"mapqc_{col}"
            params.adata.obs[col_name] = pd.Series(np.nan, index=params.adata.obs.index, dtype="Int64")
            params.adata.obs.loc[nhood_info.index, col_name] = nhood_info[col]
    # 3. add parameters to adata.uns:
    params_to_leave_out = [
        "adata",
        "overwrite",
        "samples_r",
        "samples_q",
        "return_nhood_info_df",
        "verbose",
    ]
    params.adata.uns["mapqc_params"] = {k: v for k, v in params.__dict__.items() if k not in params_to_leave_out}

    if return_nhood_info_df:
        # change knn_idc to knn_barcodes, that way we can still use the df even when adata order has changed or adata has been subsetted
        nhood_info["knn_barcodes"] = nhood_info["knn_idc"].apply(lambda x: params.adata.obs.index[x].tolist())
        del nhood_info["knn_idc"]
    if return_sample_dists_to_ref_df:
        sample_dists_to_ref_df = pd.DataFrame(
            dists_to_ref,
            index=params.samples_r + params.samples_q,
            columns=nhood_info.index,
        )
        if return_nhood_info_df:
            return nhood_info, sample_dists_to_ref_df
        else:
            return sample_dists_to_ref_df
    elif return_nhood_info_df:
        return nhood_info
