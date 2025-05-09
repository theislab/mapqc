"""
Main entry point for the mapqc package.

This module provides the primary function that users will interact with.
"""

from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc

from mapqc.center_cells.sampling import sample_center_cells_by_group
from mapqc.distances.normalized_distances import get_normalized_dists_to_ref
from mapqc.mapqc_scores import calculate_mapqc_scores
from mapqc.params import MapQCParams
from mapqc.process_nhood import process_neighborhood
from mapqc.utils.validation import validate_input_params


def run_mapqc(
    adata: sc.AnnData,
    adata_emb_loc: str,
    ref_q_key: str,
    q_cat: str,
    r_cat: str,
    sample_key: str,
    n_nhoods: int,
    k_min: int,
    k_max: int,
    min_n_cells: int,
    min_n_samples_r: int,
    study_key: str = None,
    exclude_same_study: bool = True,
    grouping_key: str = None,
    distance_metric: Literal["energy_distance", "pairwise_euclidean"] = "energy_distance",
    seed: int = None,
    overwrite: bool = False,
):
    """
    Run mapqc on an AnnData object.

    This function modifies the input AnnData object in-place by adding two new columns to adata.obs:
    - 'mapqc_score': Contains the mapqc scores for query cells (NaN for reference cells)
    - 'mapqc_filtering': Contains filtering information for query cells (None for reference cells)
    It also adds a dictionary including the input parameter values to adata.uns['mapqc_params']

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object including both the reference and the query cells.
        This object will be modified in-place.
    adata_emb_loc : str
        The location of the embeddings in adata.obsm or "X" if the embedding is in adata.X
    ref_q_key : str
        Key in adata.obs that contains reference/query labels
    q_cat : str
        Category label for query samples
    r_cat : str
        Category label for reference samples
    sample_key : str
        Key in adata.obs that contains sample identifiers
    n_nhoods : int
        Number of neighborhoods to analyze
    k_min : int
        Minimum number of cells per neighborhood
    k_max : int
        Maximum number of cells per neighborhood, if the neighborhood of size k_min does not fulfill
        filtering criteria.
    min_n_cells : int
        Minimum number of cells required per sample, in a neighborhood
    min_n_samples_r : int
        Minimum number of reference samples (with at least min_n_cells cells) required per neighborhood
    exclude_same_study : bool = True
        Whether to exclude samples from the same study when calculating distances
        between reference samples. To prevent bias in inter-sample distances within
        the reference, we recommend excluding inter-sample distances between samples
        from the same study, i.e. setting this argument to True.
    study_key : str = None
        Key in adata.obs that contains study identifiers (needed if exclude_same_study is True)
    grouping_key : str = None
        Key in adata.obs that contains grouping information, which will be used to sample
        center cells (i.e. the centers of neighborhoods). If not provided, center cells will
        be sampled randomly from the query. If provided, center cells will be sampled based
        on query and reference cell proportions per group of the grouping key. This can be
        set to e.g. a clustering performed on the joint reference and query, or a (preliminary)
        cell type annotation of reference and query.
    distance_metric: Literal["energy_distance", "pairwise_euclidean"] = "energy_distance"
        Distance metric to use to calculate distances between samples (i.e. between
        two sets of cells).
    seed: int = None
        Seed for random number generator. Set the seed to ensure reproducibility of results.
    overwrite: bool = False
        Whether to overwrite existing mapqc_score and mapqc_filtering columns in adata.obs.

    Returns
    -------
    None
        This function modifies the input AnnData object in-place by adding 'mapqc_score' and
        'mapqc_filtering' columns to adata.obs. It furthermore adds a dictionary including the
        input parameter values to adata.uns['mapqc_params']. No values are returned.
    """
    # Create parameter object for internal use
    params = MapQCParams(
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
    )

    # validate input
    validate_input_params(params)
    # now prepare run
    center_cells = sample_center_cells_by_group(
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

    for i, cell in enumerate(center_cells):
        nhood_info.loc[cell], dists[:, :, i] = process_neighborhood(params=params, center_cell=cell)
        nhood_info.loc[cell, "nhood_number"] = i

    dists_to_ref = get_normalized_dists_to_ref(params, dists)
    mapqc_scores, filtering_info_per_cell = calculate_mapqc_scores(
        params=params,
        sample_dist_to_ref_per_nhood=dists_to_ref,
        nhood_info_df=nhood_info,
    )
    # modify input adata object, adding mapqc scores and filtering info to adata.obs,
    # and adding parameters to adata.uns
    params.adata.obs["mapqc_score"] = np.nan
    params.adata.obs.loc[params.adata.obs[params.ref_q_key] == params.q_cat, "mapqc_score"] = mapqc_scores
    params.adata.obs["mapqc_filtering"] = None
    params.adata.obs.loc[params.adata.obs[params.ref_q_key] == params.q_cat, "mapqc_filtering"] = (
        filtering_info_per_cell
    )
    params_to_leave_out = ["adata", "overwrite", "samples_r", "samples_q"]
    params.adata.uns["mapqc_params"] = {k: v for k, v in params.__dict__.items() if k not in params_to_leave_out}
    return nhood_info
