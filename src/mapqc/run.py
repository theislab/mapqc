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
    adaptive_k_margin: float = 0.1,
    distance_metric: Literal["energy_distance", "pairwise_euclidean"] = "energy_distance",
    seed: int = None,
):
    """
    Run mapqc on an AnnData object.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object including both the reference and the query cells.
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
        Maximum number of cells per neighborhood
    min_n_cells : int
        Minimum number of cells required per sample, in a neighborhood
    min_n_samples_r : int
        Minimum number of reference samples (with at least min_n_cells cells) required per neighborhood
    exclude_same_study : bool = True
        Whether to exclude samples from the same study when calculating distances
        between reference samples.
    study_key : str = None
        Key in adata.obs that contains study identifiers
    grouping_key : str = None
        Key in adata.obs that contains grouping information
    adaptive_k_margin : float = 0.1
        Margin for adaptive k selection
    distance_metric: Literal["energy_distance", "pairwise_euclidean"] = "energy_distance"
        Distance metric to use to calculate distances between samples (i.e. between
        two sets of cells).
    seed: int = None
        Seed for random number generator.

    Returns
    -------
    tuple
        (mapqc_scores, filtering_info_per_cell)
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
        adaptive_k_margin=adaptive_k_margin,
        distance_metric=distance_metric,
        seed=seed,
    )

    # validate input
    validate_input_params(params)

    # set adapt_k parameter:
    if params.k_max > params.k_min:
        params.adapt_k = True
    else:
        params.adapt_k = False

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

    nhood_info = pd.DataFrame(
        columns=[
            "nhood_number",
            "filter_info",
            "k",
            "knn_idc",
        ]
    )
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
        dists_to_ref=dists_to_ref,
        nhood_info=nhood_info,
    )
    return mapqc_scores, filtering_info_per_cell
