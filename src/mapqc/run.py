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
    )

    # validate input
    validate_input_params(params)

    # now prepare run
    center_cells = sample_center_cells_by_group(
        adata_obs=params.adata.obs,
        ref_q_key=params.ref_q_key,
        q_cat=params.q_cat,
        grouping_cat=params.grouping_key,
        n_cells=params.n_nhoods,
        seed=42,
    )

    samples_r = sorted(
        params.adata.obs.loc[params.adata.obs[params.ref_q_key] == params.r_cat, params.sample_key].unique().tolist()
    )
    samples_q = sorted(
        params.adata.obs.loc[params.adata.obs[params.ref_q_key] == params.q_cat, params.sample_key].unique().tolist()
    )

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
        nhood_info.loc[cell], dists[:, :, i] = process_neighborhood(
            center_cell=cell,
            adata_emb=params.adata.X if params.adata_emb_loc == "X" else params.adata.obsm[params.adata_emb_loc],
            adata_obs=params.adata.obs,
            samples_r_all=samples_r,
            samples_q_all=samples_q,
            k_min=params.k_min,
            k_max=params.k_max,
            sample_key=params.sample_key,
            ref_q_key=params.ref_q_key,
            q_cat=params.q_cat,
            r_cat=params.r_cat,
            min_n_cells=params.min_n_cells,
            min_n_samples_r=params.min_n_samples_r,
            exclude_same_study=params.exclude_same_study,
            adaptive_k_margin=params.adaptive_k_margin,
            study_key=params.study_key,
            distance_metric=params.distance_metric,
        )
        nhood_info.loc[cell, "nhood_number"] = i

    dists_to_ref = get_normalized_dists_to_ref(dists, samples_r)
    mapqc_scores, filtering_info_per_cell = calculate_mapqc_scores(
        dists_to_ref,
        nhood_info,
        params.adata.obs,
        params.ref_q_key,
        params.q_cat,
        params.sample_key,
        samples_q,
    )
    return mapqc_scores, filtering_info_per_cell
