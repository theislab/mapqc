"""Parameter definitions for the mapqc package."""

from dataclasses import dataclass
from typing import Literal

import scanpy as sc


@dataclass
class MapQCParams:
    """
    Parameters for running MAPQC analysis.

    Attributes
    ----------
    adata : sc.AnnData
        The main AnnData object containing the data
    adata_emb_loc : str
        The location of the embedding in the adata.obsm or "X" if the embedding is in adata.X
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
        Minimum number of neighbors to consider
    k_max : int
        Maximum number of neighbors to consider
    min_n_cells : int
        Minimum number of cells required in a neighborhood
    min_n_samples_r : int
        Minimum number of reference samples required
    exclude_same_study : bool
        Whether to exclude samples from the same study
    study_key : str
        Key in adata.obs that contains study identifiers
    grouping_key : str
        Key in adata.obs that contains grouping information
    adaptive_k_margin : float
        Margin for adaptive k selection
    distance_metric : Literal["energy_distance", "pairwise_euclidean"]
        Distance metric to use to calculate distances between samples
    """

    adata: sc.AnnData | None = None
    adata_emb_loc: str | None = None
    ref_q_key: str | None = None
    q_cat: str | None = None
    r_cat: str | None = None
    sample_key: str | None = None
    n_nhoods: int | None = None
    k_min: int | None = None
    k_max: int | None = None
    adapt_k: bool | None = None
    min_n_cells: int | None = None
    min_n_samples_r: int | None = None
    exclude_same_study: bool | None = None
    study_key: str | None = None
    grouping_key: str | None = None
    adaptive_k_margin: float | None = None
    distance_metric: Literal["energy_distance", "pairwise_euclidean"] | None = None
    seed: int | None = None
    samples_r: list[str] | None = None
    samples_q: list[str] | None = None
