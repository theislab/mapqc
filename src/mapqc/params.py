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

    adata: sc.AnnData
    adata_emb_loc: str
    ref_q_key: str
    q_cat: str
    r_cat: str
    sample_key: str
    n_nhoods: int
    k_min: int
    k_max: int
    min_n_cells: int
    min_n_samples_r: int
    exclude_same_study: bool
    study_key: str
    grouping_key: str
    adaptive_k_margin: float
    distance_metric: Literal["energy_distance", "pairwise_euclidean"]


# @dataclass
# class NeighborhoodParams:
#     """
#     Parameters specific to neighborhood processing.
#     Only includes parameters needed for process_neighborhood function.
#     """

#     adata_emb: sc.AnnData
#     adata_obs: pd.DataFrame
#     ref_q_key: str
#     q_cat: str
#     r_cat: str
#     sample_key: str
#     k_min: int
#     k_max: int
#     min_n_cells: int
#     min_n_samples_r: int
#     exclude_same_study: bool
#     study_key: str
#     adaptive_k_margin: float

#     @classmethod
#     def from_mapqc_params(cls, params: MapQCParams) -> "NeighborhoodParams":
#         """Create NeighborhoodParams from MapQCParams."""
#         return cls(
#             adata_emb=params.adata_emb,
#             adata_obs=params.adata.obs,
#             ref_q_key=params.ref_q_key,
#             q_cat=params.q_cat,
#             r_cat=params.r_cat,
#             sample_key=params.sample_key,
#             k_min=params.k_min,
#             k_max=params.k_max,
#             min_n_cells=params.min_n_cells,
#             min_n_samples_r=params.min_n_samples_r,
#             exclude_same_study=params.exclude_same_study,
#             study_key=params.study_key,
#             adaptive_k_margin=params.adaptive_k_margin,
#         )


# @dataclass
# class CenterCellParams:
#     """
#     Parameters specific to center cell sampling.
#     Only includes parameters needed for sample_center_cells_by_group function.
#     """

#     adata_obs: pd.DataFrame
#     ref_q_key: str
#     q_cat: str
#     grouping_cat: str
#     n_cells: int

#     @classmethod
#     def from_mapqc_params(cls, params: MapQCParams) -> "CenterCellParams":
#         """Create CenterCellParams from MapQCParams."""
#         return cls(
#             adata_obs=params.adata.obs,
#             ref_q_key=params.ref_q_key,
#             q_cat=params.q_cat,
#             grouping_cat=params.grouping_key,
#             n_cells=params.n_nhoods,
#         )


# # Example of how to use these in your code:
# """
# # In your main function:
# params = MapQCParams(...)

# # For neighborhood processing:
# nhood_params = NeighborhoodParams.from_mapqc_params(params)
# result = process_neighborhood(
#     center_cell=cell,
#     **nhood_params.__dict__  # This unpacks all parameters
# )

# # For center cell sampling:
# center_params = CenterCellParams.from_mapqc_params(params)
# center_cells = sample_center_cells_by_group(
#     **center_params.__dict__,
#     seed=42  # Additional parameters can be added
# )
# """"
