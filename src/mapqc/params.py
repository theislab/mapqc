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
    (as run.py arguments)

    samples_r: list[str]
        List of reference sample names in adata, ordered alphabetically.
    samples_q: list[str]
        List of query sample names in adata, ordered alphabetically.
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
    overwrite: bool | None = None
    samples_r: list[str] | None = None
    samples_q: list[str] | None = None
