"""Parameter definitions for the mapqc package."""

from dataclasses import dataclass
from typing import Literal

import scanpy as sc


@dataclass
class _MapQCParams:
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
    verbose: bool = True

    def __post_init__(self):
        # set adapt_k parameter
        if self.k_min is not None and self.k_max is not None:
            if self.k_min < self.k_max:
                if self.adapt_k is None:
                    self.adapt_k = True
                elif not self.adapt_k:
                    raise ValueError("adapt_k must be True if k_min < k_max")
                self.adapt_k = True
            elif self.k_min == self.k_max:
                if self.adapt_k:
                    raise ValueError("adapt_k must be False if k_min == k_max")
                self.adapt_k = False
            else:
                raise ValueError("k_min must be less than or equal to k_max")
