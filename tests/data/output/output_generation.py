# code used to generate output:
import numpy as np
import pandas as pd
import scanpy as sc

from mapqc._mapqc_scores import _calculate_mapqc_scores
from mapqc._params import _MapQCParams

adata = sc.read_h5ad("./tests/data/input/mapped_q_and_r.h5ad")
n_nhoods = 30
params = _MapQCParams(
    adata=adata,
    ref_q_key="r_or_q",
    r_cat="r",
    q_cat="q",
    grouping_key="leiden",
    n_nhoods=n_nhoods,
    seed=42,
    adata_emb_loc="X",
    k_min=100,
    k_max=150,
    sample_key="sample",
    min_n_cells=3,
    min_n_samples_r=3,
    exclude_same_study=True,
    study_key="study",
    adaptive_k_margin=0.1,
    adapt_k=True,
    distance_metric="energy_distance",
    samples_r=sorted(adata.obs.loc[adata.obs.r_or_q == "r", "sample"].unique().tolist()),
    samples_q=sorted(adata.obs.loc[adata.obs.r_or_q == "q", "sample"].unique().tolist()),
)
dists_to_ref = np.load("./tests/data/intermediate/dists_to_ref.npy")
nhood_info = pd.read_pickle("./tests/data/intermediate/nhood_info.pkl")
mapqc_scores, filtering_info_per_cell = _calculate_mapqc_scores(
    params=params, sample_dist_to_ref_per_nhood=dists_to_ref, nhood_info_df=nhood_info
)
np.save("./tests/data/intermediate/mapqc_scores.npy", mapqc_scores)
np.save("./tests/data/intermediate/filtering_info_per_cell.npy", filtering_info_per_cell)
