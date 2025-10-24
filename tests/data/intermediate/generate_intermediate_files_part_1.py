# code used to generate intermediate files dists_to_ref.npy and nhood_info.pkl:
import numpy as np
import pandas as pd
import scanpy as sc

from mapqc._center_cells._sampling import _sample_center_cells_by_group
from mapqc._distances._normalized_distances import _get_normalized_dists_to_ref
from mapqc._params import _MapQCParams
from mapqc._process_nhood import _process_neighborhood

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
    k_min=500,
    k_max=2000,
    sample_key="sample",
    min_n_cells=3,
    min_n_samples_r=3,
    exclude_same_study=True,
    study_key="study",
    adaptive_k_margin=0.1,
    adapt_k=True,
    distance_metric="energy_distance",
)
center_cells = _sample_center_cells_by_group(
    params
)  # adata_obs=adata.obs,ref_q_key='r_or_q',q_cat='q',grouping_cat='leiden',n_cells=n_nhoods,seed=42)
params.samples_r = sorted(adata.obs.loc[adata.obs.r_or_q == "r", "sample"].unique().tolist())
params.samples_q = sorted(adata.obs.loc[adata.obs.r_or_q == "q", "sample"].unique().tolist())
nhood_info = pd.DataFrame(columns=["nhood_number", "filter_info", "k", "knn_idc", "samples_q"])
dists = np.full(
    shape=(
        len(params.samples_r),
        len(params.samples_r) + len(params.samples_q),
        len(center_cells),
    ),
    fill_value=np.nan,
)
for i, cell in enumerate(center_cells):
    nhood_dict, dists[:, :, i] = _process_neighborhood(params, cell)
    nhood_info.loc[cell] = nhood_dict
    nhood_info.loc[cell, "nhood_number"] = i
dists_to_ref = _get_normalized_dists_to_ref(params, dists)
np.save("./tests/data/intermediate/dists_to_ref.npy", dists_to_ref)
nhood_info.to_pickle("./tests/data/intermediate/nhood_info.pkl")
