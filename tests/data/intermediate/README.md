code used to generate intermediate files from input file:
```py
import numpy as np
import pandas as pd
import scanpy as sc
from mapqc.process_nhood import process_neighborhood
from mapqc.center_cells.sampling import sample_center_cells_by_group
from mapqc.distances.normalized_distances import get_normalized_dists_to_ref
adata = sc.read_h5ad("./mapqc/tests/data/input/mapped_q_and_r.h5ad")
n_nhoods = 10
center_cells = sample_center_cells_by_group(adata_obs=adata.obs,ref_q_key='r_or_q',q_cat='q',grouping_cat='leiden',n_cells=n_nhoods,seed=42)
samples_r = sorted(adata.obs.loc[adata.obs.r_or_q == "r","sample"].unique().tolist())
samples_q = sorted(adata.obs.loc[adata.obs.r_or_q == "q","sample"].unique().tolist())
nhood_info = pd.DataFrame(columns=['nhood_number','filter_info','k', 'knn_idc', ])
dists = np.full(shape=(len(samples_r), len(samples_r) + len(samples_q), len(center_cells)), fill_value=np.nan)
for i, cell in enumerate(center_cells):
    nhood_dict, dists[:,:,i] = process_neighborhood(
        center_cell = cell,
        adata_emb = adata.X,
        adata_obs = adata.obs,
        samples_r_all = samples_r,
        samples_q_all = samples_q,
        k_min = 40,
        k_max = 50,
        sample_key = "sample",
        ref_q_key = "r_or_q",
        q_cat = "q",
        r_cat = "r",
        min_n_cells= 3,
        min_n_samples_r = 3,
        exclude_same_study= True,
        adaptive_k_margin = 0.1,
        study_key = "study",
    )
    nhood_info.loc[cell] = nhood_dict
    nhood_info.loc[cell, 'nhood_number'] = i
dists_to_ref = get_normalized_dists_to_ref(dists, samples_r)
np.save("./mapqc/tests/data/intermediate/dists_to_ref.npy", dists_to_ref)
nhood_info.to_pickle("./mapqc/tests/data/intermediate/nhood_info.pkl")
```
