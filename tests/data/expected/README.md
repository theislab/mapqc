code used to generate output files from intermediate files:
```py
import numpy as np
import pandas as pd
from mapqc.mapqc_scores import calculate_mapqc_scores
dists_to_ref = np.load("./mapqc/tests/data/intermediate/dists_to_ref.npy")
nhood_info = pd.read_pickle("./mapqc/tests/data/intermediate/nhood_info.pkl")
ref_q_key = "r_or_q"
q_cat = "q"
nhood_info_df = nhood_info
adata_obs = adata.obs
sample_key='sample'
sample_dist_to_ref_per_nhood = dists_to_ref
mapqc_scores, filtering_info_per_cell = calculate_mapqc_scores(dists_to_ref, nhood_info, adata.obs, 'r_or_q','q','sample',samples_q)
np.save("./mapqc/tests/data/expected/mapqc_scores.npy", mapqc_scores)
np.save("./mapqc/tests/data/expected/filtering_info_per_cell.npy", filtering_info_per_cell)
```
