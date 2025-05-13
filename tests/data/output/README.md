code used to generate output:
```py
import scanpy as sc
import pickle
import mapqc
import os

path_input_data = "./tests/data/input/mapped_q_and_r.h5ad"
path_output_dir = "./tests/data/output/"
adata = sc.read_h5ad(path_input_data)
cols_before = adata.obs.columns
nhood_info_df, sample_dists = mapqc.run_mapqc(
    adata=adata,
    adata_emb_loc="X",
    ref_q_key="r_or_q",
    q_cat="q",
    r_cat="r",
    sample_key="sample",
    n_nhoods=30,
    k_min=100,
    k_max=150,
    min_n_cells=10,
    min_n_samples_r=3,
    exclude_same_study=True,
    study_key="study",
    grouping_key="leiden",
    seed=10,
    return_nhood_info_df=True,
    return_sample_dists_to_ref_df=True,
)
stats = mapqc.evaluate(adata, case_control_key="lung_condition", case_cats=["IPF"], control_cats=['Healthy'])
cols_added = [col for col in adata.obs if col not in cols_before]
obs_added = adata.obs.loc[:, cols_added]
params = adata.uns["mapqc_params"]
# store all output:
with open(os.path.join(path_output_dir, "nhood_info_df.pkl"), "wb") as f:
    pickle.dump(nhood_info_df, f)
sample_dists.to_csv(os.path.join(path_output_dir, "sample_dists_df.csv"))
with open(os.path.join(path_output_dir, "obs_df.pkl"), "wb") as f:
    pickle.dump(obs_added, f)
with open(os.path.join(path_output_dir, "params.pkl"), "wb") as f:
    pickle.dump(params, f)
with open(os.path.join(path_output_dir, "evaluate_stats.pkl"), "wb") as f:
    pickle.dump(stats, f)
```
