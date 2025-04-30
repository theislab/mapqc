code used to generate input data:

```py
import scanpy as sc
path_mapped_emb_good = "/lustre/groups/ml01/workspace/lisa.sikkema/mapqc/data/HLCA/HLCA_and_Sheppard_2020_original_counts_batch_sample_kl1.0_latent.h5ad"
path_adata_counts = "/lustre/groups/ml01/workspace/lisa.sikkema/mapqc/data/HLCA/HLCA_and_Sheppard_2020_counts.h5ad"
emb_good = sc.read_h5ad(path_mapped_emb_good)
adata_full = sc.read_h5ad(path_adata_counts,backed='r')
metadata_to_keep = ['study','donor_id','sample','r_or_q']
for cat in metadata_to_keep:
    emb_good.obs[cat] = None
emb_good.obs.loc[:,metadata_to_keep] = adata_full.obs.loc[emb_good.obs.index, metadata_to_keep]
studies_to_keep = ["Meyer_2019", "Banovich_Kropski_2020","Misharin_Budinger_2018","Sheppard_2020"]
subjects_per_study = emb_good.obs.groupby("study").agg({"donor_id":"unique"}).loc[studies_to_keep,:]
subjects_to_keep = list()
for study in studies_to_keep:
    study_subjects = list(subjects_per_study.loc[study,"donor_id"])
    if len(study_subjects) < 5:
        subjects_to_keep += study_subjects
    else:
        subjects_to_keep += study_subjects[:5]
ct_to_keep = ["Myeloid"]
mask = emb_good.obs.study.isin(studies_to_keep) & emb_good.obs.donor_id.isin(subjects_to_keep) & (emb_good.obs.ann_level_2.isin(ct_to_keep) | emb_good.obs.ann_level_2_pred.isin(ct_to_keep))
adata = emb_good[mask,:].copy()
obs_col_to_keep = ['study','donor_id','sample','leiden','ann_level_2','ann_level_2_pred','r_or_q']
adata.obs = adata.obs.loc[:,obs_col_to_keep].copy()
adata = sc.AnnData(adata.X[:,:10],obs=adata.obs)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
adata.write("/lustre/groups/ml01/code/lisa.sikkema/mapqc/tests/data/input/mapped_q_and_r.h5ad")
```
