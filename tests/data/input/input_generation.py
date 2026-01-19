# code used to generate input data:
import scanpy as sc


def clean_up_df_column(df, column_name, min_n_cells_per_cat):
    new_column_name = f"{column_name}_cleaned"
    df[new_column_name] = df[column_name].copy()
    group_count = df[new_column_name].value_counts()
    low_n_groups = group_count.index[group_count < min_n_cells_per_cat]
    renamer = {(ct): ("Other" if ct in low_n_groups else ct) for ct in df[new_column_name].unique()}
    df[new_column_name] = df[new_column_name].map(renamer)
    return df[new_column_name].values


path_mapped_emb_good = "/lustre/groups/ml01/workspace/lisa.sikkema/mapqc/data/HLCA/HLCA_and_Sheppard_2020_original_counts_batch_sample_kl1.0_latent.h5ad"
path_adata_counts = "/lustre/groups/ml01/workspace/lisa.sikkema/mapqc/data/HLCA/HLCA_and_Sheppard_2020_counts.h5ad"
emb_good = sc.read_h5ad(path_mapped_emb_good)
adata_full = sc.read_h5ad(path_adata_counts, backed="r")
metadata_to_keep_from_full_adata = [
    "study",
    "donor_id",
    "sample",
    "r_or_q",
    "lung_condition",
]
for cat in metadata_to_keep_from_full_adata:
    emb_good.obs[cat] = None
emb_good.obs.loc[:, metadata_to_keep_from_full_adata] = adata_full.obs.loc[
    emb_good.obs.index, metadata_to_keep_from_full_adata
]
studies_to_keep = [
    "Meyer_2019",
    "Banovich_Kropski_2020",
    "Misharin_Budinger_2018",
    "Sheppard_2020",
]
subjects_per_study = emb_good.obs.groupby("study").agg({"donor_id": "unique"}).loc[studies_to_keep, :]
subjects_to_keep = []
for study in studies_to_keep:
    study_subjects = list(subjects_per_study.loc[study, "donor_id"])
    if len(study_subjects) < 5:
        subjects_to_keep += study_subjects
    else:
        subjects_to_keep += study_subjects[:5]
ct_to_keep = ["Stroma"]  # (ann level 1)
mask = (
    emb_good.obs.study.isin(studies_to_keep)
    & emb_good.obs.donor_id.isin(subjects_to_keep)
    & (emb_good.obs.ann_level_1.isin(ct_to_keep) | emb_good.obs.ann_level_1_pred.isin(ct_to_keep))
)
adata = emb_good[mask, :].copy()
# create merged cell type embedding:
for lev in ["3", "4"]:
    adata.obs[f"ann_level_{lev}"] = adata.obs[f"ann_level_{lev}"].tolist()
    adata.obs.loc[adata.obs.r_or_q == "q", f"ann_level_{lev}"] = adata.obs.loc[
        adata.obs.r_or_q == "q", f"ann_level_{lev}_pred"
    ]
    # clean up column:
    adata.obs[f"ann_level_{lev}"] = clean_up_df_column(adata.obs, f"ann_level_{lev}", 20)

obs_col_to_keep = metadata_to_keep_from_full_adata + [
    "leiden",
    "ann_level_3",
    "ann_level_4",
]
adata.obs = adata.obs.loc[:, obs_col_to_keep].copy()
adata = sc.AnnData(adata.X[:, :10], obs=adata.obs)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
adata.write("/lustre/groups/ml01/code/lisa.sikkema/mapqc/tests/data/input/mapped_q_and_r.h5ad")
