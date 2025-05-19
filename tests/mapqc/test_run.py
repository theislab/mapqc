import os
import pickle

import pandas as pd

import mapqc


def test_run_mapqc(adata, output_data_dir):
    # create adata copy object so that session object is not modified:
    adata_test = adata.copy()
    path_output_dir = output_data_dir
    cols_before = adata_test.obs.columns
    nhood_info_df, sample_dists = mapqc.run_mapqc(
        adata=adata_test,
        adata_emb_loc="X",
        ref_q_key="r_or_q",
        q_cat="q",
        r_cat="r",
        sample_key="sample",
        n_nhoods=30,
        k_min=500,
        k_max=2000,
        min_n_cells=10,
        min_n_samples_r=3,
        exclude_same_study=True,
        study_key="study",
        grouping_key="leiden",
        seed=10,
        return_nhood_info_df=True,
        return_sample_dists_to_ref_df=True,
    )
    cols_added = [col for col in adata_test.obs if col not in cols_before]
    obs_added = adata_test.obs.loc[:, cols_added]
    params = adata_test.uns["mapqc_params"]
    # load expected output:
    nhood_info_df_expected = pickle.load(open(os.path.join(path_output_dir, "nhood_info_df.pkl"), "rb"))
    sample_dists_expected = pd.read_csv(os.path.join(path_output_dir, "sample_dists_df.csv"), index_col=0)
    obs_added_expected = pickle.load(open(os.path.join(path_output_dir, "obs_df.pkl"), "rb"))
    # for obs added expeced, remove the columns added by mapqc.evaluate():
    obs_added_expected = obs_added_expected.loc[
        :, ~obs_added_expected.columns.isin(["case_control", "mapqc_score_binary"])
    ]
    params_expected = pickle.load(open(os.path.join(path_output_dir, "params.pkl"), "rb"))
    # check if expected output is equal to actual output:
    pd.testing.assert_frame_equal(nhood_info_df, nhood_info_df_expected)
    pd.testing.assert_frame_equal(sample_dists, sample_dists_expected)
    pd.testing.assert_frame_equal(obs_added, obs_added_expected)
    assert params == params_expected
