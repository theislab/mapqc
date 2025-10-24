import os
import pickle

import pandas as pd

import mapqc


def test_evaluate(mapqc_output_adata, output_data_dir):
    # create adata copy object so that session object is not modified:
    adata_test = mapqc_output_adata.copy()
    stats = mapqc.evaluate(
        adata=adata_test,
        case_control_key="lung_condition",
        case_cats=["IPF"],
        control_cats=["Healthy"],
    )
    # get expected stats:
    expected_stats = pickle.load(open(os.path.join(output_data_dir, "evaluate_stats.pkl"), "rb"))
    # check that stats are as expected:
    assert stats == expected_stats
    # test that columns added to adata.obs are as expected:
    expected_cols = pickle.load(open(os.path.join(output_data_dir, "obs_df.pkl"), "rb"))
    cols_added = ["case_control", "mapqc_score_binary"]
    for col in cols_added:
        pd.testing.assert_series_equal(adata_test.obs[col], expected_cols[col])
