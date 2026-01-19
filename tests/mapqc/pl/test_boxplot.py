import os
import warnings

import pandas as pd
import pytest

import mapqc.pl.boxplot as boxplot

# Note that we're just testing that the plotting functions don't throw errors or warnings for now.


@pytest.fixture
def sample_dists_df(output_data_dir):
    return pd.read_csv(os.path.join(output_data_dir, "sample_dists_df.csv"), index_col=0)


def test_mapqc_scores_boxplot(mapqc_output_adata):
    """Test that mapqc_scores boxplot function runs without warnings or errors."""
    with warnings.catch_warnings():
        # We're ignoring this warning as it is due to an internal code conflict in seaborn,
        # in version 0.13.2, that was fixed in a later version. It is not relevant for us.
        warnings.filterwarnings(
            "ignore",
            category=PendingDeprecationWarning,
            message="vert: bool will be deprecated in a future version",
        )
        boxplot.mapqc_scores(mapqc_output_adata, grouping_key="ann_level_3")


def test_sample_dists_to_ref(mapqc_output_adata, sample_dists_df):
    """Test that sample_dists_to_ref_per_nhood plotting function runs without warnings or errors."""
    with warnings.catch_warnings():
        # We're ignoring this warning as it is due to an internal code conflict in seaborn,
        # in version 0.13.2, that was fixed in a later version. It is not relevant for us.
        warnings.filterwarnings(
            "ignore",
            category=PendingDeprecationWarning,
            message="vert: bool will be deprecated in a future version",
        )
        boxplot.sample_dists_to_ref_per_nhood(
            mapqc_output_adata,
            sample_dists_to_ref_df=sample_dists_df,
            label_xticks_by="ann_level_3",
        )
