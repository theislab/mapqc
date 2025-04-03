import pandas as pd
import pytest

from mapqc.neighbors.adaptive_k import filter_and_get_adaptive_k


@pytest.fixture
def cell_df_small():
    return pd.DataFrame(
        data={
            "s": ["s1", "s1", "s2", "s3", "s3", "s2", "s2", "s4", "s4", "s4"],
            "re_qu": ["re", "re", "qu", "qu", "qu", "qu", "qu", "re", "re", "re"],
            "paper": ["a", "a", "b", "b", "b", "b", "b", "a", "a", "a"],
        },
        index=["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"],
    )


def test_filter_and_get_adaptive_k(cell_df_small):
    sample_df = cell_df_small.groupby("s").agg({"re_qu": "first", "paper": "first"})
    # first try out failing query
    assert filter_and_get_adaptive_k(
        cell_df=cell_df_small,
        ref_q_key="re_qu",
        sample_key="s",
        sample_df=sample_df,
        q_cat="qu",
        r_cat="re",
        min_n_cells=4,
        min_n_samples_r=1,
        k_min=10,
        adapt_k=False,
        exclude_same_study=False,
    ) == (False, None, "not enough query cells")
    # now try out passing query, but failing reference
    # (reduce min_n_cells, exclude same study pairs)
    assert filter_and_get_adaptive_k(
        cell_df=cell_df_small,
        ref_q_key="re_qu",
        sample_key="s",
        sample_df=sample_df,
        q_cat="qu",
        r_cat="re",
        min_n_cells=2,
        min_n_samples_r=1,
        k_min=10,
        adapt_k=False,
        exclude_same_study=True,
        study_key="paper",
    ) == (False, None, "not enough reference samples from different studies")
    # now try out passing both, using adaptive k. Set k_min to 5,
    # and min_n_samples_r to 2, while setting exclude_same_study to False.
    # It would need 9 cells, plus margin 0.1, is 9.9, rounded up to 10
    # cells, so output k should be 10.
    # (include same study pairs)
    assert filter_and_get_adaptive_k(
        cell_df=cell_df_small,
        ref_q_key="re_qu",
        sample_key="s",
        sample_df=sample_df,
        q_cat="qu",
        r_cat="re",
        min_n_cells=2,
        min_n_samples_r=2,
        k_min=5,
        adapt_k=True,
        adaptive_k_margin=0.1,
        exclude_same_study=False,
        study_key="paper",
    ) == (True, 10, "pass")
