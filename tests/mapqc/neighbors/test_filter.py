import pandas as pd
import pytest

from mapqc._neighbors._filter import (
    _create_difference_matrix,
    _filter_and_get_min_k_query,
    _filter_and_get_min_k_ref,
    _get_idc_nth_instances,
)
from mapqc._params import _MapQCParams


@pytest.fixture
def cell_df_small():
    return pd.DataFrame(
        data={
            "s": ["s1", "s1", "s2", "s3", "s3", "s2", "s2", "s4", "s4", "s4"],
            "re_qu": ["re", "re", "qu", "qu", "qu", "qu", "qu", "re", "re", "re"],
        },
        index=["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"],
    )


@pytest.fixture
def cell_df_large():
    # extend by one extra sample
    return pd.DataFrame(
        data={
            "s": [
                "s1",
                "s1",
                "s2",
                "s3",
                "s3",
                "s2",
                "s2",
                "s4",
                "s4",
                "s4",
                "s5",
                "s5",
            ],
            "re_qu": [
                "re",
                "re",
                "qu",
                "qu",
                "qu",
                "qu",
                "qu",
                "re",
                "re",
                "re",
                "re",
                "re",
            ],
        },
        index=[
            "c1",
            "c2",
            "c3",
            "c4",
            "c5",
            "c6",
            "c7",
            "c8",
            "c9",
            "c10",
            "c11",
            "c12",
        ],
    )


@pytest.fixture
def sample_df_small(cell_df_small):
    sample_df = cell_df_small.groupby("s").agg({"re_qu": "first"})
    sample_df["paper"] = ["a", "b", "b", "a"]
    return sample_df


@pytest.fixture
def sample_df_large(cell_df_large):
    sample_df = cell_df_large.groupby("s").agg({"re_qu": "first"})
    sample_df["paper"] = ["a", "b", "b", "a", "c"]
    return sample_df


def test_get_idc_nth_instances():
    seq = pd.Categorical(["a", "b", "a", "c", "b", "c", "c", "c", "a"])
    idc = _get_idc_nth_instances(seq, 3)
    assert idc.equals(pd.Series(data=[6, 8], index=["c", "a"]))


def test_create_difference_matrix():
    lst = ["a", "b", "b"]
    assert _create_difference_matrix(lst).tolist() == [
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
    ]


def test_create_differene_matrix_nones():
    lst = [None, None]
    assert _create_difference_matrix(lst).tolist() == [
        [0, 0],
        [0, 0],
    ]


def test_filter_and_get_min_k_query():
    cell_info = pd.DataFrame(
        data={
            "s": ["s1", "s1", "s2", "s3", "s3", "s2", "s2"],
            "re_qu": ["re", "re", "qu", "qu", "qu", "qu", "qu"],
        },
        index=["c1", "c2", "c3", "c4", "c5", "c6", "c7"],
    )
    params = _MapQCParams(
        ref_q_key="re_qu",
        sample_key="s",
        q_cat="qu",
        k_min=3,
        adaptive_k_margin=0.1,
        min_n_cells=3,
    )
    # check situation where sample s2 is the first to satisfy
    # min_n_cells condition, at index 6 (7th position). min_k
    # returned should be (6+1) * (1 + margin) = 7 * 1.1 = 7.7
    # rounded up should be 8 (integer)
    assert _filter_and_get_min_k_query(
        params=params,
        cell_df=cell_info,
    ) == (True, 8, "pass")
    # check if we change k_min to higher, then we should
    # get input k_min
    params.k_min = 10
    assert _filter_and_get_min_k_query(
        params=params,
        cell_df=cell_info,
    ) == (True, 10, "pass")
    # check if we change min_n_cells to higher, then we do
    # not have enough cells for any query sample and we do
    # not pass filtering, and no min_k is returned
    params.min_n_cells = 4
    params.k_min = 20
    assert _filter_and_get_min_k_query(
        params=params,
        cell_df=cell_info,
    ) == (False, None, "not enough query cells")


def test_filter_and_get_min_k_ref_simple(cell_df_small, sample_df_small):
    # case 1: fixed (non-adaptive) k, min_n_cells 2, do not exclude same study
    # Note that when setting non-adaptive k, n_rows of cell_df should always
    # be same as k_min
    # this first case satisfies the conditions already below k_min, and
    # should therefore return k_min and pass
    params = _MapQCParams(
        ref_q_key="re_qu",
        sample_key="s",
        r_cat="re",
        adapt_k=False,
        min_n_cells=2,
        min_n_samples_r=1,
        exclude_same_study=False,
    )
    assert _filter_and_get_min_k_ref(
        params=params,
        cell_df=cell_df_small,
        sample_df=sample_df_small,
        k_min_query=10,
    ) == (True, 10, "pass")
    # case 2: same, but min_n_samples_r 2 (should have same output)
    # This should be satisfied at cell number 9 (two samples with
    # 2 cells), so that k_min (>9) is returned
    params.min_n_samples_r = 2
    assert _filter_and_get_min_k_ref(
        params=params,
        cell_df=cell_df_small,
        sample_df=sample_df_small,
        k_min_query=10,
    ) == (True, 10, "pass")
    # case 3: same, but with higher min_n_cells, should not pass filter
    params.min_n_cells = 5
    assert _filter_and_get_min_k_ref(
        params=params,
        cell_df=cell_df_small,
        sample_df=sample_df_small,
        k_min_query=10,
    ) == (False, None, "not enough reference samples")


def test_filter_and_get_min_k_ref_exclude_same_study_pairs(
    cell_df_small, sample_df_small, cell_df_large, sample_df_large
):
    # we are now excluding same-study pairs, unlike the previous case
    # where we had identical argument settings otherwise. This should
    # result in case not passing filter, as we have 2 samples, each from
    # a different study, and min_n_samples_r is 2. We therefore need
    # (2**2 - 2)/2 = 1 valid pairs, and have 1 valid pair.
    params = _MapQCParams(
        ref_q_key="re_qu",
        sample_key="s",
        r_cat="re",
        min_n_cells=2,
        min_n_samples_r=2,
        adapt_k=False,
        exclude_same_study=True,
        study_key="paper",
    )
    assert _filter_and_get_min_k_ref(
        params=params,
        cell_df=cell_df_small,
        sample_df=sample_df_small,
        k_min_query=10,
    ) == (False, None, "not enough reference samples from different studies")
    # when changing k_min to 12, we include a sample from a different
    # batch, so that this should now pass
    params.min_n_samples_r = 1
    assert _filter_and_get_min_k_ref(
        params=params,
        cell_df=cell_df_large,
        sample_df=sample_df_large,
        k_min_query=12,
    ) == (True, 12, "pass")


def test_filter_and_get_min_k_ref_adaptive_k(cell_df_small, sample_df_small, cell_df_large, sample_df_large):
    # allow for k adaptation, with exclude_same_study set to True,
    # such that increasing k is needed to satisfy conditions
    params = _MapQCParams(
        ref_q_key="re_qu",
        sample_key="s",
        r_cat="re",
        min_n_cells=2,
        min_n_samples_r=1,
        adapt_k=True,
        adaptive_k_margin=0.1,
        exclude_same_study=True,
        study_key="paper",
    )
    assert _filter_and_get_min_k_ref(
        params=params,
        cell_df=cell_df_large,
        sample_df=sample_df_large,
        k_min_query=10,
    ) == (True, 14, "pass")
    # check that without adaptive k this would have failed:
    # note that without adaptive k, our cell_df should have
    # n_rows = k_min, so we use cell_df_small here
    params.adapt_k = False
    params.adaptive_k_margin = None
    assert _filter_and_get_min_k_ref(
        params=params,
        cell_df=cell_df_small,
        sample_df=sample_df_small,
        k_min_query=10,
    ) == (False, None, "not enough reference samples from different studies")
