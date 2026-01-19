import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from mapqc._distances._raw_distances import (
    _distance_between_cell_sets,
    _pairwise_sample_distances,
)
from mapqc._params import _MapQCParams


@pytest.fixture
def cell_info():
    return pd.DataFrame(
        data={
            "s": ["s1", "s1", "s3", "s2", "s2", "s2", "s4", "s4", "s4", "s5", "s5"],
            "re_qu": [
                "re",
                "re",
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
            "paper": ["a", "a", "b", "b", "b", "b", "a", "a", "a", "c", "c"],
            "emb0": [0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0],
            "emb1": [0, 0, 0, 2, 2, 2, 0, 0, 0, -2, -1],
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
        ],
    )


def test_distance_between_cell_sets_pairwise_euclidean_simple():
    """Test pairwise euclidean distance with simple 2D points"""
    cell_set_1 = np.array(
        [
            [
                0,
                0,
            ],
            [np.sqrt(2), np.sqrt(2)],
            [np.sqrt(8), np.sqrt(8)],
        ]
    )
    cell_set_2 = np.array([[0, 0]])

    distance = _distance_between_cell_sets(cell_set_1, cell_set_2, distance_metric="pairwise_euclidean")

    # The mean distance between all pairs should be:
    # (0 + √(2+2) + √(8+8)) / 3 = (0 + 2 + 4) / 3 = 2
    expected = 2
    np.testing.assert_almost_equal(distance, expected, decimal=6)


def test_distance_between_cell_sets_energy_distance_simple():
    """Test energy distance with simple 2D points"""
    cell_set_1 = np.array(
        [
            [
                0,
                0,
            ],
            [np.sqrt(2), np.sqrt(2)],
            [np.sqrt(8), np.sqrt(8)],
        ]
    )
    cell_set_2 = np.array([[0, 0], [0, 0]])

    distance = _distance_between_cell_sets(cell_set_1, cell_set_2, distance_metric="energy_distance")

    # Calculate expected components:
    # sigma: mean distance within each set, excluding distances to self
    # delta: mean distance between sets (same as pairwise_euclidean)
    sigma_1 = np.mean(
        [
            np.sqrt(2 + 2),  # cell 0 to 1
            np.sqrt(8 + 8),  # cell 0 to 2
            np.sqrt(2 * (np.sqrt(8) - np.sqrt(2)) ** 2),
        ]  # cell 1 to 2
    )
    sigma_2 = 0  # twice the same point
    delta = 2  # (see test_pairwise_euclidean_simple)

    expected = 2 * delta - sigma_1 - sigma_2
    np.testing.assert_almost_equal(distance, expected, decimal=6)


def test_distance_between_cell_sets_with_precomputed_distances():
    """Test that using precomputed distances gives same result"""
    cell_set_1 = np.array(
        [
            [
                0,
                0,
            ],
            [np.sqrt(2), np.sqrt(2)],
            [np.sqrt(8), np.sqrt(8)],
        ]
    )
    cell_set_2 = np.array([[0, 0]])

    # Precompute distances
    precomputed = np.array([[0], [2], [4]])

    d1 = _distance_between_cell_sets(cell_set_1, cell_set_2, distance_metric="pairwise_euclidean")

    d2 = _distance_between_cell_sets(
        cell_set_1,
        cell_set_2,
        distance_metric="pairwise_euclidean",
        precomputed_distance_matrix=precomputed,
    )

    np.testing.assert_almost_equal(d1, d2, decimal=6)


def test_distance_between_cell_sets_invalid_precomputed_shape():
    """Test that wrong shape of precomputed distances raises error"""
    cell_set_1 = np.array([[0, 0], [1, 1]])
    cell_set_2 = np.array([[2, 2], [3, 3]])

    wrong_shape = np.zeros((3, 2))  # Wrong shape

    with pytest.raises(ValueError):
        _distance_between_cell_sets(
            cell_set_1,
            cell_set_2,
            precomputed_distance_matrix=wrong_shape,
            distance_metric="energy_distance",
        )


def test_pairwise_sample_distances_simple(cell_info):
    """Test pairwise sample distances with simple 2D points"""
    # set samples all to larger set than present in cell_info
    samples_r_all = [
        "s0",
        "s1",
        "s5",
        "s4",
    ]  # ordered non-alphabetically, and in different order than cell_df, to make sure this doesn't give problems
    samples_q_all = ["s2", "s3", "s6"]
    min_n_cells = 2
    samples_r = [
        s for s in cell_info["s"] if ((cell_info["s"].value_counts()[s] >= min_n_cells) and (s in samples_r_all))
    ]
    samples_q = [
        s for s in cell_info["s"] if ((cell_info["s"].value_counts()[s] >= min_n_cells) and (s in samples_q_all))
    ]
    samples_q_set = sorted(set(samples_q))
    # create params:
    params = _MapQCParams(
        adata=sc.AnnData(cell_info.loc[:, ["emb0", "emb1"]].values, obs=cell_info),
        adata_emb_loc="X",
        samples_r=samples_r_all,
        samples_q=samples_q_all,
        sample_key="s",
        min_n_cells=min_n_cells,
        exclude_same_study=False,
        distance_metric="energy_distance",
    )
    # calculate simplest pairwise distances (i.e. also keep
    # pairs from same study)
    test_samples_q, test_out = _pairwise_sample_distances(
        params=params,
        emb=cell_info.loc[:, ["emb0", "emb1"]].values,
        obs=cell_info,
    )
    # expectations:
    # first calculate pairwise distances semi-manually:
    dists_manual = {}
    for s1 in samples_r + samples_q:
        for s2 in samples_r + samples_q:
            dists_manual[(s1, s2)] = _distance_between_cell_sets(
                cell_set_1=cell_info.loc[cell_info["s"] == s1, ["emb0", "emb1"]].values,
                cell_set_2=cell_info.loc[cell_info["s"] == s2, ["emb0", "emb1"]].values,
                distance_metric="energy_distance",
            )
    # expectations are:
    # samples with all nans: s0, s6 (both not in nhood),
    # s3 (not enough cells)
    # lower triangle should be all nans
    # our rows are samples_r_all, our columns
    # samples_r_all + samples_q_all, so we expect the following:
    expected_out = pd.DataFrame(
        data={
            "s0": [np.nan, np.nan, np.nan, np.nan],  # missing sample
            "s1": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],  # first one because pair with s0, second as pair with self (henceforth included in "lower traingle"), rest as lower triangle
            "s5": [
                np.nan,
                dists_manual[("s1", "s5")],
                np.nan,
                np.nan,
            ],  # nans due to missing sample (s0) and lower triangle
            "s4": [
                np.nan,
                dists_manual[("s1", "s4")],
                dists_manual[("s4", "s5")],
                np.nan,
            ],
            "s2": [
                np.nan,
                dists_manual[("s1", "s2")],
                dists_manual[("s2", "s5")],
                dists_manual[("s2", "s4")],
            ],
            "s3": [np.nan, np.nan, np.nan, np.nan],  # s3 not enough cells
            "s6": [np.nan, np.nan, np.nan, np.nan],  # missing sample
        },
        index=samples_r_all,
    )
    # check that the two are the same
    assert np.array_equal(test_samples_q, samples_q_set)
    np.testing.assert_almost_equal(test_out, expected_out.values)
    # now test with exclude_same_study=True
    sample_info = cell_info.groupby("s").agg({"paper": "first"})
    params.exclude_same_study = True
    params.study_key = "paper"
    test_samples_q_2, test_out_2 = _pairwise_sample_distances(
        params=params,
        sample_df=sample_info,
        emb=cell_info.loc[:, ["emb0", "emb1"]].values,
        obs=cell_info,
    )
    # expectations:
    # first copy original expected out:
    expected_out_2 = expected_out.copy()
    # then set all distances between samples of same study to nan:
    # note that this is only relevant for reference samples,
    # as query samples are only compared to reference anyway.
    # Also, we only need to check for samples in the nhood,
    # as the rest was already set to nan.
    for s1 in samples_r:
        for s2 in samples_r:
            if sample_info.loc[s1, "paper"] == sample_info.loc[s2, "paper"]:
                expected_out_2.loc[s1, s2] = np.nan
    print(expected_out)
    print(expected_out_2)
    print(pd.DataFrame(data=test_out_2, index=samples_r_all, columns=samples_r_all + samples_q_all))
    # check that the two are the same
    assert np.array_equal(test_samples_q_2, samples_q_set)
    np.testing.assert_almost_equal(test_out_2, expected_out_2.values)
