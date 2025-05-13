import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy.spatial.distance import cdist

from mapqc._distances._raw_distances import _pairwise_sample_distances
from mapqc._params import _MapQCParams
from mapqc._process_nhood import _process_neighborhood


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
            "emb0": [0, 0, 1.5, 0, 3, 0, -2, -1, -1, 0, 0],
            "emb1": [-0.5, 0.5, 0, 3, 2, 1, 0, 0, 1, -2.5, -1],
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


def test_nhood_passing_filter(cell_info):
    center_cell = "c6"
    # for testing exploration/understanding, add the following:
    # from scipy.spatial.distance import cdist
    # dists = cdist(
    #     np.array(cell_info.loc[center_cell,['emb0','emb1']].values[None,:],dtype=float),
    #     cell_info.loc[:,['emb0','emb1']].values
    # )
    # cell_info['dist_to_cc'] = dists[0]
    # cell_info.sort_values(by='dist_to_cc') # print/show sorted
    # samples_r_all = ["s1", "s5", "s4"]  # order differently than adata_obs etc.
    # samples_q_all = ["s3", "s2"]
    params = _MapQCParams(
        adata=sc.AnnData(cell_info.loc[:, ["emb0", "emb1"]].values, obs=cell_info.loc[:, ["s", "re_qu"]]),
        adata_emb_loc="X",
        ref_q_key="re_qu",
        q_cat="qu",
        r_cat="re",
        sample_key="s",
        k_min=5,
        k_max=9,
        exclude_same_study=False,
        adaptive_k_margin=0.1,
        samples_r=["s1", "s5", "s4"],  # ["s1", "s4", "s5"], # have to be ordered alphabetically
        samples_q=[
            "s3",
            "s2",
        ],  # ["s2", "s3"], # note that both of these sample lists represent all samples in the data
        min_n_cells=2,
        min_n_samples_r=2,
        adapt_k=True,
        distance_metric="energy_distance",
    )
    nhood_info_dict, pw_dists = _process_neighborhood(
        params=params,
        center_cell=center_cell,
    )
    # Expected output (use sorted df as in commented-out code above)
    # Note that as we have different k_max from k_min, k can be adapted.
    # Query minimum n cells from a single sample fullfilled at idx 7
    # i.e. the 8th cell (sample s2)
    # Reference minimum n samples (with at least min_n_cells each)
    # fulfilled at idx 3 (4th cell) (note that these are from the same study).
    # So k expected to be adapted to (7 + 1) * 1.1 = 9 (rounded up)
    expected_nhood_info_dict = {"center_cell": center_cell, "k": 9, "filter_info": "pass", "samples_q": ["s2"]}
    # and for the knn idc, let's calculate distances to center cell:
    dists = cdist(
        np.array(cell_info.loc[center_cell, ["emb0", "emb1"]].values[None, :], dtype=float),
        cell_info.loc[:, ["emb0", "emb1"]].values,
    )
    knn_idc = np.argsort(dists[0])[: expected_nhood_info_dict["k"]]
    expected_nhood_info_dict["knn_idc"] = knn_idc
    # check equality of dictionaries
    _compare_dictionaries(expected_nhood_info_dict, nhood_info_dict)
    # now check the pairwise distances
    # as we already included tests for the pairwise distance calculation
    # itself, we just need to check that the input is as we expect it,
    # pass that through the pw distance function and check that the
    # output is the same as the output we get here
    expected_pw_dist_input = cell_info.iloc[knn_idc, :]
    expected_samples_q, expected_pw_dists = _pairwise_sample_distances(
        params=params,
        emb=expected_pw_dist_input.loc[:, ["emb0", "emb1"]].values,
        obs=expected_pw_dist_input.loc[:, ["s", "re_qu"]],
    )
    assert np.array_equal(expected_samples_q, nhood_info_dict["samples_q"])
    np.testing.assert_almost_equal(pw_dists, expected_pw_dists)


def test_nhood_failing_query_filter(cell_info):
    # Note that this does not only test proper handling of query filter
    # not passing, but also of indexing: the cell we need to pass the
    # query filter is the 10th cell, but our max_k is set to 9.
    # Indexing is a bit confusing as sometimes we have to add
    # or subtract 1 due to python indexing (e.g. needed k is
    # idx of passing cell + 1)
    kmin = 5
    center_cell = "c6"
    samples_r_all = ["s1", "s5", "s4"]  # order differently than adata_obs etc.
    samples_q_all = ["s3", "s2"]
    params = _MapQCParams(
        adata=sc.AnnData(cell_info.loc[:, ["emb0", "emb1"]].values, obs=cell_info.loc[:, ["s", "re_qu"]]),
        adata_emb_loc="X",
        ref_q_key="re_qu",
        q_cat="qu",
        r_cat="re",
        sample_key="s",
        k_min=kmin,
        k_max=9,
        min_n_cells=3,
        min_n_samples_r=2,
        exclude_same_study=False,
        adaptive_k_margin=0.1,
        samples_r=samples_r_all,
        samples_q=samples_q_all,
        distance_metric="energy_distance",
    )
    nhood_info_dict, pw_dists = _process_neighborhood(
        params=params,
        center_cell=center_cell,
    )
    # this should fail, as query only has a sample with three cells
    # at cell number 10 (idx 9)
    expected_nhood_info_dict = {
        "center_cell": center_cell,
        "k": np.nan,
        "filter_info": "not enough query cells",
        "samples_q": [],
    }
    # for the knn idc, let's calculate distances to center cell:
    dists = cdist(
        np.array(cell_info.loc[center_cell, ["emb0", "emb1"]].values[None, :], dtype=float),
        cell_info.loc[:, ["emb0", "emb1"]].values,
    )
    knn_idc = np.argsort(dists[0])[:kmin]
    expected_nhood_info_dict["knn_idc"] = knn_idc
    # check equality of dictionaries
    _compare_dictionaries(expected_nhood_info_dict, nhood_info_dict)
    # and we expect an empty pw_dists matrix
    expected_pw_dists = np.full((len(samples_r_all), len(samples_r_all) + len(samples_q_all)), np.nan)
    np.testing.assert_equal(pw_dists, expected_pw_dists)


def test_nhood_failing_reference_filter(cell_info):
    center_cell = "c6"
    samples_r_all = ["s1", "s5", "s4"]  # order differently than adata_obs etc.
    samples_q_all = ["s3", "s2"]
    kmin = 5
    params = _MapQCParams(
        adata=sc.AnnData(cell_info.loc[:, ["emb0", "emb1"]].values, obs=cell_info.loc[:, ["s", "re_qu", "paper"]]),
        adata_emb_loc="X",
        ref_q_key="re_qu",
        q_cat="qu",
        r_cat="re",
        sample_key="s",
        k_min=kmin,
        k_max=10,
        exclude_same_study=True,
        adaptive_k_margin=0.1,
        study_key="paper",
        samples_r=samples_r_all,
        samples_q=samples_q_all,
        min_n_samples_r=2,
        min_n_cells=2,
        distance_metric="energy_distance",
    )
    nhood_info_dict, pw_dists = _process_neighborhood(
        params=params,
        center_cell=center_cell,
    )
    # Expectation:
    # As we now do not count samples from the same study, our first two
    # reference samples with at least two cells are from the same
    # study (s1 and s4) and will thus not be compared anymore.
    # The first referencesample from a different study (s5) only has
    # it's 2nd cell at index 10 (11th cell!), which is too far from the
    # center cell given our k_max of 10 (Note that we put the max_k
    # only 1 below our needed k, to also test that nothing has gone
    # wrong with indexing). Therefore, while the query filter should pass,
    # the reference filter should fail.
    expected_nhood_info_dict = {
        "center_cell": center_cell,
        "k": np.nan,
        "filter_info": "not enough reference samples from different studies",
        "samples_q": [],
    }
    # for the knn idc, let's calculate distances to center cell:
    dists = cdist(
        np.array(cell_info.loc[center_cell, ["emb0", "emb1"]].values[None, :], dtype=float),
        cell_info.loc[:, ["emb0", "emb1"]].values,
    )
    knn_idc = np.argsort(dists[0])[:kmin]
    expected_nhood_info_dict["knn_idc"] = knn_idc
    # check equality of dictionaries
    _compare_dictionaries(expected_nhood_info_dict, nhood_info_dict)
    # and we expect an empty pw_dists matrix
    expected_pw_dists = np.full((len(samples_r_all), len(samples_r_all) + len(samples_q_all)), np.nan)
    np.testing.assert_equal(pw_dists, expected_pw_dists)


def _compare_dictionaries(expected_dict, output_dict):
    for key, true_value in output_dict.items():
        if isinstance(true_value, np.ndarray):
            if np.issubdtype(true_value.dtype, np.floating):
                # For floating point arrays, use np.isclose to handle NaN values
                assert np.isclose(true_value, expected_dict[key], equal_nan=True).all()
            else:
                # For non-floating point arrays, use direct comparison
                assert (true_value == expected_dict[key]).all()
        elif isinstance(true_value, (float | np.floating)):
            # For scalar floats, use np.isclose to handle NaN values
            assert np.isclose(true_value, expected_dict[key], equal_nan=True)
        else:
            # For other types, use direct comparison
            assert true_value == expected_dict[key]
