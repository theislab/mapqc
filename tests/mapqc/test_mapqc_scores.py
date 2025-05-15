import numpy as np
import pandas as pd
import scanpy as sc

from mapqc._mapqc_scores import (
    _calculate_mapqc_scores,
    _create_sample_and_nhood_based_cell_mask,
    _get_per_cell_filtering_info,
)
from mapqc._params import _MapQCParams


def test_mapqc_scores(adata, intermediate_data_dir):
    # load intermediate output:
    dists_to_ref = np.load(intermediate_data_dir / "dists_to_ref.npy")
    nhood_info = pd.read_pickle(intermediate_data_dir / "nhood_info.pkl")
    # load expected final output:
    mapqc_scores_exp = np.load(intermediate_data_dir / "mapqc_scores.npy")
    filtering_info_per_cell_exp = np.load(intermediate_data_dir / "filtering_info_per_cell.npy", allow_pickle=True)
    # calculate mapqc_socres
    params = _MapQCParams(
        adata=adata,
        ref_q_key="r_or_q",
        r_cat="r",
        q_cat="q",
        grouping_key="leiden",
        n_nhoods=10,
        seed=42,
        adata_emb_loc="X",
        k_min=500,
        k_max=2000,
        sample_key="sample",
        min_n_cells=3,
        min_n_samples_r=3,
        exclude_same_study=True,
        study_key="study",
        adaptive_k_margin=0.1,
        adapt_k=True,
        distance_metric="energy_distance",
        samples_r=sorted(adata.obs.loc[adata.obs.r_or_q == "r", "sample"].unique().tolist()),
        samples_q=sorted(adata.obs.loc[adata.obs.r_or_q == "q", "sample"].unique().tolist()),
    )
    # samples_q = sorted(adata.obs.loc[adata.obs[ref_q_key] == q_cat, sample_key].unique().tolist())
    mapqc_scores, filtering_info_per_cell = _calculate_mapqc_scores(
        params=params,
        sample_dist_to_ref_per_nhood=dists_to_ref,
        nhood_info_df=nhood_info,
    )
    # check that the output matches the expected output:
    assert np.array_equal(mapqc_scores, mapqc_scores_exp, equal_nan=True)
    assert np.array_equal(filtering_info_per_cell, filtering_info_per_cell_exp)


def test_create_sample_and_nhood_based_cell_mask():
    # create test input:
    # order samples non-alphabetically, to test that alphabetical
    # order of samples is implemented correctly.
    obs = pd.DataFrame(
        data={
            "ref_or_que": ["ref"] * 3 + ["que"] * 7,
            "sample_id": ["A"] * 3 + ["C"] * 3 + ["B"] * 3 + ["D"],
        },
        index=range(1, 11),
    )
    # create nhood info df, with knn_idc specifying which cells are in each neighborhood (2 neighborhoods in total)
    nhood_info_df = pd.DataFrame(
        data={"knn_idc": [[3, 4, 7], [3, 5, 6, 7, 8]], "samples_q": [["B", "C"], ["B", "C"]]}, index=[4, 7]
    )
    # expected output (there are only 7 query cells):
    nhood_mask = np.zeros((2, 7))
    # set values to 1 for neighborhoods in which cells occur:
    nhood_mask[0, np.array([3, 4, 7]) - 3] = 1  # correct indices for removal of reference cells
    nhood_mask[1, np.array([3, 5, 6, 7, 8]) - 3] = 1
    # create sample mask. There are only 3 query samples, B, C and D.
    # there are only 7 query cells. Note that samples should be
    # ordered alphabetically in the mask.
    sample_mask = np.zeros((3, 7))
    # set values to 1 for samples in which cells occur:
    sample_mask[1, [0, 1, 2]] = 1  # first 3 query cells in sample C (row 1)
    sample_mask[0, [3, 4, 5]] = 1  # next 3 query cells in sample B (row 0)
    sample_mask[2, [6]] = 1  # last query cell in sample D
    # now calculate full mask, which should have shape n_query_samples, n_nhoods, n_query_cells
    full_mask = nhood_mask[np.newaxis, :, :] * sample_mask[:, np.newaxis, :]
    # To prevent scanpy warning, we convert our float indices to strings:
    obs.index = obs.index.astype(str)
    # now calculate true output using function:
    params = _MapQCParams(
        adata=sc.AnnData(obs=obs),
        ref_q_key="ref_or_que",
        q_cat="que",
        sample_key="sample_id",
        samples_q=["B", "C", "D"],  # should be sorted alphabetically
    )
    (
        true_mask,
        true_nhood_mask,
        true_sample_mask,
    ) = _create_sample_and_nhood_based_cell_mask(
        params=params,
        nhood_info_df=nhood_info_df,
    )
    # check that the output matches the expected output:
    assert np.array_equal(full_mask, true_mask)
    assert np.array_equal(nhood_mask, true_nhood_mask)
    assert np.array_equal(sample_mask, true_sample_mask)


def test_get_per_cell_filtering_info():
    # define test input, with 5 neighborhoods and 10 cells:
    mapqc_scores = np.array(
        [
            np.nan,
            np.nan,
            np.nan,
            -0.97625909,
            -0.77165746,
            0.0911037,
            np.nan,
            np.nan,
            0.0911037,
            np.nan,
        ]
    )
    nhood_mask = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        ]
    )
    # Example DataFrame for testing
    # knn_idc can be disregarded, this is based on the full data
    nhood_info_df = pd.DataFrame(
        {
            "nhood_number": [0.0, 1.0, 2.0, 3.0, 4.0],
            "filter_info": [
                "pass",
                "pass",
                "pass",
                "not enough reference samples",
                "not enough reference samples from different studies",
            ],
            "k": [41, 40, 40, np.nan, np.nan],
            "knn_idc": [[11], [], [29, 37, 52], [6, 39], [11, 37, 52]],
        }
    )
    # expected output:
    # cells with only zeros in nhood mask are not sampled:
    expected_output = np.full_like(mapqc_scores, fill_value=None, dtype=object)
    expected_output[[0, 1, 7, 9]] = "not sampled"
    # cells that include a neighborhood that passed filtering,
    # should be set to pass. I.e. neighborhoods 0-2
    expected_output[[3, 4, 5, 8]] = "pass"
    # cells that were part of a neighborhood, but only nhoods
    # that did not pass filtering (nhood 3 and 4), should have the most common
    # reason for failing filtering, with ties solved alphabetically.
    expected_output[[2, 6]] = "not enough reference samples"
    # get true output:
    true_output = _get_per_cell_filtering_info(mapqc_scores, nhood_mask, nhood_info_df)
    # check that the output matches the expected output:
    assert np.array_equal(expected_output, true_output)
