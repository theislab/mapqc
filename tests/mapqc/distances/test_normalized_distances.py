import warnings

import numpy as np
import pytest

from mapqc._distances._normalized_distances import (
    _fill_lower_triangle,
    _get_normalized_dists_to_ref,
    _identify_outliers,
    _mask_rows_and_columns,
)
from mapqc._params import _MapQCParams


@pytest.fixture
def pw_dists_triu():
    # create mock pw distances tensor with only upper
    # triangle filled, and np.nans for missing samples
    # and neighborhoods that did not pass filtering:
    n_samples_ref = 6
    n_samples_q = 2
    n_nhoods = 3
    pw_dists_triu = np.full(shape=(n_samples_ref, n_samples_ref + n_samples_q, n_nhoods), fill_value=np.nan)
    # fill in neighborhood 0: all reference samples present,
    # but one is an outlier, and one query sample present.
    # Query sample is far from the ference.
    dists_nhood_0 = np.array(
        [
            [np.nan, 20, 2, 1, 1, 0.1, np.nan, 2],  # r1
            [np.nan, np.nan, 22, 25, 20, 24, np.nan, 28],  # r2
            [np.nan, np.nan, np.nan, 3, 2, 1, np.nan, 5],  # r3
            [np.nan, np.nan, np.nan, np.nan, 0.5, 0.2, np.nan, 1],  # r4
            [np.nan, np.nan, np.nan, np.nan, np.nan, 1, np.nan, 7],  # r5
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8],  # r6
        ]
    )
    # neighborhood 1 is missing (did not pass filtering): leave as is.
    # neighborhood 2: nhood 2: well-integrated query. Includes results for
    # r2, r3, r4, r5 and q1.
    dists_nhood_2 = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # r1
            [np.nan, np.nan, 10, 15, 20, np.nan, 18, np.nan],  # r2
            [np.nan, np.nan, np.nan, 12, 16, np.nan, 14, np.nan],  # r3
            [np.nan, np.nan, np.nan, np.nan, 21, np.nan, 17, np.nan],  # r4
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 18, np.nan],  # r5
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # r6
        ]
    )
    pw_dists_triu[:, :, 0] = dists_nhood_0
    pw_dists_triu[:, :, 2] = dists_nhood_2
    return pw_dists_triu


@pytest.fixture
def pw_dists_valid(pw_dists_triu):
    pw_dists = _fill_lower_triangle(pw_dists_triu)
    # remove invalid neighborhoods (as in the normalize function):
    # Get all neighborhoods that have at least one entry that is not nan:
    valid_nhoods = ~(np.isnan(pw_dists).all(axis=(0, 1)))
    # Store indices of these valid neighborhoods:
    valid_nhood_idc = np.where(valid_nhoods)[0]
    # Now continue with only the valid neighborhoods:
    pw_dists_valid = pw_dists[:, :, valid_nhood_idc].copy()
    return pw_dists_valid


@pytest.fixture
def r_dists_to_ref_raw(pw_dists_valid):
    n_samples_ref = 6
    # Subset to reference samples only:
    pw_dists_valid_r = pw_dists_valid[:, :n_samples_ref, :]
    # Calculate mean distance to the reference for each
    # reference sample, per neighborhood:
    with warnings.catch_warnings():
        # note that we expect some empty slices here: not all
        # reference samples are present in each neighborhood.
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        r_dists_to_ref_raw = np.nanmean(pw_dists_valid_r, axis=0)
    return r_dists_to_ref_raw


def test_fill_lower_triangle(pw_dists_triu):
    n_samples_ref = 6
    pw_dists_triu_filled = _fill_lower_triangle(pw_dists_triu)
    # check manually:
    for i in range(n_samples_ref):
        for j in range(i + 1, n_samples_ref):
            # check that original entries did not change
            np.testing.assert_equal(pw_dists_triu_filled[i, j, :], pw_dists_triu[i, j, :])
            # check that mirror worked correctly
            np.testing.assert_equal(pw_dists_triu_filled[j, i, :], pw_dists_triu[i, j, :])
    # also check that query distances did not change:
    np.testing.assert_equal(pw_dists_triu_filled[:, n_samples_ref:, :], pw_dists_triu[:, n_samples_ref:, :])


def test_identify_outliers(r_dists_to_ref_raw):
    outlier_mask = _identify_outliers(r_dists_to_ref_raw)
    # there should not be any outliers except for nhood 0, sample 2 (idx 1):
    # and neighborhood 1 should have been removed as all np.nans:
    expected_outlier_mask = np.array(
        [
            [True, False, True, True, True, True],  # nhood 0
            [True, True, True, True, True, True],  # nhood 2
        ]
    ).T
    assert np.all(outlier_mask == expected_outlier_mask)


def test_mask_rows_and_columns(pw_dists_valid, r_dists_to_ref_raw):
    outlier_mask = _identify_outliers(r_dists_to_ref_raw)
    masked_pw_dists = _mask_rows_and_columns(pw_dists_valid, outlier_mask)
    # we expect rows and columns for reference sample 2 to be masked,
    # and only in neighborhood 0:
    expected_masked_pw_dists = pw_dists_valid.copy()
    expected_masked_pw_dists[1, :, 0] = np.nan
    expected_masked_pw_dists[:, 1, 0] = np.nan
    # check that the masked pw_dists are correct:
    np.testing.assert_equal(masked_pw_dists, expected_masked_pw_dists)


def test_get_normalized_dists_to_ref(pw_dists_triu, pw_dists_valid, r_dists_to_ref_raw):
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    n_samples_ref = 6
    # get baseline mean and stdev:
    outlier_mask = _identify_outliers(r_dists_to_ref_raw)
    masked_pw_dists_valid = _mask_rows_and_columns(pw_dists_valid, outlier_mask)
    row_masked_pw_dists_valid = _mask_rows_and_columns(pw_dists_valid, outlier_mask, rows_only=True)
    r_dists_to_ref_outliers_removed = np.nanmean(masked_pw_dists_valid[:, :n_samples_ref, :], axis=0)
    baseline_mean = np.nanmean(r_dists_to_ref_outliers_removed, axis=0)
    baseline_stdev = np.nanstd(r_dists_to_ref_outliers_removed, axis=0, ddof=1)
    # create params object:
    params = _MapQCParams(samples_r=list(range(n_samples_ref)))
    # get output from normalize_pw_dists:
    normalized_dists_to_ref = _get_normalized_dists_to_ref(params=params, pw_dists_triu=pw_dists_triu)
    # predict individual values (in case our matrix subtraction/division below is incorrect):
    # note that neighborhood 1 in the row_masked_pw_dists is neighborhood 2 in the original
    # pw_dists tensor, as we removed neighborhood 1 from the original tensor as it had
    # all np.nans
    # Note also that we use row_masked, as we want to disregard outlier reference samples
    # when calculating a sample's distance to the reference
    # Let's calculate for sample idx 2 in neighborhood 0 and sample idx 4 in neighborhood 2:
    norm_val_1 = np.nanmean((row_masked_pw_dists_valid[:, 2, 0]) - baseline_mean[0]) / baseline_stdev[0]
    norm_val_2 = np.nanmean((row_masked_pw_dists_valid[:, 4, 1]) - baseline_mean[1]) / baseline_stdev[1]
    np.testing.assert_equal(norm_val_1, normalized_dists_to_ref[2, 0])
    np.testing.assert_equal(norm_val_2, normalized_dists_to_ref[4, 2])
    # now predict results per neighborhood (idc for nh2 are 1, see note above).
    expected_pw_dists_normalized_nh0 = (
        np.nanmean(row_masked_pw_dists_valid[:, :, 0], axis=0) - baseline_mean[0]
    ) / baseline_stdev[0]
    expected_pw_dists_normalized_nh2 = (
        np.nanmean(row_masked_pw_dists_valid[:, :, 1], axis=0) - baseline_mean[1]
    ) / baseline_stdev[1]
    # check that the normalized pw_dists are correct:
    np.testing.assert_equal(expected_pw_dists_normalized_nh0, normalized_dists_to_ref[:, 0])
    np.testing.assert_equal(expected_pw_dists_normalized_nh2, normalized_dists_to_ref[:, 2])
    # check that mean and standard deviation of the reference are indeed 0 and 1,
    # excluding reference outliers:
    # subset to reference samples only:
    r_normalized_dists_to_ref = normalized_dists_to_ref[:n_samples_ref, :]
    # check that mean and standard deviation of the reference are indeed 0 and 1,
    # excluding reference outliers:
    # start with nh0:
    nh0_r_normalized_dists_to_ref_outliers_removed = r_normalized_dists_to_ref[outlier_mask[:, 0], 0]
    np.testing.assert_almost_equal(np.nanmean(nh0_r_normalized_dists_to_ref_outliers_removed, axis=0), 0)
    np.testing.assert_almost_equal(np.nanstd(nh0_r_normalized_dists_to_ref_outliers_removed, axis=0, ddof=1), 1)
    # now nh2:
    nh2_r_normalized_dists_to_ref_outliers_removed = r_normalized_dists_to_ref[outlier_mask[:, 1], 2]
    np.testing.assert_almost_equal(np.nanmean(nh2_r_normalized_dists_to_ref_outliers_removed, axis=0), 0)
    np.testing.assert_almost_equal(np.nanstd(nh2_r_normalized_dists_to_ref_outliers_removed, axis=0, ddof=1), 1)
