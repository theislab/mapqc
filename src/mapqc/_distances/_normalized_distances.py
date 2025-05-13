import warnings

import numpy as np

from mapqc._params import _MapQCParams


def _get_normalized_dists_to_ref(
    params: _MapQCParams,
    pw_dists_triu: np.ndarray,
) -> np.ndarray:
    """Calculate normalized distance to the reference for each sample in each nhood.

    Parameters
    ----------
    params: _MapQCParams
        MapQC parameters object
    pw_dists_triu: np.ndarray
        Tensor of shape (n_ref_samples, n_ref_samples + n_query_samples, n_neighborhoods)
        containing pairwise distances between samples for each neighborhood. Neighborhoods
        that did not pass filtering are expected to have all np.nans. Order of samples
        in rows and columns is fixed (across this package).

    Returns
    -------
    sample_dists_to_ref_normalized: np.ndarray
        Matrix of shape (n_ref_samples + n_query_samples, n_neighborhoods)
        containing normalized distance to the reference for each sample in each nhood.
        The normalized distances are based on the raw distances, but Z-scoring them using
        the mean and standard deviation of the reference samples' distance to the
        reference, excluding outliers (see code for outlier definition).
    """
    n_ref_samples = len(params.samples_r)
    # note that we were working with only the upper triangle
    # (hence triu suffix above) of the pairwise distances between reference
    # samples for each neighborhood - the rest was set to nan for
    # computational efficiency (as dist(i,j) = dist(j,i)).
    # However, we now need to calcualte the mean distance of each sample
    # to *all* other reference samples, and we'll do that by taking the
    # mean across the axis 0, so we need to fill in our lower triangle as well.
    # We'll do that here:
    pw_dists = _fill_lower_triangle(pw_dists_triu)
    # Note that we have slices in this 3D tensor (n_ref_samples x n_ref_samples x n_neighborhoods)
    # that have all nans; these are neighborhoods that did not pass filtering.
    # We'll therefore subset to only valid neighborhoods. We will add the
    # invalid neighborhoods back in at the end.
    # Get all neighborhoods that have at least one entry that is not nan:
    valid_nhoods = ~(np.isnan(pw_dists).all(axis=(0, 1)))
    # Store indices of these valid neighborhoods:
    valid_nhood_idc = np.where(valid_nhoods)[0]
    # Now continue with only the valid neighborhoods:
    pw_dists_valid = pw_dists[:, :, valid_nhood_idc].copy()
    # Subset to reference only for identifying outliers
    # (we are only interested in identifying reference outliers),
    # and calculate distance to the reference (which is the mean
    # distance to all (other) reference
    # samples for each sample), i.e. mean along axis 0.
    pw_dists_r_to_r_valid = pw_dists_valid[:, :n_ref_samples, :]
    with warnings.catch_warnings():
        # note that we expect some empty slices here: not all
        # reference samples are present in each neighborhood.
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
        r_dists_to_ref_raw = np.nanmean(pw_dists_r_to_r_valid, axis=0)
    # Now identify outlier reference samples:
    outlier_mask = _identify_outliers(r_dists_to_ref_raw)
    # Note that this is a 1-D mask (specifying which reference
    # samples to keep). To apply this to both our rows and columns
    # of our 3D tensor (rows and first (n_samples_r) columns are
    # the same reference samples), we'll use our masking function
    # below. We want to mask both rows and columns to calculate
    # our baseline characteristics (mean and stdev) excluding
    # outliers.
    pw_dists_valid_outliers_removed = _mask_rows_and_columns(pw_dists_valid, outlier_mask)
    # for the final normalized distances to the reference, we do
    # want to include the outlier samples (just to keep track of
    # all distances to the reference). We will therefore also
    # create a tensor with the outlier sample *rows* excluded,
    # such that this sample is not taken into account when calculating
    # the mean absolute distance of a sample to the reference.
    # However, we do want to take the sample into account when calculating
    # each samples distance to the reference (this is done per column),
    # just for plotting purposes later, to understand what each neighborhood
    # contains. We will therefore also create a tensor with the outlier
    # samples only excluded in the rows. Note that this does not
    # change/affect final mapqc scores (i.e. distances for the query samples).
    pw_dists_valid_outlier_rows_removed = _mask_rows_and_columns(pw_dists_valid, outlier_mask, rows_only=True)
    # calculate sample distances to the reference now,
    # which we define as a sample's mean distance to
    # all (other) reference samples, per neighborhood
    with warnings.catch_warnings():
        # filter out warnings caused by empty slices, we expect these
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        # calculate for all samples:
        sample_dists_to_ref = np.nanmean(pw_dists_valid_outlier_rows_removed, axis=0)
        # calculate for all samples, excluding outliers, to calculate our baseline
        # characteristics (mean and stdev):
        sample_dists_to_ref_outliers_removed = np.nanmean(pw_dists_valid_outliers_removed, axis=0)
    # Now calculate the baseline/null distribution characteristics;
    # these are based on the reference samples only.
    r_sample_dists_to_ref_outliers_removed = sample_dists_to_ref_outliers_removed[:n_ref_samples, :]
    with warnings.catch_warnings():
        # filter out warnings caused by empty slices, we expect these
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")
        baseline_mean_dist_to_ref = np.nanmean(r_sample_dists_to_ref_outliers_removed, axis=0)
        # subtract 1 from degrees of freedom as we're looking at sample and not population standard deviation
        baseline_stdev_dist_to_ref = np.nanstd(r_sample_dists_to_ref_outliers_removed, axis=0, ddof=1)
    # Now, normalize the sample distances based on the baseline
    # mean and standard deviation. Note that we include the outliers
    # here, we only wanted to exclude them for mean/stdev calculation.
    sample_dists_to_ref_normalized = (sample_dists_to_ref - baseline_mean_dist_to_ref) / baseline_stdev_dist_to_ref
    # create an nan array with sample distances to the reference
    # per neighborhood; here we go back to the original shape,
    # including neighborhoods we excluded above:
    sample_dists_to_ref_normalized_full = np.full(pw_dists.shape[1:], np.nan)
    # add in normalized distances:
    sample_dists_to_ref_normalized_full[:, valid_nhood_idc] = sample_dists_to_ref_normalized
    # return normalized distances:
    return sample_dists_to_ref_normalized_full


def _fill_lower_triangle(arr: np.ndarray) -> np.ndarray:
    """Fills lower triangle (excluding diagonal) with values from upper triangle.

    If input tensor is not square but includes more columns (which represent query
    sample distances to ref samples), the extra columns will stay as is.

    Parameters
    ----------
    arr: np.ndarray
        Input tensor of shape (n_r, n_all, n_n)

    Returns
    -------
    arr_full: np.ndarray
        Output tensor of shape (n_r, n_all, n_n)
        where the lower triangle (excluding diagonal) is filled with values from the upper triangle.
        The extra columns (representing query sample distances to ref samples) stay as the were,
        they were never upper-triangulated anyway.
    """
    # first copy the tensor:
    arr_full = arr.copy()
    # Get the indices of the lower triangle, excluding the diagonal
    # (we still want to disregard distances to same sample)
    i_lower = np.tril_indices(arr.shape[0], -1)  # -1 excludes diagonal
    # by swapping the indices, we basically mirror the upper triangle
    arr_full[i_lower[0], i_lower[1], :] = arr[i_lower[1], i_lower[0], :]
    return arr_full


def _identify_outliers(
    r_sample_dist_to_ref: np.ndarray,
    max_n_interquartile_range_from_hinge: float = 3.5,
) -> np.ndarray:
    """Identify reference outlier samples based on their distance to other reference samples.

    Parameters
    ----------
    r_sample_dist_to_ref: np.ndarray
        Array of shape (n_r, n_n) where n_r is the number of reference samples and n_n
        is the number of neighborhoods. The array represents the mean distance of each
        reference sample to all other reference samples, per neighborhood.
    max_n_interquartile_range_from_hinge: float, optional
        The maximum number of interquartile ranges from the hinge (i.e. above the 75th
        percentile, or below the 25th percentile) that a sample's mean distance to the
        reference can be away from the hinge before being considered an outlier.
        Default is 3.5.

    Returns
    -------
    mask: np.ndarray
        Boolean mask of shape (n_r, n_n) set to True for samples to keep, and False
        for outlier samples, for each neighborhood.
    """
    nhood_25percentiles = np.nanpercentile(r_sample_dist_to_ref, q=25, axis=0)
    nhood_75percentiles = np.nanpercentile(r_sample_dist_to_ref, q=75, axis=0)
    nhood_interquartile_range = nhood_75percentiles - nhood_25percentiles
    upper_threshold_outliers = nhood_75percentiles + max_n_interquartile_range_from_hinge * nhood_interquartile_range
    lower_threshold_outliers = nhood_25percentiles - max_n_interquartile_range_from_hinge * nhood_interquartile_range
    upper_outliers = r_sample_dist_to_ref > upper_threshold_outliers
    lower_outliers = r_sample_dist_to_ref < lower_threshold_outliers
    outliers = upper_outliers | lower_outliers
    mask = ~outliers
    return mask


def _mask_rows_and_columns(tensor: np.ndarray, mask: np.ndarray, rows_only: bool = False) -> np.ndarray:
    """
    Mask specific rows and columns in a 3D tensor based on a boolean mask.

    Args:
        tensor: 3D numpy array of shape (n_r, n_all, n_n)
        mask: 2D boolean array of shape (n_r, n_n) where False indicates rows/columns to mask
        rows_only: bool, optional
            If True, only mask the rows of the tensor. If False, mask both rows and columns.
            Default is False.

    Returns
    -------
        Modified tensor with specified rows and columns set to np.nan
    """
    # Create a copy to avoid modifying the original
    masked_tensor = tensor.copy()

    # If the shape of the mask and tensor differ (looking only
    # at dim 2 and 3 of the tensor), this is because
    # the mask was calculated on the reference sample only, and
    # the tensor includes reference and query samples. As we want to
    # keep all our query samples (no filtering there), we can set
    # all of those to True. Therefore, let's augment our mask,
    # for the columns only:
    if mask.shape != tensor.shape[1:]:
        n_missing_rows = tensor.shape[1] - mask.shape[0]
        augmentation = np.full((n_missing_rows, mask.shape[1]), True)
        col_mask = np.append(mask, augmentation, axis=0)
    else:
        col_mask = mask
    # Create 3D masks for rows and columns
    # For rows: expand mask to (n_r, n_all, n_n), which is
    # currently (n_r, n_n) (i.e. info about which reference
    # samples did not pass outlier filtering for each neighborhood).
    # First, we add a new axis for the missing dimension. As we
    # want to apply our row mask to the rows, our missing dimension
    # are the columns of the tensor, so we add that using np.newaxis.
    # Then, we broadcast (i.e. stack-repeat) the mask to match the
    # dimension of the actual tensor
    row_mask = np.broadcast_to(mask[:, np.newaxis, :], tensor.shape)
    # Apply row mask: set entire rows to nan where mask is False
    masked_tensor = np.where(row_mask, masked_tensor, np.nan)
    if rows_only:
        return masked_tensor
    # For columns: we now add an extra dimension for the rows,
    # and follow the same logic as above.
    col_mask = np.broadcast_to(col_mask[np.newaxis, :, :], tensor.shape)
    # Apply column mask: set entire columns to nan where mask is False
    masked_tensor = np.where(col_mask, masked_tensor, np.nan)
    return masked_tensor
