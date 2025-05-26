# min n_samples r should be at least 3, as with two samples, there will only be one distance which is not enough to
# calculate a standard deviation. Consider even setting the minimum to 4...
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from mapqc._params import _MapQCParams


def _validate_input_params(params: _MapQCParams):
    """Validate input parameters"""
    _check_adata(params.adata, params.adata_emb_loc)
    _check_ref_q_arguments(params.adata, params.ref_q_key, params.q_cat, params.r_cat)
    _check_sample_information(params.adata, params.sample_key, params.ref_q_key, params.r_cat)
    _check_study_arguments(params.adata, params.study_key, params.exclude_same_study, params.ref_q_key, params.r_cat)
    _check_n_nhoods(params.adata, params.n_nhoods, params.ref_q_key, params.q_cat)
    _check_k_arguments(
        params.adata,
        params.ref_q_key,
        params.q_cat,
        params.k_min,
        params.k_max,
        params.adaptive_k_margin,
    )
    _check_min_n_cells_and_samples(params.min_n_cells, params.min_n_samples_r)
    _check_grouping_key(params.adata, params.grouping_key)
    _check_distance_metric(params.distance_metric)
    _check_seed(params.seed)
    _check_overwrite(params.overwrite, params.adata)
    _check_verbose(params.verbose)


def _check_adata(adata, adata_emb_loc):
    """Check if adata and adata_emb_loc are valid"""
    if not isinstance(adata, sc.AnnData):
        raise ValueError("adata must be a sc.AnnData object")
    if not isinstance(adata_emb_loc, str):
        raise ValueError(
            "adata_emb_loc must be a string, specifying the location of the embedding in obsm, or X if in adata.X."
        )
    if adata_emb_loc != "X":
        if adata_emb_loc not in adata.obsm.keys():
            raise ValueError("adata_emb_loc must be a key in adata.obsm, or 'X' if embedding is adata.X.")
    if adata_emb_loc == "X":
        adata_emb = adata.X
    else:
        adata_emb = adata.obsm[adata_emb_loc]
    if not isinstance(adata_emb, np.ndarray) and not isinstance(adata_emb, sparse.spmatrix):
        raise ValueError("Your embedding must be a numpy array or a scipy sparse matrix.")
    if adata_emb.shape[1] > 500:
        warnings.warn(
            "Your embedding has more than 500 dimensions. Are you sure this is the low-dimensional mapping-based embedding?",
            stacklevel=4,
        )


def _check_ref_q_arguments(adata, ref_q_key, q_cat, r_cat):
    """Check if ref_q_key, q_cat, and r_cat are valid"""
    if not isinstance(ref_q_key, str):
        raise ValueError(
            "ref_q_key must be a string, specifying the column in adata.obs that contains the reference and query categories."
        )
    if not isinstance(q_cat, str):
        raise ValueError("q_cat must be a string, specifying the query category.")
    if not isinstance(r_cat, str):
        raise ValueError("r_cat must be a string, specifying the reference category.")
    if pd.isnull(adata.obs[ref_q_key]).any() or (adata.obs[ref_q_key] == "nan").any():
        raise ValueError("ref_q_key must not contain any null values.")
    if q_cat == r_cat:
        raise ValueError("q_cat and r_cat must be different.")
    if q_cat not in adata.obs[ref_q_key].unique():
        raise ValueError("q_cat must be a category in adata.obs[ref_q_key].")
    if r_cat not in adata.obs[ref_q_key].unique():
        raise ValueError("r_cat must be a category in adata.obs[ref_q_key].")
    if adata.obs[ref_q_key].unique().size > 2:
        raise ValueError(
            "ref_q_key must have exactly two unique values, corresponding to the reference and query categories."
        )


def _check_sample_information(adata, sample_key, ref_q_key, r_cat):
    """Check if sample_key is valid"""
    if not isinstance(sample_key, str):
        raise ValueError(
            "sample_key must be a string, specifying the column in adata.obs that contains the sample information."
        )
    if sample_key not in adata.obs.columns:
        raise ValueError("sample_key must be a column in adata.obs.")
    if pd.isnull(adata.obs[sample_key]).any() or (adata.obs[sample_key] == "nan").any():
        raise ValueError("The values in your sample_key adata.obs columnmust not contain any null values.")
    if (adata.obs.groupby(sample_key, observed=True).agg({ref_q_key: "nunique"}).values > 1).any():
        raise ValueError("Each sample must have only one unique value of ref_q_key.")
    # check how many reference samples are present in the data:
    n_ref_samples = adata.obs.groupby(ref_q_key, observed=True).agg({sample_key: "nunique"}).loc[r_cat].values[0]
    if n_ref_samples < 3:
        raise ValueError(
            "You have only {n_ref_samples} reference samples in your data. You need at least 3 to be able to run mapQC."
        )
    elif n_ref_samples < 10:
        warnings.warn(
            f"You have only {n_ref_samples} reference samples in your data. Note that mapQC is meant to be used on mappings to large references, and its assumptions are unlikely to hold.",
            stacklevel=4,
        )


def _check_study_arguments(adata, study_key, exclude_same_study, ref_q_key, r_cat):
    """Check if study_key and exclude_same_study are valid"""
    if not isinstance(exclude_same_study, bool):
        raise ValueError("exclude_same_study must be a boolean.")
    if not exclude_same_study:
        if study_key is not None:
            warnings.warn(
                "study_key argument was set, but will be ignored as exclude_same_study was set to False.",
                stacklevel=4,
            )
    else:
        if not isinstance(study_key, str):
            raise ValueError(
                "study_key must be a string, specifying the column in adata.obs that contains the study information."
            )
        if study_key not in adata.obs.columns:
            raise ValueError("study_key must be a column in adata.obs.")
        if pd.isnull(adata.obs[study_key]).any() or (adata.obs[study_key] == "nan").any():
            raise ValueError("The values in your study_key adata.obs column must not contain any null values.")
        # check number of reference studies:
        n_ref_studies = adata.obs.groupby(ref_q_key, observed=True).agg({study_key: "nunique"}).loc[r_cat].values[0]
        if n_ref_studies < 3:
            warnings.warn(
                f"You have only {n_ref_studies} reference studies in your data. Note that mapQC is meant to be used on mappings to large references, and its assumptions might not hold in your context.",
                stacklevel=4,
            )
            if exclude_same_study:
                warnings.warn(
                    "Given your low number of reference studies, we recommend setting the exclude_same_study argument to False.",
                    stacklevel=4,
                )


def _check_n_nhoods(adata, n_nhoods, ref_q_key, q_cat):
    """Check if n_nhoods is valid"""
    if not isinstance(n_nhoods, int):
        raise ValueError("n_nhoods must be an integer.")
    if n_nhoods < 1:
        raise ValueError("n_nhoods must be greater than 0.")
    n_query_cells = adata.obs[ref_q_key].value_counts()[q_cat]
    if n_nhoods > n_query_cells:
        raise ValueError("n_nhoods must be less than the number of query cells.")
    if n_query_cells / n_nhoods < 50:
        warnings.warn(
            "Note that your number of neighborhoods n_nhoods is relatively high given the number of query cells in your data. This is likely unnecessary and will slow down calculation of mapQC scores.",
            stacklevel=4,
        )


def _check_k_arguments(adata, ref_q_key, q_cat, k_min, k_max, adaptive_k_margin):
    """Check if k_min, k_max, and adaptive_k_margin are valid"""
    n_query_cells = adata.obs[ref_q_key].value_counts()[q_cat]
    if not isinstance(k_min, int):
        raise ValueError("k_min must be an integer.")
    if not isinstance(k_max, int):
        raise ValueError("k_max must be an integer.")
    if k_max < k_min:
        raise ValueError("k_max must be greater than or equal to k_min.")
    elif k_max > k_min:
        if not isinstance(adaptive_k_margin, float):
            raise ValueError("adaptive_k_margin must be a float.")
        if adaptive_k_margin < 0:
            raise ValueError("adaptive_k_margin must be greater than 0.")
        if adaptive_k_margin > 1:
            raise ValueError("adaptive_k_margin must be less than 1.")
        if k_max / k_min > 20:
            warnings.warn(
                "We do not recommend setting k_max to more than 20 times k_min. This is likely to result in very large k values, which may slow down calculation of mapQC scores.",
                stacklevel=4,
            )
    if n_query_cells / k_min < 50:
        warnings.warn(
            f"Your total number of query cells is {n_query_cells}, while you set your k_min to {k_min}. This could be a too large k_min given the number of query cells, unless you expect very little cellular heterogeneity in your data (e.g. a limited number of cell types).",
            stacklevel=4,
        )


def _check_min_n_cells_and_samples(min_n_cells, min_n_samples_r):
    """Check if min_n_cells and min_n_samples_r are valid"""
    if not isinstance(min_n_cells, int):
        raise ValueError("min_n_cells must be an integer.")
    if not isinstance(min_n_samples_r, int):
        raise ValueError("min_n_samples_r must be an integer.")
    if min_n_cells < 5:
        raise ValueError("min_n_cells must be at least 5.")
    if min_n_samples_r < 3:
        raise ValueError("min_n_samples_r must be at least 3.")


def _check_grouping_key(adata, grouping_key):
    """Check if grouping_key is valid"""
    if grouping_key is not None:
        if not isinstance(grouping_key, str):
            raise ValueError(
                "grouping_key must be a string, specifying the column in adata.obs that contains the grouping information."
            )
        if grouping_key not in adata.obs.columns:
            raise ValueError("grouping_key must be a column in adata.obs.")
        if pd.isnull(adata.obs[grouping_key]).any() or (adata.obs[grouping_key] == "nan").any():
            raise ValueError("The values in your grouping_key adata.obs column must not contain any null values.")


def _check_distance_metric(distance_metric):
    """Check if distance_metric is valid"""
    if distance_metric not in ["energy_distance", "pairwise_euclidean"]:
        raise ValueError("distance_metric must be one of 'energy_distance' or 'pairwise_euclidean'.")


def _check_seed(seed):
    """Check if seed is valid"""
    if seed is not None:
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer.")
        if seed < 0:
            raise ValueError("seed must be greater than 0.")


def _check_overwrite(overwrite, adata):
    """Check if overwrite is valid"""
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean.")
    if not overwrite:
        if "mapqc_score" in adata.obs.columns:
            raise ValueError("mapqc_score column already exists in adata.obs. Set overwrite to True to overwrite it.")
        if "mapqc_filtering" in adata.obs.columns:
            raise ValueError(
                "mapqc_filtering column already exists in adata.obs. Set overwrite to True to overwrite it."
            )


def _check_verbose(verbose):
    """Check if verbose is valid"""
    if not isinstance(verbose, bool):
        raise ValueError("verbose must be a boolean.")


def _validate_return_parameters(return_nhood_info_df, return_sample_dists_to_ref_df):
    """Check if return_nhood_info_df and return_sample_dists_to_ref_df are valid"""
    if not isinstance(return_nhood_info_df, bool):
        raise ValueError("return_nhood_info_df must be a boolean.")
    if not isinstance(return_sample_dists_to_ref_df, bool):
        raise ValueError("return_sample_dists_to_ref_df must be a boolean.")
