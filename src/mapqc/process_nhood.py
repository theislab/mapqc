from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from mapqc.distances.raw_distances import pairwise_sample_distances
from mapqc.neighbors.adaptive_k import filter_and_get_adaptive_k


def process_neighborhood(
    center_cell: str,
    adata_emb: np.ndarray,
    adata_obs: pd.DataFrame,
    samples_r_all: list[str],
    samples_q_all: list[str],
    k_min: int,
    k_max: int,
    sample_key: str,
    ref_q_key: str,
    q_cat: str,
    r_cat: str,
    min_n_cells: int,
    min_n_samples_r: int,
    exclude_same_study: bool,
    distance_metric: Literal["energy_distance", "pairwise_euclidean"] = "energy_distance",
    adaptive_k_margin: float = None,
    study_key: str = None,
):
    """Check if nhood passes filtering and calculate pairwise distances.

    Parameters
    ----------
    center_cell: str
        Center cell of the neighborhood (cell's row name in adata.obs).
    adata_emb: np.ndarray
        Embedding of all cells in the data (not just the neighborhood).
    adata_obs: pd.DataFrame
        Metadata of all cells in the data (not just the neighborhood).
        Should include ref_q column, sample column, and study column
        if exclude_same_study is True.
    samples_r_all: list[str]
        List of all reference sample names, ordered consistently across neighborhoods,
        including also samples that are not in the neighborhood.
    samples_q_all: list[str]
        List of all query sample names, ordered consistently across neighborhoods,
        including also samples that are not in the neighborhood.
    k_min: int
        Minimum number of cells per neighborhood.
    k_max: int
        Maximum number of cells per neighborhood.
    sample_key: str
        Column name of the sample column in adata_obs.
    ref_q_key: str
        Column name of the reference query column in adata_obs.
    q_cat: str
        Category of the query cells in adata_obs[ref_q_key].
    r_cat: str
        Category of the reference cells in adata_obs[ref_q_key].
    min_n_cells: int
        Minimum number of cells per sample.
    min_n_samples_r: int
        Minimum number of samples of the reference that have at least
        min_n_cells, per neighborhood.
    exclude_same_study: bool
        Whether to exclude pairs of samples from the same study when
        calculating pairwise distances between reference samples.
    distance_metric: Literal["energy_distance", "pairwise_euclidean"] = "energy_distance"
        Distance metric to use to calculate distances between samples (i.e. between
        two sets of cells).
    adaptive_k_margin: float = None
        Margin to add(as a fraction of the minimum number of cells) to the minimum
        number of cells needed to pass filtering when adapting k. Only needed if
        k can be adapted, i.e. if k_max != k_min.
    study_key: str = None
        Column name of the study column in adata_obs. Only needed if
        exclude_same_study is True.

    Returns
    -------
    (nhood_info_dict: dict, nhood_sample_pw_dists: np.ndarray)
        nhood_info_dict: Dictionary containing information about the neighborhood,
        specifically:
            center_cell: Center cell of the neighborhood (row name in adata.obs).
            k: Number of cells in this neighborhood (possibly adapted to pass filtering).
            knn_idc: Indices of the cells in the neighborhood (as row number in adata_obs).
            filter_info: Filtering outcome ('pass' or reason for failing).
        nhood_sample_pw_dists: Matrix of pairwise distances, with *all* reference samples
            in the rows and *all* samples (reference and query, respectively)in the columns,
            according to the order of the input lists samples_r_all and samples_q_all.
            Samples (or sample pairs) that did not pass filtering or were not present in
            the neighborhood are set to NaN. If the neighborhood did not pass filtering,
            all values are set to NaN.
    """
    n_dims_total = adata_emb.shape[1]
    n_samples_r_all = len(samples_r_all)
    n_samples_q_all = len(samples_q_all)
    cc_idx = np.where(adata_obs.index == center_cell)[0][0]
    # get distances of all cells to center cell
    dists_to_cc = cdist(
        adata_emb[cc_idx, :].reshape((1, n_dims_total)),
        adata_emb,
    )[0]
    # sort cell idc by distance:
    cell_idc_by_dist = np.argsort(dists_to_cc)
    # keep only cells relevant for the neighborhood.
    # if we use an adaptive k, we want to keep the maximum
    # number of cells that might be included in our final nhood.
    # Note that as we add a margin of adaptive_k_margin to the
    # minimum number of cells needed to pass filtering, we only
    # need to check k_max/(1+adaptive_k_margin) cells, so we'll
    # only include those to limit computation time.
    if k_max != k_min:
        k_max_minus_margin = int(max(k_min, np.floor(k_max / (1 + adaptive_k_margin))))
    else:
        k_max_minus_margin = k_min
    if k_min != k_max:
        adapt_k = True
    else:
        adapt_k = False
    # get cell_dataframe with relevant information to do filtering
    metadata_to_keep = [ref_q_key]
    if exclude_same_study:
        metadata_to_keep.append(study_key)
    cell_df = adata_obs.iloc[cell_idc_by_dist[:(k_max_minus_margin)], :].loc[
        :, metadata_to_keep + [sample_key]
    ]  # we add 1 to include the center cell
    sample_df = cell_df.groupby(sample_key, observed=False).agg({cat: "first" for cat in metadata_to_keep})
    # filter and adapt k if wanted and needed (note that k will automatically not be adapted if cell_df has n_rows=min_k)
    filter_pass, adapted_k, filter_info = filter_and_get_adaptive_k(
        cell_df=cell_df,
        ref_q_key=ref_q_key,
        sample_key=sample_key,
        sample_df=sample_df,
        q_cat=q_cat,
        r_cat=r_cat,
        min_n_cells=min_n_cells,
        min_n_samples_r=min_n_samples_r,
        k_min=k_min,
        adapt_k=adapt_k,
        exclude_same_study=exclude_same_study,
        adaptive_k_margin=adaptive_k_margin,
        study_key=study_key,
    )
    if not filter_pass:
        nhood_info_dict = {
            "center_cell": center_cell,
            "k": None,
            "knn_idc": cell_idc_by_dist[:k_min],
            "filter_info": filter_info,
        }
        nhood_sample_pw_dists = np.full((n_samples_r_all, n_samples_r_all + n_samples_q_all), np.nan)
        return (nhood_info_dict, nhood_sample_pw_dists)
    else:
        knn_idc = cell_idc_by_dist[:adapted_k]  # note that we include the center cell in our k count
        nhood_emb = adata_emb[knn_idc, :]
        nhood_obs = adata_obs.iloc[knn_idc, :]
        if exclude_same_study:
            sample_df = sample_df
            study_key = study_key
        else:
            sample_df = None
            study_key = None
        # calculate pairwise distances between all samples in the neighborhood
        nhood_sample_pw_dists = pairwise_sample_distances(
            emb=nhood_emb,
            obs=nhood_obs,
            samples_r_all=samples_r_all,
            samples_q_all=samples_q_all,
            sample_key=sample_key,
            min_n_cells=min_n_cells,
            exclude_same_study=exclude_same_study,
            sample_df=sample_df,
            study_key=study_key,
            distance_metric=distance_metric,
        )
        nhood_info_dict = {
            "center_cell": center_cell,
            "k": adapted_k,
            "knn_idc": knn_idc,
            "filter_info": filter_info,
        }
        return (nhood_info_dict, nhood_sample_pw_dists)
