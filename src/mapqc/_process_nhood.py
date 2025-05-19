import numpy as np
from scipy.spatial.distance import cdist

from mapqc._distances._raw_distances import _pairwise_sample_distances
from mapqc._neighbors._adaptive_k import _filter_and_get_adaptive_k
from mapqc._params import _MapQCParams


def _process_neighborhood(
    params: _MapQCParams,
    center_cell: str,
):
    """Check if nhood passes filtering and calculate pairwise distances.

    Parameters
    ----------
    params: _MapQCParams
        MapQC parameters object.
    center_cell: str
        Center cell of the neighborhood (cell's row name in adata.obs).

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
    adata_emb = params.adata.X if params.adata_emb_loc == "X" else params.adata.obsm[params.adata_emb_loc]
    adata_obs = params.adata.obs
    n_dims_total = adata_emb.shape[1]
    n_samples_r_all = len(params.samples_r)
    n_samples_q_all = len(params.samples_q)
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
    if params.k_max != params.k_min:
        k_max_minus_margin = int(max(params.k_min, np.floor(params.k_max / (1 + params.adaptive_k_margin))))
    else:
        k_max_minus_margin = params.k_min
    # get cell_dataframe with relevant information to do filtering
    metadata_to_keep = [params.ref_q_key]
    if params.exclude_same_study:
        metadata_to_keep.append(params.study_key)
    cell_df = adata_obs.iloc[cell_idc_by_dist[:(k_max_minus_margin)], :].loc[
        :, metadata_to_keep + [params.sample_key]
    ]  # we add 1 to include the center cell
    sample_df = cell_df.groupby(params.sample_key, observed=False).agg(dict.fromkeys(metadata_to_keep, "first"))
    # filter and adapt k if wanted and needed (note that k will automatically not be adapted if cell_df has n_rows=min_k)
    filter_pass, adapted_k, filter_info = _filter_and_get_adaptive_k(
        params=params,
        cell_df=cell_df,
        sample_df=sample_df,
    )
    if not filter_pass:
        # get query samples in neighborhood that have sufficient number of cells:
        knn_idc = cell_idc_by_dist[: params.k_min]
        query_sample_cell_counts = (
            params.adata.obs.iloc[knn_idc, :]
            .groupby(params.sample_key, observed=True)
            .agg({params.ref_q_key: "first", params.sample_key: "size"})
        )
        samples_q_sufficient_cells = sorted(
            query_sample_cell_counts.index[
                (query_sample_cell_counts[params.ref_q_key] == params.q_cat)
                & (query_sample_cell_counts[params.sample_key] >= params.min_n_cells)
            ]
        )
        nhood_info_dict = {
            "center_cell": center_cell,
            "k": np.nan,
            "knn_idc": knn_idc,
            "filter_info": filter_info,
            "samples_q": samples_q_sufficient_cells,
        }
        nhood_sample_pw_dists = np.full((n_samples_r_all, n_samples_r_all + n_samples_q_all), np.nan)
        return (nhood_info_dict, nhood_sample_pw_dists)
    else:
        knn_idc = cell_idc_by_dist[:adapted_k]  # note that we include the center cell in our k count
        nhood_emb = adata_emb[knn_idc, :]
        nhood_obs = adata_obs.iloc[knn_idc, :]
        if params.exclude_same_study:
            sample_df = sample_df
            # study_key = params.study_key
        else:
            sample_df = None
            # study_key = None
        # calculate pairwise distances between all samples in the neighborhood
        samples_q, nhood_sample_pw_dists = _pairwise_sample_distances(
            params=params,
            emb=nhood_emb,
            obs=nhood_obs,
            sample_df=sample_df,
        )
        nhood_info_dict = {
            "center_cell": center_cell,
            "k": adapted_k,
            "knn_idc": knn_idc,
            "filter_info": filter_info,
            "samples_q": samples_q,
        }
        return (nhood_info_dict, nhood_sample_pw_dists)
