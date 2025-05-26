"""
Evaluation module for mapqc package.

This module provides functions for evaluating the performance of mapqc.
"""

import anndata
import pandas as pd
import scanpy as sc


def evaluate(
    adata: anndata.AnnData,
    case_control_key: str,
    case_cats: list[str],
    control_cats: list[str],
) -> dict:
    """
    Evaluate and summarize mapQC output.

    Parameters
    ----------
    adata
        AnnData object with mapQC output (i.e. mapqc.run() has been run)
    case_control_key
        Column name in adata.obs that contains the case-control information
    case_cats
        Unique case categories, i.e. the non-control categories to be evaluated
        for the query cells only, as a list (e.g. ["IPF", "COPD"]). Each
        category will be evaluated separately. If only one category exists,
        still provide as a list (e.g. ["IPF"]).
    control_cats
        Unique control categories in adata.obs[case_control_key] for the query
        cells only, as a list (e.g. ["Control", "Control_2"]). These will be
        evaluated as one group. The controls are considered to be the same as
        the reference used to map against.

    Returns
    -------
    dict of statistics:
        Dictionary containing the following statistics:

        * perc_nhoods_pass : float
            Percentage of neighborhoods that passed filtering
        * perc_cells_sampled : float
            Percentage of cells that were sampled
        * perc_sampled_cells_pass : float
            Percentage of sampled cells that passed filtering
        * perc_[control_cat]_cells_dist_to_ref : float
            Percentage of [control_cat] cells that passed filtering that were
            distant to the reference (mapQC score > 2)
        * perc_[case_cat]_cells_dist_to_ref : float
            Percentage of [case_cat] cells that passed filtering that were
            distant to the reference (mapQC score > 2), for each case_cat
            included in case_cats

    Notes
    -----
    In addition to returning these statistics as a dictionary, the function
    prints each statistic to the console as it is calculated.

    """
    _validate_input(adata, case_control_key, case_cats, control_cats)
    params = adata.uns["mapqc_params"]
    _format_case_control_key(adata, case_control_key, case_cats, control_cats, params)
    _binarize_mapqc_scores(adata)
    ref_q_key = params["ref_q_key"]
    q_cat = params["q_cat"]
    stats = {}
    # collect neighborhood-level statistics
    n_nhoods_total = params["n_nhoods"]
    n_nhoods_pass = (adata.obs["mapqc_nhood_filtering"] == "pass").sum()
    perc_nhoods_pass = round(n_nhoods_pass / n_nhoods_total * 100, 1)
    stats["perc_nhoods_pass"] = perc_nhoods_pass
    print(f"{perc_nhoods_pass}% of neighborhoods passed filtering ({n_nhoods_pass} out of {n_nhoods_total}).")
    if perc_nhoods_pass < 50:
        print(
            "For more details on filtering, run mapqc.pl.umap.neighborhood_filtering(adata), or set return_nhood_info_df=True when running mapQC, and check out your nhood_info_df."
        )
    if perc_nhoods_pass == 0:
        print(
            "Try running with a larger k_min and/or k_max, looser filtering thresholds, and/or a larger number of neighborhoods."
        )
        return stats
    # collect cell-level filtering statistics
    cell_filtering_info = adata.obs.loc[adata.obs[ref_q_key] == q_cat, "mapqc_filtering"]
    if "nan" in cell_filtering_info:
        cell_filtering_info.loc[cell_filtering_info.values == "nan"] = None
    n_cells_total = len(cell_filtering_info)
    n_cells_not_sampled = (cell_filtering_info == "not sampled").sum()
    n_cells_sampled = n_cells_total - n_cells_not_sampled
    stats["perc_cells_sampled"] = round(n_cells_sampled / n_cells_total * 100, 1)
    print(f"{stats['perc_cells_sampled']}% of cells were sampled ({n_cells_sampled} out of {n_cells_total}).")
    n_cells_pass = (cell_filtering_info == "pass").sum()
    perc_cells_pass = round(n_cells_pass / n_cells_sampled * 100, 1)
    stats["perc_sampled_cells_pass"] = perc_cells_pass
    print(
        f"{perc_cells_pass}% of sampled cells passed filtering ({n_cells_pass} out of {n_cells_sampled} sampled cells)."
    )
    # collect cell-level mapqc statistics
    cell_mapqc_stats = pd.crosstab(
        adata.obs.loc[adata.obs[ref_q_key] == q_cat, "case_control"],
        adata.obs.loc[adata.obs[ref_q_key] == q_cat, "mapqc_score_binary"],
    )
    cell_mapqc_stats = cell_mapqc_stats.div(cell_mapqc_stats.sum(axis=1), axis=0)
    control_cat = [cat for cat in adata.obs.case_control.unique() if cat.startswith("Control")][0]
    perc_ctr_cells_dist_to_ref = round(cell_mapqc_stats.loc[control_cat, ">2"] * 100, 1)
    print(
        f"Percentage of {control_cat} cells (that passed filtering) distant to the reference (mapQC score > 2): {perc_ctr_cells_dist_to_ref}%"
    )
    stats[f"perc_{control_cat}_cells_dist_to_ref"] = perc_ctr_cells_dist_to_ref
    case_cats = [cat for cat in adata.obs.case_control.unique() if cat.startswith("Case")]
    for case_cat in case_cats:
        perc_case_cells_dist_to_ref = round(cell_mapqc_stats.loc[case_cat, ">2"] * 100, 1)
        print(
            f"Percentage of {case_cat} cells (that passed filtering) distant to the reference (mapQC score > 2): {perc_case_cells_dist_to_ref}%"
        )
        stats[f"perc_{case_cat}_cells_dist_to_ref"] = perc_case_cells_dist_to_ref
    return stats


def _validate_input(
    adata: sc.AnnData,
    case_control_key: str,
    case_cats: list[str],
    control_cats: list[str],
):
    # check that adata has mapqc_params:
    if "mapqc_params" not in adata.uns:
        raise ValueError("There is no mapqc_params in your adata.uns. Please run mapqc.run() first.")
    params = adata.uns["mapqc_params"]
    ref_q_key = params["ref_q_key"]
    q_cat = params["q_cat"]
    # check if case_control_key is a column in adata.obs
    if case_control_key not in adata.obs.columns:
        raise ValueError(f"case_control_key {case_control_key} not found in adata.obs")
    # check that all query cells have values for case_control_key
    if pd.isnull(adata.obs.loc[adata.obs[ref_q_key] == q_cat, case_control_key]).any():
        raise ValueError(f"There are query cells with null values for {case_control_key} in the adata.obs")
    # check that case_cats and control_cats are lists
    if not isinstance(case_cats, list):
        raise ValueError("case_cats must be a list")
    if not isinstance(control_cats, list):
        raise ValueError("control_cats must be a list")
    # check that case_cats and control_cats cover all query values for case_control_key
    observed_query_case_control_values = adata.obs.loc[adata.obs[ref_q_key] == q_cat, case_control_key].unique()
    if not set(observed_query_case_control_values) == set(case_cats + control_cats):
        raise ValueError(
            f"case_cats and control_cats must cover all query values for {case_control_key}. Observed values: {observed_query_case_control_values}"
        )
    # check that adata has not been subsetted:
    n_nhoods_expected = params["n_nhoods"]
    n_non_nan_nhood = adata.obs["mapqc_nhood_filtering"].notna().sum()
    n_non_nan_str_nhood = (adata.obs["mapqc_nhood_filtering"] != "nan").sum()
    if (n_non_nan_nhood != n_nhoods_expected) and (n_non_nan_str_nhood != n_nhoods_expected):
        raise ValueError(
            f"It looks like the adata object has been subsetted. (Filtering info is available for only {n_non_nan_nhood} neighborhoods). Please run mapqc.run() and mapqc.evaluate() on the same adata object."
        )


def _format_case_control_key(
    adata: sc.AnnData,
    case_control_key: str,
    case_cats: list[str],
    control_cats: list[str],
    params: dict,
):
    ref_q_key = params["ref_q_key"]
    r_cat = params["r_cat"]
    control_cat_name = f"Control ({', '.join(control_cats)})"
    case_control_mapping = {
        cat: f"Case ({cat})" if cat in case_cats else control_cat_name for cat in adata.obs[case_control_key].unique()
    }
    adata.obs["case_control"] = adata.obs[case_control_key].map(case_control_mapping)
    adata.obs.case_control = adata.obs.case_control.tolist()
    adata.obs.loc[adata.obs[ref_q_key] == r_cat, "case_control"] = "Reference"


def _binarize_mapqc_scores(
    adata: sc.AnnData,
):
    # add binarized version of mapqc_score to adata.obs
    adata.obs["mapqc_score_binary"] = None
    adata.obs.loc[adata.obs.mapqc_filtering.isna(), "mapqc_score_binary"] = "Reference"
    adata.obs.loc[adata.obs.mapqc_score > 2, "mapqc_score_binary"] = ">2"
    adata.obs.loc[adata.obs.mapqc_score <= 2, "mapqc_score_binary"] = "<=2"
    adata.obs.loc[adata.obs.mapqc_filtering == "not sampled", "mapqc_score_binary"] = "not sampled"
    adata.obs.loc[
        adata.obs.mapqc_filtering.isin(
            [
                "not enough query cells",
                "not enough reference samples",
                "not enough reference samples from different studies",
            ]
        ),
        "mapqc_score_binary",
    ] = "filtered out"
