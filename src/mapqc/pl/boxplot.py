import random
import warnings

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def sample_dists_to_ref_per_nhood(
    adata: anndata.AnnData,
    sample_dists_to_ref_df: pd.DataFrame,
    figsize: tuple[float, float] = (24, 8),
    boxplot_kwargs: dict = None,
    palette: dict = None,
    label_xticks_by: str = None,
    dotsize: float = 1,
    max_n_nhoods: int = 80,
    ylim: tuple[float, float] = None,
    return_fig: bool = False,
):
    """
    Boxplot of sample distances to reference per neighborhood, split by case/control status.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object. Both mapqc.run_mapqc() and mapqc.evaluate() should have been run on
        the adata object.
    sample_dists_to_ref_df : pd.DataFrame
        DataFrame with sample distances to reference. This is an optional output of mapqc.run(),
        when the return_sample_dists_to_ref_df parameter is set to True.
    figsize : tuple[float, float], optional
        Figure size. Default is (24, 8).
    boxplot_kwargs : dict, optional
        Keyword arguments for seaborn's boxplot.
    palette : dict, optional
        Color palette. Default is None.
    label_xticks_by : str, optional
        Label to use for x-axis ticks, based on its center cell (e.g. an annotation in adata.obs).
        Default is None.
    dotsize : float, optional
        Dot size for the dots showing individual data points in the boxplot. Default is 1.
    max_n_nhoods : int, optional
        Maximum number of neighborhoods to plot. Default is 80. These are randomly sampled (with
        a fixed seed).
    ylim: tuple[float, float], optional
        y-axis limits. Default is None.
    return_fig : bool, optional
        Return the figure object. Default is False.
    """
    if boxplot_kwargs is None:
        boxplot_kwargs = {}
    seed = 109
    ref_q_key = adata.uns["mapqc_params"]["ref_q_key"]
    sample_key = adata.uns["mapqc_params"]["sample_key"]
    # drop columns that are all NaN (these are neighborhoods that were filtered out)
    sample_dists_pass_filter = sample_dists_to_ref_df.dropna(axis=1, how="all")
    n_nhoods_total = sample_dists_pass_filter.shape[1]
    # subset to a random subset of filters, so that the plot does not get too crowded
    if n_nhoods_total > max_n_nhoods:
        random.seed(seed)
        random_nhood_numbers = random.sample(range(n_nhoods_total), k=max_n_nhoods)
        random_nhood_names = sample_dists_pass_filter.columns[random_nhood_numbers]
    else:
        # random_nhood_numbers = list(np.arange(n_nhoods_total))
        random_nhood_names = sample_dists_pass_filter.columns.tolist()
    # Create dataframe to plot, combatible with sns.boxplot
    plotting_df = pd.DataFrame(sample_dists_pass_filter.loc[:, random_nhood_names].unstack()).rename(
        columns={0: "dist"}
    )
    plotting_df["nhood"] = plotting_df.index.get_level_values(0)
    # Add case_control information to the plotting dataframe
    sample_df = adata.obs.groupby(sample_key, observed=True).agg({"case_control": "first", ref_q_key: "first"})
    sample_df["case_control"] = sample_df["case_control"].tolist()
    plotting_df["case_control"] = sample_df.loc[plotting_df.index.get_level_values(1), "case_control"].values.flatten()
    plotting_df[ref_q_key] = sample_df.loc[plotting_df.index.get_level_values(1), ref_q_key].values  # .flatten()
    # set colors for the plot
    if palette is None:
        palette = _create_case_control_palette(adata)
    # determine neighborhood order:
    # calculate median distance to the reference of the query samples per neighborhood:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        nhood_median_dist_to_ref = (
            plotting_df.groupby(["nhood", "case_control"], observed=True)
            .agg({"dist": lambda x: np.nanmedian(x)})
            .unstack()
        )
    # remove second level column names:
    nhood_median_dist_to_ref.columns = nhood_median_dist_to_ref.columns.droplevel(0)
    # sort neighborhoods based on median distance to the reference for the query:
    control_cat = [cat for cat in adata.obs.case_control.unique() if cat.startswith("Control")][0]
    nhoods_ordered = nhood_median_dist_to_ref.sort_values(by=control_cat, ascending=False).index

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    x = "nhood"
    y = "dist"
    hue = "case_control"
    sns.boxplot(
        data=plotting_df,
        x=x,
        y=y,
        order=nhoods_ordered,
        hue=hue,
        palette=palette,
        ax=ax,
        saturation=0.5,
        fliersize=0,
        orient="vertical",
        **boxplot_kwargs,
    )
    sns.stripplot(
        data=plotting_df,
        x=x,
        y=y,
        order=nhoods_ordered,
        hue=hue,
        dodge=True,
        ax=ax,
        size=dotsize * 3,
        palette="dark:black",
        legend=False,
        orient="vertical",
    )
    # draw horizontal, dashed red line at y=2:
    plt.axhline(y=2, linestyle="--", alpha=0.8, linewidth=2, color="red")
    leg = plt.legend(loc=(1.01, 0.4), frameon=False, title="")
    leg._legend_box.align = "left"
    if label_xticks_by is None:
        ax.set_xticks(range(len(nhoods_ordered)))
        ax.set_xticklabels(nhoods_ordered, rotation=90)
        ax.set_xlabel("Neighborhood, named by its center cell")
    else:
        nhood_labels = adata.obs.loc[nhoods_ordered, label_xticks_by]
        ax.set_xticks(range(len(nhoods_ordered)))
        ax.set_xticklabels(nhood_labels, rotation=90)
        ax.set_xlabel(f"Neighborhood, named by {label_xticks_by} of its center cell")
    ax.set_ylabel("Sample distance to reference")
    if ylim is not None:
        ax.set_ylim(ylim)
    if return_fig:
        return fig
    else:
        plt.show()


def mapqc_scores(
    adata: anndata.AnnData,
    grouping_key: str,
    group_order: list[str] = None,
    min_n_cells_per_box: int = 10,
    figsize: tuple[float, float] = (24, 8),
    dotsize: float = 1,
    return_fig: bool = False,
    palette_case_control: dict = None,
    color_dots_by: str = None,
    palette_dots: dict = None,
    ylim: tuple[float, float] = None,
    boxplot_kwargs: dict = None,
):
    """
    Boxplot of MapQC scores according to a grouping key, split by case/control status.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object. Both mapqc.run_mapqc() and mapqc.evaluate() should have been run on
        the adata object.
    grouping_key : str
        Key in adata.obs to group the cells by. Note that each group in turn will be split
        by case/control status.
    group_order : list[str], optional
        Order of the groups to plot. If not provided, the groups will be plotted alphabetically.
        Note that this list can also be a subset of the groups in adata.obs[grouping_key].
    min_n_cells_per_box : int, optional
        Minimum number of cells per box. Default is 10. Boxes based on groups with fewer than
        min_n_cells_per_box cells are excluded from the plot.
    figsize : tuple[float, float], optional
        Figure size. Default is (24, 8).
    dotsize : float, optional
        Dot size for the dots showing individual data points in the boxplot. Default is 1.
    return_fig : bool, optional
        Return the figure object. Default is False.
    palette_case_control : dict, optional
        Color palette for the boxes (controls and case categories). Default is None.
    color_dots_by : str, optional
        Metadata category (column in adata.obs) to color the dots by. Default is None.
    palette_dots : dict, optional
        Color palette for the dots, if color_dots_by is provided. Default is None.
    ylim : tuple[float, float], optional
        y-axis limits. Default is None.
    boxplot_kwargs : dict, optional
        Keyword arguments for seaborn's boxplot.
    """
    if boxplot_kwargs is None:
        boxplot_kwargs = {}
    _validate_grouping(adata, grouping_key, group_order)
    ref_q_key = adata.uns["mapqc_params"]["ref_q_key"]
    q_cat = adata.uns["mapqc_params"]["q_cat"]
    control_cat_name = [cat for cat in adata.obs.case_control.unique() if cat.startswith("Control")][0]
    case_cats = [cat for cat in adata.obs.case_control.unique() if cat.startswith("Case")]
    obs_cols_to_keep = [grouping_key, ref_q_key, "mapqc_score", "case_control"]
    if color_dots_by is not None:
        obs_cols_to_keep.append(color_dots_by)
    plotting_df = adata.obs.loc[:, obs_cols_to_keep]
    plotting_df["case_control"] = plotting_df["case_control"].tolist()
    # subset to query only
    plotting_df = plotting_df.loc[plotting_df[ref_q_key] == q_cat]
    # subset to only cells that have mapqc scores:
    plotting_df = plotting_df.loc[plotting_df["mapqc_score"].notna()]
    plotting_df[grouping_key] = plotting_df[grouping_key].tolist()
    # exclude groups with less than min_n_cells_per_box cells:
    cell_count_per_group = (
        plotting_df.groupby([grouping_key, "case_control"], observed=True).size().reset_index(name="n_cells")
    )
    groups_to_keep = cell_count_per_group.loc[
        cell_count_per_group["n_cells"] >= min_n_cells_per_box,
        [grouping_key, "case_control"],
    ]
    groups_to_keep = [
        (group_cat, case_control_cat)
        for group_cat, case_control_cat in zip(
            groups_to_keep[grouping_key], groups_to_keep["case_control"], strict=False
        )
    ]
    plotting_df = plotting_df.loc[
        [
            (group_cat, case_control_cat) in groups_to_keep
            for group_cat, case_control_cat in zip(plotting_df[grouping_key], plotting_df["case_control"], strict=False)
        ],
        :,
    ]
    # keep only groups in groups_to_order and set order:
    if group_order is not None:
        plotting_df = plotting_df.loc[plotting_df[grouping_key].isin(group_order)]
        # keep only groups in group_order that are still in plotting_df:
        group_order = [group for group in group_order if group in plotting_df[grouping_key].unique()]
    else:
        group_order = sorted(plotting_df[grouping_key].unique())
    # create color vector for stripplot, i.e. for the dots
    if color_dots_by is not None:
        plotting_df[color_dots_by] = plotting_df[color_dots_by].tolist()

    if palette_case_control is None:
        palette_case_control = _create_case_control_palette(adata)
    hue_order = [control_cat_name] + case_cats
    if color_dots_by is None:
        fig, ax = plt.subplots(figsize=figsize)
        # create a single boxplot for all case control groups
        sns.boxplot(
            data=plotting_df,
            x=grouping_key,
            y="mapqc_score",
            hue="case_control",
            order=group_order,
            hue_order=hue_order,
            palette=palette_case_control,
            ax=ax,
            saturation=0.5,
            fliersize=0,
            whis=(5, 95),  # make whiskers extend to 5th and 95th percentiles
            orient="vertical",
            **boxplot_kwargs,
        )
        sns.stripplot(
            data=plotting_df,
            x=grouping_key,
            y="mapqc_score",
            hue="case_control",
            order=group_order,
            hue_order=hue_order,
            dodge=True,
            ax=ax,
            size=dotsize,
            palette="dark:black",
            legend=False,
            orient="vertical",
        )
        leg = plt.legend(loc=(1.01, 0.4), frameon=False, title="Control/case category", markerscale=10)
        leg._legend_box.align = "left"
        plt.axhline(y=2, linestyle="--", alpha=0.8, linewidth=2, color="red")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.tick_params(axis="x", rotation=90)
        ax.set_ylabel("MapQC score")
    else:
        plots_to_generate = [control_cat_name] + case_cats
        fig, axs = plt.subplots(len(plots_to_generate), 1, figsize=(figsize[0], figsize[1] * len(plots_to_generate)))
        if palette_dots is None:
            palette_dots = "tab10"
        for case_control_group, ax in zip(plots_to_generate, axs, strict=False):
            sns.boxplot(
                data=plotting_df.loc[plotting_df["case_control"] == case_control_group],
                x=grouping_key,
                y="mapqc_score",
                ax=ax,
                saturation=0.5,
                fliersize=0,
                color=palette_case_control[case_control_group],
                orient="vertical",
                legend=False,
                **boxplot_kwargs,
            )
            sns.stripplot(
                data=plotting_df.loc[plotting_df["case_control"] == case_control_group],
                x=grouping_key,
                y="mapqc_score",
                hue=color_dots_by,
                dodge=False,
                ax=ax,
                size=dotsize,
                palette=palette_dots,
                legend=True,
                orient="vertical",
            )
            ax.set_title(case_control_group)
            leg = ax.legend(loc=(1.01, 0.4), frameon=False, title=color_dots_by, markerscale=10)
            leg._legend_box.align = "left"
            ax.axhline(y=2, linestyle="--", alpha=0.8, linewidth=2, color="red")
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.tick_params(axis="x", rotation=90)
            ax.set_ylabel("MapQC score")

    plt.tight_layout()
    if return_fig:
        return fig
    else:
        plt.show()


def _create_case_control_palette(
    adata: anndata.AnnData,
):
    """Create a color palette for the case/control status."""
    case_cats = [cat for cat in adata.obs.case_control.unique() if cat.startswith("Case")]
    control_cat = [cat for cat in adata.obs.case_control.unique() if cat.startswith("Control")][0]
    if len(case_cats) < 5:
        control_color = "dodgerblue"
        case_colors = ["midnightblue", "blue", "slateblue", "rebeccapurple"]
        palette = {control_cat: control_color}
        palette.update(dict(zip(case_cats, case_colors, strict=False)))
        palette["Reference"] = "darkgray"
        return palette
    else:
        return None


def _validate_grouping(
    adata: anndata.AnnData,
    grouping_key: str,
    group_order: list[str] = None,
):
    """Validate the grouping key and group order."""
    # check that grouping_key is in adata.obs
    ref_q_key = adata.uns["mapqc_params"]["ref_q_key"]
    q_cat = adata.uns["mapqc_params"]["q_cat"]
    if grouping_key not in adata.obs.columns:
        raise ValueError(f"Grouping key {grouping_key} not found in adata.obs.")
    # check that all query cells have a value for the grouping key:
    if (
        pd.isnull(adata.obs.loc[adata.obs[ref_q_key] == q_cat, grouping_key]).any()
        or "nan" in adata.obs.loc[adata.obs[ref_q_key] == q_cat, grouping_key].unique()
    ):
        raise ValueError(f"All query cells must have a value for the grouping key {grouping_key}.")
    # check that group_order is a subset of the groups in adata.obs[grouping_key]
    if group_order is not None:
        groups_in_query = adata.obs.loc[adata.obs[ref_q_key] == q_cat, grouping_key].unique()
        if not all(group in groups_in_query for group in group_order):
            raise ValueError(f"Group order {group_order} is not a subset of the groups in adata.obs[{grouping_key}].")
        if len(group_order) != len(groups_in_query):
            warnings.warn(
                f"Note: your group_order does not include all groups in {grouping_key}. We are therefore not showing all groups.",
                stacklevel=2,
            )
