from ast import literal_eval

import anndata
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.gridspec import GridSpec


def mapqc_scores(
    adata: anndata.AnnData,
    vmin: float = -4,
    vmax: float = 4,
    figsize: tuple[float, float] = (6, 5),
    return_fig: bool = False,
    umap_kwargs: dict = None,
):
    """
    UMAP colored by MapQC scores, split by case/control status.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object with MapQC scores in adata.obs.mapqc_score. Both
        mapqc.run_mapqc() and mapqc.evaluate() should have been run on
        the adata object, as well as sc.tl.umap().
    vmin : float, optional
        Minimum value for the colorbar. Default is -4.
    vmax : float, optional
        Maximum value for the colorbar. Default is 4.
    figsize : tuple[float, float], optional
        Figure size per panel. Default is (6, 5).
    return_fig : bool, optional
        Return the figure object. Default is False.
    umap_kwargs : dict, optional
        Keyword arguments for scanpy's UMAP plotting.
    """
    if umap_kwargs is None:
        umap_kwargs = {}
    cmap_name = "coolwarm"
    colors = _translate_values_to_colors_rgba(
        point_color_values=adata.obs.mapqc_score,
        point_ref_q_values=adata.obs[adata.uns["mapqc_params"]["ref_q_key"]],
        r_cat=adata.uns["mapqc_params"]["r_cat"],
        point_filtering_values=adata.obs.mapqc_filtering,
        vmin=vmin,
        vmax=vmax,
        cmap_name=cmap_name,
    )

    # Create figure with extra space for colorbar and legend
    fig, gs = _umap_base(adata, colors, figsize, umap_kwargs, extra_width=0.75)

    # Add colorbar
    if gs is not None:
        # Create a smaller axis for the colorbar
        cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]

        # Add legend at the top
        legend_elements = [
            plt.scatter([], [], c="black", marker="o", label="Not sampled"),
            plt.scatter([], [], c="darkolivegreen", marker="o", label="Filtered out"),
            plt.scatter([], [], color=(0.5, 0.5, 0.5, 1.0), marker="o", label="Reference"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            fontsize=10,
            frameon=False,
            bbox_to_anchor=(1.0, 0.95),
        )  # Position legend above colorbar

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax, extend="both")  # 'both' creates triangular ends
        cbar.set_label("MapQC score", fontsize=10)

    if return_fig:
        return fig
    else:
        plt.show()


def mapqc_scores_binary(
    adata: anndata.AnnData,
    figsize: tuple[float, float] = (6, 5),
    return_fig: bool = False,
    umap_kwargs: dict = None,
):
    """
    UMAP colored by MapQC scores (binary, >2 or <=2), split by case/control status.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object with MapQC scores in adata.obs.mapqc_score. Both
        mapqc.run_mapqc() and mapqc.evaluate() should have been run on
        the adata object, as well as sc.tl.umap().
    figsize : tuple[float, float], optional
        Figure size per panel. Default is (6, 5).
    return_fig : bool, optional
        Return the figure object. Default is False.
    umap_kwargs : dict, optional
        Keyword arguments for scanpy's UMAP plotting.
    """
    if umap_kwargs is None:
        umap_kwargs = {}
    # Define palette
    palette = {
        "Reference": "grey",
        ">2": "red",
        "<=2": "antiquewhite",
        "not sampled": "black",
        "filtered out": "darkolivegreen",
    }

    # Create color array based on binary categories
    colors = np.array([mcolors.to_rgba(palette[cat]) for cat in adata.obs.mapqc_score_binary])

    # Create figure with extra space for legend
    fig, gs = _umap_base(adata, colors, figsize, umap_kwargs, extra_width=0.75)

    # Add legend
    if gs is not None:
        # Create legend elements
        legend_elements = [plt.scatter([], [], c=color, marker="o", label=cat) for cat, color in palette.items()]

        # Add legend at the top
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            fontsize=10,
            frameon=False,
            bbox_to_anchor=(1.0, 0.95),
        )

    if return_fig:
        return fig
    else:
        plt.show()


def neighborhood_filtering(
    adata: anndata.AnnData,
    figsize: tuple[float, float] = (6, 5),
    return_fig: bool = False,
    umap_kwargs: dict = None,
    dotsize: float = 1,
):
    """
    UMAP coloring neighborhood center cells by filtering status.

    UMAP shows neighborhood filtering for all center cells of the
    neighborhoods used for mapQC score calculation.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object with neighborhood filtering in adata.obs.mapqc_nhood_filtering.
        Both mapqc.run_mapqc() and mapqc.evaluate() should have been run on
        the adata object, as well as sc.tl.umap().
    figsize : tuple[float, float], optional
        Figure size. Default is (6, 5).
    return_fig : bool, optional
        Return the figure object. Default is False.
    umap_kwargs : dict, optional
        Keyword arguments for scanpy's UMAP plotting.
    dotsize : float, optional
        Dot size. Default is 1.
    """
    if umap_kwargs is None:
        umap_kwargs = {}
    palette_filtering = {
        "not sampled": "black",
        "pass": "tab:green",
        "not enough query cells": "tab:orange",
        "not enough reference samples": "tab:red",
        "not enough reference samples from different studies": "salmon",
        "nan": "darkgrey",
    }
    fig, ax = plt.subplots(figsize=figsize)
    dotsizes = np.ones(adata.n_obs) * dotsize
    dotsizes[~pd.isnull(adata.obs.mapqc_nhood_filtering)] = dotsize * 30
    sc.pl.umap(
        adata,
        color="mapqc_nhood_filtering",
        frameon=False,
        ax=ax,
        show=False,
        palette=palette_filtering,
        size=dotsizes,
        title="Neighborhood filtering",
        **umap_kwargs,
    )
    if return_fig:
        return fig
    else:
        plt.show()


def neighborhood_center_cell(
    adata: anndata.AnnData,
    center_cell: str,
    figsize: tuple[float, float] = (6, 5),
    return_fig: bool = False,
    umap_kwargs: dict = None,
    dotsize: float = 1,
):
    """
    UMAP highlighting a single neighborhood center cell.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object. Both mapqc.run_mapqc() and mapqc.evaluate() should have been run on
        the adata object, as well as sc.tl.umap().
    center_cell : str
        Center cell to plot.
    figsize : tuple[float, float], optional
        Figure size. Default is (6, 5).
    return_fig : bool, optional
        Return the figure object. Default is False.
    umap_kwargs : dict, optional
        Keyword arguments for scanpy's UMAP plotting.
    dotsize : float, optional
        Dot size. Default is 1.
    """
    if umap_kwargs is None:
        umap_kwargs = {}
    _check_center_cell(adata, center_cell)
    # plot
    center_cell_idx = np.where(adata.obs.index == center_cell)[0][0]
    colors = [mcolors.to_rgba("darkgrey")] * adata.n_obs
    colors[center_cell_idx] = mcolors.to_rgba("black")
    dotsizes = np.ones(adata.n_obs) * dotsize
    dotsizes[center_cell_idx] = dotsize * 50
    if "title" not in umap_kwargs:
        umap_kwargs["title"] = f"Center cell {center_cell}"
    fig, _ = _umap_base(
        adata,
        colors=colors,
        figsize=figsize,
        dotsizes=dotsizes,
        umap_kwargs=umap_kwargs,
        split_by_case_control=False,
    )
    if return_fig:
        return fig
    else:
        plt.show()


def neighborhood_cells(
    adata: anndata.AnnData,
    center_cell: str,
    nhood_info_df: pd.DataFrame,
    color_by: str = None,
    color_by_palette: str = "tab10",
    figsize: tuple[float, float] = (6, 5),
    dotsize: float = 1,
    return_fig: bool = False,
    umap_kwargs: dict = None,
):
    """
    UMAP highlighting all cells in the neighborhood of a given center cell.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object. Both mapqc.run_mapqc() and mapqc.evaluate() should have been run on
        the adata object, as well as sc.tl.umap().
    center_cell : str
        Center cell to plot.
    nhood_info_df : pd.DataFrame
        DataFrame with neighborhood information. This is an optional output of mapqc.run(),
        when the return_sample_dists_to_ref_df parameter is set to True.
    color_by : str, optional
        Metadata category (column name in adata.obs) to color the neighborhood cells by.
    color_by_palette : str, optional
        Name of a matplotlib color palette to use for coloring the neighborhood cells by. Default is "tab10".
    figsize : tuple[float, float], optional
        Figure size. Default is (6, 5).
    dotsize : float, optional
        Dot size. Default is 1.
    return_fig : bool, optional
        Return the figure object. Default is False.
    umap_kwargs : dict, optional
        Keyword arguments for scanpy's UMAP plotting.
    """
    if umap_kwargs is None:
        umap_kwargs = {}
    _check_center_cell(adata, center_cell)
    # Convert knn_barcodes to list if stored as string
    if isinstance(nhood_info_df["knn_barcodes"].iloc[0], str):
        nhood_info_df.loc[:, "knn_barcodes"] = nhood_info_df["knn_barcodes"].apply(literal_eval)
    nhood_cell_barcodes = nhood_info_df.loc[center_cell, "knn_barcodes"]
    nhood_cell_idc = np.where(adata.obs.index.isin(nhood_cell_barcodes))[0]
    colors = np.array([mcolors.to_rgba("darkgrey")] * adata.n_obs)
    if color_by is None:
        colors[nhood_cell_idc] = mcolors.to_rgba("black")
    else:
        color_vector = adata.obs[color_by].values[nhood_cell_idc]
        colors[nhood_cell_idc], value_to_color = _categorical_to_colors_rgba(
            color_vector, palette_name=color_by_palette
        )

    dotsizes = np.ones(adata.n_obs) * dotsize
    dotsizes[nhood_cell_idc] = dotsize * 10
    if "title" not in umap_kwargs:
        umap_kwargs["title"] = f"Neighborhood cells of center cell\n{center_cell}"
    fig, _ = _umap_base(
        adata,
        colors=colors,
        figsize=figsize,
        dotsizes=dotsizes,
        umap_kwargs=umap_kwargs,
        split_by_case_control=False,
    )
    if color_by is not None:
        legend_elements = [
            plt.scatter([], [], color=color, marker="o", label=cat) for cat, color in value_to_color.items()
        ]

        # Add legend at the top
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            fontsize=10,
            frameon=False,
            bbox_to_anchor=(1.0, 0.95),
        )
    if return_fig:
        return fig
    else:
        plt.show()


def _umap_base(
    adata: anndata.AnnData,
    colors: np.ndarray,
    figsize: tuple[float, float],
    umap_kwargs: dict = None,
    dotsizes: np.ndarray = None,
    extra_width: float = 0,
    split_by_case_control: bool = True,
):
    """
    Base function for UMAP plotting with flexible layout control.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object with case_control annotation in adata.obs.case_control
    colors : np.ndarray
        Array of colors for each point
    figsize : tuple[float, float]
        Base figure size (will be adjusted for extra_width if needed)
    umap_kwargs : dict, optional
        Additional arguments for scanpy's UMAP plotting
    dotsizes : np.ndarray, optional
        Array of dot sizes for each point
    extra_width : float, default 0
        Extra width to add to the figure (e.g., for colorbar/legend)
    split_by_case_control : bool, default True
        Whether to split the UMAP by case/control status
    """
    if umap_kwargs is None:
        umap_kwargs = {}
    if split_by_case_control:
        # Calculate figure size to maintain square subplots
        control_cat = [cat for cat in adata.obs.case_control.unique() if cat.startswith("Control")][0]
        case_cats = [cat for cat in adata.obs.case_control.unique() if cat.startswith("Case")]
        n_figures = len(case_cats) + 1
    else:
        n_figures = 1

    # Adjust figure size if extra width is needed
    fig_width = figsize[0] * n_figures + extra_width
    fig_height = figsize[1]
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create GridSpec with extra space if needed
    if extra_width > 0:
        gs = GridSpec(1, n_figures + 1, width_ratios=[1] * n_figures + [0.15])
    else:
        gs = GridSpec(1, n_figures)

    # Plot UMAPs
    if split_by_case_control:
        for i, cat in enumerate([control_cat] + case_cats):
            ax = fig.add_subplot(gs[0, i])
            subset = adata.obs.case_control.isin(["Reference", cat]).values
            # Create a copy of the subset to avoid the warning
            adata_subset = adata[subset].copy()
            color_subset = colors[subset]

            sc.pl.umap(
                adata_subset,  # Use the copy instead of the view
                color=None,
                frameon=False,
                sort_order=False,
                ax=ax,
                show=False,
                title=cat,
                **umap_kwargs,
            )
            scatter = ax.collections[0]
            scatter.set_facecolors(color_subset)
            scatter.set_edgecolors("none")
            if dotsizes is not None:
                dotsize_subset = dotsizes[subset]
                scatter.set_sizes(dotsize_subset)
    else:
        ax = fig.add_subplot(gs[0, 0])
        sc.pl.umap(
            adata,
            color=None,
            frameon=False,
            sort_order=False,
            ax=ax,
            show=False,
            **umap_kwargs,
        )
        scatter = ax.collections[0]
        scatter.set_facecolors(colors)
        scatter.set_edgecolors("none")
        if dotsizes is not None:
            scatter.set_sizes(dotsizes)
    return (fig, gs)


def _check_center_cell(adata: anndata.AnnData, center_cell: str):
    """Some checks to make sure the center cell is valid."""
    # check if center_cell is in adata.obs.index
    if center_cell not in adata.obs.index:
        raise ValueError(f"Center cell {center_cell} not found in adata.obs.index")
    # check if center_cell is in adata.obs.mapqc_nhood_filtering
    if adata.obs.loc[center_cell, "mapqc_nhood_filtering"] is None:
        raise ValueError(f"Cell {center_cell} was not a center cell.")


def _categorical_to_colors_rgba(
    values: np.ndarray,
    palette_name: str = "tab10",
) -> tuple[np.ndarray, dict]:
    """
    Convert categorical values to RGBA colors using a matplotlib color palette.

    Parameters
    ----------
    values : np.ndarray
        Array of categorical values (can be strings, numbers, etc.)
    palette_name : str, optional
        Name of the matplotlib color palette to use. Default is "tab10".
        Common options include: "tab10", "tab20", "Set1", "Set2", "Set3", "Paired"

    Returns
    -------
    tuple
        - np.ndarray: Array of RGBA colors, one for each input value
        - dict: Dictionary mapping unique values to their RGBA colors
    """
    # Get unique values and their indices
    unique_values = np.unique(values)
    value_to_idx = {val: idx for idx, val in enumerate(unique_values)}

    # Get the colormap
    cmap = plt.get_cmap(palette_name)

    # Map each value to a color
    n_colors = len(unique_values)
    # If we have more unique values than colors in the palette, we'll cycle through the palette
    colors = cmap(np.arange(n_colors) % cmap.N)

    # Create the output array
    rgba_colors = np.zeros((len(values), 4))
    for i, val in enumerate(values):
        rgba_colors[i] = colors[value_to_idx[val]]

    # Create dictionary mapping values to colors
    value_to_color = {val: colors[idx] for val, idx in value_to_idx.items()}

    return rgba_colors, value_to_color


def _translate_values_to_colors_rgba(
    point_color_values,
    point_ref_q_values,
    r_cat,
    point_filtering_values,
    vmin,
    vmax,
    cmap_name,
) -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = np.array(
        [
            _value_to_color(point, r_or_q, r_cat, f, cmap, norm)
            for point, r_or_q, f in zip(
                point_color_values,
                point_ref_q_values,
                point_filtering_values,
                strict=False,
            )
        ]
    )
    return colors


def _value_to_color(value, r_or_q, r_cat, filtering, cmap, norm):
    if r_or_q == r_cat:
        return (0.5, 0.5, 0.5, 1.0)  # Grey for reference
    elif isinstance(value, float) and not np.isnan(value):
        return cmap(norm(value))
    elif filtering == "not sampled":
        return mcolors.to_rgba("black")
    else:
        return mcolors.to_rgba("darkolivegreen")
