import os

import pandas as pd
import pytest

import mapqc.pl.umap as umap

# for now we're just testing that plotting funcitons don't throw errors or warnings


@pytest.fixture
def nhood_info_df(output_data_dir):
    return pd.read_pickle(os.path.join(output_data_dir, "nhood_info_df.pkl"))


def test_mapqc_scores(mapqc_output_adata):
    """Test that mapqc_scores plotting function runs without warnings or errors."""
    umap.mapqc_scores(mapqc_output_adata)


def test_mapqc_scores_binary(mapqc_output_adata):
    """Test that mapqc_scores_binary plotting function runs without warnings or errors."""
    umap.mapqc_scores_binary(mapqc_output_adata)


def test_neighborhood_filtering(mapqc_output_adata):
    """Test that neighborhood_filtering plotting function runs without warnings or errors."""
    umap.neighborhood_filtering(mapqc_output_adata)


def test_neighborhood_cells(mapqc_output_adata, nhood_info_df):
    """Test that neighborhood_cells plotting function runs without warnings or errors."""
    umap.neighborhood_cells(
        mapqc_output_adata,
        center_cell=nhood_info_df.index[0],
        nhood_info_df=nhood_info_df,
    )


def test_neighborhood_center_cell(mapqc_output_adata, nhood_info_df):
    """Test that neighborhood_center_cell plotting function runs without warnings or errors."""
    umap.neighborhood_center_cell(mapqc_output_adata, center_cell=nhood_info_df.index[0])
