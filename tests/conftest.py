import os
import pickle
from pathlib import Path

import pandas as pd
import pytest
import scanpy as sc


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the absolute path to the test data directory."""
    # Get the directory where this conftest.py file is located
    test_dir = Path(__file__).parent
    # Return the absolute path to the data directory
    return test_dir / "data"


@pytest.fixture(scope="session")
def input_data_dir(test_data_dir):
    """Return the absolute path to the input test data directory."""
    return test_data_dir / "input"


@pytest.fixture(scope="session")
def intermediate_data_dir(test_data_dir):
    """Return the absolute path to the intermediate test data directory."""
    return test_data_dir / "intermediate"


@pytest.fixture(scope="session")
def output_data_dir(test_data_dir):
    """Return the absolute path to the output test data directory."""
    return test_data_dir / "output"


@pytest.fixture(scope="session")
def adata(input_data_dir):
    """Session-scoped fixture that loads the AnnData object once for all tests.

    This fixture reads the AnnData object
    """
    adata = sc.read_h5ad(os.path.join(input_data_dir, "mapped_q_and_r.h5ad"))
    return adata


@pytest.fixture(scope="session")
def mapqc_output_adata(adata, output_data_dir):
    # columns added by mapqc.run():
    adata_output = adata.copy()
    obs_added = pickle.load(open(os.path.join(output_data_dir, "obs_df.pkl"), "rb"))
    adata_output.obs = pd.concat([adata_output.obs, obs_added], axis=1)
    # params added by mapqc.run()
    adata_output.uns["mapqc_params"] = pickle.load(open(os.path.join(output_data_dir, "params.pkl"), "rb"))
    return adata_output
