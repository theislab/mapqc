from dataclasses import dataclass

import pandas as pd
import pytest
import scanpy as sc

from mapqc._center_cells._sampling import _sample_center_cells_by_group


@pytest.fixture
def mock_adata():
    # same amount of reference and query cells in total (180 each)
    # cluster 1: cluster with only reference cells
    # cluster 2: many more query than reference cells
    # cluster 3: many more reference than query cells
    # cluster 4: only query cells
    # cluster 5: same number of query cells as cluster 4, but now with the same number of reference cells as well
    n_cells_per_cl_r = {1: 60, 2: 20, 3: 60, 4: 0, 5: 40}
    n_cells_per_cl_q = {1: 0, 2: 80, 3: 20, 4: 40, 5: 40}
    n_cells_r = sum(n_cells_per_cl_r.values())
    n_cells_q = sum(n_cells_per_cl_q.values())
    cl_col = []
    for cl, i in n_cells_per_cl_r.items():
        cl_col.extend(i * [cl])
    for cl, i in n_cells_per_cl_q.items():
        cl_col.extend(i * [cl])
    adata_obs = pd.DataFrame(
        index=[f"c{n}" for n in range(n_cells_r + n_cells_q)],
        data={"ref_or_qu": n_cells_r * ["r"] + n_cells_q * ["q"], "cl": cl_col},
    )
    return sc.AnnData(obs=adata_obs)


# create simplified version of parameter class, that requires fewer arguments:
@dataclass
class _MapQCParams:
    adata: sc.AnnData
    ref_q_key: str
    q_cat: str
    grouping_key: str
    n_nhoods: int
    seed: int


def test_sample_center_cells_by_group(mock_adata):
    params = _MapQCParams(adata=mock_adata, ref_q_key="ref_or_qu", q_cat="q", grouping_key="cl", n_nhoods=30, seed=42)
    sampled_cells = _sample_center_cells_by_group(
        params=params,
    )
    n_cells_per_group = mock_adata.obs.loc[sampled_cells, params.grouping_key].value_counts()
    # check that more cells are sampled from cluster 5 than from cluster 4
    assert n_cells_per_group[5] > n_cells_per_group[4]
    # check that only query cells are sampled:
    assert (mock_adata.obs.loc[sampled_cells, params.ref_q_key] == params.q_cat).all()
    # check that the same number of cells is sampled for cluster 3 and 5;
    # these clusters have the same number of cells in total, but with
    # a different distribution across reference and query cells
    # (note that this outcome would not be the same if the total
    # number of cells differed between reference and query)
    # account for rounding artifacts
    assert abs(n_cells_per_group[3] - n_cells_per_group[5]) <= 1
    # check that number of cells sampled is correct:
    assert len(sampled_cells) == params.n_nhoods
    # finally, check that outcome is as expected, take a low number
    # of sampled cells here, just to check that outcome does not
    # change with code updates etc.
    # sampled_cells_small = _sample_center_cells_by_group(
    #     adata_obs=adata_obs, ref_q_key="ref_or_qu", q_cat="q", grouping_cat="cl", n_cells=5, seed=42
    # )
    # assert sampled_cells_small == ["c210", "c260", "c299", "c339", "c336"]
