import numpy as np
import pandas as pd

from mapqc._params import _MapQCParams


def _sample_center_cells_by_group(
    params: _MapQCParams,
) -> list[str]:
    """Sample n query cells, based on cell proportions per group for both reference and query.

    Parameters
    ----------
    params: _MapQCParams parameters object.

    Returns
    -------
    sampled_cells: list
        List of cell indices (row names of adata.obs) that were sampled.
    """
    # calculate proportions per group for reference and query:
    ref_q_group_n = params.adata.obs.groupby([params.ref_q_key, params.grouping_key], observed=True).size().unstack()
    ref_q_group_n = ref_q_group_n.fillna(0)
    ref_q_group_props = ref_q_group_n.div(ref_q_group_n.sum(axis=1), axis=0)
    # as the target group proportions, take the mean of the reference and
    # query proportions. In this way, we take both abundance of cell types
    # in the reference and the query into account:
    # import pdb; pdb.set_trace()
    target_props = ref_q_group_props.mean(axis=0)
    # calculate the target number of cells per group:
    target_n_cells_per_group = np.rint(target_props * params.n_nhoods).astype(int)
    # in case we don't have enough cells for a specific group in the query,
    # (e.g. because a cluster contained no or few query cells), we need
    # to sample those from the remaining groups.
    # If we do have enough cells, this will just return the target_n_cells_per_group.
    target_n_cells_per_group = _redistribute_missing_cells(
        params.n_nhoods, target_n_cells_per_group, target_props, ref_q_group_n.loc[params.q_cat, :]
    )
    # correct for rounding errors:
    n_too_many_cells = (target_n_cells_per_group.sum() - params.n_nhoods).astype(int)
    if n_too_many_cells > 0:
        # remove cells from groups, starting with the groups with the
        groups_to_remove_from = target_n_cells_per_group.sort_values(ascending=False).index[:n_too_many_cells]
        target_n_cells_per_group.loc[groups_to_remove_from] -= 1
    # now sample query cells from each group:
    sampled_cells = pd.concat(
        [
            group.sample(target_n_cells_per_group[group_name], replace=False, random_state=params.seed)
            for group_name, group in params.adata.obs.loc[params.adata.obs[params.ref_q_key] == params.q_cat].groupby(
                params.grouping_key, observed=True
            )
        ]
    ).index.tolist()
    # return result:
    return sampled_cells


def _redistribute_missing_cells(
    n_cells_to_sample: int,
    target_n_cells_per_group: pd.Series,
    target_group_props: pd.Series,
    true_n_cells_per_group: pd.Series,
) -> pd.Series:
    """
    Redistributes "missing" number of cells from low n groups across remaining groups.

    Parameters
    ----------
    n_cells_to_sample: int
        Number of cells to sample from the query.
    target_n_cells_per_group: pd.Series
        Target number of cells per group.
    target_group_props: pd.Series
        Target group proportions.
    true_n_cells_per_group: pd.Series
        True number of query cells per group.

    Returns
    -------
    clipped_n_cells_per_group: pd.Series
        Updated number of cells per group. If target number of cells per group could be
        reached, returns unchanged target_n_cells_per_group. If target number of cells
        could not be reached due to too low query cell numbers in specific groups, returns
        updated number of cells per group, redistributing "missing" number of cells across
        groups with sufficient cells.
    """
    # Clip n cells per group based on true n cells per group
    clipped_n_cells_per_group = target_n_cells_per_group.clip(upper=true_n_cells_per_group)
    # calculate how many cells need to be redistributed
    n_cells_to_redistribute = n_cells_to_sample - clipped_n_cells_per_group.sum()
    n_cells_to_redistribute_previous = None
    while n_cells_to_redistribute > 0:
        # Calculate remaining capacity for each group
        remaining_capacity = true_n_cells_per_group - clipped_n_cells_per_group
        groups_with_capacity = remaining_capacity[remaining_capacity > 0]
        # if not enough query cells exist, raise error:
        if len(groups_with_capacity) == 0:
            raise ValueError(f"Not enough query cells exist to sample n_cells = {n_cells_to_sample}.")
        # re-normalize proportions to total 1 for the groups with capacity only:
        target_props_remaining_groups = target_group_props[groups_with_capacity.index]
        redistribution_props = target_props_remaining_groups.div(target_props_remaining_groups.sum())
        # calculate number of additional cells to sample from each group.
        # If the number of cells to redistribute in this round is the same as in the previous round,
        # we round up (ceil) here to avoid infinite loop due to rounding artefacts.
        if n_cells_to_redistribute == n_cells_to_redistribute_previous:
            additional_cells_per_group = np.ceil(n_cells_to_redistribute * redistribution_props).astype(int)
        else:
            additional_cells_per_group = np.rint(n_cells_to_redistribute * redistribution_props).astype(int)
        # clip according to capacity again:
        clipped_additional_cells_per_group = additional_cells_per_group.clip(
            upper=remaining_capacity[additional_cells_per_group.index]
        )
        # now check if we still lack cells:
        n_cells_to_redistribute_previous = n_cells_to_redistribute
        n_cells_to_redistribute -= clipped_additional_cells_per_group.sum()
        # update the number of cells per group:
        clipped_n_cells_per_group[additional_cells_per_group.index] += clipped_additional_cells_per_group
    return clipped_n_cells_per_group
