import pandas as pd

from mapqc.neighbors.adaptive_k import get_idc_nth_instances


def test_get_idc_nth_instances():
    seq = pd.Categorical(["a", "b", "a", "c", "b", "c", "c", "c", "a"])
    idc = get_idc_nth_instances(seq, 3)
    assert idc.equals(pd.Series(data=[6, 8], index=["c", "a"]))
