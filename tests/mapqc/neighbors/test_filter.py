import pandas as pd

from mapqc.neighbors.filter import _create_difference_matrix, check_n_ref_samples


def test_create_difference_matrix():
    lst = ["a", "b", "b"]
    assert _create_difference_matrix(lst).tolist() == [
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
    ]


def test_check_n_ref_samples():
    sample_info_df = pd.DataFrame(
        data={
            "st": ["st1", "st1", "st2", "st2", "st3"],
            "re_qu": ["re", "re", "re", "re", "qu"],
        },
        index=["s1", "s2", "s3", "s4", "s5"],
    )
    sample_set_list = ["s1", "s3", "s2", "s4", "s5"]
    ref_q_key = "re_qu"
    ref_cat = "re"
    st_key = "st"
    assert check_n_ref_samples(
        sample_set=sample_set_list,
        sample_info=sample_info_df,
        r_q_key=ref_q_key,
        r_cat=ref_cat,
        min_n_samples_r=3,
        exclude_same_study=True,
        study_key=st_key,
    ) == (True, "pass")
    assert check_n_ref_samples(
        sample_set=sample_set_list,
        sample_info=sample_info_df,
        r_q_key=ref_q_key,
        r_cat=ref_cat,
        min_n_samples_r=5,
        exclude_same_study=True,
        study_key=st_key,
    ) == (False, "too few reference samples")
    assert check_n_ref_samples(
        sample_set=sample_set_list,
        sample_info=sample_info_df,
        r_q_key=ref_q_key,
        r_cat=ref_cat,
        min_n_samples_r=4,
        exclude_same_study=False,
        study_key=st_key,
    ) == (True, "pass")
    assert check_n_ref_samples(
        sample_set=sample_set_list,
        sample_info=sample_info_df,
        r_q_key=ref_q_key,
        r_cat=ref_cat,
        min_n_samples_r=4,
        exclude_same_study=True,
        study_key=st_key,
    ) == (False, "too few reference samples from different batches")
