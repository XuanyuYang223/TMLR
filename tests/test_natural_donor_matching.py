import numpy as np

from experiments.reviewer_revision_diagnostics import donor_permutations


def test_natural_donor_controls_preserve_marginals_and_change_only_requested_answer():
    labels=np.array([[0,a,0,b] for a in range(3) for b in range(4) for _ in range(4)])
    orbit=np.arange(len(labels))
    different,same,eligible=donor_permutations(labels,orbit,912)
    assert eligible.all()
    for ids in [different,same]:
        np.testing.assert_array_equal(np.sort(ids),np.arange(len(labels)))
        np.testing.assert_array_equal(labels[ids,1],labels[:,1])
        assert (orbit[ids]!=orbit).all()
    assert (labels[different,3]!=labels[:,3]).all()
    assert (labels[same,3]==labels[:,3]).all()


def test_singleton_bucket_removed_from_both_arms_without_model_scores():
    labels=np.array([[0,0,0,b] for b in range(3) for _ in range(4)]+[[0,1,0,5]])
    different,same,eligible=donor_permutations(labels,np.arange(len(labels)),18)
    assert eligible[:-1].all() and not eligible[-1]
    np.testing.assert_array_equal(np.sort(different[eligible]),np.flatnonzero(eligible))
    np.testing.assert_array_equal(np.sort(same[eligible]),np.flatnonzero(eligible))
