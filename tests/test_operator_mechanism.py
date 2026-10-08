import numpy as np
import torch

from experiments.operator_mechanism_diagnostic import swap_null, fit_affine_known, design, nested_interval
from experiments.relation_error_localization import row_projection


def test_predicted_null_swap_preserves_all_logits_and_transmits_downstream_signal():
    w = np.array([[1., 0., 0.], [-1., 0., 0.]])
    p = row_projection(w)
    recipient = np.array([[1., 0., 3.], [2., 0., 7.]])
    donor = np.array([[9., 2., -4.], [-8., 4., -6.]])
    hybrid = swap_null(recipient, donor, p)
    np.testing.assert_allclose(hybrid@w.T, recipient@w.T, atol=1e-12)
    np.testing.assert_allclose(hybrid[:, 1:], donor[:, 1:], atol=1e-12)
    b = np.array([[1., 0., 0.], [-2., 1., 0.], [0., 0., 1.]])
    assert not np.array_equal((recipient@b@w.T).argmax(1), (hybrid@b@w.T).argmax(1))


def test_convex_affine_solver_recovers_exact_single_step_linear_truth():
    torch.set_num_threads(1)
    rng = np.random.default_rng(13)
    h = rng.normal(size=(64, 4)); theta = rng.normal(size=(5, 4))
    target = design(h)@theta
    w, bw = rng.normal(size=(3, 4)), rng.normal(size=3)
    config = {'solver': {'svd_relative_rank_threshold': 1e-12, 'max_iterations': 30,
              'gradient_tolerance': 1e-9, 'change_tolerance': 1e-14, 'gap_bound_tolerance': 1e-8},
              'temperature': 2., 'geometry_weight': .25, 'output_kd_weight': 1., 'ridge_candidates': [0., 1e-3]}
    ols, convex, ridge, metadata = fit_affine_known(h, target, target, w, bw, float(target.var(0).mean()), config)
    np.testing.assert_allclose(ols, theta, atol=1e-10)
    np.testing.assert_allclose(convex, theta, atol=1e-9)
    assert metadata['convergence_certificate_pass']
    assert metadata['no_compound_fit_targets']
    assert metadata['ols_whitened_normal_residual'] < 1e-10


def test_nested_bootstrap_keeps_comparisons_paired_within_source():
    assert nested_interval(np.full((3, 9), 7.), 100, 12) == [7., 7.]
    values = np.array([[-5.]*9, [0.]*9, [5.]*9])
    lo, hi = nested_interval(values, 1000, 12)
    assert lo <= 0 <= hi
