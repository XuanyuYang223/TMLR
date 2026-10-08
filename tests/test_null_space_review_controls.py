import numpy as np

from experiments.null_space_review_controls import norm_error_calibrate


def test_norm_error_control_changes_orientation_without_changing_readout():
    rng = np.random.default_rng(3)
    w = rng.normal(size=(3, 9))
    _, _, vt = np.linalg.svd(w, full_matrices=True)
    null = vt[3:].T
    truth = rng.normal(size=(25, 6))
    donor = rng.normal(size=(25, 6))
    direction = rng.normal(size=(25, 6))
    new = norm_error_calibrate(donor, truth, direction)
    np.testing.assert_allclose(np.linalg.norm(new, axis=1), np.linalg.norm(donor, axis=1), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(new-truth, axis=1), np.linalg.norm(donor-truth, axis=1), atol=1e-12)
    row = rng.normal(size=(25, 9))
    np.testing.assert_allclose((row+new@null.T)@w.T, (row+donor@null.T)@w.T, atol=1e-12)
    downstream = rng.normal(size=(9, 3))
    assert np.linalg.norm((new-donor)@null.T@downstream) > 1


def test_calibration_degenerate_truth_and_direction_are_finite():
    donor = np.array([[1., 2., 3.], [4., 0., 0.]])
    truth = np.array([[0., 0., 0.], [2., 0., 0.]])
    result = norm_error_calibrate(donor, truth, truth.copy())
    assert np.isfinite(result).all()
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), np.linalg.norm(donor, axis=1))
    np.testing.assert_allclose(np.linalg.norm(result-truth, axis=1), np.linalg.norm(donor-truth, axis=1))
