import numpy as np

from experiments.relation_readout_diagnostics import fit_readouts


def known_examples():
    rng = np.random.default_rng(108)
    labels = np.tile(np.array([1, 2, 3]), (30, 1))
    val_labels = np.tile(np.array([3, 2, 1]), (10, 1))
    codes = np.eye(5)
    train = codes[labels] + rng.normal(scale=.005, size=(30, 3, 5))
    val = codes[val_labels] + rng.normal(scale=.005, size=(10, 3, 5))
    return train, labels, val, val_labels


def test_known_only_readouts_recover_source_labels_and_mask_unseen_prototype_classes():
    train, labels, val, val_labels = known_examples()
    maps, metadata = fit_readouts(train, labels, val, val_labels, [1e-6, 1e-2])
    for weight, bias in maps.values():
        prediction = (val @ weight.T + bias).argmax(-1)
        np.testing.assert_array_equal(prediction, val_labels)
    assert metadata['known_training_label_classes'] == [1, 2, 3]
    bias = maps['known_prototype_validation_selected'][1]
    assert np.all(bias[[0, 4, 5, 30]] < -1e9)


def test_affine_readout_center_conversion_preserves_decisions_after_translation():
    train, labels, val, val_labels = known_examples()
    shift = np.array([20., -9., 13., 18., -5.])
    original, _ = fit_readouts(train, labels, val, val_labels, [1e-6, 1e-2])
    changed, _ = fit_readouts(train + shift, labels, val + shift, val_labels, [1e-6, 1e-2])
    for label in original:
        wa, ba = original[label]
        wb, bb = changed[label]
        np.testing.assert_array_equal((val @ wa.T + ba).argmax(-1), ((val + shift) @ wb.T + bb).argmax(-1))
