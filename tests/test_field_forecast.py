import unittest
import numpy as np

from experiments.field_forecast import design, fit_ridge, predict


class FieldForecastTests(unittest.TestCase):
    def test_unseen_world_uses_average_training_fixed_effect(self):
        config = {'targets': [[1, 0, 0, 0], [0, 1, 0, 0]]}
        source = {'world_seeds': [17, 42, 101], 'model_seeds': [17, 42, 101]}
        row = {'source_final_accuracy': 1., 'source_auc': .9, 'source_physical_complexity': 4.,
               'target_physical_complexity': 3., 'target_id': 0, 'world_seed': 271, 'model_seed': 17,
               'target_composition_size': 2, 'target_character_energy_fraction': .1, 'oracle_occupancy': .7}
        x = design([row], 'baseline', config, source)
        np.testing.assert_allclose(x[0, 5:7], [1/3, 1/3])
        np.testing.assert_array_equal(x[0, 7:], [0., 0.])

    def test_fitted_scaling_is_reused_for_future_features(self):
        x = np.arange(12).reshape(6, 2).astype(float)
        truth = x[:, 0]*.1
        fit = fit_ridge(x, truth, 1.)
        before = {k: list(v) for k, v in fit.items()}
        result = predict(fit, np.array([[100., 101.]]))
        self.assertEqual(fit, before)
        self.assertTrue(np.isfinite(result).all())


if __name__ == '__main__': unittest.main()
