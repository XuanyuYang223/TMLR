import unittest

from experiments.field_forecast_report import match_outcomes


class FrozenForecastEvaluationTests(unittest.TestCase):
    def test_uses_exact_paired_random_and_preserves_negative_gain(self):
        common = {'world_seed': 271, 'model_seed': 17, 'target_id': 2, 'budget': 25, 'mode': 'linear'}
        rows = [{**common, 'group': 'P', 'accuracy': .15},
                {**common, 'group': 'random', 'accuracy': .30},
                {**common, 'group': 'random', 'model_seed': 42, 'accuracy': .99}]
        forecast = {**common, 'group': 'P', 'variant': 'baseline', 'predicted_outcome': .1}
        result = match_outcomes([forecast], rows)[0]
        self.assertAlmostEqual(result['actual_outcome'], -.15)
        self.assertAlmostEqual(result['error'], .25)

    def test_missing_baseline_and_duplicate_endpoint_fail(self):
        row = {'group': 'P', 'world_seed': 271, 'model_seed': 17, 'target_id': 2, 'budget': 25, 'mode': 'linear', 'accuracy': .15}
        forecast = {**row, 'variant': 'baseline', 'predicted_outcome': .1}
        with self.assertRaises(AssertionError): match_outcomes([forecast], [row])
        with self.assertRaises(AssertionError): match_outcomes([forecast], [row, row])


if __name__ == '__main__': unittest.main()
