import unittest

from experiments.six_hour_report import transfer_summaries


class SixHourReportTests(unittest.TestCase):
    def test_gain_requires_same_seed_target_budget_and_readout_baseline(self):
        rows = [dict(group='random', seed=17, target='a', budget=64, mode='finetune', test_accuracy=.3),
                dict(group='x', seed=17, target='a', budget=64, mode='finetune', test_accuracy=.5),
                dict(group='x', seed=42, target='a', budget=64, mode='finetune', test_accuracy=.9),
                dict(group='x', seed=17, target='a', budget=256, mode='finetune', test_accuracy=.9),
                dict(group='x', seed=17, target='a', budget=64, mode='linear_query', test_accuracy=.9)]
        gains, summary = transfer_summaries(rows)
        self.assertEqual(len(gains), 1)
        self.assertAlmostEqual(summary[0]['gain'], .2)
        self.assertEqual(summary[0]['paired_cells'], 1)
        self.assertEqual(summary[0]['seeds'], 1)
        self.assertIsNone(summary[0]['gain_seed_sd'])


if __name__ == '__main__':
    unittest.main()
