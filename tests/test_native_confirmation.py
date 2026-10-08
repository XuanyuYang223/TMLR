import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.native_confirmation import jobs, feature_vector, paths, prepare_data, setup, permutation_row
from experiments.native_confirmation_analysis import paired_summary
from experiments.permworld_combinations import select_groups


class NativeConfirmationTests(unittest.TestCase):
    def test_catalog_has_complete_paired_candidate_and_single_controls(self):
        plan,config,_=paths()
        catalog=jobs(plan,select_groups(config))
        self.assertEqual(sum(r['phase']=='multi' for r in catalog),26)
        self.assertEqual(sum(r['phase']=='single' for r in catalog),20)
        for seed in plan['model_seeds']:
            selected=[r for r in catalog if r['seed']==seed and (r['phase']=='single' or r['id']=='interior_none')]
            self.assertEqual(len(selected),5)
            self.assertEqual({r['id'] for r in selected},{'interior_none',*plan['single_tasks']})
        self.assertFalse(set(plan['model_seeds']) & {17,42,101})

    def test_new_seed_features_exclude_old_initialization_indicators(self):
        row={'baseline':list(map(float,range(12))),'math':[5.,0.,.5,.25,.25]}
        self.assertEqual(feature_vector(row,'source_learning'),[0.,1.,7.,8.,9.])
        values=feature_vector(row,'source_learning_label_stats_plus_relations')
        self.assertNotIn(10.,values);self.assertNotIn(11.,values)
        self.assertEqual(values[-4],1.)

    def test_uncertainty_uses_seed_gains_and_rejects_endpoint_matrix(self):
        summary=paired_summary([.01,.02,-.01,.03,0.],seed=7,repetitions=1000)
        self.assertEqual(summary['seeds'],5)
        self.assertEqual(summary['positive_seeds'],3)
        self.assertAlmostEqual(summary['mean_gain'],.01)
        with self.assertRaises(AssertionError):paired_summary(np.ones((5,4)),7)

    def test_fresh_generator_rejects_exact_prior_permutation_and_keeps_splits_disjoint(self):
        plan,config,_=paths()
        config={**config,'lengths':[10,10]}
        names,functions,tokens,one_line=setup(config)
        seed=71
        first=tuple(map(int,np.random.default_rng(seed).permutation(10)+1))
        prefix=[tokens['<BOS>'],tokens['<SIZE>'],10]+[tokens[t] for t in one_line(first)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prior=root/'prior.npz'
            np.savez(prior,train_input=np.array([prefix]),train_lengths=np.array([10]))
            small={**plan,'data_seed':seed,'excluded_datasets':[str(prior)],
                   'examples_per_length':{'train':3,'support_pool':2,'target_test':3}}
            prepare_data(small,config,root,names,functions,tokens,one_line)
            with np.load(root/'dataset/data.npz') as data:
                permutations=[permutation_row(row,10) for split in ('train','support_pool','target_test') for row in data[split+'_input']]
                self.assertEqual(len(set(permutations)),8)
                self.assertNotIn(first,permutations)
                for split in ('train','target_test'):
                    for row,labels in zip(data[split+'_input'],data[split+'_labels']):
                        p=permutation_row(row,10)
                        np.testing.assert_array_equal(labels,[functions[t](p) for t in names])


if __name__=='__main__':unittest.main()
