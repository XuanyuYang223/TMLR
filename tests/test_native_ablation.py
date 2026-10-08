import itertools
import json
from pathlib import Path
import unittest

import numpy as np
from experiments.native_ablation import jobs,sample
from experiments.native_ablation_math import actions
from experiments.native_confirmation import setup
from experiments.native_transform_audit import certificates
from experiments.permutation_audit import transform
from experiments.permworld_combinations import select_groups


class NativeAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=json.loads(Path('configs/native_ablation.json').read_text())
        cls.config=json.loads(Path(cls.plan['base_config']).read_text())
        cls.names,cls.functions,_,_=setup(cls.config)

    def test_budget_and_leave_one_controls(self):
        rows=jobs(self.plan);self.assertEqual(len(rows),42)
        for seed in self.plan['seeds']:
            selected=[x for x in rows if x['seed']==seed]
            equal=[x for x in selected if x['id'].endswith('_b96')]
            self.assertEqual(len(equal),9)
            self.assertTrue(all(x['steps']*x['batch']*len(x['tasks'])==1920000 for x in equal))
            drop=[x for x in equal if x['id'].startswith('drop_')]
            self.assertEqual({frozenset(x['tasks']) for x in drop},{frozenset(set(self.plan['candidate_tasks'])-{t}) for t in self.plan['candidate_tasks']})

    def test_nested_sampler_preserves_legacy_core(self):
        buckets={n:np.arange(n*100,n*100+50) for n in range(10,31)}
        legacy=np.random.default_rng(1009+20261005)
        streams={b:(np.random.default_rng(1009+20261005),np.random.default_rng(1009+202610061)) for b in (8,24,32,96)}
        for _ in range(100):
            n=int(legacy.integers(10,31));base=legacy.choice(buckets[n],32,replace=True)
            for batch,(core,extra) in streams.items():
                length,ids=sample(core,extra,buckets,batch,10,30)
                self.assertEqual(length,n);np.testing.assert_array_equal(ids[:min(batch,32)],base[:min(batch,32)])
                self.assertEqual(len(ids),batch);self.assertEqual(core.bit_generator.state,legacy.bit_generator.state)

    def test_generic_rules_reproduce_old_certificates(self):
        for g in select_groups(self.config):
            a=actions(g['tasks']);old=certificates(g);self.assertEqual(set(a),set(old))
            for op in a:np.testing.assert_array_equal(a[op],old[op][0])

    def test_novel_actions_against_exact_permutations(self):
        old={frozenset(g['tasks']) for g in select_groups(self.config)}
        for g in self.plan['novel_groups']:
            self.assertNotIn(frozenset(g['tasks']),old)
            a=actions(g['tasks'])
            for n in range(1,6):
                for value in itertools.permutations(range(1,n+1)):
                    y=np.array([self.functions[t](value) for t in g['tasks']])
                    for op,m in a.items():
                        other=np.array([self.functions[t](transform(value,op)) for t in g['tasks']]);np.testing.assert_array_equal(y@m.T,other)


if __name__=='__main__':unittest.main()
