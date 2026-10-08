import json
from pathlib import Path
import unittest
import numpy as np
from experiments.algebra import projective
from experiments.controlled_target_followup import future_design,select_targets
from experiments.field_symmetry import group_sources


class NewTargetFollowupTests(unittest.TestCase):
    def test_new_targets_selected_only_by_algebra_and_physical_support(self):
        source=json.loads(Path('configs/field_matched_support.json').read_text())
        old=json.loads(Path('configs/field_matched_support_transfer.json').read_text())
        candidates,selected=select_targets(source,old)
        self.assertEqual(len(candidates),40);self.assertEqual(len(selected),8)
        excluded={projective(t,5) for t in old['targets']}|{projective(t,5) for g in source['groups'] for t in group_sources(g,5)}
        self.assertTrue(all(projective(r['target'],5) not in excluded for r in selected))
        self.assertTrue(all(min(r['physical_support'].values())>=3 for r in selected))
        self.assertEqual([min(r['source_orders'].values()) for r in selected],[2]*4+[3]*4)

    def test_new_target_identity_gets_mean_old_target_effect(self):
        config={'targets':[[1,0,0,0]]*8};source={'world_seeds':[17,42,101],'model_seeds':[17,42,101]}
        row={'source_final_accuracy':1.,'source_auc':.9,'source_physical_complexity':4.,'target_physical_complexity':3.,'target_id':0,'world_seed':271,'model_seed':17}
        x=future_design([row],'baseline',config,source)
        np.testing.assert_allclose(x[0,4:11],np.ones(7)/8)


if __name__=='__main__':unittest.main()
