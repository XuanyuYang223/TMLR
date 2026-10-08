import unittest

import numpy as np
import torch

from experiments.answer_prompt_baseline import answer_features,remove_linear_answer_component
from experiments.hidden_relation_train import AffineOperators,collision_pair,relation_loss,visible_key
from experiments.hidden_relation_evaluate import partial_probe,score_method
from experiments.permutation_audit import transform
from experiments.representation_algebra import permutation_action_table


def lrmax(p):
    maximum=0;count=0
    for value in p:
        if value>maximum:count+=1;maximum=value
    return count


class HiddenRelationTests(unittest.TestCase):
    def test_collision_answers_cannot_determine_hidden_target(self):
        seen=set();a,b=collision_pair(np.random.default_rng(8),7,seen,{},lrmax)
        self.assertEqual(visible_key(a,lrmax),visible_key(b,lrmax))
        self.assertNotEqual(lrmax(a[5]),lrmax(b[5]))
        self.assertEqual(lrmax(a[5]),lrmax(a[2]))
        self.assertEqual(lrmax(b[5]),lrmax(b[2]))
        self.assertEqual(len(seen),16)

    def test_input_words_c_then_i_differ_from_i_then_c(self):
        p=(4,1,6,2,7,3,5)
        self.assertNotEqual(transform(transform(p,'complement'),'inverse'),transform(transform(p,'inverse'),'complement'))

    def test_relation_loss_recovers_independent_known_pairs(self):
        base=torch.tensor([[1.,2.,3.],[3.,5.,7.],[2.,7.,1.]])
        ops=AffineOperators(3)
        with torch.no_grad():
            ops.offset[0].copy_(torch.diag(torch.tensor([-2.,0.,-2.])))
            ops.offset[1].copy_(torch.diag(torch.tensor([0.,-2.,-2.])))
        h=torch.stack([base,ops(base,0),ops(base,1)],dim=1)
        self.assertLess(float(relation_loss(h,ops,torch.arange(3)).detach()),1e-12)
        self.assertGreater(float(relation_loss(h,ops,torch.tensor([1,2,0])).detach()),.1)

    def test_answer_prompt_features_have_no_other_input_information(self):
        labels=np.array([[[2,3],[2,3]],[[4,1],[4,1]]])
        h=answer_features(labels,['task_a','task_b'],17,8)
        np.testing.assert_array_equal(h[:,0],h[:,1])

    def test_answer_residual_removes_exact_additive_answer_code(self):
        rng=np.random.default_rng(19);labels=rng.integers(1,6,size=(80,8,4));codes=np.eye(31)[labels].reshape(80,8,-1)
        h=codes @ rng.normal(size=(124,12))+rng.normal(size=12)
        data={'split':np.array([0]*50+[1]*15+[2]*15),'lengths':np.full(80,10)}
        residual,audit,_=remove_linear_answer_component(h,labels,data,[1e-9])
        self.assertLess(audit['heldout_h_reconstruction_nmse'],1e-12)

    def test_compound_states_cannot_change_generator_fitting(self):
        rng=np.random.default_rng(27);table=permutation_action_table();base=rng.normal(size=(50,8))
        h=np.array([[row[table[g]] for g in range(8)] for row in base])
        data={'split':np.array([0]*30+[1]*10+[2]*10),'lengths':np.full(50,12)}
        a,means,alphas=partial_probe(h,data,[1e-8,.01],17)
        corrupted=h.copy();corrupted[:,[2,3,5,6,7]]=rng.normal(size=(50,5,8))*100
        b,other,selected=partial_probe(corrupted,data,[1e-8,.01],17)
        self.assertEqual(alphas,selected)
        for g in a:
            np.testing.assert_array_equal(a[g][0],b[g][0]);np.testing.assert_array_equal(a[g][1],b[g][1])
        np.testing.assert_array_equal(means[12],other[12])

    def test_compound_prediction_does_not_forward_or_use_compound_target(self):
        rng=np.random.default_rng(29);h=rng.normal(size=(30,8,8))
        data={'split':np.array([0]*10+[1]*4+[2]*6+[3]*10),'lengths':np.full(30,12),
            'labels':h.argmax(-1),'pair_ids':np.array([-1]*20+[0,0,1,1,2,2,3,3,4,4])}
        maps={'c':(rng.normal(size=(8,8)),np.zeros(8)),'i':(rng.normal(size=(8,8)),np.zeros(8))}
        _,a=score_method(h,data,maps,{},np.eye(8),np.zeros(8),'query')
        h[:,5]+=100
        _,b=score_method(h,data,maps,{},np.eye(8),np.zeros(8),'query')
        np.testing.assert_array_equal(a['answer_collisions_ci_hidden'],b['answer_collisions_ci_hidden'])
        np.testing.assert_array_equal(a['answer_collisions_ci_answers'],b['answer_collisions_ci_answers'])


if __name__=='__main__':unittest.main()
