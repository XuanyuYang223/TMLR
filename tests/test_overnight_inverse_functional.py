import unittest

import numpy as np
import torch

from experiments.overnight_inverse_diagnostic import stratified
from experiments.overnight_inverse_functional import cohorts, losses, schedules


class OvernightTests(unittest.TestCase):
    def test_strata_preserve_weighted_accuracy(self):
        hit=np.array([True,False,True,True,False]);modal=np.array([True,True,False,False,False])
        result=stratified(hit,modal,np.array([10,10,15,15,15]))
        self.assertEqual(result['modal']['count'],2);self.assertEqual(result['nonmodal']['count'],3)
        self.assertEqual(result['accuracy'],.6);self.assertEqual(result['modal']['accuracy'],.5)
        self.assertEqual(result['nonmodal']['accuracy'],2/3)

    def test_ordinary_unlabeled_rows_have_no_gradients(self):
        torch.manual_seed(7);logits=torch.randn(8,6,requires_grad=True);h=torch.randn(8,5,requires_grad=True)
        plan={'labeled_batch':4,'geometry_weight':.3,'distillation_weight':1.,'distillation_temperature':2.}
        loss,_=losses(logits,h,torch.tensor([0,1,2,3]),torch.randn(8,5),torch.randn(8,5),torch.randn(8,6),torch.ones(8,dtype=torch.bool),'ordinary',plan)
        loss.backward();self.assertEqual(float(logits.grad[4:].abs().sum()),0.);self.assertEqual(float(h.grad.abs().sum()),0.)
        self.assertGreater(float(logits.grad[:4].abs().sum()),0.)

    def test_geometry_changes_eligible_rows_only(self):
        torch.manual_seed(9);logits=torch.randn(8,6,requires_grad=True);h=torch.randn(8,5,requires_grad=True)
        plan={'labeled_batch':4,'geometry_weight':.3,'distillation_weight':1.,'distillation_temperature':2.}
        eligible=torch.tensor([True,True,True,False,True,True,False,False])
        loss,_=losses(logits,h,torch.tensor([0,1,2,3]),torch.randn(8,5),torch.randn(8,5),torch.randn(8,6),eligible,'correct_geometry',plan)
        loss.backward();self.assertGreater(float(h.grad[eligible].abs().sum()),0.)
        self.assertEqual(float(h.grad[~eligible].abs().sum()),0.);self.assertEqual(float(logits.grad[4:].abs().sum()),0.)

    def test_both_distillation_arms_have_unlabeled_output_gradients(self):
        for condition in ['distillation_geometry','distillation_mismatch']:
            torch.manual_seed(7);logits=torch.randn(8,6,requires_grad=True);h=torch.randn(8,5,requires_grad=True)
            plan={'labeled_batch':4,'geometry_weight':.3,'distillation_weight':1.,'distillation_temperature':2.}
            loss,_=losses(logits,h,torch.tensor([0,1,2,3]),torch.randn(8,5),torch.randn(8,5),torch.randn(8,6),torch.ones(8,dtype=torch.bool),condition,plan)
            loss.backward();self.assertGreater(float(logits.grad[4:].abs().sum()),0.);self.assertGreater(float(h.grad.abs().sum()),0.)

    def test_schedule_excludes_validation_and_pairs_lengths(self):
        plan={'updates':20,'lengths':[10,15],'labeled_batch':3,'unlabeled_batch':4}
        rep={'training_seed':3,'unlabeled_seed':5}
        support={'lengths':np.array([10]*5+[15]*5),'split':np.array([0,0,0,0,1]*2)}
        pool={'lengths':np.array([10]*8+[15]*8)}
        ls,us=schedules(plan,rep,support,pool);self.assertFalse(support['split'][ls].any())
        for l,u in zip(ls,us):self.assertTrue((support['lengths'][l]==pool['lengths'][u][0]).all())
        l2,u2=schedules(plan,rep,support,pool);np.testing.assert_array_equal(ls,l2);np.testing.assert_array_equal(us,u2)

    def test_only_support_has_validation_splits(self):
        plan={'confirmation_seed':1,'final_test_seed':2,'test_per_length':10,'unlabeled_per_repeat':13,'train_counts':[2]*5,'validation_counts':[1]*5,
              'replicates':[{'id':'n0','support_seed':3,'unlabeled_seed':4}]}
        specs=cohorts(plan);self.assertEqual(sum(specs[-1]['counts']),13);self.assertFalse(specs[-1]['labeled'])
        self.assertTrue(specs[-2]['support']);self.assertTrue(specs[-2]['labeled'])


if __name__=='__main__':unittest.main()
