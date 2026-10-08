import unittest

import numpy as np
import torch

from experiments.overnight_inverse_residual import answer_residual, loss_terms, schedule


class ResidualGeometryTests(unittest.TestCase):
    def test_answer_offsets_cancel_and_singletons_excluded(self):
        h=torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.],[9.,10.]])
        answers=torch.tensor([2,2,2,3,4]);centered,eligible=answer_residual(h,answers)
        shifted=h+torch.where((answers==2)[:,None],torch.tensor([40.,-23.]),torch.zeros(2))
        result,mask=answer_residual(shifted,answers)
        torch.testing.assert_close(centered,result);self.assertTrue(torch.equal(mask,eligible))
        self.assertEqual(eligible.tolist(),[True,True,True,False,False]);torch.testing.assert_close(centered[eligible].sum(0),torch.zeros(2))

    def test_residual_geometry_has_no_labeled_hidden_gradient(self):
        torch.manual_seed(33);logits=torch.randn(12,8,requires_grad=True);h=torch.randn(12,6,requires_grad=True)
        answers=torch.tensor([1,1,1,2,2,2,3,4]);plan={'geometry_weight':.3,'distillation_temperature':2.}
        loss,_=loss_terms(logits,h,torch.tensor([0,1,2,3]),torch.randn(12,6),torch.randn(12,6),torch.randn(12,8),answers,4,'residual_correct',plan)
        loss.backward();self.assertEqual(float(h.grad[:4].abs().sum()),0.);self.assertEqual(float(h.grad[10:].abs().sum()),0.)
        self.assertGreater(float(h.grad[4:10].abs().sum()),0.);self.assertGreater(float(logits.grad[4:].abs().sum()),0.)

    def test_distillation_baseline_has_no_hidden_alignment_gradient(self):
        torch.manual_seed(33);logits=torch.randn(12,8,requires_grad=True);h=torch.randn(12,6,requires_grad=True)
        plan={'geometry_weight':.3,'distillation_temperature':2.}
        loss,_=loss_terms(logits,h,torch.tensor([0,1,2,3]),torch.randn(12,6),torch.randn(12,6),torch.randn(12,8),torch.tensor([1,1,1,2,2,2,3,4]),4,'distillation',plan)
        loss.backward();self.assertEqual(float(h.grad.abs().sum()),0.);self.assertGreater(float(logits.grad[4:].abs().sum()),0.)

    def test_schedule_pairs_all_rows_and_excludes_validation(self):
        plan={'updates':10,'labeled_batch':3,'unlabeled_batch':6};bp={'lengths':[10,15]};rep={'training_seed':7,'unlabeled_seed':11}
        support={'lengths':np.array([10]*5+[15]*5),'split':np.array([0,0,0,0,1]*2)};pool={'lengths':np.array([10]*8+[15]*8)}
        ls,us=schedule(plan,bp,rep,support,pool);self.assertFalse(support['split'][ls].any())
        for a,b in zip(ls,us):self.assertTrue((support['lengths'][a]==pool['lengths'][b][0]).all())


if __name__=='__main__':unittest.main()
