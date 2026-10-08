import copy
import unittest
import numpy as np
import torch

from experiments.longrun_attention import accelerate
from experiments.longrun_engine import learning_rate
from experiments.longrun_transfer import normalized_features, query_features
from experiments.permworld_combinations import features
from neurips_permutations.models import CausalTransformer
from neurips_permutations.passage import TOKEN_TO_ID, one_line_tokens


class LongRunTransferTests(unittest.TestCase):
    def test_support_normalization_does_not_use_validation_or_test_distribution(self):
        support=np.array([[1.,3.],[3.,7.]])
        test=np.array([[100.,1000.]])
        x,(xt,)=normalized_features(support,[test])
        np.testing.assert_allclose(x,[[-1.,-1.],[1.,1.]])
        np.testing.assert_allclose(xt,[[98.,497.5]])
        np.testing.assert_array_equal(test,[[100.,1000.]])

    def test_accelerated_model_adaptation_copy_cannot_change_source_weights(self):
        torch.manual_seed(9)
        model=accelerate(CausalTransformer(30,16,d_model=16,layers=2,n_heads=2))
        learner=copy.deepcopy(model)
        before={k:v.clone() for k,v in model.state_dict().items()}
        x=torch.randint(0,30,(2,12))
        opt=torch.optim.AdamW(learner.parameters(),lr=.01)
        learner(x).square().sum().backward();opt.step()
        for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertFalse(torch.equal(model.token_embedding.weight,learner.token_embedding.weight))

    def test_query_and_task_free_features_need_no_answer_labels(self):
        permutation=tuple(range(1,11))
        tokens=['<BOS>','<SIZE>','10',*one_line_tokens(permutation)]
        inputs=torch.tensor([[TOKEN_TO_ID[t] for t in tokens]])
        data={'x_input':inputs,'x_lengths':torch.tensor([10])}
        model=accelerate(CausalTransformer(len(TOKEN_TO_ID),26,d_model=16,layers=1,n_heads=2))
        self.assertEqual(features(model,data,'x',TOKEN_TO_ID).shape,(1,16))
        self.assertEqual(query_features(model,data,'x','descents',TOKEN_TO_ID).shape,(1,16))

    def test_learning_rate_warmup_then_decay_is_fixed_by_step(self):
        plan={'warmup_steps':500,'source_learning_rate':.0003,'minimum_learning_rate_ratio':.1}
        self.assertAlmostEqual(learning_rate(500,20000,plan),.0003)
        self.assertAlmostEqual(learning_rate(20000,20000,plan),.00003)
        self.assertLess(learning_rate(1,20000,plan),learning_rate(400,20000,plan))


if __name__=='__main__':unittest.main()
