import unittest
import numpy as np
from experiments.native_output_consistency import score_identity


class MathematicalConsistencyTests(unittest.TestCase):
    def test_algebraic_consistency_can_hold_for_incorrect_answers(self):
        predicted=np.array([[1,4,5],[1,3,6],[1,3,5]])
        result=score_identity(predicted,np.array([10,10,10]),{'fixed':1,'above':1,'below':1},1,0,{'fixed':0,'above':1,'below':2})
        np.testing.assert_array_equal(result,[True,True,False])

    def test_unsupported_answer_token_cannot_count_as_consistent(self):
        predicted=np.array([[100,100,0]])
        result=score_identity(predicted,np.array([10]),{'a':1,'b':-1,'c':1},0,0,{'a':0,'b':1,'c':2})
        self.assertFalse(result[0])


if __name__=='__main__':unittest.main()
