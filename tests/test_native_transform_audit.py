import unittest
import numpy as np

from experiments.native_transform_audit import token_transform


class NativeTransformAuditTests(unittest.TestCase):
    def test_every_operator_is_involutive_and_preserves_nonvalue_tokens(self):
        values = [3, 1, 4, 2]
        prefix = np.full((1, 20), 150, dtype=np.int64)
        prefix[0, 4+2*np.arange(4)] = values
        original = prefix.copy()
        n = np.array([4])
        for operator in ('complement', 'reverse', 'reverse_complement', 'inverse'):
            changed = token_transform(prefix, n, operator)
            np.testing.assert_array_equal(token_transform(changed, n, operator), original)
            mask = np.ones(20, dtype=bool); mask[4+2*np.arange(4)] = False
            np.testing.assert_array_equal(changed[:, mask], original[:, mask])
        np.testing.assert_array_equal(prefix, original)


if __name__ == '__main__': unittest.main()
