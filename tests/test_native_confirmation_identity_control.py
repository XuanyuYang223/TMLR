import unittest
from experiments.native_confirmation_identity_control import identity_features


class IdentityControlTests(unittest.TestCase):
    def test_old_seed_indicators_do_not_become_new_group_features(self):
        row={'group':'b','baseline':list(map(float,range(12)))}
        values=identity_features(row,['a','b','c'])
        self.assertEqual(values,[0.,1.,7.,8.,9.,1.,0.])
        row['group']='a'
        self.assertEqual(identity_features(row,['a','b','c'])[-2:],[0.,0.])


if __name__=='__main__':unittest.main()
