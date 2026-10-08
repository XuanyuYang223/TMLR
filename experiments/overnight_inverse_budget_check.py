"""Independently regenerate input schedules and answer-matched derangements."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .inverse_functional_alignment import now
from .longrun_engine import atomic_json
from .overnight_inverse_functional import initialize
from .permworld_combinations import sha


def run():
    plan,root,sig=initialize();supp=Path('results/overnight_inverse_residual_geometry');sp=json.loads(Path('configs/overnight_inverse_residual.json').read_text());checked=0;pairings=0
    registration=root/'budget_verification_protocol.json'
    if not registration.exists():atomic_json(registration,{'registered_utc':now(),'code_sha256':sha(__file__),'rules':'Regenerate exact40k/20k schedules with independent bucket lists and RNG; prove validation excluded, lengths identical, within-batch distinct indices, and source/pseudo-answer preserving derangement on full valid pools. No new U oracle labels.'})
    assert json.loads(registration.read_text())['code_sha256']==sha(__file__)
    for rep in plan['replicates']:
        support=np.load(root/'dataset'/rep['id']/'support/dataset.npz');pool=np.load(root/'dataset'/rep['id']/'unlabeled/dataset.npz');assert 'labels'not in pool
        for part,data in [('support',support),('unlabeled',pool)]:
            teacher=np.load(root/'teacher'/f"{rep['id']}_{part}.npz");np.testing.assert_array_equal(teacher['predicted_answer'],teacher['logits'].argmax(-1))
            p=np.load(root/'teacher'/f"{rep['id']}_{part}_mismatch.npz");partner=p['partner'];valid=p['eligible']
            answers=data['labels']if part=='support'else teacher['predicted_answer'];candidate=data['split']==0 if part=='support'else np.ones(len(answers),dtype=bool)
            counts=Counter((int(n),int(y))for n,y,m in zip(data['lengths'],answers,candidate)if m)
            expected=np.array([bool(m)and counts[int(n),int(y)]>=2 for n,y,m in zip(data['lengths'],answers,candidate)])
            np.testing.assert_array_equal(valid,expected);rows=np.flatnonzero(valid);assert np.all(partner[rows]!=rows)and np.all(candidate[partner[rows]])
            np.testing.assert_array_equal(answers[rows],answers[partner[rows]]);np.testing.assert_array_equal(data['lengths'][rows],data['lengths'][partner[rows]])
            assert len(np.unique(partner[rows]))==len(rows);pairings+=1
        for folder,p,offset in [(root,plan,81001),(supp,sp,91001)]:
            file=folder/'training'/f"{rep['id']}_schedule.npz"
            if not file.exists():continue
            lgen=np.random.default_rng(rep['training_seed']+offset);ugen=np.random.default_rng(rep['unlabeled_seed']+offset)
            labeled={n:np.array([i for i,(nn,s)in enumerate(zip(support['lengths'],support['split']))if nn==n and s==0])for n in plan['lengths']}
            unlabeled={n:np.array([i for i,nn in enumerate(pool['lengths'])if nn==n])for n in plan['lengths']};saved=np.load(file)
            for j in range(p['updates']):
                n=int(lgen.choice(plan['lengths']));l=lgen.choice(labeled[n],p['labeled_batch'],replace=False);u=ugen.choice(unlabeled[n],p['unlabeled_batch'],replace=False)
                np.testing.assert_array_equal(l,saved['labeled'][j]);np.testing.assert_array_equal(u,saved['unlabeled'][j]);assert len(set(l))==len(l)and len(set(u))==len(u)
                assert np.all(support['split'][l]==0)and np.all(support['lengths'][l]==n)and np.all(pool['lengths'][u]==n)
            for condition in p['conditions']:
                record=json.loads((folder/'training'/f"{rep['id']}_{condition}.json").read_text());assert record['replicate']==rep
                if record['status']!='partial_deadline':
                    assert record['schedule_sha256']==sha(file);target=next(s for s in sig['targets']if s['seed']==rep['target_pretrain_seed']);assert record['initialization_sha256']==target['checkpoint_sha256']
                    assert record['forward_rows']==record['step']*(p['labeled_batch']+p['unlabeled_batch'])
            checked+=1
    atomic_json(root/'budget_verification.json',{'completed_utc':now(),'paired_schedules_independently_reconstructed':checked,'answer_preserving_derangements_verified':pairings,'unlabeled_oracle_answers_read':False})
    print({'schedules':checked,'pairings':pairings},flush=True)


if __name__=='__main__':run()
