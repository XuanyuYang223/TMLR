"""Verify selection, frozen readout, affine composition and paired counts."""
import json
from pathlib import Path

import numpy as np
import torch

from .longrun_engine import atomic_json
from .permworld_combinations import sha

ROOT=Path('results/frozen_operator_constraint_ablation')


def run():
    sig=json.loads((ROOT/'protocol.json').read_text())['signature'];d=dict(np.load(ROOT/'dataset.npz'))
    checks=0;maximum_val_difference=0.
    for source in sig['sources']:
        assert sha(source['checkpoint'])==source['checkpoint_sha256']
        assert sha(source['known_features'])==source['known_features_sha256']
        state=torch.load(source['checkpoint'],weights_only=True,map_location='cpu')
        m=dict(np.load(ROOT/'maps'/f"{source['name']}.npz"));rec=json.loads((ROOT/'maps'/f"{source['name']}.json").read_text())
        known=dict(np.load(source['known_features']));h=np.load(ROOT/'features'/f"{source['name']}.npz")['hidden'].astype(np.float64)
        out=dict(np.load(ROOT/'evaluations'/f"{source['name']}.npz"));ev=json.loads((ROOT/'evaluations'/f"{source['name']}.json").read_text())['rows']
        np.testing.assert_array_equal(m['readout_weight'],state['model']['lm_head.weight'][:31].numpy())
        np.testing.assert_array_equal(m['native_frozen_rho'],np.eye(256)[None]+state['operators']['offset'].numpy())
        np.testing.assert_array_equal(m['native_frozen_bias'],state['operators']['bias'].numpy())
        scale=np.square(known['train_hidden'].astype(np.float64)-known['train_hidden'].astype(np.float64).reshape(-1,256).mean(0)).mean()
        for family in rec['candidates']:
            j=int(np.argmin([x['visible_validation_generator_mse'] for x in family['candidates']]))
            assert j==rec['selected'][family['family']]['candidate']
            for k,meta in enumerate(family['candidates']):
                rho=m[f"{family['family']}_{k}_rho"];b=m[f"{family['family']}_{k}_bias"]
                if meta['norm_cap'] is not None:assert np.linalg.svd(rho,compute_uv=False).max()<=meta['norm_cap']+1e-5
                error=0.
                for g,a in enumerate([1,2]):
                    x=np.concatenate([known['validation_hidden'][:,0],known['validation_hidden'][:,a]]).astype(np.float64)
                    y=np.concatenate([known['validation_hidden'][:,a],known['validation_hidden'][:,0]]).astype(np.float64)
                    error+=np.square(x@rho[g]+b[g]-y).mean()/scale/2
                maximum_val_difference=max(maximum_val_difference,abs(error-meta['visible_validation_generator_mse']))
                np.testing.assert_allclose(error,meta['visible_validation_generator_mse'],atol=2e-6,rtol=2e-5)
        for method in sig['plan']['methods']:
            p='native_frozen' if method=='native_frozen' else f"{method}_{rec['selected'][method]['candidate']}"
            ac,ai=m[p+'_rho'];bc,bi=m[p+'_bias']
            ci=(h[:,0]@ac+bc)@ai+bi
            ici=((h[:,0]@ai+bi)@ac+bc)@ai+bi
            for word,pred,action in [('ci',ci,5),('ici',ici,2)]:
                answers=(pred@m['readout_weight'].astype(np.float64).T+m['readout_bias']).argmax(-1)
                np.testing.assert_array_equal(answers,out[f'{method}_{word}_answers'])
                truth=d['labels'][:,action];hit=(answers==truth).reshape(-1,2)
                swapped=(answers.reshape(-1,2)==truth.reshape(-1,2)[:,::-1]).all(1)
                row=next(x for x in ev if x['method']==method and x['word']==word)
                for key,val in [('answer_accuracy',hit.mean()),('both_correct',hit.all(1).mean()),('neither_correct',(~hit).all(1).mean()),('orientation_excess',hit.all(1).mean()-swapped.mean())]:
                    np.testing.assert_allclose(row[key],val,atol=1e-15)
                checks+=1
    atomic_json(ROOT/'verification.json',{'status':'passed','endpoints_verified':checks,
        'all_candidate_visible_selection_scores_checked':48,'maximum_visible_validation_float32_vs_float64_error':maximum_val_difference,
        'source_checkpoint_hashes_verified':3,'frozen_readout_weights_verified':True,'all_cap_constraints_verified':True,
        'composition_and_pair_counts_verified':True})


if __name__=='__main__':run()
