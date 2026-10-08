"""Independent stored-state intervention and affine solver verification."""
import json
from pathlib import Path

import numpy as np
import torch

from .longrun_engine import atomic_json
from .operator_capacity_verify import orbit_key, word, encoder_replay
from .permworld_combinations import sha
from .two_step_relation_factorial import now


ROOT=Path('results/operator_mechanism_diagnostic')
PARENT=Path('results/operator_capacity_confirmation')


def softmax(z):
    z=z-z.max(-1,keepdims=True);v=np.exp(z);return v/v.sum(-1,keepdims=True)


def run():
    config=json.loads(Path('configs/operator_mechanism_diagnostic.json').read_text())
    protocol=json.loads((ROOT/'protocol.json').read_text())
    results=json.loads((ROOT/'results.json').read_text());fit_summary=json.loads((ROOT/'fit_summary.json').read_text())
    assert sha('experiments/operator_mechanism_diagnostic.py')==protocol['signature']['code_sha256']
    checks=0
    for paths in [protocol['signature']['parent_artifacts_sha256'],protocol['signature']['dependencies_sha256'],
                  json.loads((ROOT/'data_audit.json').read_text())['dataset_sha256']]:
        for p,expected in paths.items():assert sha(p)==expected;checks+=1
    excluded={orbit_key(x) for file in ['results/algebra_relation_v3/matrix/dataset/test.npz',
        'results/readout_null_confirmation/matrix/dataset/test.npz',str(PARENT/'dataset/test.npz')]
        for x in np.load(file)['x'][:,0]}
    common={orbit_key(x) for name in ['source','validation','pilot_validation']
            for x in np.load(PARENT/'dataset'/f'{name}.npz')['x'][:,0]}
    used=set();max_logit=0.;max_encoder=0.;max_identity=0.;max_gap=0.;contrasts={}
    opened=json.loads((ROOT/'fresh_opened.json').read_text())
    assert len(opened['fit_record_sha256'])==6
    for i in range(3):
        data=dict(np.load(ROOT/'datasets'/f's{i}_fresh.npz'))
        own={orbit_key(x) for x in np.load(PARENT/'dataset'/f'support{i}.npz')['x'][:,0]}
        keys={orbit_key(x) for x in data['x'][:,0]}
        assert len(keys)==384 and not keys&(excluded|common|own|used)
        used|=keys
        for path,labels in zip(data['x'],data['labels']):
            for x,y,w in zip(path,labels,['','a','b','ab']):
                expected=word(path[0],w)
                assert tuple(x)==expected and (expected[0]+expected[3])%13==y;checks+=1
        for pid in range(128):
            ids=np.flatnonzero(data['pair_ids']==pid);assert len(ids)==2
            assert np.array_equal(data['labels'][ids[0],:3],data['labels'][ids[1],:3])
            assert data['labels'][ids[0],3]!=data['labels'][ids[1],3];checks+=3
        state=torch.load(PARENT/'models'/f'n{i}_both_correct.pt',map_location='cpu',weights_only=True)
        z=dict(np.load(PARENT/'states'/f's{i}_known.npz'))
        w,bw,b,bb=[z[k].astype(float) for k in ['w','bias_w','b','bias_b']]
        # Independent QR projector, rather than the analysis's SVD projector.
        basis=np.linalg.qr(w.T,mode='reduced')[0];p=basis@basis.T;q=np.eye(128)-p
        support=dict(np.load(PARENT/'dataset'/f'support{i}.npz'))
        train=z['train'].astype(float)
        x=np.column_stack([train[:,0],np.ones(1024)])
        u,s,v=np.linalg.svd(x,full_matrices=False);features=u*np.sqrt(1024)
        scale=float(np.var(z['train'][:,1],axis=0).mean())
        for pairing in ['correct','wrong']:
            file=ROOT/'fits'/f's{i}_{pairing}.json';rec=json.loads(file.read_text())
            assert sha(file)==opened['fit_record_sha256'][str(file)]
            assert rec['completed_utc']<opened['opened_utc']
            target=train[:,1] if pairing=='correct' else train[support['pc'],1]
            params=dict(np.load(ROOT/'fits'/f's{i}_{pairing}.npz'))
            assert sha(ROOT/'fits'/f's{i}_{pairing}.npz')==rec['checkpoint_sha256']
            pred=x@params['state_ols']
            orthogonality=features.T@(pred-target)/1024
            np.testing.assert_allclose(orthogonality,0,atol=1e-7)
            pred=x@params['convex_kd_geometry']
            t=config['temperature']
            pp=softmax((pred@w.T+bw)/t);pt=softmax((train[:,1]@w.T+bw)/t)
            grad_pred=config['output_kd_weight']*t*(pp-pt)@w/1024+2*config['geometry_weight']*(pred-target)/(1024*128*scale)
            # Full affine coefficient gradient in an orthonormal training-design basis.
            gradient=features.T@grad_pred
            mu=2*config['geometry_weight']/(scale*128)
            gap=float(np.square(gradient).sum()/(2*mu))
            assert gap<1e-8
            max_gap=max(max_gap,gap);checks+=4
            assert rec['solver']['no_compound_fit_targets']
            # Every direct affine map can be represented by the original Identity residual.
            for kind,theta in params.items():
                delta=theta[:128]-z['a'].astype(float);bias=theta[128]-z['bias_a'].astype(float)
                represented=train[:4,0]@z['a'].astype(float)+z['bias_a'].astype(float)+train[:4,0]@delta+bias
                np.testing.assert_allclose(represented,x[:4]@theta,atol=1e-7,rtol=1e-7);checks+=1
        for test in ['existing','fresh']:
            if test=='existing':
                td=dict(np.load(PARENT/'dataset/test.npz'));h=np.load(PARENT/'states'/f's{i}_test.npz')['hidden'].astype(float)
            else:
                td=data;h=np.load(ROOT/'states'/f's{i}_fresh.npz')['hidden'].astype(float)
                native=encoder_replay(td['x'],state['model'])
                np.testing.assert_allclose(native,h,atol=5e-5,rtol=5e-5)
                max_encoder=max(max_encoder,float(np.abs(native-h).max()));checks+=1
            caches={}
            for file in (ROOT/'evaluations').glob(f'{test}_s{i}_*.npz'):
                name=file.stem[len(f'{test}_s{i}_'):];cached=dict(np.load(file));first=cached['first_pred']
                truth=h[:,1];error=first-truth;en=error@q;er=error@p
                oracle=(truth@b+bb)@w.T+bw
                logits=(first@b+bb)@w.T+bw
                np.testing.assert_allclose(logits-oracle,en@b@w.T+er@b@w.T,atol=1e-8,rtol=1e-8)
                hits=logits.argmax(1)==td['labels'][:,3]
                np.testing.assert_array_equal(hits,cached['compound_hit'])
                np.testing.assert_allclose(cached['hidden_error'],np.square(error).mean(1)/scale,atol=1e-10)
                def centered(a):return a-a.mean(1,keepdims=True)
                dn,dr=centered(en@b@w.T),centered(er@b@w.T)
                total=np.square(centered(logits-oracle)).sum(1)
                decomposition=np.square(dn).sum(1)+np.square(dr).sum(1)+2*(dn*dr).sum(1)
                np.testing.assert_allclose(total,decomposition,atol=1e-7,rtol=1e-8)
                max_identity=max(max_identity,float(np.abs(total-decomposition).max()))
                for sid,split in [(0,'iid'),(1,'collisions')]:
                    r=next(x for x in results['records'] if (x['test'],x['source'],x['condition'],x['split'])==(test,i,name,split))
                    use=td['split']==sid;assert r['accuracy']==float(hits[use].mean());checks+=1
                    if sid:
                        pairs=np.array([hits[td['pair_ids']==j] for j in range(128)])
                        assert r['pair_both_correct']==float(pairs.all(1).mean())
                        contrasts[(test,i,name)]=pairs;checks+=1
                caches[name]=cached;checks+=4
            for recipient,donor in config['hybrids']:
                r=caches[recipient]['first_pred'];d=caches[donor]['first_pred']
                name=f'swap_{recipient}_null_{donor}'
                hybrid=r+(d-r)@q
                np.testing.assert_allclose(hybrid,caches[name]['first_pred'],atol=1e-10,rtol=1e-10)
                logit=(hybrid-r)@w.T;max_logit=max(max_logit,float(np.abs(logit).max()))
                assert np.abs(logit).max()<1e-9;checks+=2
    for test in ['existing','fresh']:
        for a,c,name in [('swap_linear_correct_null_nonlinear_correct','linear_correct','correct_null_swap_gain'),
                         ('convex_kd_geometry_correct','nonlinear_correct','convex_linear_vs_gelu')]:
            value=np.array([100*(contrasts[(test,i,a)].mean(1)-contrasts[(test,i,c)].mean(1)) for i in range(3)])
            rec=next(x for x in results['contrasts'] if (x['test'],x['contrast'],x['endpoint'])==(test,name,'accuracy'))
            np.testing.assert_allclose(value.mean(1),rec['source_effects_pp'],atol=1e-12)
            np.testing.assert_allclose(value.mean(),rec['mean_pp'],atol=1e-12);checks+=2
    atomic_json(ROOT/'independent_verification.json',{'status':'passed','verified_utc':now(),'checks':checks,
        'max_first_logit_change_qr_projector':max_logit,'max_hidden_encoder_replay_difference':max_encoder,
        'max_error_energy_identity_residual':max_identity,'max_recomputed_full_training_gap_bound':max_gap,
        'fresh_orbits_checked':len(used),'original_parent_artifacts_preserved':len(protocol['signature']['parent_artifacts_sha256']),
        'scope':'Independent QR nullspace, analytic softmax gradients and convex-gap bound, scalar group actions, double encoder replay and predicted-state intervention arithmetic. Fresh tests reuse three sources.'})
    print(json.dumps({'status':'passed','checks':checks,'max_gap':max_gap,'max_swap_logit':max_logit}),flush=True)


if __name__=='__main__':run()
