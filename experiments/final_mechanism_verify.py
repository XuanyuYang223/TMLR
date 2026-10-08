"""Independent audit for final new-source confirmation and same-loss solves."""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import torch

from .longrun_engine import atomic_json
from .operator_capacity_verify import orbit_key,word,encoder_replay,replay
from .permworld_combinations import sha
from .two_step_relation_factorial import now

ROOT=Path('results/final_mechanism_confirmation')


def soft(z):
    z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)


def audit_data(config):
    sets={};checks=0
    for file in (ROOT/'dataset').glob('*.npz'):
        d=dict(np.load(file));words=['','a','b','ab'] if file.stem=='test' else ['','a','b']
        for path,label in zip(d['x'],d['labels']):
            for x,y,w in zip(path,label,words):
                v=word(path[0],w);assert tuple(x)==v and (v[0]+v[3])%13==y;checks+=1
        sets[file.stem]={orbit_key(x) for x in d['x'][:,0]}
        if 'pc' in d:
            ids=np.arange(len(d['labels']))
            assert np.array_equal(np.sort(d['pc']),ids) and (d['pc']!=ids).all()
            assert np.array_equal(d['labels'][d['pc']],d['labels']);checks+=3
    train=set.union(sets['source'],*[sets[f'support{i}'] for i in range(5)])
    val=sets['validation']|sets['pilot_validation'];test=sets['test']
    assert not train&val and not train&test and not val&test
    old={orbit_key(x) for file in config['excluded_test_datasets'] for x in np.load(file)['x'][:,0]}
    assert len(old)==2688 and not test&old
    d=np.load(ROOT/'dataset/test.npz')
    for pid in range(128):
        ids=np.flatnonzero(d['pair_ids']==pid);assert len(ids)==2
        assert np.array_equal(d['labels'][ids[0],:3],d['labels'][ids[1],:3])
        assert d['labels'][ids[0],3]!=d['labels'][ids[1],3];checks+=3
    return checks


def run():
    torch.set_num_threads(1)
    config=json.loads(Path('configs/final_mechanism_confirmation.json').read_text())
    protocol=json.loads((ROOT/'protocol.json').read_text());results=json.loads((ROOT/'results.json').read_text())
    checks=audit_data(config)
    for paths in [protocol['signature']['code_sha256'],protocol['signature']['historical_test_sha256']]:
        for p,expected in paths.items():assert sha(p)==expected;checks+=1
    preserved=0
    for file in ['results/operator_capacity_confirmation/delivery.json','results/operator_mechanism_diagnostic/delivery.json']:
        assert sha(file)==protocol['signature']['previous_delivery_sha256'][file]
        for p,expected in json.loads(Path(file).read_text())['artifact_sha256'].items():assert sha(p)==expected;preserved+=1
    opened=json.loads((ROOT/'test_opened.json').read_text())
    assert opened['all20_budget_and20_convex_fits_complete']
    assert sha('experiments/final_mechanism_evaluate.py')==opened['evaluation_code_sha256']
    for p,expected in opened['fit_sha256'].items():
        assert sha(p)==expected
        assert json.loads(Path(p).read_text())['completed_utc']<opened['opened_utc'];checks+=2
    data=dict(np.load(ROOT/'dataset/test.npz'));max_gap=0.;max_logit=0.;source_hashes=[];pair_hits={}
    for i,seed in enumerate(config['source_seeds']):
        ordinary=torch.load(ROOT/'sources'/f's{i}.pt',map_location='cpu',weights_only=True)
        assert ordinary['source_seed']==seed
        assert seed not in [131,271,389,8123,9133,10151,11261,12373,13487]
        backbone=torch.load(ROOT/'models'/f'n{i}_both_correct.pt',map_location='cpu',weights_only=True)
        source_hashes.append(sha(ROOT/'sources'/f's{i}.pt'))
        for k in ['readout.weight','readout.bias']:assert torch.equal(ordinary['model'][k],backbone['model'][k]);checks+=1
        z=dict(np.load(ROOT/'states'/f's{i}_known.npz'));h=np.load(ROOT/'states'/f's{i}_test.npz')['hidden'].astype(float)
        np.testing.assert_allclose(encoder_replay(data['x'],backbone['model']),h,atol=5e-5,rtol=5e-5);checks+=1
        w,bw,b,bb=[z[k].astype(float) for k in ['w','bias_w','b','bias_b']]
        basis=np.linalg.qr(w.T,mode='reduced')[0];q=np.eye(128)-basis@basis.T
        rng=np.random.default_rng(config['schedule_seed']+i);digest=sha256()
        for _ in range(2000):digest.update(rng.integers(0,1024,128,dtype=np.int64).tobytes())
        scale=None
        for c in config['conditions']:
            file=ROOT/'predictors'/f's{i}_{c}.pt';state=torch.load(file,map_location='cpu',weights_only=True)
            rec=json.loads((ROOT/'predictor_fits'/f's{i}_{c}.json').read_text())
            assert rec['steps']==2000 and rec['anchor_exposures']==256000 and rec['parameters']==65920
            assert rec['schedule_sha256']==digest.hexdigest() and rec['checkpoint_sha256']==sha(file)
            assert rec['backbone_sha256']==sha(ROOT/'models'/f'n{i}_both_correct.pt')
            assert sum(v.numel() for k,v in state['state_dict'].items() if k.startswith('layers.'))==65920
            if scale is None:scale=rec['geometry_scale']
            assert rec['geometry_scale']==scale;checks+=6
        tx=np.column_stack([z['train'][:,0].astype(float),np.ones(1024)])
        u,_,_=np.linalg.svd(tx,full_matrices=False);f=u*np.sqrt(1024)
        teacher=np.load(ROOT/'states'/f's{i}_fixed_supervision.npz')['teacher_logits'].astype(float)
        np.testing.assert_allclose(z['train'][:,1].astype(float)@w.T+bw,teacher,atol=1e-4,rtol=1e-5)
        pc=np.load(ROOT/'dataset'/f'support{i}.npz')['pc']
        for pairing in ['correct','wrong']:
            target=z['train'][:,1].astype(float)
            if pairing=='wrong':target=target[pc]
            for kind in config['linear_solvers']:
                cp=ROOT/'affine_fits'/f's{i}_{pairing}_{kind}.npz';theta=np.load(cp)['theta']
                rec=json.loads(cp.with_suffix('.json').read_text());assert sha(cp)==rec['checkpoint_sha256']
                pred=tx@theta;t=config['temperature']
                pp,pt=soft((pred@w.T+bw)/t),soft(teacher/t)
                g=t*(pp-pt)@w/1024+2*.25*(pred-target)/(1024*128*scale)
                grad=f.T@g;mu=2*.25/(128*scale);gap=float(np.square(grad).sum()/(2*mu))
                assert gap<1e-8;max_gap=max(max_gap,gap);checks+=2
                htest=np.column_stack([h[:,0],np.ones(len(h))])
                expected=htest@theta
                cached=np.load(ROOT/'evaluations'/f's{i}_affine_{pairing}_{kind}.npz')['first_pred']
                np.testing.assert_allclose(expected,cached,atol=1e-9,rtol=1e-9);checks+=1
        caches={}
        for file in (ROOT/'evaluations').glob(f's{i}_*.npz'):
            name=file.stem[len(f's{i}_'):];cache=dict(np.load(file));first=cache['first_pred']
            logits=(first@b+bb)@w.T+bw;hit=logits.argmax(1)==data['labels'][:,3]
            np.testing.assert_array_equal(hit,cache['compound_hit']);checks+=1
            if name in config['conditions']:
                state=torch.load(ROOT/'predictors'/f's{i}_{name}.pt',map_location='cpu',weights_only=True)['state_dict']
                np.testing.assert_allclose(first,replay(h[:,0],state,name.startswith('nonlinear')),atol=1e-10,rtol=1e-10);checks+=1
            error=first-h[:,1]
            np.testing.assert_allclose(cache['hidden_error'],np.square(error).mean(1)/scale,atol=1e-10)
            en=error@q;dn=en@b@w.T;dn-=dn.mean(1,keepdims=True)
            np.testing.assert_allclose(cache['downstream_null_sq'],np.square(dn).sum(1),atol=1e-7,rtol=1e-8);checks+=2
            for sid,split in [(0,'iid'),(1,'collisions')]:
                row=next(r for r in results['records'] if (r['source'],r['condition'],r['split'])==(i,name,split))
                use=data['split']==sid;assert row['accuracy']==float(hit[use].mean());checks+=1
                if sid:
                    hh=np.array([hit[data['pair_ids']==pid] for pid in range(128)])
                    assert row['pair_both_correct']==float(hh.all(1).mean());pair_hits[(i,name)]=hh;checks+=1
            caches[name]=cache
        for recipient,donor in config['hybrids']:
            original=caches[recipient]['first_pred'];d=caches[donor]['first_pred']
            name=f'swap_{recipient}_null_{donor}';hybrid=original+(d-original)@q
            np.testing.assert_allclose(hybrid,caches[name]['first_pred'],atol=1e-10,rtol=1e-10)
            change=(hybrid-original)@w.T;max_logit=max(max_logit,float(np.abs(change).max()))
            assert np.abs(change).max()<1e-9;checks+=2
    assert len(set(source_hashes))==5
    for name,coefficients in {
        'same_budget_interaction':{'nonlinear_correct':1,'nonlinear_wrong':-1,'linear_correct':-1,'linear_wrong':1},
        'correct_null_swap_gain':{'swap_linear_correct_null_nonlinear_correct':1,'linear_correct':-1},
        'remaining_relation_interaction':{'nonlinear_correct':1,'nonlinear_wrong':-1,'affine_correct_ols_initialization':-1,'affine_wrong_ols_initialization':1}}.items():
        values=100*sum(c*np.array([pair_hits[(i,key)].mean(1) for i in range(5)]) for key,c in coefficients.items())
        record=next(r for r in results['contrasts'] if r['contrast']==name and r['endpoint']=='accuracy')
        np.testing.assert_allclose(values.mean(1),record['source_effects_pp'],atol=1e-12);checks+=1
    atomic_json(ROOT/'independent_verification.json',{'status':'passed','verified_utc':now(),'checks':checks,
        'old_artifacts_preserved':preserved,'new_independent_source_hashes':source_hashes,
        'max_recomputed_global_gap_bound':max_gap,'max_independent_swap_logit_change':max_logit,
        'verified_new_source_seeds':config['source_seeds'],'scope':'Independent scalar algebra, QR projectors, analytic shared-loss gradients and strong-convexity bounds; full frozen state, budget, hashes and predicted-state replay.'})
    print(json.dumps({'status':'passed','checks':checks,'max_gap':max_gap}),flush=True)


if __name__=='__main__':run()
