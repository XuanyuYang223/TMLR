"""Independent reconstruction of new-world primary endpoints and split audit."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
import torch
from torch.nn import functional as F

from . import algebra_relation_worlds as worlds
from .algebra_relation_feasibility import apply_encoding
from .longrun_engine import atomic_json
from .operator_capacity_verify import replay
from .public_reproduction import check_manifest
from .reviewer_revision_statistics import holm
from .two_step_relation_factorial import now


def run(output=Path('results/reviewer_fresh_audit')):
    from . import algebra_relation_campaign as core
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;apply_encoding('numeric_gelu')
    c=json.loads(Path('configs/reviewer_fresh_confirmation.json').read_text());root=Path(c['output'])
    r=json.loads((root/'results.json').read_text());manifest=check_manifest(root/'delivery.json')
    opened=json.loads((root/'test_opened.json').read_text());checks=0;gates=[];effects=[[],[]]
    for i in range(c['independent_units']):
        local=root/f'u{i:02d}';parts={}
        p=c['field_prime'];key=lambda x:worlds.key(tuple(x),'matrix',p)
        for file in (local/'dataset').glob('*.npz'):
            d=dict(np.load(file));parts[file.stem]=d
            words=['','a','b','ab'] if file.stem=='test' else ['','a','b']
            for j,word in enumerate(words):
                for anchor,state,label in zip(d['x'][:,0],d['x'][:,j],d['labels'][:,j]):
                    expected=worlds.transform(tuple(anchor),word,'matrix',p)
                    assert tuple(state)==expected and label==worlds.answer(expected,'matrix',p);checks+=1
        sets={name:{key(x) for x in d['x'][:,0]} for name,d in parts.items()}
        for name in sets:
            for other in sets:
                if name!=other:assert not sets[name]&sets[other];checks+=1
        support=parts['support0'];pc=support['pc']
        np.testing.assert_array_equal(np.sort(pc),np.arange(len(pc)))
        np.testing.assert_array_equal(support['labels'][pc],support['labels']);assert (pc!=np.arange(len(pc))).all();checks+=3
        data=parts['test'];collision=data['split']==1
        for pid in range(c['collision_pairs']):
            ids=np.flatnonzero(data['pair_ids']==pid);assert len(ids)==2
            np.testing.assert_array_equal(data['labels'][ids[0],:3],data['labels'][ids[1],:3])
            assert data['labels'][ids[0],3]!=data['labels'][ids[1],3];checks+=3
        complete=json.loads((local/'unit_complete.json').read_text());assert complete['completed_utc']<opened['opened_utc'];checks+=1
        assert complete['source_initialization']==c['source_seed_base']+1009*i
        data_audit=json.loads((local/'data_audit.json').read_text())
        assert data_audit['data_seed']==c['data_seed_base']+2003*i;checks+=1
        source=torch.load(local/'sources/s0.pt',map_location='cpu',weights_only=True)
        state=torch.load(local/'models/n0_both_correct.pt',map_location='cpu',weights_only=True)
        assert source['source_seed']==complete['source_initialization']
        for name in ['readout.weight','readout.bias']:assert torch.equal(source['model'][name],state['model'][name]);checks+=1
        model=core.Encoder(p,c['hidden_width']).to('cuda');model.load_state_dict(state['model']);model.eval()
        with torch.no_grad():h=model(torch.as_tensor(worlds.feature(data['x'],p),device='cuda')).cpu().numpy().astype(float)
        z=dict(np.load(local/'states/s0_known.npz'));w,bw=[z[k].astype(float) for k in ['w','bias_w']]
        selected=complete['selected_second_noise']
        candidates=[json.loads((local/'seconds'/f'noise_{noise}.json').read_text()) for noise in c['robust_second_noise']]
        # Recompute the full known-step selection objective, not just its ordering.
        vx=torch.as_tensor(z['validation'],device='cuda')
        tx=torch.as_tensor(z['train'],device='cuda')
        wt=torch.as_tensor(z['w'],device='cuda');bt=torch.as_tensor(z['bias_w'],device='cuda')
        scale=float(tx[:,2].var(0,unbiased=False).mean());temperature=c['temperature']
        teacher=F.softmax(F.linear(vx[:,2],wt,bt)/temperature,dim=-1)
        for candidate in candidates:
            cp=torch.load(local/'seconds'/f'noise_{candidate["noise"]}.pt',map_location='cuda',weights_only=True)['state_dict']
            generator=torch.Generator(device='cuda').manual_seed(c['source_seed_base']+1009*i+401+2011)
            scores=[]
            for sigma in c['robust_second_noise']:
                eps=torch.randn(vx[:,0].shape,generator=generator,device='cuda')*sigma*np.sqrt(scale)
                prediction=F.linear(vx[:,0]+eps,cp['weight'],cp['bias'])
                logits=F.linear(prediction,wt,bt)
                kd=F.kl_div(F.log_softmax(logits/temperature,dim=-1),teacher,reduction='batchmean')*temperature**2
                score=kd+c['geometry_weight']*(prediction-vx[:,2]).square().mean()/scale
                scores.append(float(score))
            np.testing.assert_allclose(scores,candidate['known_noise_validation_scores'],atol=1e-6,rtol=1e-5)
            assert abs(np.mean(scores)-candidate['known_validation_criterion'])<1e-5;checks+=4
        assert selected==min(candidates,key=lambda a:a['known_validation_criterion'])['noise'];checks+=1
        second=torch.load(local/'seconds'/f'noise_{selected}.pt',map_location='cpu',weights_only=True)['state_dict']
        b=second['weight'].numpy().astype(float).T;bb=second['bias'].numpy().astype(float)
        hits={};schedules=[]
        for condition in c['conditions']:
            fit=json.loads((local/'predictor_fits'/f's0_{condition}.json').read_text())
            assert fit['parameters']==1050112 and fit['steps']==4000 and fit['anchor_exposures']==1024000
            schedules.append((fit['schedule_sha256'],fit['exposure_count_sha256'],fit['backbone_sha256'],fit['known_states_sha256']))
            predictor=torch.load(local/'predictors'/f's0_{condition}.pt',map_location='cpu',weights_only=True)['state_dict']
            first=replay(h[:,0],predictor,condition.startswith('nonlinear'))
            hit=((first@b+bb)@w.T+bw).argmax(1)==data['labels'][:,3];hits[condition]=hit
            saved=np.load(local/'evaluations'/f'known_selected_{condition}.npz')['hits']
            np.testing.assert_array_equal(hit,saved);checks+=len(hit)
            original=((first@z['b'].astype(float)+z['bias_b'].astype(float))@w.T+bw).argmax(1)==data['labels'][:,3]
            np.testing.assert_array_equal(original,np.load(local/'evaluations'/f'original_{condition}.npz')['hits']);checks+=len(hit)
            for split,mask in [('iid',~collision),('collisions',collision)]:
                record=next(a for a in r['records'] if (a['unit'],a['second'],a['condition'],a['split'])==(i,'known_selected',condition,split))
                first_hit=(first@w.T+bw).argmax(1)[mask]==data['labels'][mask,1]
                assert abs(first_hit.mean()-record['first_accuracy'])<1e-12
                assert abs(np.square(first[mask]-h[mask,1]).mean()-record['first_state_mse'])<1e-12;checks+=2
        assert len(set(schedules))==1;checks+=1
        effects[0].append(100*(hits['nonlinear_correct'][collision].mean()-hits['nonlinear_wrong'][collision].mean()))
        # Independent pseudoinverse construction checks the complete-logit invariant.
        p_row=np.linalg.pinv(w,rcond=1e-10)@w;q=np.eye(len(p_row))-p_row;use=data['exchange_eligible'];natural=h[use,1]
        values={}
        for name,ids,part in [('null_different',data['different_donor'],'null'),
                             ('row_different',data['different_donor'],'row'),('null_same',data['same_donor'],'null')]:
            ids=ids[use];donor=h[ids,1]
            np.testing.assert_array_equal(np.sort(ids),np.flatnonzero(use));checks+=len(ids)
            np.testing.assert_array_equal(data['labels'][ids,1],data['labels'][use,1]);checks+=len(ids)
            assert (data['orbit_ids'][ids]!=data['orbit_ids'][use]).all()
            assert ((data['labels'][ids,3]==data['labels'][use,3]) if name=='null_same' else
                    (data['labels'][ids,3]!=data['labels'][use,3])).all();checks+=len(ids)
            mixed=natural@p_row+donor@q if part=='null' else donor@p_row+natural@q
            if part=='null':np.testing.assert_allclose((mixed-natural)@w.T,0,atol=1e-8);checks+=len(ids)
            classes=((mixed@b+bb)@w.T+bw).argmax(1)
            hit=classes==data['labels'][ids,3]
            record=next(a for a in r['interchanges'] if a['unit']==i and a['condition']==name)
            assert abs(hit.mean()-record['donor_target_accuracy'])<1e-12;checks+=1
            base=((natural@b+bb)@w.T+bw).argmax(1)
            assert abs((classes!=base).mean()-record['prediction_changed_fraction'])<1e-12;checks+=1
            values[name]=hit.mean()
        effects[1].append(100*(values['null_different']-values['row_different']))
        grade=complete['source_grade'];gates.append({'unit':i,'native_min':min(grade['native']),'generator_min':min(grade['generators']),
                                                   'passed':min(grade['native'])>=.95 and min(grade['generators'])>=.9})
    pvalues=[]
    for expected,record in zip(effects,r['primary_contrasts']):
        np.testing.assert_allclose(expected,record['source_world_effects_pp'],atol=1e-12)
        pvalues.append(float(stats.ttest_1samp(expected,0).pvalue));checks+=c['independent_units']
    for actual,record in zip(holm(pvalues),r['primary_contrasts']):assert abs(actual-record['holm_adjusted_p'])<1e-12;checks+=1
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'verification.json').exists():
        raise FileExistsError('Preserve the previous audit; choose a new --output directory.')
    atomic_json(out/'verification.json',{'status':'passed','completed_utc':now(),'checks':checks,'manifest':manifest,
        'source_primitive_gates':gates,'all_gates_pass':all(a['passed'] for a in gates),
        'world_source_units':c['independent_units'],'all_parent_files_preserved':True})
    print(json.dumps({'status':'passed','checks':checks,'all_source_gates_pass':all(a['passed'] for a in gates)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/reviewer_fresh_audit'))
    run(parser.parse_args().output)
