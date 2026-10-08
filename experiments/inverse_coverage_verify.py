"""Independent validation of coverage, pseudo-label controls and all endpoints."""
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.nn import functional as F

from .inverse_alignment_coverage import initialize
from .inverse_functional_alignment import configure, now
from .inverse_functional_verify import direct, gram, inverse, key
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .specialist_cka_controls import api, old_inputs


def data_check(plan,root,sig):
    audit = json.loads((root/'dataset/audit.json').read_text()); _,tokens,_,_,_ = api(plan); all_keys = set(); examples = 0
    for cohort in audit['cohorts']:
        file = root/'dataset'/cohort['cohort']/'dataset.npz'; assert sha(file)==cohort['sha256']; d = dict(np.load(file)); labeled = cohort['cohort']=='test'
        assert ('labels' in d)==labeled and cohort['oracle_labels_present']==labeled
        assert len(d['lengths'])==(2560 if labeled else 4096)
        for i,n in enumerate(d['lengths']):
            p = d['permutations'][i,0,:n]; assert sorted(p)==list(range(1,n+1)); c=n+1-p
            orbit=[p,c,p[::-1],c[::-1],inverse(p),inverse(c),inverse(p[::-1]),inverse(c[::-1])]
            for j,q in enumerate(orbit):
                np.testing.assert_array_equal(q,d['permutations'][i,j,:n]); assert np.all(d['permutations'][i,j,n:]==0)
                kk=key(q); assert kk not in all_keys; all_keys.add(kk)
            for action,q in enumerate([p,orbit[4]]):
                row=d['input'][i,action]; assert row[0]==tokens['<BOS>'] and row[1]==tokens['<SIZE>'] and row[2]==tokens[f'{n:02d}'] and row[3]==tokens['<ONE_START>']
                assert row[2*n+3]==tokens['<ONE_END>'] and np.all(row[2*n+4:]==tokens['<PAD>'])
                np.testing.assert_array_equal(row[4:4+2*n:2],[tokens[f'{v:02d}'] for v in q])
            if labeled: assert int(np.sum(orbit[4][:-1]>orbit[4][1:]))==int(d['labels'][i])
            examples+=1
    for entry in sig['excluded_archives']: assert sha(entry['path'])==entry['sha256']
    prior=old_inputs(sig['excluded_archives']); assert not all_keys.intersection(prior)
    manifest=Path(plan['repository'])/'data/permutation-properties-16m-v1/manifest.json'; assert sha(manifest)==audit['original_manifest_sha256']
    parent=json.loads(manifest.read_text()); pattern=re.compile(rb'"primary"\s*:\s*(\[[^]]+\])'); scanned=0
    for j,shard in enumerate(parent['shards']):
        path=manifest.parent/shard['filename']; assert sha(path)==shard['sha256']
        with gzip.open(path,'rb') as handle:
            for line in handle:
                match=pattern.search(line); assert match is not None; assert match[1].replace(b' ',b'') not in all_keys; scanned+=1
        if j%50==0: print({'independent_parent_inputs_rescanned':scanned},flush=True)
    assert scanned==16000000
    return {'new_examples':examples,'new_orbit_states_checked':len(all_keys),'original_inputs_rescanned':scanned,'prior_local_inputs_rechecked':len(prior),'unlabeled_archives_have_no_oracle_answers':True,'unlabeled_oracle_answers_computed_by_verifier':False}


def schedules_check(plan,root,sig):
    old=Path(plan['previous_study']); exposures={}
    for rep,seed in zip(sig['replicates'],plan['unlabeled_seeds']):
        support=np.load(old/'dataset'/rep['id']/'dataset.npz'); pool=np.load(root/'dataset'/rep['id']/'dataset.npz'); rng=np.random.default_rng(rep['data_seed']+81001); ur=np.random.default_rng(seed+81001)
        ls=[];us=[]
        for _ in range(1200):
            n=rng.choice(plan['lengths']); a=np.flatnonzero((support['split']==0)&(support['lengths']==n)); b=np.flatnonzero(pool['lengths']==n)
            ls.append(rng.choice(a,32,replace=False));us.append(ur.choice(b,32,replace=False))
        ls=np.asarray(ls);us=np.asarray(us); file=root/'training'/f"{rep['id']}_schedule.npz"; saved=np.load(file)
        np.testing.assert_array_equal(ls,saved['labeled']);np.testing.assert_array_equal(us,saved['unlabeled']);assert np.all(support['split'][ls]==0)
        np.testing.assert_array_equal(support['lengths'][ls],pool['lengths'][us])
        label_hash=hashlib.sha256(ls.tobytes()).hexdigest(); previous=json.loads((old/'training'/f"{rep['id']}_ordinary.json").read_text());assert label_hash==previous['schedule_sha256']
        teacher=np.load(root/'teacher'/f"{rep['id']}.npz"); predictions=teacher['logits'].argmax(-1);np.testing.assert_array_equal(predictions,teacher['predicted_answer'])
        pairing=np.load(root/'teacher'/f"{rep['id']}_mismatch.npz");p=pairing['partner'];eligible=pairing['eligible'];rows=np.flatnonzero(eligible)
        assert np.all(p[rows]!=rows);np.testing.assert_array_equal(predictions[rows],predictions[p[rows]]);np.testing.assert_array_equal(pool['lengths'][rows],pool['lengths'][p[rows]])
        expected=np.zeros(4096,dtype=bool)
        for n in plan['lengths']:
            for y in np.unique(predictions[pool['lengths']==n]):
                ix=np.flatnonzero((pool['lengths']==n)&(predictions==y))
                if len(ix)>=2:expected[ix]=True
        np.testing.assert_array_equal(eligible,expected)
        for condition in plan['conditions']:
            record=json.loads((root/'training'/f"{rep['id']}_{condition}.json").read_text())
            assert record['schedule_sha256']==sha(file) and record['labeled_schedule_bytes_sha256']==label_hash
            assert record['initialization_sha256']==sha(old/'initializations'/f"{rep['id']}.pt")
            assert record['distinct_unlabeled_exposed']==len(np.unique(us))
            assert record['labeled_forward_examples']==record['unlabeled_forward_examples']==38400
            assert record['target_labeled_examples']==192 and record['target_validation_examples']==64
        exposures[rep['id']]=len(np.unique(us))
    return {'schedules_recreated':6,'labeled_schedules_match_previous_study':True,'all_36_exposures_and_initializations_matched':True,'distinct_unlabeled_exposed':exposures,'teacher_predicted_answer_mismatch_preserves_length_and_prediction':True}


def geometry_check(plan,root,sig,test):
    records=json.loads((root/'geometry.json').read_text())['records']; errors=[]
    for rec in records:
        rep=next(r for r in sig['replicates'] if r['id']==rec['replicate']);name=rep['id']+'_'+rec['condition']
        source=np.load(root/'teacher'/f"test_s{rep['source_seed']}.npz")['hidden']
        if rec['condition']=='initialization':hidden=np.load(root/'evaluations'/f'{name}.npz')['hidden']
        else:hidden=np.load(root/'evaluations'/f'{name}.npz')[rec['endpoint']+'_hidden']
        for row in rec['geometry']:
            ix=np.flatnonzero(test['lengths']==row['length'])[:128];a=hidden[ix];b=source[ix];labels=test['labels'][ix]
            keep=np.array([(labels==v).sum()>=3 for v in labels]);a0=a[keep];b0=b[keep];labels=labels[keep];ar=a0.astype(float).copy();br=b0.astype(float).copy();perm=np.arange(len(ar))
            for y in set(labels):
                j=np.flatnonzero(labels==y);perm[j]=np.r_[j[-1],j[:-1]];ar[j]-=ar[j].mean(0);br[j]-=br[j].mean(0)
            expected={'correct_cka':gram(a,b),'matched_correct_cka':gram(a0,b0),'matched_wrong_cka':gram(a0,b0[perm]),'answer_residual_correct_cka':gram(ar,br),'answer_residual_wrong_cka':gram(ar,br[perm])}
            assert row['rows']==128 and row['matched_rows']==int(keep.sum())
            for name,value in expected.items():np.testing.assert_allclose(value,row[name],atol=2e-12,rtol=2e-12);errors.append(abs(value-row[name]))
        pf=np.load(root/'evaluations'/f"{rec['replicate']}_{rec['condition']}_{rec['endpoint']}_pool.npz");pool=np.load(root/'dataset'/rep['id']/'dataset.npz');teacher=np.load(root/'teacher'/f"{rep['id']}.npz")['hidden'];ix=pf['pool_indices']
        expected_ix=np.concatenate([np.flatnonzero(pool['lengths']==n)[:128] for n in plan['lengths']]);np.testing.assert_array_equal(ix,expected_ix)
        for n in plan['lengths']:
            use=pool['lengths'][ix]==n; value=gram(pf['hidden'][use],teacher[ix][use]);wanted=rec['pool_geometry'][str(n)]
            np.testing.assert_allclose(value,wanted,atol=2e-12,rtol=2e-12);errors.append(abs(value-wanted))
        for name in ['correct_cka','matched_correct_cka','matched_wrong_cka','answer_residual_correct_cka','answer_residual_wrong_cka']:
            np.testing.assert_allclose(rec['mean'][name],np.mean([s[name]for s in rec['geometry']]),atol=1e-15)
    return {'independent_gram_scores':len(errors),'maximum_gram_covariance_error':max(errors)}


def summary_check(plan,root,sig):
    summary=json.loads((root/'summary.json').read_text()); rows=json.loads((root/'geometry.json').read_text())['records']; errors=[]
    for endpoint in ['final','selected']:
        for contrast,metrics in summary['contrasts'][endpoint].items():
            a,b=contrast.split('-')
            for metric,s in metrics.items():
                values=[]
                for rep in sig['replicates']:
                    left=next(r for r in rows if r['replicate']==rep['id'] and r['condition']==a and r['endpoint']==endpoint)
                    right=next(r for r in rows if r['replicate']==rep['id'] and r['condition']==b and r['endpoint']==endpoint)
                    if metric=='accuracy_pp':values.append(100*(left['accuracy']-right['accuracy']))
                    else:
                        name='correct_cka' if metric=='cka' else 'answer_residual_contrast';values.append(left['mean'][name]-right['mean'][name])
                np.testing.assert_allclose(values,s['paired_deltas'],atol=1e-14);np.testing.assert_allclose(np.mean(values),s['mean'],atol=1e-14)
                clusters=[np.mean([values[i] for i,r in enumerate(sig['replicates']) if r['source_seed']==seed]) for seed in sorted({r['source_seed']for r in sig['replicates']})]
                for sample,wanted in [(values,s['conditional_95']),(clusters,s['teacher_cluster_95'])]:
                    # Independent recursive sum construction, not producer tuple enumeration.
                    distribution=np.array([0.])
                    for _ in sample:distribution=(distribution[:,None]+np.asarray(sample)[None,:]).ravel()
                    actual=np.quantile(distribution/len(sample),[.025,.975]);np.testing.assert_allclose(actual,wanted,atol=1e-12);errors.extend(np.abs(actual-np.asarray(wanted)))
    return {'bootstrap_bounds_independently_checked':len(errors),'maximum_bootstrap_bound_error':float(max(errors))}


def run():
    plan,root,sig=initialize();device=configure();_,tokens,_,TrainConfig,factory=api(plan);old=Path(plan['previous_study'])
    marker=json.loads((root/'test_opened.json').read_text());assert marker['all_36_fits_complete'] and len(marker['candidate_sha256'])==108
    assert json.loads((root/'training/state.json').read_text())['completed_utc']<marker['opened_utc']
    analysis=json.loads((root/'analysis_protocol.json').read_text());assert analysis['registered_utc']<marker['opened_utc'];assert analysis['code_sha256']==sha('experiments/inverse_coverage_analysis.py')
    dc=data_check(plan,root,sig);sc=schedules_check(plan,root,sig);test=dict(np.load(root/'dataset/test/dataset.npz'));sample=np.concatenate([np.flatnonzero(test['lengths']==n)[::128] for n in plan['lengths']]);errors=[];validations=0;endpoints=0
    for source in sig['sources']:
        assert sha(source['checkpoint'])==source['checkpoint_sha256'];cp=torch.load(source['checkpoint'],weights_only=True,map_location='cpu');cfg=TrainConfig.from_value(cp['config']);model=factory(cfg);model.load_state_dict(cp['model']);del cp;model.to(device)
        out=direct(model,test,plan['source_task'],tokens,1,sample);cached=dict(np.load(root/'teacher'/f"test_s{source['seed']}.npz"))
        for branch in ['hidden','logits']:
            expected=cached[branch][sample];np.testing.assert_allclose(out[branch],expected,atol=3e-4,rtol=3e-4);errors.append(float(np.max(np.abs(out[branch]-expected))))
        for rep in sig['replicates']:
            if rep['source_seed']!=source['seed']:continue
            pool=dict(np.load(root/'dataset'/rep['id']/'dataset.npz'));ix=np.concatenate([np.flatnonzero(pool['lengths']==n)[::256] for n in plan['lengths']]);out=direct(model,pool,plan['source_task'],tokens,1,ix);cached=dict(np.load(root/'teacher'/f"{rep['id']}.npz"))
            for branch in ['hidden','logits']:
                np.testing.assert_allclose(out[branch],cached[branch][ix],atol=3e-4,rtol=3e-4);errors.append(float(np.max(np.abs(out[branch]-cached[branch][ix]))))
        del model
    for rep in sig['replicates']:
        source=next(s for s in sig['sources']if s['seed']==rep['source_seed']);cp=torch.load(source['checkpoint'],weights_only=True,map_location='cpu');cfg=TrainConfig.from_value(cp['config']);del cp
        support=dict(np.load(old/'dataset'/rep['id']/'dataset.npz'));val=np.flatnonzero(support['split']==1)
        for condition in plan['conditions']:
            name=rep['id']+'_'+condition;record=json.loads((root/'training'/f'{name}.json').read_text());assert record['status']=='complete' and record['updated_utc']<marker['opened_utc']
            assert [g['step']for g in record['curve']]==[400,800,1200];selected=min(record['curve'],key=lambda g:(-g['accuracy'],g['cross_entropy'],g['step']));assert selected==record['selected']
            evaluation=json.loads((root/'evaluations'/f'{name}.json').read_text());archive=root/'evaluations'/f'{name}.npz';assert sha(archive)==evaluation['archive_sha256'];cached=dict(np.load(archive))
            for grade in record['curve']:
                step=grade['step'];path=root/'checkpoints'/f'{name}_u{step}.pt';assert sha(path)==record['candidate_sha256'][str(step)]==marker['candidate_sha256'][path.stem]
                cp=torch.load(path,weights_only=True,map_location='cpu');assert cp['step']==step;model=factory(cfg);model.load_state_dict(cp['model']);del cp;model.to(device)
                v=direct(model,support,plan['target_task'],tokens,0,val)['logits'];accuracy=float((v.argmax(-1)==support['labels'][val]).mean());ce=float(F.cross_entropy(torch.tensor(v),torch.tensor(support['labels'][val])))
                assert accuracy==grade['accuracy'];np.testing.assert_allclose(ce,grade['cross_entropy'],atol=5e-5,rtol=5e-5);validations+=1
                for endpoint,epstep in [('final',1200),('selected',selected['step'])]:
                    if step!=epstep:continue
                    out=direct(model,test,plan['target_task'],tokens,0,sample)
                    for branch in ['hidden','logits']:
                        expected=cached[endpoint+'_'+branch][sample];np.testing.assert_allclose(out[branch],expected,atol=3e-4,rtol=3e-4);errors.append(float(np.max(np.abs(out[branch]-expected))))
                    answers=cached[endpoint+'_logits'].argmax(-1);hit=answers==test['labels'];assert float(hit.mean())==evaluation['results'][endpoint]['accuracy']
                    for n in plan['lengths']:assert float(hit[test['lengths']==n].mean())==evaluation['results'][endpoint]['per_length'][str(n)]
                    np.testing.assert_array_equal(out['logits'].argmax(-1),answers[sample]);endpoints+=1
                    pf=np.load(root/'evaluations'/f'{name}_{endpoint}_pool.npz');pool=dict(np.load(root/'dataset'/rep['id']/'dataset.npz'));ix=pf['pool_indices'][::64];po=direct(model,pool,plan['target_task'],tokens,0,ix)['hidden']
                    np.testing.assert_allclose(po,pf['hidden'][::64],atol=3e-4,rtol=3e-4);errors.append(float(np.max(np.abs(po-pf['hidden'][::64]))))
                del model
        print({'replicate_verified':rep['id'],'validation_checkpoints':validations,'test_endpoints':endpoints},flush=True)
    gc=geometry_check(plan,root,sig,test);bc=summary_check(plan,root,sig)
    atomic_json(root/'verification.json',{'status':'passed','completed_utc':now(),'data':dc,'budget_controls':sc,
        'accuracy_endpoints_checked':endpoints,'validation_candidates_replayed':validations,'maximum_original_model_replay_error':max(errors),
        'source_checkpoint_hashes_verified':3,'all_training_finished_before_test_opening':True,'analysis_registered_before_test':True,**gc,**bc})


if __name__=='__main__':run()
