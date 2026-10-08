"""Independent inputs, masks, original-forward replays and Gram verification."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.nn import functional as F

from .inverse_functional_alignment import configure, now
from .inverse_functional_report import paired_interval
from .inverse_functional_verify import direct, gram, inverse, key
from .longrun_engine import atomic_json
from .overnight_inverse_functional import initialize
from .permworld_combinations import sha
from .specialist_cka_controls import api, old_inputs


def register(root):
    file=root/'verification_protocol.json';signature={'code_sha256':sha(__file__),'main_protocol_sha256':sha(root/'protocol.json'),
        'checks':['Independently reconstruct all eight orbit states and token prefixes; rescan original16M using regex.',
                  'Check all old artifacts and frozen checkpoint hashes.',
                  'Reconstruct paired schedules and training/pseudo-answer preserving mismatch masks.',
                  'Replay original unaccelerated full LM forward and final_norm hook on test subsets and all64 validation rows.',
                  'Recompute accuracies, subgroup counts/decomposition, length modes with Counter and CKA using Gram matrices.']}
    if file.exists():assert json.loads(file.read_text())['signature']==signature;return
    atomic_json(file,{'registered_utc':now(),'signature':signature,'main_test_opened':(root/'test_opened.json').exists()})


def preserve(plan,root,sig):
    checks=0
    for file,digest in sig['prior_artifacts_sha256'].items():assert sha(file)==digest,file;checks+=1
    assert sha(Path(plan['previous_study'])/'completion.json')==sig['prior_completion_sha256']
    for entry in sig['sources']+sig['targets']:assert sha(entry['checkpoint'])==entry['checkpoint_sha256']
    for name,digest in sig['upstream_sha256'].items():assert sha(Path(plan['upstream_python'])/'neurips_permutations'/name)==digest
    return {'previous_artifacts_preserved':checks,'frozen_checkpoints_preserved':6,'original_code_preserved':True}


def data_check(plan,root,sig):
    audit=json.loads((root/'dataset/audit.json').read_text());_,tokens,_,_,_=api(plan);seen=set();examples=0;unlabeled=0
    for entry in audit['cohorts']:
        file=Path(entry['path']);assert sha(file)==entry['sha256'];d=dict(np.load(file));labeled=entry['oracle_labels_present'];assert ('labels'in d)==labeled
        if not labeled:unlabeled+=len(d['lengths'])
        if '/support'in entry['cohort']:assert int((d['split']==0).sum())==192 and int((d['split']==1).sum())==64
        for i,n in enumerate(d['lengths']):
            p=d['permutations'][i,0,:n];assert sorted(p)==list(range(1,n+1));c=n+1-p
            orbit=[p,c,p[::-1],c[::-1],inverse(p),inverse(c),inverse(p[::-1]),inverse(c[::-1])]
            for j,q in enumerate(orbit):
                assert np.array_equal(q,d['permutations'][i,j,:n]);assert not d['permutations'][i,j,n:].any()
                kk=key(q);assert kk not in seen;seen.add(kk)
            for action,q in enumerate([p,orbit[4]]):
                row=d['input'][i,action];assert list(row[:4])==[tokens['<BOS>'],tokens['<SIZE>'],tokens[f'{n:02d}'],tokens['<ONE_START>']]
                assert row[2*n+3]==tokens['<ONE_END>'] and np.all(row[2*n+4:]==tokens['<PAD>'])
                assert np.array_equal(row[4:4+2*n:2],[tokens[f'{v:02d}'] for v in q])
            # Oracle answers are checked only in labeled archives, never U.
            if labeled:assert int(np.sum(orbit[4][:-1]>orbit[4][1:]))==int(d['labels'][i])
            examples+=1
        print({'verified_new_cohort':entry['cohort'],'examples':examples},flush=True)
    for entry in sig['excluded_archives']:assert sha(entry['path'])==entry['sha256']
    prior=old_inputs(sig['excluded_archives']);assert not seen.intersection(prior)
    manifest=Path(plan['repository'])/'data/permutation-properties-16m-v1/manifest.json';assert sha(manifest)==audit['original_manifest_sha256'];parent=json.loads(manifest.read_text())
    pattern=re.compile(rb'"primary"\s*:\s*(\[[^]]+\])');scanned=0
    for j,shard in enumerate(parent['shards']):
        file=manifest.parent/shard['filename'];assert sha(file)==shard['sha256']
        with gzip.open(file,'rb') as handle:
            for line in handle:
                match=pattern.search(line);assert match is not None;assert match[1].replace(b' ',b'')not in seen;scanned+=1
        if j%50==0:print({'independent_parent_rescan':scanned},flush=True)
    assert scanned==16000000
    return {'new_examples':examples,'all_orbit_states_checked':len(seen),'original_inputs_rescanned':scanned,
            'prior_local_input_keys_checked':len(prior),'unlabeled_examples_without_oracle':unlabeled,'unlabeled_oracle_answers_computed':False}


def check_confirmation(plan,root,sig):
    folder=root/'confirmation';data=np.load(folder/'dataset/test/dataset.npz');y=data['labels'];n=data['lengths']
    summary=json.loads((folder/'diagnostic/summary.json').read_text());modes=json.loads((Path(plan['previous_study'])/'length_priors/summary.json').read_text())['records'];checks=0
    previous=json.loads((Path(plan['previous_study'])/'protocol.json').read_text())['signature']
    for row in summary['records']:
        rep=next(r for r in previous['replicates'] if r['id']==row['replicate'])
        mapping=next(r['modes']for r in modes if r['replicate']==row['replicate'] and r['condition']==row['prior'])
        p=folder/'teacher'/f"test_s{rep['source_seed']}.npz" if row['condition']=='teacher' else folder/'evaluations'/f"{row['replicate']}_{row['condition']}.npz"
        pred=np.load(p)['logits'if row['condition']=='teacher'else'final_logits'].argmax(-1)
        hits=[int(a)==int(b)for a,b in zip(pred,y)];assert sum(hits)/len(hits)==row['accuracy'];checks+=1
        for group,flag in [('modal',True),('nonmodal',False)]:
            values=[h for h,nn,yy in zip(hits,n,y)if (mapping[str(int(nn))]==int(yy))==flag]
            assert len(values)==row[group]['count'] and sum(values)/len(values)==row[group]['accuracy'];checks+=1
    return {'confirmation_accuracy_and_subgroup_checks':checks,'new_test_examples':len(y),'frozen_target_models':36}


def replay(plan,root,sig):
    device=configure();_,tokens,_,TrainConfig,factory=api(plan);data=dict(np.load(root/'dataset/test/dataset.npz'))
    ix=np.concatenate([np.flatnonzero(data['lengths']==n)[::128]for n in plan['lengths']]);errors=[];validations=0;endpoints=0
    for source in sig['sources']:
        cp=torch.load(source['checkpoint'],map_location='cpu',weights_only=True);model=factory(TrainConfig.from_value(cp['config']));model.load_state_dict(cp['model']);del cp;model.to(device)
        out=direct(model,data,plan['source_task'],tokens,1,ix);saved=np.load(root/'teacher'/f"test_s{source['seed']}.npz")
        for branch in ['hidden','logits']:
            np.testing.assert_allclose(out[branch],saved[branch][ix],atol=3e-4,rtol=3e-4);errors.append(float(np.abs(out[branch]-saved[branch][ix]).max()))
        del model
    for rep in plan['replicates']:
        target=next(s for s in sig['targets']if s['seed']==rep['target_pretrain_seed']);support=dict(np.load(root/'dataset'/rep['id']/'support/dataset.npz'));valix=np.flatnonzero(support['split']==1)
        for condition in plan['conditions']:
            name=rep['id']+'_'+condition;record=json.loads((root/'training'/f'{name}.json').read_text());assert record['status']=='complete'and record['step']==plan['updates']
            assert record['initialization_sha256']==target['checkpoint_sha256'];assert record['schedule_sha256']==sha(root/'training'/f"{rep['id']}_schedule.npz")
            assert record['forward_rows']==40000*64
            for step in plan['checkpoint_updates']:
                file=root/'checkpoints'/f'{name}_u{step}.pt';assert sha(file)==record['checkpoint_sha256'][str(step)]
                cp=torch.load(target['checkpoint'],map_location='cpu',weights_only=True);model=factory(TrainConfig.from_value(cp['config']));del cp
                cp=torch.load(file,map_location='cpu',weights_only=True);model.load_state_dict(cp['model']);del cp;model.to(device)
                cached=np.load(root/'evaluations'/f'{name}_u{step}.npz');out=direct(model,data,plan['target_task'],tokens,0,ix)
                for branch in ['hidden','logits']:
                    np.testing.assert_allclose(out[branch],cached[branch][ix],atol=3e-4,rtol=3e-4);errors.append(float(np.abs(out[branch]-cached[branch][ix]).max()))
                out=direct(model,support,plan['target_task'],tokens,0,valix);hit=out['logits'].argmax(-1)==support['labels'][valix]
                grade=next(g for g in record['curve']if g['step']==step);assert float(hit.mean())==grade['accuracy']
                ce=float(F.cross_entropy(torch.tensor(out['logits']),torch.tensor(support['labels'][valix])))
                np.testing.assert_allclose(ce,grade['cross_entropy'],atol=3e-4,rtol=3e-4);validations+=1;endpoints+=1;del model
            print({'independently_replayed':name},flush=True)
    return {'original_forward_endpoints_replayed':endpoints,'validation_checkpoints_replayed':validations,'maximum_cached_feature_difference':max(errors)}


def summaries(plan,root,sig):
    summary=json.loads((root/'summary.json').read_text());data=np.load(root/'dataset/test/dataset.npz');n=data['lengths'];y=data['labels'];modes=json.loads((root/'length_modes.json').read_text())['records'];checks=0;grams=0
    for row in modes:
        rep=row['replicate'];support=np.load(root/'dataset'/rep/'support/dataset.npz');pool=np.load(root/'dataset'/rep/'unlabeled/dataset.npz')
        nn=pool['lengths']if row['prior']=='teacher'else support['lengths'][support['split']==0]
        answers=np.load(root/'teacher'/f'{rep}_unlabeled.npz')['predicted_answer']if row['prior']=='teacher'else support['labels'][support['split']==0]
        for length in plan['lengths']:
            counted=Counter(map(int,answers[nn==length]));mode=max(counted,key=lambda v:(counted[v],-v));assert mode==row['modes'][str(length)];checks+=1
    for row in summary['records']:
        rep=next(r for r in plan['replicates']if r['id']==row['replicate']);file=root/'evaluations'/(f"{rep['id']}_initial.npz"if row['condition']=='initial'else f"{rep['id']}_{row['condition']}_u{row['step']}.npz")
        assert sha(file)==row['archive_sha256'];saved=np.load(file);teacher=np.load(root/'teacher'/f"test_s{rep['source_seed']}.npz")
        hit=saved['logits'].argmax(-1)==y;assert float(hit.mean())==row['accuracy'];checks+=1
        mapping=next(r['modes']for r in modes if r['replicate']==rep['id']and r['prior']=='teacher');modal=np.array([mapping[str(int(a))]==int(b)for a,b in zip(n,y)])
        for group,mask in [('modal',modal),('nonmodal',~modal)]:assert int(mask.sum())==row[group]['count']and float(hit[mask].mean())==row[group]['accuracy'];checks+=1
        for branch,records in [('hidden',row['per_length_geometry']),('logits',row['per_length_output_geometry'])]:
            for cell in records:
                ix=np.flatnonzero(n==cell['length'])[:128];a=saved[branch][ix];b=teacher[branch][ix];labels=y[ix];keep=np.array([(labels==v).sum()>=3for v in labels]);a0=a[keep];b0=b[keep];lab=labels[keep]
                ar=a0.astype(float).copy();br=b0.astype(float).copy();perm=np.arange(len(ar))
                for v in set(lab):
                    j=np.flatnonzero(lab==v);ar[j]-=ar[j].mean(0);br[j]-=br[j].mean(0);perm[j]=np.r_[j[-1],j[:-1]]
                expected={'correct_cka':gram(a,b),'matched_correct_cka':gram(a0,b0),'matched_wrong_cka':gram(a0,b0[perm]),'answer_residual_correct_cka':gram(ar,br),'answer_residual_wrong_cka':gram(ar,br[perm])}
                for k,value in expected.items():np.testing.assert_allclose(value,cell[k],atol=2e-12,rtol=2e-12);grams+=1
    return {'independent_accuracy_subgroup_and_mode_checks':checks,'independent_gram_scores':grams}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','data','confirmation','main']);args=parser.parse_args();plan,root,sig=initialize();register(root)
    if args.phase=='data':atomic_json(root/'data_verification.json',{'completed_utc':now(),**data_check(plan,root,sig),**preserve(plan,root,sig)})
    elif args.phase=='confirmation':atomic_json(root/'confirmation/verification.json',{'completed_utc':now(),**check_confirmation(plan,root,sig)})
    elif args.phase=='main':atomic_json(root/'verification.json',{'completed_utc':now(),**summaries(plan,root,sig),**replay(plan,root,sig),**preserve(plan,root,sig)})
