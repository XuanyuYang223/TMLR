"""Cache-only complementary metrics for the 50 original Property32 positions."""
from collections import defaultdict
import hashlib
import time
import numpy as np
import torch
from inventory import ROOT,WS,csv_read,csv_write,save,sha
from evaluate import verify
from metrics import center,scores,cka
from experiments.permutation_audit import transform,TRANSFORMS


def run():
    verify();start=time.monotonic();parent=WS/'diagnostics/permworld_cka_20261009'
    data=dict(np.load(parent/'k_probe_inputs.npz'));ns=data['lengths'];perms=data['permutations']
    keys=[min(transform(tuple(map(int,p[:n])),t) for t in TRANSFORMS) for p,n in zip(perms,ns)]
    split=np.array([int(hashlib.sha256(repr(k).encode()).hexdigest(),16)%2 for k in keys])
    assert not ({k for k,s in zip(keys,split) if s==0}&{k for k,s in zip(keys,split) if s==1})
    jobs=[r for r in csv_read(parent/'model_inventory.csv') if r['family']=='k_series']
    init={s:np.load(parent/f'k_initial_reference_seed{s}.npy') for s in [17,42,101]}
    groups=defaultdict(dict)
    for j in jobs:groups[j['replicate'],int(j['k'])][j['pool']]=j
    known={(r['replicate'],int(r['k'])):r for r in csv_read(parent/'k_pooled_control.csv')}
    out=[];summaries=[];checks=[];shared=[];splitrows=[]
    for n in range(2,31):
        splitrows.append(dict(length=n,fit_anchors=int(((ns==n)&(split==0)).sum()),test_anchors=int(((ns==n)&(split==1)).sum()),
            fit_unique_orbits=len({k for k,s,l in zip(keys,split,ns) if s==0 and l==n}),
            test_unique_orbits=len({k for k,s,l in zip(keys,split,ns) if s==1 and l==n})))
    for (rep,k),pools in groups.items():
        a=pools['a'];b=pools['b'];seed=int(a['seed']);assert seed==int(b['seed'])
        assert a['initial_weight_hash_rebuilt']==b['initial_weight_hash_rebuilt']
        x=torch.load(a['activation_cache'],map_location='cpu',weights_only=False)['layers']['final_norm'].numpy()
        y=torch.load(b['activation_cache'],map_location='cpu',weights_only=False)['layers']['final_norm'].numpy();z=init[seed]
        shared.append(dict(replicate=rep,k=k,seed=seed,rebuilt_initial_weights_identical=True,original_initial_identity_verified=False,
            rebuilt_initial_parameter_hash=a['initial_weight_hash_rebuilt'],initialization_grade='C'))
        block=[]
        for comparison,u,v in [('between_pools',x,y),('pool_a_own_initial',x,z),('pool_b_own_initial',y,z)]:
            # Pooled contrast reuses all original anchors, with no fit operation.
            val=cka(u,v);old=known[rep,k][{'between_pools':'intermodel_cka','pool_a_own_initial':'pool_a_self_initial_cka','pool_b_own_initial':'pool_b_self_initial_cka'}[comparison]]
            checks.append(abs(val-float(old)));assert checks[-1]<1e-10
            block.append(dict(domain='Property32_k',cohort='property32_k',replicate=rep,seed=seed,k=k,
                comparison=comparison,metric='linear_cka',length='pooled',fit_anchors=0,test_anchors=len(ns),
                score=val,raw_error=None,initialization_grade='C',scope='Pooled-length contrast; not main length-stratified estimate'))
            for n in range(2,31):
                fi=np.flatnonzero((ns==n)&(split==0));ti=np.flatnonzero((ns==n)&(split==1))
                values,rank=scores(center(u[fi]),center(v[fi]),center(u[ti]),center(v[ti]),ids=ti,duplicate_keys=perms[ti])
                for metric,(value,error) in values.items():
                    block.append(dict(domain='Property32_k',cohort='property32_k',replicate=rep,seed=seed,k=k,
                        comparison=comparison,metric=metric,length=n,fit_anchors=len(fi),test_anchors=len(ti),
                        score=value,raw_error=error,procrustes_rank=rank if metric=='procrustes' else '',initialization_grade='C',
                        scope='Orbit-split cached inputs; R3/R4 are fixedseed subset sensitivity'))
        out.extend(block)
        for comparison in ['between_pools','pool_a_own_initial','pool_b_own_initial']:
            for metric in ['linear_cka','debiased_cka','knn_overlap','procrustes']:
                cells=[r for r in block if r['comparison']==comparison and r['metric']==metric and r['length']!='pooled']
                good=[r['score'] for r in cells if r['score'] is not None]
                summaries.append(dict(replicate=rep,seed=seed,k=k,comparison=comparison,metric=metric,
                    defined_lengths=len(good),planned_lengths=29,fixed_weight='1/29;complete29 required',
                    fixed_length_mean=float(np.mean(good)) if len(good)==29 else None,
                    repetition_type='source_seed' if rep in ['r0','r1','r2'] else 'task_subset_same_seed17'))
        print({'k_cached_pair_complete':rep,'k':k},flush=True)
    csv_write('k_metric_results.csv',out);csv_write('k_metric_summary.csv',summaries)
    csv_write('k_orbit_split_audit.csv',splitrows);csv_write('k_shared_initialization_audit.csv',shared)
    save('k_evaluation_audit.json',{'model_positions':50,'pool_pairs':25,'cached_source_forward':0,
        'fit_test_orbit_overlap':0,'cached_input_sha256':sha(parent/'k_probe_inputs.npz'),
        'max_original_pooled_cka_error':max(checks),'original_pooled_cells_recomputed':len(checks),
        'elapsed_seconds':time.monotonic()-start,'new_source_training':0,
        'degenerate_lengths':'No adaptive alternative lengths/ranks/metrics when undefined.'})


if __name__=='__main__':run()
