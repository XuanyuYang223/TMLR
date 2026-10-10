"""Independent real-data calculations, immutable-artifact guards, completion audit."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from inventory import ROOT,WS,NR,sha,js,save,csv_read,csv_write
from evaluate import verify,weight_hash,make_source,forward
from metrics import verify as verify_toys


def independent_checks():
    # Fixed first relation/seed/length, not chosen using metric outcomes.
    left=dict(np.load(ROOT/'features/specialist16_descents_s17_trained.npz'))['prefix_final_norm']
    right=dict(np.load(ROOT/'features/specialist16_recoils_s17_trained.npz'))['prefix_final_norm']
    d=dict(np.load(ROOT/'evaluation_inputs.npz'))
    fi=np.flatnonzero((d['lengths']==10)&(d['split']==0));ti=np.flatnonzero((d['lengths']==10)&(d['split']==1))
    xf=left[fi,0].astype(float);yf=right[fi,2].astype(float)
    xt=left[ti,0].astype(float);yt=right[ti,2].astype(float)
    for a in [xf,yf,xt,yt]:a-=a.mean(0)
    c=np.square(xt.T@yt).sum()/np.sqrt(np.square(xt.T@xt).sum()*np.square(yt.T@yt).sum())
    def slow_neighbors(x):
        return [set(sorted((j for j in range(len(x)) if j!=i),key=lambda j:(float(np.square(x[i]-x[j]).sum()),int(ti[j])))[:10]) for i in range(len(x))]
    aa=slow_neighbors(xt);bb=slow_neighbors(yt);kn=float(np.mean([len(a&b)/10 for a,b in zip(aa,bb)]))
    # Torch double SVD/equations provide a separate computational path from NumPy metric implementation.
    ft=[torch.from_numpy(a) for a in [xf,yf]];tt=[torch.from_numpy(a) for a in [xt,yt]]
    zz=[];tz=[]
    for x,y in zip(ft,tt):
        _,sv,vh=torch.linalg.svd(x,full_matrices=False);basis=vh[:8].T
        z=x@basis;scale=torch.sqrt(z.square().sum()/len(z));zz.append(z/scale);tz.append(y@basis/scale)
    u,s,vh=torch.linalg.svd(zz[0].T@zz[1]);rot=u@vh
    error=float(torch.sqrt((tz[0]@rot-tz[1]).square().sum()/tz[1].square().sum()))
    records=csv_read(ROOT/'permutation_metric_results.csv')
    selected={r['metric']:r for r in records if r['cohort']=='specialist16' and r['relation']=='descent_inverse' and r['seed']=='17'
        and r['length']=='10' and r['branch']=='prefix_final_norm' and r['control']=='within_length'}
    errs={'independent_covariance_cka':abs(c-float(selected['linear_cka']['trained_correct'])),
        'independent_slow_knn':abs(kn-float(selected['knn_overlap']['trained_correct'])),
        'independent_torch_procrustes':abs(error-float(selected['procrustes']['trained_correct_raw_error']))}
    assert errs['independent_covariance_cka']<1e-10
    assert errs['independent_slow_knn']<1e-12
    assert errs['independent_torch_procrustes']<1e-8
    return errs


@torch.inference_mode()
def check_forward_and_causality():
    rows=csv_read(ROOT/'training_inventory.csv')
    row=next(r for r in rows if r['cohort']=='specialist16' and r['tasks']=='descents' and r['seed']=='17')
    data=dict(np.load(ROOT/'evaluation_inputs.npz'));cache=dict(np.load(ROOT/'features/specialist16_descents_s17_trained.npz'))
    ids=np.flatnonzero((data['lengths']==10)&(data['split']==1))[:4]
    inputs=data['input'][ids][:,[0,2,3]].reshape(-1,64);ns=np.repeat(data['lengths'][ids],3)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,ph=make_source(row,'trained');actual=forward(model,inputs,ns,'descents',device)
    errors={k:float(np.max(np.abs(actual[k]-cache[k][ids][:,[0,2,3]].reshape(actual[k].shape)))) for k in actual}
    assert max(errors.values())<2e-5,errors
    end=2*ns+3;prefix=torch.tensor(inputs[:,:int(end.max())+1],device=device,dtype=torch.long)
    from neurips_permutations.passage import TOKEN_TO_ID
    # Explicit padding mask, independent of PAD's numeric ID.
    h,valid=model._embed_inputs(prefix,prefix.ne(TOKEN_TO_ID['<PAD>']))
    for block in model.blocks:h=block(h,valid)
    h=model.final_norm(h);p=h[torch.arange(len(ns),device=device),torch.tensor(end,device=device)].cpu().numpy()
    causal=float(np.max(np.abs(p-actual['prefix_final_norm'])))
    assert causal<2e-5,causal
    return dict(recomputed_existing_and_new_actions_max_errors=errors,task_suffix_exclusion_max_error=causal,samples=12)


def run():
    verify();freeze=js(ROOT/'protocol_freeze.json')
    assert sha(ROOT/'evaluate.py')==freeze['evaluation_code_sha256']
    assert sha(ROOT/'summary_protocol.json')==freeze['summary_protocol_sha256']
    checks=independent_checks();checks['fixed_toys']=verify_toys();checks['forward_and_causality']=check_forward_and_causality()
    rows=csv_read(ROOT/'metric_results.csv');identities=0;error_checks=0
    for r in rows:
        if r['delta']:
            d=float(r['delta']);assert abs(d-(float(r['Bt'])-float(r['B0'])))<1e-10
            assert abs(d-(float(r['delta_correct'])-float(r['delta_wrong'])))<1e-10;identities+=1
        if r['metric']=='procrustes':
            for c in ['trained','initial']:
                for role in ['correct','wrong','identity']:
                    v=r[f'{c}_{role}'];e=r[f'{c}_{role}_raw_error']
                    if v:assert abs(float(v)+float(e))<1e-12;error_checks+=1
        if r['branch'].startswith('answer_'):
            assert not r['B0'] and not r['delta'] and not r['initial_correct']
            if r['control']=='answer_strata':assert not r['trained_correct']
    # Check selected immutable source weight files, not every unselected container.
    inventory=csv_read(ROOT/'training_inventory.csv')
    selected=[r for r in inventory if r['parameter_sha256'] or r['cohort'] in ['F5_rank4_symmetry45','property32_k']]
    guarded={r['checkpoint']:r['checkpoint_sha256'] for r in selected}
    for p,h in guarded.items():assert sha(p)==h,p
    # Record parameter digest of selected field states used above; do not load unselected inventory states.
    for r in inventory:
        if r['cohort']=='F5_rank4_symmetry45':
            state=torch.load(r['checkpoint'],map_location='cpu',weights_only=True)
            r['parameter_sha256']=weight_hash(state);r['parameter_hash_status']='Canonical digest of selected ordinary F5 state_dict; encoder/readout source used in current evaluation'
    csv_write('training_inventory.csv',inventory)
    changed=[];protected_count=0
    manifests=[(WS,js(WS/'diagnostics/permworld_cka_20261009/protected_before.json')['files']),
        (WS/'diagnostics/permworld_cka_20261009',js(WS/'diagnostics/permworld_cka_20261009/deliverable_manifest.json')['files']),
        (WS/'diagnostics/permworld_encoding_plan_20261009',js(WS/'diagnostics/permworld_encoding_plan_20261009/manifest.json')['files'])]
    for base,files in manifests:
        for p,h in files.items():
            path=base/p
            if not path.exists() or sha(path)!=h:changed.append(str(path))
            protected_count+=1
    assert not changed,changed
    assert not subprocess.check_output(['git','diff','--name-only'],cwd=WS,text=True).strip()
    assert not subprocess.check_output(['git','diff','--name-only'],cwd=NR,text=True).strip()
    assert not csv_read(ROOT/'additional_training_manifest.csv')
    planned=WS/'results/permworld_encoding_preflight_20261009'
    assert not list(planned.rglob('*.pt'))
    required=['training_inventory.csv','checkpoint_dedup.csv','relation_task_graph.csv','metric_protocol.json','metric_results.csv',
        'reuse_and_missing_report.txt','additional_training_manifest.csv','cost_estimate.csv','report.html']
    for p in required:assert (ROOT/p).is_file() and (ROOT/p).stat().st_size>0
    for fname in ['source_forward_audit.json','permutation_evaluation_audit.json','field_evaluation_audit.json','k_evaluation_audit.json']:
        assert js(ROOT/fname)['new_source_training']==0
    script_files=sorted(ROOT.glob('*.py'));metric_files=sorted(ROOT.glob('*.csv'))+sorted(ROOT.glob('*.json'))+sorted(ROOT.glob('*.html'))+sorted(ROOT.glob('*.txt'))
    save('audit.json',dict(status='complete_and_stopped_before_new_source_training',new_source_training=0,paused_training_resumed=False,published=False,
        repository_heads=js(ROOT/'inventory_audit.json')['repository_heads'],protocol_freeze=freeze,
        independent_real_data_metric_checks=checks,decomposition_identities_checked=identities,negative_error_score_identities_checked=error_checks,
        known_original_cka_validation=js(ROOT/'permutation_evaluation_audit.json'),F5_validation=js(ROOT/'field_evaluation_audit.json'),
        k_validation=js(ROOT/'k_evaluation_audit.json'),selected_source_files_guarded=len(guarded),
        protected_files_checked=protected_count,protected_files_changed=changed,tracked_repository_files_changed=[],
        formal_encoding_models=0,development_encoding_models=0,initialization_original_weight_identity_not_verified=True,
        script_sha256={p.name:sha(p) for p in script_files},
        deliverable_sha256={p.name:sha(p) for p in metric_files if p.name!='audit.json'},
        stop_scope='No source train/paused resume/encoding batch/F17 experiment/release.'))
    print('All checks passed; protected files unchanged:',protected_count,'independent errors:',checks,flush=True)


if __name__=='__main__':run()
