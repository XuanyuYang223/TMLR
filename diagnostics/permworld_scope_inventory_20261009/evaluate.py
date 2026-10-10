"""Existing-checkpoint forward/cache evaluation; no optimizer or source updates."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from inventory import ROOT,WS,NR,sha,js,save,csv_read,csv_write
from prepare_eval import ACTIONS
from metrics import center,scores,cka
from experiments.longrun_attention import accelerate
from neurips_permutations.passage import TOKEN_TO_ID,VOCABULARY

BRANCHES=('prefix_final_norm','query_final_norm','output_prob')


def verify():
    f=js(ROOT/'protocol_freeze.json')
    for key,path in [('metric_protocol_sha256','metric_protocol.json'),('metrics_code_sha256','metrics.py'),
                     ('relation_code_sha256','relations.py'),('evaluation_inputs_sha256','evaluation_inputs.npz')]:
        assert sha(ROOT/path)==f[key],path


def weight_hash(state):
    import hashlib
    h=hashlib.sha256()
    for name,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy()
        for text in [name,str(a.dtype),str(a.shape)]:h.update(text.encode())
        h.update(a.tobytes())
    return h.hexdigest()


@torch.inference_mode()
def forward(model,inputs,ns,task,device):
    model=accelerate(model).to(device).eval();out={k:[] for k in BRANCHES}
    for start in range(0,len(ns),256):
        n=torch.tensor(ns[start:start+256],device=device,dtype=torch.long)
        x=torch.tensor(inputs[start:start+256],device=device,dtype=torch.long)
        stop=int((2*n+6).max());ids=x[:,:stop].clone()
        if ids.shape[1]<stop:
            extra=torch.full((len(ids),stop-ids.shape[1]),TOKEN_TO_ID['<PAD>'],device=device,dtype=torch.long)
            ids=torch.cat([ids,extra],1)
        ar=torch.arange(len(ids),device=device);end=2*n+3;query=end+2
        ids[ar,end+1]=TOKEN_TO_ID[f'<{task.upper()}>'];ids[ar,query]=TOKEN_TO_ID['=']
        h,valid=model._embed_inputs(ids,ids.ne(TOKEN_TO_ID['<PAD>']))
        for block in model.blocks:h=block(h,valid)
        h=model.final_norm(h)
        out['prefix_final_norm'].append(h[ar,end].cpu().numpy())
        out['query_final_norm'].append(h[ar,query].cpu().numpy())
        out['output_prob'].append(model.lm_head(h[ar,query])[:,:31].softmax(-1).cpu().numpy())
    return {k:np.concatenate(v) for k,v in out.items()}


def make_source(row,condition):
    cp=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
    if row['cohort']=='specialist16':
        from neurips_permutations.training import TrainConfig,_default_model_factory
        cfg=TrainConfig.from_value(cp['config']);torch.manual_seed(int(row['seed']))
        model=_default_model_factory(cfg)
    else:
        from experiments.permworld_combinations import new_model
        cfg=js(WS/'configs/permworld_combinations.json');arch=json.loads(row['architecture'])
        cfg.update({k:arch[k] for k in ['d_model','layers','heads']})
        model=new_model(cfg,int(row['seed']),'cpu')
    if condition=='trained':model.load_state_dict(cp['model'],strict=True)
    return model,weight_hash(model.state_dict())


def name(row,condition):return f'{row["cohort"]}_{row["tasks"]}_s{row["seed"]}_{condition}'


def extract_sources():
    verify();data=dict(np.load(ROOT/'evaluation_inputs.npz'));pairs=js(ROOT/'evaluation_pairs.json')
    inventory=csv_read(ROOT/'training_inventory.csv')
    needed={(p['cohort'],p[f'{s}_id']) for p in pairs for s in ['left','right']}
    rows=[r for r in inventory if (r['cohort'],r['logical_position']) in needed]
    cache=ROOT/'features';cache.mkdir(exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();forward_sequences=0;reused_sequences=0;facts=[]
    for i,row in enumerate(rows):
        assert sha(row['checkpoint'])==row['checkpoint_sha256']
        for condition in ['initial','trained']:
            target=cache/f'{name(row,condition)}.npz';meta=target.with_suffix('.json')
            if target.exists():
                record=js(meta);assert sha(target)==record['archive_sha256']
                assert record['data_sha256']==sha(ROOT/'evaluation_inputs.npz')
                facts.append(record);continue
            model,ph=make_source(row,condition)
            dims={'prefix_final_norm':model.d_model if hasattr(model,'d_model') else model.token_embedding.embedding_dim,
                  'query_final_norm':model.token_embedding.embedding_dim,'output_prob':31}
            arrays={k:np.zeros((len(data['lengths']),4,d),np.float32) for k,d in dims.items()}
            need=np.ones((len(data['lengths']),4),bool);old_cache='';reuse_count=0
            if row['cohort']=='specialist16':
                parent=WS/'results/specialist_cka_controls/features'
                label='matched_init' if condition=='initial' else 'trained'
                oldpath=parent/f'{label}_{row["tasks"]}_s{row["seed"]}.npz';om=js(oldpath.with_suffix('.json'))
                assert sha(oldpath)==om['archive_sha256'];old_cache=str(oldpath)
                with np.load(oldpath) as a:
                    use=data['old_indices']>=0;indices=data['old_indices'][use]
                    for j,action in enumerate(a['actions']):
                        for k in BRANCHES:arrays[k][use,int(action)]=a[k][indices,j]
                        need[use,int(action)]=False
                reuse_count=int((~need).sum())
            selected=np.flatnonzero(need.ravel());flat=data['input'].reshape(-1,64)
            result=forward(model,flat[selected],np.repeat(data['lengths'],4)[selected],row['tasks'],device)
            for k in BRANCHES:arrays[k].reshape(-1,dims[k])[selected]=result[k]
            np.savez_compressed(target,**arrays)
            rec={'archive_sha256':sha(target),'data_sha256':sha(ROOT/'evaluation_inputs.npz'),
                'checkpoint_sha256':row['checkpoint_sha256'],'parameter_sha256':ph,'condition':condition,
                'cohort':row['cohort'],'logical_position':row['logical_position'],'task':row['tasks'],'seed':int(row['seed']),
                'initialization_grade':'C','original_initial_weight_identity_verified':False,
                'reused_cache':old_cache,'reused_sequences':reuse_count,'forward_sequences':len(selected),
                'landmarks':'prefix_final_norm before task; query_final_norm separate auxiliary',
                'formal_or_development_source_training':0}
            save(str(meta.relative_to(ROOT)),rec);facts.append(rec)
            forward_sequences+=len(selected);reused_sequences+=reuse_count
            del model,result,arrays;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
        print({'source_features_complete':i+1,'sources':len(rows),'cohort':row['cohort'],'task':row['tasks'],'seed':row['seed']},flush=True)
    save('source_forward_audit.json',{'sources':len(rows),'feature_conditions':len(facts),'forward_sequences_this_run':forward_sequences,
        'reused_sequences_this_run':reused_sequences,'elapsed_seconds':time.monotonic()-started,'records':facts,'new_source_training':0})
    # Only selected states were loaded. Inventory rows elsewhere retain an explicit missing-state-hash status.
    for r in inventory:
        for f in facts:
            if (r['cohort'],r['logical_position'])==(f['cohort'],f['logical_position']) and f['condition']=='trained':
                r.update(parameter_sha256=f['parameter_sha256'],parameter_hash_status='Computed from selected existing trained state_dict',
                    initialization_grade='C',initialization_reason='Exact documented constructor/vocabulary/manual_seed path rebuilt; not verified original initial weights.',
                    cache=str(cache/f'{name(r,"trained")}.npz'))
    csv_write('training_inventory.csv',inventory)


def subtract(a,b):return float(a-b) if a is not None and b is not None else None


def decomposed(values):
    a=values['trained'];b=values.get('initial',{})
    out={}
    for condition,vs in [('trained',a),('initial',b)]:
        for role in ['correct','wrong','identity']:
            value,error=vs.get(role,(None,None));out[f'{condition}_{role}']=value;out[f'{condition}_{role}_raw_error']=error
        out[f'{condition}_preference']=subtract(out[f'{condition}_correct'],out[f'{condition}_wrong'])
    out['B0']=out['initial_preference'];out['Bt']=out['trained_preference']
    out['delta_correct']=subtract(out['trained_correct'],out['initial_correct'])
    out['delta_wrong']=subtract(out['trained_wrong'],out['initial_wrong'])
    out['delta']=subtract(out['Bt'],out['B0'])
    if out['delta'] is not None:assert abs(out['delta']-(out['delta_correct']-out['delta_wrong']))<1e-10
    return out


def comparison_rows(metadata,fit,test,features,labels,duplicate_keys=None):
    """All inputs already relation-ordered as identity/correct/wrong per side."""
    rows=[];nfit=len(fit);ntest=len(test)
    joint=np.column_stack([labels['left'],labels['right']])
    for control in ['within_length','answer_strata']:
        selections={};keys={}
        for split,ids in [('fit',fit),('test',test)]:
            k=joint[ids];_,inv,count=np.unique(k,axis=0,return_inverse=True,return_counts=True)
            use=ids[count[inv]>=3] if control=='answer_strata' else ids
            selections[split]=use;keys[split]=joint[use] if control=='answer_strata' else None
        fi=selections['fit'];ti=selections['test'];collected={}
        ranks={}
        for branch,conditions in features.items():
            collected[branch]={m:{} for m in ['linear_cka','debiased_cka','knn_overlap','procrustes']}
            for condition,sides in conditions.items():
                for role,ai in [('identity',0),('correct',1),('wrong',2)]:
                    xf=center(sides['left'][fi],keys['fit']);yf=center(sides['right'][fi,ai],keys['fit'])
                    xt=center(sides['left'][ti],keys['test']);yt=center(sides['right'][ti,ai],keys['test'])
                    v,r=scores(xf,yf,xt,yt,ids=ti,duplicate_keys=duplicate_keys[ti] if duplicate_keys is not None else None)
                    ranks[branch]=r
                    for m,value in v.items():collected[branch][m].setdefault(condition,{})[role]=value
            for metric,values in collected[branch].items():
                rows.append(dict(**metadata,branch=branch,control=control,metric=metric,fit_anchors=len(fi),test_anchors=len(ti),
                    planned_fit_anchors=nfit,planned_test_anchors=ntest,procrustes_rank=ranks[branch] if metric=='procrustes' else '',
                    **decomposed(values)))
    return rows


def evaluate_permutations():
    verify();data=dict(np.load(ROOT/'evaluation_inputs.npz'));tasks=list(data['tasks'])
    pairs=js(ROOT/'evaluation_pairs.json');inventory=csv_read(ROOT/'training_inventory.csv')
    lookup={(r['cohort'],r['logical_position']):r for r in inventory};rows=[];began=time.monotonic()
    original=csv_read(WS/'diagnostics/permworld_cka_20261009/relation_length_cka.csv')
    known={(r['relation'],int(r['left_seed']),int(r['length']),r['control']):r for r in original
        if r['left_seed']==r['right_seed'] and r['metric']=='ordinary'};checks=[]
    for pi,pair in enumerate(pairs):
        left=lookup[pair['cohort'],pair['left_id']];right=lookup[pair['cohort'],pair['right_id']]
        action=[0,ACTIONS.index(pair['correct']),ACTIONS.index(pair['wrong'])]
        arrays={};features={k:{} for k in BRANCHES}
        for cond in ['initial','trained']:
            a=dict(np.load(ROOT/'features'/f'{name(left,cond)}.npz'))
            b=a if left['logical_position']==right['logical_position'] else dict(np.load(ROOT/'features'/f'{name(right,cond)}.npz'))
            for k in BRANCHES:features[k][cond]={'left':a[k][:,0],'right':b[k][:,action]}
        yl=data['labels'][:,0,tasks.index(pair['left'])];yr=data['labels'][:,action,tasks.index(pair['right'])]
        for kind in ['answer_onehot','answer_scalar']:
            goldleft=np.eye(31)[yl] if kind=='answer_onehot' else yl[:,None]
            goldright=np.eye(31)[yr] if kind=='answer_onehot' else yr[:,:,None]
            features[kind]={'trained':{'left':goldleft,'right':goldright}}
        for n in range(10,31):
            fit=np.flatnonzero((data['lengths']==n)&(data['split']==0));test=np.flatnonzero((data['lengths']==n)&(data['split']==1))
            meta=dict(domain='PermWorld',cohort=pair['cohort'],relation=pair['id'],family=pair['family'],kind=pair['kind'],
                left_task=pair['left'],right_task=pair['right'],source_unit=f'seed{pair["seed"]}',seed=pair['seed'],world='one_fixed_cohort_world',
                length=n,correct_transform=pair['correct'],wrong_transform=pair['wrong'],initialization_grade='C',
                original_initial_weight_identity_verified=False,interpretation='exploratory within-cohort matched seed; seed is not new data world')
            newrows=comparison_rows(meta,fit,test,features,{'left':yl,'right':yr})
            if pair['cohort']=='specialist16' and pair['origin']=='original8':
                for row in newrows:
                    if row['metric']!='linear_cka' or row['branch']!='prefix_final_norm':continue
                    old=known[pair['id'],pair['seed'],n,'length' if row['control']=='within_length' else 'answer_strata']
                    errors=[abs(row[f]-float(old[g])) for f,g in [('trained_correct','trained_correct'),('trained_wrong','trained_wrong'),
                        ('initial_correct','init_correct'),('initial_wrong','init_wrong'),('delta','delta')]]
                    assert max(errors)<5e-10,('cached CKA mismatch',pair,n,max(errors))
                    checks.append(max(errors))
            rows.extend(newrows)
        del features,arrays;a=None;b=None;gc.collect()
        print({'metric_pair_complete':pi+1,'pairs':len(pairs),'cohort':pair['cohort'],'relation':pair['id'],'seed':pair['seed']},flush=True)
    csv_write('permutation_metric_results.csv',rows)
    save('permutation_evaluation_audit.json',{'rows':len(rows),'relation_source_pairs':len(pairs),
        'known_original_per_length_cka_cells_recomputed':len(checks),'maximum_known_cka_error':max(checks),
        'elapsed_seconds':time.monotonic()-began,'new_source_training':0})


def evaluate_field():
    verify();from experiments.models import SourceModel
    from experiments.algebra import world
    from experiments.field_symmetry import group_sources
    root=WS/'results/field_symmetry';meta=js(root/'metadata.json');cfg=meta['config']
    assert sha(WS/'experiments/models.py')==meta['model_sha256']
    assert sha(WS/'experiments/algebra.py')==meta['algebra_sha256']
    torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32=False;rows=[];checks=[];start=time.monotonic()
    for w in cfg['world_seeds']:
        physical,_,encoded,basis=world(5,4,w);z=physical@basis.T%5;lookup={tuple(v):i for i,v in enumerate(z)}
        order=np.random.default_rng(w+40000).permutation(4)
        fi=np.flatnonzero(np.isin(z[:,0],[0,1]));ti=np.flatnonzero(np.isin(z[:,0],[2,3,4]))
        for seed in cfg['model_seeds']:
            h={};probs={}
            for group in cfg['groups']:
                record=js(root/f'{group}_w{w}_m{seed}.json');cp=root/'checkpoints'/f'{group}_w{w}_m{seed}.pt'
                assert record['status']=='complete' and sha(cp)==record['checkpoint_sha256']
                for condition in ['initial','trained']:
                    torch.manual_seed(seed);model=SourceModel(20,cfg['hidden'],cfg['features'],5).to(device).eval()
                    if condition=='trained':model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
                    with torch.inference_mode():
                        xx=torch.tensor(encoded,device=device);actual=model.encoder(xx).cpu().numpy()
                        probabilities=model(xx).softmax(-1).flatten(1).cpu().numpy()
                    original=np.load(root/f'{group}_w{w}_m{seed}_step{6000 if condition=="trained" else 0}_features.npy')
                    error=float(np.max(np.abs(actual-original)));assert error<2e-5
                    checks.append(error);h[group,condition]=original;probs[group,condition]=probabilities
                    del model
            for k in [1,2,3,4]:
                group=f'M{k}';nextk=1+k%4
                # Exact inverse of Gk: t3=z3;tk4=k^-1*(z4-z3).
                def indices(coefficient):
                    changed=z.copy();changed[:,3]=pow(coefficient,-1,5)*(z[:,3]-z[:,2])%5
                    ids=np.array([lookup[tuple(v)] for v in changed]);assert np.array_equal(z[ids,0],z[:,0])
                    return ids
                correct=indices(k);wrong=indices(nextk);indices3=np.column_stack([np.arange(625),correct,wrong])
                yl=z[:,order];gr=group_sources(group,5)
                yr=(z[indices3]@gr.T%5)[:,:,order]
                assert np.array_equal(yl,yr[:,1]);assert not np.array_equal(yl,yr[:,2])
                features={}
                for branch,container in [('encoder_final',h),('output_prob',probs)]:
                    features[branch]={c:{'left':container['P',c],'right':container[group,c][indices3]} for c in ['initial','trained']}
                features['answer_onehot']={'trained':{'left':np.eye(5)[yl].reshape(625,20),
                    'right':np.eye(5)[yr].reshape(625,3,20)}}
                features['answer_scalar']={'trained':{'left':yl,'right':yr}}
                # Flatten all four visible answers per action into the common exact answer key.
                meta_row=dict(domain='F5',cohort='F5_rank4_symmetry45',relation=f'P_M{k}_basis_inverse',family='F5_coordinate_basis',kind='cross_task_set',
                    left_task='P four coordinates',right_task=group+' four linear coordinates',source_unit=f'w{w}_s{seed}',seed=seed,world=w,
                    length=4,correct_transform=f'inverse_G{k}',wrong_transform=f'inverse_G{nextk}',
                    initialization_grade='B_encoder_C_initial_output',original_initial_weight_identity_verified=False,
                    interpretation='All625 source inputs seen; fit/test only analysis-map orbit holdout; physical basis/world is not an independent input universe')
                newrows=comparison_rows(meta_row,fi,ti,features,{'left':yl,'right':yr.reshape(625,12)})
                assert all(r['test_anchors']==0 for r in newrows if r['control']=='answer_strata')
                rows.extend(newrows)
            print({'field_world_complete':w,'seed':seed},flush=True)
    csv_write('field_metric_results.csv',rows)
    save('field_evaluation_audit.json',{'source_checkpoints':45,'comparisons':36,'rows':len(rows),
        'original_step0_encoder_snapshots_reused':45,'saved_feature_matches':len(checks),'max_saved_feature_recompute_error':max(checks),
        'source_inputs_seen':625,'analysis_fit':250,'analysis_test':375,'fit_test_orbit_overlap':0,
        'exact_answer_control_undefined':'Full source answer tuple uniquely identifies each input; no repeated answer strata.',
        'elapsed_seconds':time.monotonic()-start,'new_source_training':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['features','permutation-metrics','field-metrics'])
    a=p.parse_args()
    {'features':extract_sources,'permutation-metrics':evaluate_permutations,'field-metrics':evaluate_field}[a.stage]()
