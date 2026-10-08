"""Fresh cross-specialist CKA with common input/answer controls.

Protocol and source hashes are saved before data generation or inference.
The original CKA landmark is ONE_END, before any task prompt is present.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys
import tomllib

import numpy as np
import torch

from .longrun_engine import atomic_json
from .longrun_attention import accelerate
from .permworld_combinations import sha
from .permutation_audit import transform, TRANSFORMS
from .analysis import write_csv

CONFIG = Path('configs/specialist_cka_controls.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def api(plan):
    sys.path.insert(0, str(Path(plan['upstream_python']).resolve()))
    from neurips_permutations.math_ops import PROPERTY_FUNCTIONS
    from neurips_permutations.passage import TOKEN_TO_ID, one_line_tokens
    from neurips_permutations.training import TrainConfig, _default_model_factory
    return PROPERTY_FUNCTIONS, TOKEN_TO_ID, one_line_tokens, TrainConfig, _default_model_factory


def center_groups(x, keys):
    x = np.asarray(x, dtype=np.float64)
    _, inv = np.unique(np.asarray(keys), axis=0, return_inverse=True)
    counts = np.bincount(inv)
    sums = np.zeros((len(counts), x.shape[1]))
    np.add.at(sums, inv, x)
    return x - (sums / counts[:, None])[inv]


def cka(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xc, yc = x-x.mean(0), y-y.mean(0)
    # Constant-answer strata have precisely zero variance. Do not label them CKA=0.
    if np.square(xc).sum() <= 1e-20 or np.square(yc).sum() <= 1e-20:
        return None
    numerator = np.square(xc.T @ yc).sum()
    denominator = np.sqrt(np.square(xc.T @ xc).sum()*np.square(yc.T @ yc).sum())
    return float(numerator/denominator)


def common_strata(n, left, right_all, minimum):
    keys = np.column_stack([n, left, right_all])
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    keep = counts[inv] >= minimum
    return keys, keep, {'anchors': len(n), 'retained': int(keep.sum()),
        'strata': int(np.sum(counts >= minimum)), 'discarded_singleton_or_small_strata': int(np.sum(counts < minimum))}


def register():
    plan = json.loads(CONFIG.read_text())
    root, repo = Path(plan['output']), Path(plan['repository'])
    root.mkdir(parents=True, exist_ok=True)
    for sub in ['features', 'evaluations']:
        (root/sub).mkdir(exist_ok=True)
    config = tomllib.loads((repo/'configs/property_task_geometry.toml').read_text())
    relations = config['relation_pairs']
    tasks = sorted({r[k] for r in relations for k in ['left','right']})
    reuse = {(r['task'], r['seed']): r for r in config['checkpoint_reuse']}
    sources = []
    for task in tasks:
        for seed in plan['model_seeds']:
            output = reuse.get((task,seed), {}).get('output_dir',
                f'runs/property-task-geometry/specialists/geometry-specialist-{task}-transformer-seed{seed}')
            marker = repo/output/'completed.json'
            rec = json.loads(marker.read_text())
            cp = repo/rec['checkpoint']
            assert cp.is_file() and sha(cp)==rec['checkpoint_sha256']
            assert rec['tasks']==[task] and rec['seed']==seed and rec['global_step']==20000
            sources.append({'task': task, 'seed': seed, 'checkpoint': str(cp),
                'checkpoint_sha256': rec['checkpoint_sha256'], 'marker': str(marker), 'marker_sha256': sha(marker)})
    # Input-bearing archives only; do not read outcome scores to choose this protocol.
    names = {'data.npz', 'probe_dataset.npz', 'source_data.npz', 'training_orbit_audit.npz', 'dataset.npz'}
    excludes = sorted({str(p) for p in Path('results').rglob('*.npz') if p.name in names and root not in p.parents})
    seen_hashes, unique = set(), []
    for p in excludes:
        h=sha(p)
        if h not in seen_hashes:
            unique.append({'path':p, 'sha256':h}); seen_hashes.add(h)
    signature = {'plan': plan, 'code_sha256': sha(__file__), 'config_sha256': sha(CONFIG),
        'upstream_geometry_config_sha256': sha(repo/'configs/property_task_geometry.toml'),
        'upstream_code_sha256': {p:sha(Path(plan['upstream_python'])/'neurips_permutations'/p) for p in ['models.py','training.py','passage.py','math_ops.py']},
        'local_dependencies_sha256': {p:sha(p) for p in ['experiments/longrun_attention.py','experiments/permutation_audit.py']},
        'sources':sources, 'relations':relations, 'excluded_archives':unique,
        'original_parent_manifest_sha256':sha(repo/'data/permutation-properties-16m-v1/manifest.json'),
        'prior_completed_study_sha256':sha('results/output_information_followup/completion.json')}
    path=root/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())['signature']==signature
    else:
        assert not (root/'dataset.npz').exists() and not list((root/'features').glob('*'))
        atomic_json(path, {'registered_utc':now(), 'new_cka_scores':0, 'signature':signature})
    return plan,root,signature


def input_key(p):
    return b'['+b','.join(str(int(v)).encode() for v in p)+b']'


def original_input_key(line):
    start=line.index(b'[',line.index(b'"primary"'))
    stop=line.index(b']',start)+1
    return line[start:stop].replace(b' ',b'')


def old_inputs(archives):
    seen=set()
    for entry in archives:
        with np.load(entry['path']) as a:
            if 'permutations' in a.files and 'lengths' in a.files:
                for row,n in zip(a['permutations'],a['lengths']):
                    for p in np.asarray(row).reshape(-1,row.shape[-1]):
                        seen.add(input_key(p[:int(n)]))
            for key in a.files:
                if key=='input' or key.endswith('_input'):
                    nk='lengths' if key=='input' else key[:-6]+'_lengths'
                    if nk not in a.files: continue
                    for row,n in zip(a[key],a[nk]):
                        for x in np.asarray(row).reshape(-1,row.shape[-1]):
                            seen.add(input_key(x[4:4+2*int(n):2]))
    return seen


def create_data(plan,root,sig):
    target=root/'dataset.npz'
    if target.exists(): return
    functions,tokens,one_line,_,_=api(plan)
    seen=old_inputs(sig['excluded_archives'])
    rng=np.random.default_rng(plan['data_seed'])
    candidates, member_to_anchor, rejected = {}, {}, set()
    for n in range(plan['lengths'][0],plan['lengths'][1]+1):
        candidates[n]=[]
        while len(candidates[n])<plan['candidates_per_length']:
            p=tuple(map(int,rng.permutation(n)+1)); orbit=[transform(p,t) for t in TRANSFORMS]
            keys=[input_key(x) for x in orbit]
            if len(set(keys))!=8 or any(k in seen or k in member_to_anchor for k in keys): continue
            anchor=(n,len(candidates[n])); candidates[n].append(p)
            for k in keys: member_to_anchor[k]=anchor
    manifest=Path(plan['repository'])/'data/permutation-properties-16m-v1/manifest.json'
    original=json.loads(manifest.read_text())
    scanned=0
    # Scan every original parent input, including prior validation/probes, but ignore answers.
    for j,shard in enumerate(original['shards']):
        path=manifest.parent/shard['filename']
        assert sha(path)==shard['sha256']
        with gzip.open(path,'rb') as handle:
            for line in handle:
                anchor=member_to_anchor.get(original_input_key(line))
                if anchor is not None: rejected.add(anchor)
                scanned+=1
        if j%20==0: print(json.dumps({'scanned_original_inputs':scanned,'rejected_candidate_orbits':len(rejected)}),flush=True)
    tasks=sorted({r[k] for r in sig['relations'] for k in ['left','right']})
    rows,perms,ns,splits,labels=[],[],[],[],[]
    for n,values in candidates.items():
        available=[p for j,p in enumerate(values) if (n,j) not in rejected]
        needed=sum(plan['examples_per_length'].values())
        assert len(available)>=needed,(n,len(available),needed)
        offset=0
        for split,(_,count) in enumerate(plan['examples_per_length'].items()):
            for p in available[offset:offset+count]:
                orbit=[transform(p,t) for t in plan['transforms']]
                encoded=[]
                for value in orbit:
                    ids=[tokens['<BOS>'],tokens['<SIZE>'],tokens[f'{n:02d}']]+[tokens[t] for t in one_line(value)]
                    encoded.append(ids+[tokens['<PAD>']]*(64-len(ids)))
                rows.append(encoded);perms.append([list(v)+[0]*(30-n) for v in orbit])
                labels.append([[functions[t](v) for t in tasks] for v in orbit]);ns.append(n);splits.append(split)
            offset+=count
    np.savez_compressed(target,input=np.asarray(rows),permutations=np.asarray(perms),lengths=np.asarray(ns),
        split=np.asarray(splits),labels=np.asarray(labels),tasks=np.asarray(tasks),transforms=np.asarray(plan['transforms']))
    for r in sig['relations']:
        y=np.asarray(labels); ia=tasks.index(r['left']);ib=tasks.index(r['right']);a=plan['transforms'].index(r['input_transform'])
        np.testing.assert_array_equal(y[:,0,ia]+r['right_label_offset'],y[:,a,ib])
    atomic_json(root/'dataset_audit.json',{'created_utc':now(),'sha256':sha(target),'original_inputs_scanned':scanned,
        'prior_local_inputs_excluded':len(seen),'candidate_orbits':sum(map(len,candidates.values())),
        'candidate_orbits_rejected_by_original_parent':len(rejected),'orbits_selected':len(rows),
        'split_counts':dict(Counter(splits)), 'exclusion_scope':'All eight states of every selected candidate orbit were excluded against every original parent input and input-bearing local archives; only e/C/I are scored. No source retraining.'})


@torch.inference_mode()
def extract(model,inputs,n,task,tokens,batch_size):
    device=next(model.parameters()).device
    outputs={f'prefix_{name}':[] for name in ['embedding','block_01','block_02','block_03','block_04','final_norm']}
    outputs.update(query_final_norm=[],output_prob=[],confidence=[])
    flat=inputs.reshape(-1,inputs.shape[-1]);length=np.repeat(n,inputs.shape[1])
    for start in range(0,len(flat),batch_size):
        x=torch.tensor(flat[start:start+batch_size],device=device)
        ns=torch.tensor(length[start:start+batch_size],device=device)
        stop=int((2*ns+6).max().item())
        ids=torch.full((len(x),stop),tokens['<PAD>'],dtype=torch.long,device=device)
        width=min(x.shape[1],stop);ids[:,:width]=x[:,:width]
        ar=torch.arange(len(x),device=device);p=2*ns+3;q=p+2
        ids[ar,p+1]=tokens[f'<{task.upper()}>'];ids[ar,q]=tokens['=']
        h,valid=model._embed_inputs(ids,ids!=tokens['<PAD>'])
        outputs['prefix_embedding'].append(h[ar,p].float().cpu().numpy())
        for j,b in enumerate(model.blocks):
            h=b(h,valid);outputs[f'prefix_block_{j+1:02d}'].append(h[ar,p].float().cpu().numpy())
        h=model.final_norm(h)
        outputs['prefix_final_norm'].append(h[ar,p].float().cpu().numpy())
        query=h[ar,q];outputs['query_final_norm'].append(query.float().cpu().numpy())
        logits=model.lm_head(query).float()
        numeric=logits[:,:31];prob=numeric.softmax(-1)
        outputs['output_prob'].append(prob.cpu().numpy())
        top=numeric.topk(2).values
        outputs['confidence'].append(torch.stack([prob.max(-1).values,-(prob*prob.clamp_min(1e-30).log()).sum(-1),top[:,0]-top[:,1]],dim=-1).cpu().numpy())
    return {k:np.concatenate(v).reshape(len(n),inputs.shape[1],-1) for k,v in outputs.items()}


def features(plan,root,sig):
    _,tokens,_,TrainConfig,factory=api(plan)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=dict(np.load(root/'dataset.npz'));left={r['left'] for r in sig['relations']}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    jobs=[]
    for source in sig['sources']:
        jobs.append({**source,'condition':'trained','random_seed':None})
        jobs.append({**source,'condition':'matched_init','random_seed':source['seed']})
        if source['task'] not in left:
            jobs.append({**source,'condition':'independent_init','random_seed':source['seed']+plan['independent_random_seed_offset']})
    for j,job in enumerate(jobs):
        name=f"{job['condition']}_{job['task']}_s{job['seed']}";fp=root/'features'/f'{name}.npz';mp=fp.with_suffix('.json')
        if mp.exists():
            rec=json.loads(mp.read_text());assert rec['archive_sha256']==sha(fp);continue
        cp=torch.load(job['checkpoint'],map_location='cpu',weights_only=True)
        cfg=TrainConfig.from_value(cp['config'])
        torch.manual_seed(job['random_seed'] if job['random_seed'] is not None else job['seed'])
        model=factory(cfg)
        if job['condition']=='trained':model.load_state_dict(cp['model'],strict=True)
        del cp
        model=accelerate(model).to(device).eval()
        actions=[0] if job['task'] in left else [0,1,2]
        out=extract(model,data['input'][:,actions],data['lengths'],job['task'],tokens,plan['batch_size'])
        np.savez_compressed(fp,**out,actions=np.asarray(actions))
        atomic_json(mp,{'job':job,'archive_sha256':sha(fp),'dataset_sha256':sha(root/'dataset.npz'),'completed_utc':now()})
        del model
        if device=='cuda':torch.cuda.empty_cache()
        print(json.dumps({'features_complete':name,'completed':j+1,'planned':len(jobs)}),flush=True)


def confidence_residual(x,prob,confidence,labels,n,fit,test,alpha):
    conf=np.column_stack([np.eye(21)[n-10],confidence,prob[np.arange(len(n)),labels]])
    design=np.column_stack([conf,np.ones(len(n))]);a=design[fit]
    penalty=np.eye(a.shape[1])*alpha;penalty[-1,-1]=0
    w=np.linalg.solve(a.T@a+penalty,a.T@x[fit])
    return x[test]-design[test]@w


def evaluate(plan,root,sig):
    data=dict(np.load(root/'dataset.npz'));test=data['split']==2;fit=data['split']==0
    n=data['lengths'][test];tasks=list(data['tasks'])
    branches=['prefix_final_norm','query_final_norm','output_prob','answer_onehot','answer_scalar']
    details=[];contrasts=[];conditional=[]
    def load(condition,task,seed):return dict(np.load(root/'features'/f'{condition}_{task}_s{seed}.npz'))
    for relation in sig['relations']:
        pair=relation['pair_id'];la=tasks.index(relation['left']);rb=tasks.index(relation['right'])
        ya=data['labels'][:,0,la];yb=data['labels'][:,:,rb]
        keys,keep,audit=common_strata(n,ya[test],yb[test],plan['minimum_stratum_size'])
        correct=plan['transforms'].index(relation['input_transform']);wrong=1 if correct==2 else 2
        selected_keys=keys[keep]
        # Permutations are determined by true labels/data only and shared across every seed and branch.
        rng=np.random.default_rng(plan['conditional_permutation_seed']+sig['relations'].index(relation))
        _,inv=np.unique(selected_keys,axis=0,return_inverse=True)
        strata=[np.flatnonzero(inv==i) for i in np.unique(inv)]
        permutations=[]
        for _ in range(plan['conditional_permutations']):
            order=np.arange(keep.sum())
            for ids in strata:order[ids]=rng.permutation(ids)
            permutations.append(order)
        for seed in plan['model_seeds']:
            destination=root/'evaluations'/f'{pair}_s{seed}.json'
            if destination.exists():
                rec=json.loads(destination.read_text());details+=rec['scores'];contrasts+=rec['contrasts'];conditional+=rec['conditional_null'];continue
            scores=[];delta=[];null=[]
            for condition in ['trained','matched_init','independent_init']:
                a=load('matched_init' if condition=='independent_init' else condition,relation['left'],seed)
                b=load(condition,relation['right'],seed)
                names=branches+([f'prefix_{l}' for l in ['embedding','block_01','block_02','block_03','block_04']] if condition!='independent_init' else [])
                for branch in names:
                    xa= np.eye(31)[ya] if branch=='answer_onehot' else ya[:,None] if branch=='answer_scalar' else a[branch][:,0]
                    by= np.eye(31)[yb] if branch=='answer_onehot' else yb[:,:,None] if branch=='answer_scalar' else b[branch]
                    for control in ['raw','length','answer_strata','confidence']:
                        if control=='confidence' and branch not in ['prefix_final_norm','query_final_norm']:continue
                        if branch not in branches and control not in ['raw','length']:continue
                        z=[]
                        for action in range(3):
                            x=xa[test];y=by[test,action]
                            if control=='length':x=center_groups(x,n[:,None]);y=center_groups(y,n[:,None])
                            elif control=='answer_strata':x=center_groups(x[keep],selected_keys);y=center_groups(y[keep],selected_keys)
                            elif control=='confidence':
                                x=confidence_residual(xa,a['output_prob'][:,0],a['confidence'][:,0],ya,data['lengths'],fit,test,plan['confidence_ridge'])
                                y=confidence_residual(by[:,action],b['output_prob'][:,action],b['confidence'][:,action],yb[:,action],data['lengths'],fit,test,plan['confidence_ridge'])
                            value=cka(x,y);z.append(value)
                            scores.append({'relation':pair,'seed':seed,'condition':condition,'branch':branch,'control':control,
                                'transform':plan['transforms'][action],'role':'correct' if action==correct else 'wrong' if action==wrong else 'identity',
                                'cka':value,'samples':int(keep.sum()) if control=='answer_strata' else len(n)})
                        row={'relation':pair,'seed':seed,'condition':condition,'branch':branch,'control':control,
                            'correct':z[correct],'wrong':z[wrong],'identity':z[0],
                            'correct_minus_wrong':None if z[correct] is None or z[wrong] is None else z[correct]-z[wrong],
                            'correct_minus_identity':None if z[correct] is None or z[0] is None else z[correct]-z[0]}
                        delta.append(row)
                        if control=='answer_strata' and branch in ['prefix_final_norm','query_final_norm']:
                            xx=center_groups(xa[test][keep],selected_keys)
                            ys=[center_groups(by[test,action][keep],selected_keys) for action in [correct,wrong]]
                            shuffled=[cka(xx,ys[0][p])-cka(xx,ys[1][p]) for p in permutations]
                            null.append({'relation':pair,'seed':seed,'condition':condition,'branch':branch,
                                'observed_contrast':row['correct_minus_wrong'],'shuffle_mean':float(np.mean(shuffled)),
                                'shuffle_min':float(np.min(shuffled)),'shuffle_max':float(np.max(shuffled)),
                                'contrast_minus_shuffle_mean':float(row['correct_minus_wrong']-np.mean(shuffled))})
            rec={'relation':relation,'seed':seed,'stratum_audit':audit,'scores':scores,'contrasts':delta,
                'conditional_null':null,'completed_utc':now()}
            atomic_json(destination,rec);details+=scores;contrasts+=delta;conditional+=null
            print(json.dumps({'cka_complete':pair,'seed':seed,'strata':audit}),flush=True)
    write_csv(root/'scores.csv',details);write_csv(root/'contrasts.csv',contrasts);write_csv(root/'conditional_null.csv',conditional)
    # Input-only baselines, independent of network weights and labels.
    inp=data['permutations'][test].astype(np.float64);nrm=inp/n[:,None,None]
    onehot=[]
    for action in range(3):
        value=np.zeros((len(n),30,30))
        for j,nn in enumerate(n):value[j,np.arange(nn),inp[j,action,:nn].astype(int)-1]=1
        onehot.append(value.reshape(len(n),-1))
    ir=[]
    for branch,values in [('numeric_normalized',[nrm[:,a] for a in range(3)]),('permutation_matrix_onehot',onehot)]:
        for control in ['raw','length']:
            for a in range(3):
                x,y=values[0],values[a]
                if control=='length':x=center_groups(x,n[:,None]);y=center_groups(y,n[:,None])
                ir.append({'branch':branch,'control':control,'transform':plan['transforms'][a],'cka':cka(x,y)})
    write_csv(root/'input_baselines.csv',ir)
    # Accuracy is diagnostic; probabilities in CKA are normalized over the same 31 numeric tokens.
    accuracy=[]
    for source in sig['sources']:
        a=load('trained',source['task'],source['seed']);task=tasks.index(source['task'])
        for k,act in enumerate(a['actions']):
            accuracy.append({'task':source['task'],'seed':source['seed'],'transform':plan['transforms'][act],
                'test_numeric_accuracy':float(np.mean(a['output_prob'][test,k].argmax(-1)==data['labels'][test,act,task]))})
    write_csv(root/'source_accuracy.csv',accuracy)
    atomic_json(root/'state.json',{'status':'analyses_complete','relation_seed_cells':24,'score_rows':len(details),'completed_utc':now()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['register','data','features','evaluate','all'])
    args=parser.parse_args();plan,root,sig=register()
    if args.stage in ['data','all']:create_data(plan,root,sig)
    if args.stage in ['features','all']:features(plan,root,sig)
    if args.stage in ['evaluate','all']:evaluate(plan,root,sig)


if __name__=='__main__':main()
