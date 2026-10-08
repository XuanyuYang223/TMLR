"""Controlled increase of unlabeled inverse-relation supervision coverage."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .inverse_functional_alignment import configure, now, forward, extract, geometry_loss, output_loss, validation, matched_derangement
from .longrun_attention import accelerate
from .longrun_engine import atomic_json, atomic_torch
from .permworld_combinations import sha
from .permutation_audit import TRANSFORMS, transform
from .specialist_cka_controls import api, old_inputs, input_key, original_input_key

CONFIG = Path('configs/inverse_alignment_coverage.json')


def initialize():
    plan = json.loads(CONFIG.read_text()); root = Path(plan['output']); old = Path(plan['previous_study'])
    for path in [root] + [root/p for p in ['dataset', 'teacher', 'training', 'checkpoints', 'evaluations']]: path.mkdir(parents=True, exist_ok=True)
    file = root/'protocol.json'
    if file.exists():
        sig = json.loads(file.read_text())['signature']
        assert sig['plan'] == plan and sig['code_sha256'] == sha(__file__) and sig['config_sha256'] == sha(CONFIG)
        for source in sig['sources']: assert sha(source['checkpoint']) == source['checkpoint_sha256']
        for path,digest in sig['references_sha256'].items(): assert sha(path) == digest, path
        for path,digest in sig['dependencies_sha256'].items(): assert sha(path) == digest, path
        return plan, root, sig
    previous = json.loads((old/'protocol.json').read_text())['signature']; reps = previous['plan']['replicates']
    references = {}
    for rep in reps:
        for path in [old/'dataset'/rep['id']/'dataset.npz', old/'dataset'/rep['id']/'mismatch.npz', old/'teacher'/f"{rep['id']}.npz", old/'initializations'/f"{rep['id']}.pt"]: references[str(path)] = sha(path)
    references[str(old/'teacher/gate.json')] = sha(old/'teacher/gate.json')
    for source in previous['sources']: assert sha(source['checkpoint']) == source['checkpoint_sha256']
    names = {'data.npz', 'probe_dataset.npz', 'source_data.npz', 'training_orbit_audit.npz', 'dataset.npz'}
    archives = []; unique = set()
    for path in sorted(Path('results').rglob('*.npz')):
        if path.name not in names or root in path.parents: continue
        digest = sha(path)
        if digest not in unique: archives.append({'path': str(path), 'sha256': digest}); unique.add(digest)
    deps = ['experiments/inverse_functional_alignment.py','experiments/longrun_attention.py','experiments/permworld_combinations.py','experiments/permutation_audit.py','experiments/specialist_cka_controls.py']
    sig = {'plan': plan, 'replicates': reps, 'sources': previous['sources'], 'references_sha256': references,
           'code_sha256': sha(__file__), 'config_sha256': sha(CONFIG), 'excluded_archives': archives,
           'prior_completion_sha256': sha(old/'completion.json'), 'dependencies_sha256': {p:sha(p) for p in deps},
           'upstream_sha256': {p:sha(Path(plan['upstream_python'])/'neurips_permutations'/p) for p in ['models.py','training.py','math_ops.py','passage.py']}}
    assert not list((root/'training').glob('*'))
    atomic_json(file, {'registered_utc': now(), 'new_test_outcomes_observed': False, 'signature': sig})
    return plan, root, sig


def create_data(plan, root, sig):
    if (root/'dataset/audit.json').exists(): return
    functions, tokens, one_line, _, _ = api(plan); seen = old_inputs(sig['excluded_archives']); used = set(); lookup = {}; candidates = {}
    seeds = [plan['test_data_seed']] + plan['unlabeled_seeds']
    for world, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        for n, count in zip(plan['lengths'], plan['candidate_counts']):
            rows = []
            while len(rows) < count:
                p = tuple(map(int, rng.permutation(n)+1)); keys = [input_key(transform(p,t)) for t in TRANSFORMS]
                if len(set(keys)) != 8 or any(k in seen or k in used for k in keys): continue
                anchor = (world,n,len(rows)); rows.append(p)
                for k in keys: used.add(k); lookup[k] = anchor
            candidates[world,n] = rows
    manifest = Path(plan['repository'])/'data/permutation-properties-16m-v1/manifest.json'; parent = json.loads(manifest.read_text()); rejected = set(); scanned = 0
    for j, shard in enumerate(parent['shards']):
        path = manifest.parent/shard['filename']; assert sha(path) == shard['sha256']
        with gzip.open(path,'rb') as handle:
            for line in handle:
                anchor = lookup.get(original_input_key(line))
                if anchor is not None: rejected.add(anchor)
                scanned += 1
        if j%25 == 0: print({'parent_inputs_scanned':scanned,'rejected_orbits':len(rejected)},flush=True)
    records = []
    for world in range(len(seeds)):
        lengths = []; inputs = []; perms = []; labels = []
        for k,n in enumerate(plan['lengths']):
            count = plan['test_per_length'] if world == 0 else plan['unlabeled_counts'][k]
            rows = [p for j,p in enumerate(candidates[world,n]) if (world,n,j) not in rejected]; assert len(rows) >= count
            for p in rows[:count]:
                prefixes = []
                for q in [p,transform(p,'inverse')]:
                    prefix = [tokens['<BOS>'],tokens['<SIZE>'],tokens[f'{n:02d}']]+[tokens[v] for v in one_line(q)]
                    prefixes.append(prefix+[tokens['<PAD>']]*(64-len(prefix)))
                lengths.append(n); inputs.append(prefixes); perms.append([list(transform(p,t))+[0]*(30-n) for t in TRANSFORMS])
                if world == 0: labels.append(functions[plan['target_task']](p))
        name = 'test' if world == 0 else sig['replicates'][world-1]['id']; folder = root/'dataset'/name; folder.mkdir(exist_ok=True)
        arrays = {'input':np.asarray(inputs),'permutations':np.asarray(perms),'lengths':np.asarray(lengths)}
        if world == 0: arrays['labels'] = np.asarray(labels)
        assert world == 0 or 'labels' not in arrays
        file = folder/'dataset.npz'; np.savez_compressed(file,**arrays)
        records.append({'cohort':name,'examples':len(lengths),'sha256':sha(file),'oracle_labels_present':world==0})
    atomic_json(root/'dataset/audit.json', {'created_utc':now(),'original_inputs_scanned':scanned,'original_manifest_sha256':sha(manifest),
        'previous_local_inputs_excluded':len(seen),'rejected_orbits':len(rejected),'cohorts':records,'all_eight_orbit_states_excluded':True,'labeled_supports_reused':True})


def prepare_teachers(plan, root, sig):
    old = Path(plan['previous_study']); assert json.loads((old/'teacher/gate.json').read_text())['status'] == 'passed'
    device = configure(); _,tokens,_,TrainConfig,factory = api(plan)
    for source in sig['sources']:
        cp = torch.load(source['checkpoint'],map_location='cpu',weights_only=True); model = accelerate(factory(TrainConfig.from_value(cp['config']))); model.load_state_dict(cp['model']); del cp
        model.to(device).eval()
        for p in model.parameters(): p.requires_grad_(False)
        for rep,seed in zip(sig['replicates'],plan['unlabeled_seeds']):
            if rep['source_seed'] != source['seed']: continue
            file = root/'teacher'/f"{rep['id']}.npz"; data = dict(np.load(root/'dataset'/rep['id']/'dataset.npz')); assert 'labels' not in data
            if not file.exists():
                out = extract(model,data,plan['source_task'],tokens,1); out['predicted_answer'] = out['logits'].argmax(-1); np.savez_compressed(file,**out)
            out = dict(np.load(file)); partner,eligible = matched_derangement(data['lengths'],out['predicted_answer'],np.ones(len(data['lengths']),dtype=bool),seed+99001)
            paired = root/'teacher'/f"{rep['id']}_mismatch.npz"
            if not paired.exists(): np.savez_compressed(paired,partner=partner,eligible=eligible)
            print({'teacher_pool':rep['id'],'unlabeled_examples':len(eligible),'geometry_eligible':int(eligible.sum())},flush=True)
        del model
    atomic_json(root/'teacher/state.json', {'status':'ready','completed_utc':now(),'unlabeled_oracle_answers_read':False,'test_consulted':False})


def schedules(plan, rep, support, pool, seed):
    rng = np.random.default_rng(rep['data_seed']+81001); ur = np.random.default_rng(seed+81001); labeled = []; unlabeled = []
    for _ in range(plan['updates']):
        n = int(rng.choice(plan['lengths'])); a = np.flatnonzero((support['split']==0)&(support['lengths']==n)); b = np.flatnonzero(pool['lengths']==n)
        labeled.append(rng.choice(a,plan['labeled_batch'],replace=False)); unlabeled.append(ur.choice(b,plan['unlabeled_batch'],replace=False))
    return np.asarray(labeled),np.asarray(unlabeled)


def losses(logits, h, labels, correct_h, wrong_h, teacher_logits, eligible, labeled_count, condition, plan):
    ce = F.cross_entropy(logits[:labeled_count].float(),labels)
    full = geometry_loss(h[eligible],correct_h[eligible]); wrong = geometry_loss(h[eligible],wrong_h[eligible])
    use = eligible[:labeled_count]; support = geometry_loss(h[:labeled_count][use],correct_h[:labeled_count][use])
    kd = output_loss(logits,teacher_logits,plan['distillation_temperature'])
    weights = [float(condition in ['coverage_alignment','coverage_distillation_alignment']),float(condition=='coverage_matched_mismatch'),float(condition=='support_alignment')]
    loss = ce + plan['geometry_weight']*(weights[0]*full+weights[1]*wrong+weights[2]*support) + plan['distillation_weight']*float(condition in ['coverage_distillation','coverage_distillation_alignment'])*kd
    return loss, {'cross_entropy':ce,'coverage_geometry_loss':full,'mismatched_geometry_loss':wrong,'support_geometry_loss':support,'distillation_loss':kd}


def train(plan, root, sig):
    assert json.loads((root/'teacher/state.json').read_text())['status'] == 'ready'
    old = Path(plan['previous_study']); device = configure(); _,tokens,_,TrainConfig,factory = api(plan)
    for rep,seed in zip(sig['replicates'],plan['unlabeled_seeds']):
        source = next(s for s in sig['sources'] if s['seed']==rep['source_seed'])
        cp = torch.load(source['checkpoint'],map_location='cpu',weights_only=True); cfg = TrainConfig.from_value(cp['config']); del cp
        support = dict(np.load(old/'dataset'/rep['id']/'dataset.npz')); pool = dict(np.load(root/'dataset'/rep['id']/'dataset.npz')); assert 'labels' not in pool
        teacher = dict(np.load(old/'teacher'/f"{rep['id']}.npz")); unteacher = dict(np.load(root/'teacher'/f"{rep['id']}.npz"))
        matched = dict(np.load(old/'dataset'/rep['id']/'mismatch.npz')); unmatched = dict(np.load(root/'teacher'/f"{rep['id']}_mismatch.npz"))
        ls,us = schedules(plan,rep,support,pool,seed); schedule_file = root/'training'/f"{rep['id']}_schedule.npz"
        if not schedule_file.exists(): np.savez_compressed(schedule_file,labeled=ls,unlabeled=us)
        schedule_sha = sha(schedule_file); initial = old/'initializations'/f"{rep['id']}.pt"
        for condition in plan['conditions']:
            name = rep['id']+'_'+condition; file = root/'training'/f'{name}.json'; latest = root/'checkpoints'/f'{name}_latest.pt'
            if file.exists() and json.loads(file.read_text())['status']=='complete': continue
            model = accelerate(factory(cfg)); model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True)); model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
            arrays = {'x':support['input'][:,0],'n':support['lengths'],'y':support['labels'],'ux':pool['input'][:,0],'un':pool['lengths'],
                'h':teacher['hidden'],'logits':teacher['logits'],'uh':unteacher['hidden'],'ulogits':unteacher['logits'],
                'partner':matched['partner'],'upartner':unmatched['partner'],'eligible':matched['eligible'],'ueligible':unmatched['eligible']}
            a = {k:torch.tensor(v,device=device) for k,v in arrays.items()}
            torch.manual_seed(rep['target_seed']+71001)
            if device=='cuda': torch.cuda.manual_seed_all(rep['target_seed']+71001)
            start = 0; curve = []
            if latest.exists():
                cp = torch.load(latest,map_location='cpu',weights_only=True); model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['optimizer'])
                start = cp['step']; curve = cp['curve']; torch.set_rng_state(cp['torch_rng'])
                if device=='cuda': torch.cuda.set_rng_state_all(cp['cuda_rng'])
                del cp
            begun = time.monotonic()
            for step in range(start+1,plan['updates']+1):
                if step <= plan['warmup_updates']: ratio = step/plan['warmup_updates']
                else: ratio = plan['minimum_lr_ratio']+(1-plan['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*(step-plan['warmup_updates'])/(plan['updates']-plan['warmup_updates'])))
                for group in optimizer.param_groups: group['lr'] = plan['learning_rate']*ratio
                l = torch.tensor(ls[step-1],device=device); u = torch.tensor(us[step-1],device=device)
                x = torch.cat([a['x'][l],a['ux'][u]]); n = torch.cat([a['n'][l],a['un'][u]])
                right = torch.cat([a['h'][l],a['uh'][u]]); wrong = torch.cat([a['h'][a['partner'][l]],a['uh'][a['upartner'][u]]])
                teacher_logits = torch.cat([a['logits'][l],a['ulogits'][u]]); eligible = torch.cat([a['eligible'][l],a['ueligible'][u]])
                model.train(); optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'): logits,h = forward(model,x,n,plan['target_task'],tokens)
                loss,parts = losses(logits,h,a['y'][l],right,wrong,teacher_logits,eligible,plan['labeled_batch'],condition,plan)
                if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite {name} {step}')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_clip']); optimizer.step()
                if step%200 == 0: print(json.dumps({'training':name,'step':step,'elapsed_seconds':time.monotonic()-begun,**{k:float(v.detach().cpu()) for k,v in parts.items()}}),flush=True)
                if step in plan['validation_updates']:
                    grade = {'step':step,**validation(model,support,plan['target_task'],tokens,device)}; curve.append(grade)
                    candidate = root/'checkpoints'/f'{name}_u{step}.pt'; atomic_torch(candidate,{'model':model.state_dict(),'step':step,'grade':grade})
                    selected = min(curve,key=lambda g:(-g['accuracy'],g['cross_entropy'],g['step']))
                    atomic_torch(latest,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'curve':curve,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else []})
                    candidates = {str(g['step']):sha(root/'checkpoints'/f"{name}_u{g['step']}.pt") for g in curve}
                    atomic_json(file,{'status':'complete' if step==plan['updates'] else 'training','replicate':rep,'condition':condition,'curve':curve,'selected':selected,
                        'initialization_sha256':sha(initial),'schedule_sha256':schedule_sha,'labeled_schedule_bytes_sha256':hashlib.sha256(ls.tobytes()).hexdigest(),
                        'distinct_unlabeled_exposed':len(np.unique(us[:step])),'unlabeled_available':len(pool['lengths']),
                        'target_labeled_examples':192,'target_validation_examples':64,'labeled_forward_examples':step*32,'unlabeled_forward_examples':step*32,
                        'candidate_sha256':candidates,'latest_sha256':sha(latest),'updated_utc':now()})
            del model,optimizer,a
            if device=='cuda': torch.cuda.empty_cache()
    atomic_json(root/'training/state.json',{'status':'all_targets_complete','target_models':36,'completed_utc':now()})


def evaluate(plan, root, sig):
    assert json.loads((root/'training/state.json').read_text())['status']=='all_targets_complete'
    marker = root/'test_opened.json'
    if not marker.exists(): atomic_json(marker,{'opened_utc':now(),'candidate_sha256':{p.stem:sha(p) for p in (root/'checkpoints').glob('*_u*.pt')},'all_36_fits_complete':True})
    device = configure(); _,tokens,_,TrainConfig,factory = api(plan); data = dict(np.load(root/'dataset/test/dataset.npz')); old = Path(plan['previous_study'])
    for source in sig['sources']:
        file = root/'teacher'/f"test_s{source['seed']}.npz"
        if file.exists(): continue
        cp = torch.load(source['checkpoint'],map_location='cpu',weights_only=True); model = accelerate(factory(TrainConfig.from_value(cp['config']))); model.load_state_dict(cp['model']); del cp; model.to(device)
        out = extract(model,data,plan['source_task'],tokens,1); np.savez_compressed(file,**out); del model
    for rep in sig['replicates']:
        source = next(s for s in sig['sources'] if s['seed']==rep['source_seed']); cp = torch.load(source['checkpoint'],map_location='cpu',weights_only=True); cfg = TrainConfig.from_value(cp['config']); del cp
        initial = root/'evaluations'/f"{rep['id']}_initialization.npz"
        if not initial.exists():
            model = accelerate(factory(cfg)); model.load_state_dict(torch.load(old/'initializations'/f"{rep['id']}.pt",map_location='cpu',weights_only=True)); model.to(device)
            out = extract(model,data,plan['target_task'],tokens,0); np.savez_compressed(initial,**out); del model
        for condition in plan['conditions']:
            name = rep['id']+'_'+condition; file = root/'evaluations'/f'{name}.json'
            if file.exists(): continue
            record = json.loads((root/'training'/f'{name}.json').read_text()); results = {}; arrays = {}
            for endpoint,step in [('final',1200),('selected',record['selected']['step'])]:
                cp = torch.load(root/'checkpoints'/f'{name}_u{step}.pt',map_location='cpu',weights_only=True); model = accelerate(factory(cfg)); model.load_state_dict(cp['model']); del cp; model.to(device)
                out = extract(model,data,plan['target_task'],tokens,0); hit = out['logits'].argmax(-1)==data['labels']
                results[endpoint] = {'accuracy':float(hit.mean()),'per_length':{str(n):float(hit[data['lengths']==n].mean()) for n in plan['lengths']},'step':step}
                for k,v in out.items(): arrays[endpoint+'_'+k] = v
                del model
            archive = file.with_suffix('.npz'); np.savez_compressed(archive,**arrays)
            atomic_json(file,{'replicate':rep,'condition':condition,'results':results,'archive_sha256':sha(archive),'completed_utc':now()})
            print({'evaluated':name,'final_accuracy':results['final']['accuracy'],'selected_accuracy':results['selected']['accuracy']},flush=True)
    atomic_json(root/'state.json',{'status':'evaluation_complete','completed_utc':now(),'accuracy_endpoints':72})


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage',choices=['register','data','teachers','train','evaluate','all']); stage = parser.parse_args().stage
    plan,root,sig = initialize()
    if stage in ['data','all']: create_data(plan,root,sig)
    if stage in ['teachers','all']: prepare_teachers(plan,root,sig)
    if stage in ['train','all']: train(plan,root,sig)
    if stage in ['evaluate','all']: evaluate(plan,root,sig)
