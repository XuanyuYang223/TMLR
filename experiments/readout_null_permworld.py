"""Jointly learn known generators; preserve the completed frozen assay."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import two_step_relation_factorial as core
from .hidden_relation_train import AffineOperators, query_hidden
from .longrun_engine import atomic_json, atomic_torch
from .native_confirmation import setup
from .permworld_combinations import sha

from .relation_error_localization import row_projection
import os
from .longrun_engine import atomic_json as original_atomic_json
CONFIG=Path('configs/readout_null_permworld.json')
FLAGS={'full_correct':(True,True),'full_wrong':(False,False),'null_correct':(True,True),'null_wrong':(False,False)}
# Separate workers never share a temporary progress filename.
def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def initialize():
    core.CONFIG=CONFIG;core.FLAGS=FLAGS
    return core.initialize()

def joint_losses(model, ops, x, lengths, labels, teacher, ids, pc, pi, task, tokens, temperature, scale):
    # Always forward five states. Pair permutations preserve exact marginals.
    inputs = torch.stack([x[ids, 0], x[ids, 1], x[ids, 2], x[pc, 1], x[pi, 2]], dim=1)
    h = query_hidden(model, inputs.reshape(-1, inputs.shape[-1]), lengths[ids].repeat_interleave(5), task, tokens)
    h = h.float().reshape(len(ids), 5, -1)
    weights = model.lm_head.weight[:31].float()
    bias = torch.zeros(31, device=h.device) if model.lm_head.bias is None else model.lm_head.bias[:31].float()
    native = h[:, :3] @ weights.T + bias
    ce = F.cross_entropy(native.reshape(-1, 31), labels[ids].reshape(-1))
    native_kd = kd(native.reshape(-1, 31), teacher[ids].reshape(-1, 31), temperature)
    geometry, output = [], []
    for index, action, wrong_action in [(0, 1, 3), (1, 2, 4)]:
        forward = ops(h[:, 0], index)
        backward = ops(h[:, action], index)
        reverse_geometric = ops(h[:, wrong_action], index)
        errors=[forward-h[:,wrong_action],reverse_geometric-h[:,0]]
        with torch.autocast(device_type=h.device.type,enabled=False):
            geometry.extend([(e.float()@ops.geometry_projector if ops.geometry_space=='null' else e.float()).square().mean() for e in errors])
        output.extend([kd(forward @ weights.T + bias, teacher[ids, action], temperature),
                       kd(backward @ weights.T + bias, teacher[ids, 0], temperature)])
    return ce, native_kd, torch.stack(output).mean(), torch.stack(geometry).mean() / max(ops.geometry_scale, 1e-6)

@torch.no_grad()
def validate(model, ops, data, parent, tokens):
    model.eval()
    h = core.hidden_features(model, data, parent['source_task'], tokens)
    device = next(model.parameters()).device
    ht = torch.as_tensor(h, device=device)
    weights = model.lm_head.weight[:31].float()
    bias = torch.zeros(31, device=device) if model.lm_head.bias is None else model.lm_head.bias[:31].float()
    native = (ht @ weights.T + bias).argmax(-1).cpu().numpy()
    generators = [(ops(ht[:, 0], j) @ weights.T + bias).argmax(-1).cpu().numpy() for j in range(2)]
    return {'native_e_C_I_accuracy': (native == data['labels']).mean(0).tolist(),
            'generator_C_I_accuracy': [float((prediction == data['labels'][:, j + 1]).mean()) for j, prediction in enumerate(generators)]}

def train_one(plan, parent, config, root, sig, index, condition, tokens, device):
    name = f'n{index}_hidden_{condition}'; record = root / 'fits' / f'{name}.json'
    if record.exists() and json.loads(record.read_text())['status'] == 'complete':
        return
    raw = dict(np.load(root / 'dataset' / f'n{index}' / 'support' / 'dataset.npz'))
    fit = raw['split'] == 0
    validation = {k: raw[k][~fit] for k in ['input', 'lengths', 'labels']}
    pair = dict(np.load(root / 'dataset' / f'n{index}' / 'support' / 'pairings.npz'))
    teacher_cache = np.load(root / 'features' / f'n{index}_support.npz')
    scale = float(teacher_cache['train_hidden'][pair['eligible']].var(axis=(0, 1)).mean())
    source = sig['sources'][index % 3]
    model = core.source_model(source, parent, config, device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in model.lm_head.parameters():
        parameter.requires_grad_(False)
    readout_before = sha256(model.lm_head.weight.detach().cpu().numpy().tobytes()).hexdigest()
    ops = AffineOperators(parent['architecture']['d_model']).to(device)
    w=model.lm_head.weight[:31].detach().cpu().numpy().astype(float)
    q=np.eye(w.shape[1])-row_projection(w)
    ops.geometry_projector=torch.as_tensor(q,device=device,dtype=torch.float32)
    ops.geometry_space='null' if condition.startswith('null_') else 'full'
    if ops.geometry_space=='null':
        scale=float((teacher_cache['train_hidden'][pair['eligible']]@q).var(axis=(0,1)).mean())
    ops.geometry_scale=scale
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if p.requires_grad], 'lr': plan['source_learning_rate'], 'weight_decay': .01},
        {'params': list(ops.parameters()), 'lr': plan['operator_learning_rate'], 'weight_decay': plan['weight_decay']},
    ])
    tensors = {k: torch.as_tensor(raw[k][fit], device=device) for k in ['input', 'lengths', 'labels']}
    teacher = torch.as_tensor(teacher_cache['train_output'], device=device)
    schedule = list(core.epoch_schedule(raw['lengths'][fit], pair['eligible'], plan['epochs'], plan['batch_size'], plan['support_seeds'][index] + 8001))
    correct_c, correct_i = core.FLAGS[condition]
    partners = [np.arange(int(fit.sum())) if correct_c else pair['pc'],
                np.arange(int(fit.sum())) if correct_i else pair['pi']]
    checkpoint = root / 'maps' / f'{name}_resume.pt'
    start_epoch = 0; curve = []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model']); ops.load_state_dict(state['operators']); optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']; curve = state['curve']
    digest = sha256(); exposures = np.zeros(int(fit.sum()), dtype=np.int64)
    started = time.monotonic()
    for step, (epoch, ids) in enumerate(schedule, 1):
        digest.update(np.asarray([epoch], dtype=np.int64).tobytes() + ids.tobytes()); exposures[ids] += 1
        if epoch <= start_epoch:
            continue
        ids = torch.as_tensor(ids, device=device)
        pc, pi = [torch.as_tensor(p[ids.cpu().numpy()], device=device) for p in partners]
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            ce, native_kd, operator_kd, geometry = joint_losses(model, ops, tensors['input'], tensors['lengths'], tensors['labels'],
                teacher, ids, pc, pi, parent['source_task'], tokens, plan['distillation_temperature'], scale)
            loss = plan['native_ce_weight'] * ce + plan['native_distillation_weight'] * native_kd
            loss = loss + plan['operator_distillation_weight'] * operator_kd + plan['geometry_weight'] * geometry
        loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad] + list(ops.parameters()), 1.)
        ratio = min(1., step / plan['warmup_updates'])
        cosine = (1 + math.cos(math.pi * step / len(schedule))) / 2
        factor = ratio * (plan['minimum_learning_rate_ratio'] + (1 - plan['minimum_learning_rate_ratio']) * cosine)
        optimizer.param_groups[0]['lr'] = plan['source_learning_rate'] * factor
        optimizer.param_groups[1]['lr'] = plan['operator_learning_rate'] * factor
        optimizer.step()
        boundary = step == len(schedule) or schedule[step][0] != epoch
        if boundary and epoch % 5 == 0:
            grade = validate(model, ops, validation, parent, tokens)
            curve.append({'epoch': epoch, 'step': step, **grade, 'native_ce': float(ce.detach().cpu()),
                          'native_kd': float(native_kd.detach().cpu()), 'operator_kd': float(operator_kd.detach().cpu()),
                          'geometry_loss': float(geometry.detach().cpu())})
            state = {'model': model.state_dict(), 'operators': ops.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'curve': curve}
            atomic_torch(checkpoint, state)
            if epoch in plan['checkpoint_epochs']:
                atomic_torch(root / 'maps' / f'{name}_e{epoch}.pt', {'model': model.state_dict(), 'operators': ops.state_dict(), 'epoch': epoch})
            atomic_json(root / 'current_job.json', {'fit': name, 'epoch': epoch, 'epochs': plan['epochs'], **grade,
                        'elapsed_seconds': time.monotonic() - started, 'updated_utc': core.now()})
            print(json.dumps({'fit': name, 'epoch': epoch, **grade}), flush=True)
    assert np.all(exposures[pair['eligible']] == plan['epochs'])
    assert readout_before == sha256(model.lm_head.weight.detach().cpu().numpy().tobytes()).hexdigest()
    width = parent['architecture']['d_model']; offsets = ops.offset.detach().cpu().numpy().astype(float)
    bias = np.zeros(31) if model.lm_head.bias is None else model.lm_head.bias[:31].detach().cpu().numpy()
    file = root / 'maps' / f'{name}.npz'
    np.savez_compressed(file, rho_c=np.eye(width) + offsets[0], rho_i=np.eye(width) + offsets[1],
                        bias_c=ops.bias[0].detach().cpu().numpy(), bias_i=ops.bias[1].detach().cpu().numpy(),
                        readout_weight=model.lm_head.weight[:31].detach().cpu().numpy(), readout_bias=bias,
                        mean_lengths=np.asarray(plan['lengths']), mean_vectors=np.zeros((len(plan['lengths']), width)))
    final = root / 'maps' / f'{name}_e{plan["epochs"]}.pt'
    atomic_json(record, {'status': 'complete', 'replicate': f'n{index}', 'source_seed': source['seed'], 'view': 'hidden',
                'condition': condition, 'epochs': plan['epochs'], 'updates': len(schedule), 'completed_utc': core.now(),
                'anchor_exposures': int(exposures.sum()), 'schedule_sha256': digest.hexdigest(), 'map_sha256': sha(file),
                'checkpoint_sha256': sha(final), 'readout_parameter_sha256': readout_before,
                'source_initialization_sha256': source['sha256'], 'curve': curve,
                'geometry_space':ops.geometry_space,'normalization_scale':scale,'readout_rank':int(round(w.shape[1]-np.trace(q)))})
    del model, ops, optimizer


@torch.no_grad()
def evaluate(plan,parent,config,root,sig,tokens):
    files=[root/'fits'/f'n{i}_hidden_{c}.json' for i in range(3) for c in FLAGS]
    assert all(p.exists() for p in files)
    atomic_json(root/'test_opened.json',{'opened_utc':core.now(),'all12_fixed_budget_fits_complete':True,'fit_record_sha256':{str(p):sha(p) for p in files}})
    data=dict(np.load(root/'dataset/test/dataset.npz'));rows=[]
    for i in range(3):
        for c in FLAGS:
            name=f'n{i}_hidden_{c}';source=sig['sources'][i]
            model=core.source_model(source,parent,config,'cuda')
            state=torch.load(root/'maps'/f'{name}_e20.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['model'])
            h=core.hidden_features(model,data,parent['source_task'],tokens).astype(float);m=dict(np.load(root/'maps'/f'{name}.npz'))
            pred=core.apply_numpy(h[:,0],'ci',m);forced=core.apply_numpy(h[:,1],'i',m)
            w=m['readout_weight'];bw=m['readout_bias'];truth=data['labels'][:,5];ans=(pred@w.T+bw).argmax(1);hit=ans==truth
            np.savez_compressed(root/'evaluations'/f'n{i}_{c}.npz',native=h.astype(np.float32),ab_pred=pred,ab_answers=ans,
                rho_a=m['rho_c'],rho_b=m['rho_i'],bias_a=m['bias_c'],bias_b=m['bias_i'],readout_weight=w,readout_bias=bw)
            for sid,split in [(0,'iid'),(1,'collisions')]:
                use=data['split']==sid;pairs=data['pair_ids'][use]
                ck=[core.gram_cka(pred[np.flatnonzero(use & (data['lengths']==n))[:128]],h[np.flatnonzero(use & (data['lengths']==n))[:128],5]) for n in plan['lengths']]
                rows.append({'replicate':i,'source':i,'condition':c,'split':split,'word':'ci','accuracy':float(hit[use].mean()),
                    'pair_both_correct':float(np.mean([hit[use][pairs==p].all() for p in np.unique(pairs)])) if sid else None,
                    'prediction_nmse':float(np.square(pred[use]-h[use,5]).sum()/np.square(h[use,5]-h[use,0]).sum()),
                    'cka':float(np.mean(ck)),
                    'true_intermediate_accuracy':float(((forced@w.T+bw).argmax(1)[use]==truth[use]).mean()),
                    'direct_native_accuracy':float(((h[:,5]@w.T+bw).argmax(1)[use]==truth[use]).mean())})
    atomic_json(root/'evaluation_records.json',{'status':'complete','completed_utc':core.now(),'records':rows})
    atomic_json(root/'completion.json',{'status':'trained_and_evaluated','completed_utc':core.now(),'fits':12,'source_models':3})

def run():
    prereg=Path(json.loads(CONFIG.read_text())['output']);prereg.mkdir(parents=True,exist_ok=True)
    signature={'code_sha256':sha(__file__),'config_sha256':sha(CONFIG),'common_config_sha256':sha('configs/readout_null_confirmation.json')}
    pp=prereg/'direction_protocol.json'
    if pp.exists():assert json.loads(pp.read_text())['signature']==signature
    else:atomic_json(pp,{'registered_utc':core.now(),'new_compound_outcomes_observed':False,'signature':signature})
    source_done=Path(json.loads(CONFIG.read_text())['parent'])/'completion.json'
    started=time.monotonic()
    while not source_done.exists():
        if time.monotonic()-started>7200:raise TimeoutError('Ordinary source training did not complete')
        if 'Traceback' in Path('results/readout_null_sources.log').read_text():raise RuntimeError('Source process failed')
        time.sleep(10)
    plan,parent,config,root,sig=initialize();_,_,tokens,_=setup(config);torch.set_num_threads(2)
    core.prepare_data(plan,parent,config,root,sig);core.cache_supports(plan,parent,config,root,sig)
    for i in range(3):
        for c in FLAGS:train_one(plan,parent,config,root,sig,i,c,tokens,'cuda')
    evaluate(plan,parent,config,root,sig,tokens)

if __name__=='__main__':run()
