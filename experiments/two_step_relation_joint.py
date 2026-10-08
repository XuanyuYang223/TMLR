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

CONFIG = Path('configs/two_step_relation_joint.json')


def initialize():
    core.CONFIG = CONFIG
    plan, parent, config, root, sig = core.initialize()
    file = root / 'joint_protocol.json'
    signature = {'code_sha256': sha(__file__), 'base_protocol_sha256': sha(root / 'protocol.json'),
                 'source_initializations': sig['sources'], 'plan': plan}
    if file.exists():
        assert json.loads(file.read_text())['signature'] == signature
    else:
        assert not (root / 'data_manifest.json').exists()
        atomic_json(file, {'registered_utc': core.now(), 'new_outcomes_observed': False, 'signature': signature})
        (root / 'joint_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return plan, parent, config, root, sig


def kd(student, teacher, temperature):
    return F.kl_div(F.log_softmax(student / temperature, -1), F.softmax(teacher / temperature, -1),
                    reduction='batchmean') * temperature ** 2


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
        geometry.extend([(forward - h[:, wrong_action]).square().mean(),
                         (reverse_geometric - h[:, 0]).square().mean()])
        output.extend([kd(forward @ weights.T + bias, teacher[ids, action], temperature),
                       kd(backward @ weights.T + bias, teacher[ids, 0], temperature)])
    return ce, native_kd, torch.stack(output).mean(), torch.stack(geometry).mean() / max(scale, 1e-6)


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
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state['model']); ops.load_state_dict(state['operators']); optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']; curve = state['curve']
    digest = sha256(); exposures = np.zeros(int(fit.sum()), dtype=np.int64)
    started = time.monotonic()
    for step, (epoch, ids) in enumerate(schedule, 1):
        digest.update(np.asarray([epoch], dtype=np.int64).tobytes() + ids.tobytes()); exposures[ids] += 1
        if epoch <= start_epoch:
            continue
        if time.time() >= datetime.fromisoformat(plan['deadline_utc']).timestamp():
            raise TimeoutError('Joint fit incomplete by10AM; compound test remains closed')
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
                'source_initialization_sha256': source['sha256'], 'curve': curve})
    del model, ops, optimizer


def evaluate(plan, parent, config, root, sig, tokens, device):
    fits = list((root / 'fits').glob('*.json')); assert len(fits) == 24
    assert all(json.loads(p.read_text())['status'] == 'complete' for p in fits)
    atomic_json(root / 'test_opened.json', {'opened_utc': core.now(), 'all24_fits_complete': True,
                'fit_record_sha256': {p.name: sha(p) for p in fits}})
    data = dict(np.load(root / 'dataset' / 'test' / 'dataset.npz')); rows = []
    actions = {'c': 1, 'i': 4, 'ci': 5, 'ic': 6, 'cc': 0, 'ii': 0}
    for index in range(6):
        for condition in plan['conditions']:
            name = f'n{index}_hidden_{condition}'; source = sig['sources'][index % 3]
            model = core.source_model(source, parent, config, device)
            state = torch.load(root / 'maps' / f'{name}_e{plan["epochs"]}.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(state['model']); del state
            h = core.hidden_features(model, data, parent['source_task'], tokens).astype(float)
            maps = dict(np.load(root / 'maps' / f'{name}.npz'))
            output = h @ maps['readout_weight'].T + maps['readout_bias']; arrays = {}
            np.savez_compressed(root / 'features' / f'{name}_test.npz', hidden=h.astype(np.float32), output=output.astype(np.float32))
            del model
            for word in plan['words']:
                action = actions[word]; prediction = core.apply_numpy(h[:, 0], word, maps)
                answers = (prediction @ maps['readout_weight'].T + maps['readout_bias']).argmax(-1)
                arrays[word + '_hidden'] = prediction.astype(np.float32); arrays[word + '_answers'] = answers
                forced_answers = None
                if len(word) == 2:
                    intermediate = 1 if word[0] == 'c' else 4
                    forced = core.apply_numpy(h[:, intermediate], word[1], maps)
                    forced_answers = (forced @ maps['readout_weight'].T + maps['readout_bias']).argmax(-1)
                    arrays[word + '_forced_answers'] = forced_answers
                for split, tag in [(0, 'iid'), (1, 'collisions')]:
                    use = data['split'] == split; hit = answers[use] == data['labels'][use, action]
                    denominator = np.square(h[use, action] - h[use, 0]).sum()
                    ck = [core.gram_cka(prediction[np.flatnonzero(use & (data['lengths'] == n))[:128]],
                                       h[np.flatnonzero(use & (data['lengths'] == n))[:128], action]) for n in plan['lengths']]
                    row = {'replicate': f'n{index}', 'source_seed': source['seed'], 'condition': condition, 'view': 'hidden',
                           'split': tag, 'word': word, 'action': action, 'examples': int(use.sum()), 'accuracy': float(hit.mean()),
                           'teacher_forced_accuracy': float((forced_answers[use] == data['labels'][use, action]).mean()) if forced_answers is not None else None,
                           'direct_input_accuracy': float((output[use, action].argmax(-1) == data['labels'][use, action]).mean()),
                           'pair_both_correct': None, 'displacement_nmse': float(np.square(prediction[use] - h[use, action]).sum() / denominator) if denominator > 1e-20 else None,
                           'cka': float(np.mean(ck)), 'composed_displacement_energy': float(np.square(prediction[use] - h[use, 0]).sum() /
                               max(np.square(h[use, 0] - h[use, 0].mean(0)).sum(), 1e-20))}
                    if split == 1:
                        pair_ids = data['pair_ids'][use]
                        row['pair_both_correct'] = float(np.mean([hit[pair_ids == p].all() for p in np.unique(pair_ids)]))
                    rows.append(row)
            np.savez_compressed(root / 'evaluations' / f'{name}.npz', **arrays)
    atomic_json(root / 'evaluation_records.json', {'completed_utc': core.now(), 'records': rows})
    core.summarize(plan, root)
    summary = json.loads((root / 'summary.json').read_text()); grades = []
    for seed in plan['source_seeds']:
        selected = [json.loads(p.read_text())['curve'][-1]['generator_C_I_accuracy'] for p in fits
                    if json.loads(p.read_text())['source_seed'] == seed and json.loads(p.read_text())['condition'] == 'both_correct']
        grades.append(np.mean(selected, axis=0).tolist())
    summary.update(operator_fits=24, joint_encoder_learning=True, frozen_backbones=0,
                   frozen_teacher_backbones=3, trained_backbones=24, frozen_numeric_readouts=3,
                   single_generator_interpretation_gate=all(min(g) > .9 for g in grades),
                   correct_generator_three_source_validation_accuracy=grades)
    atomic_json(root / 'summary.json', summary)
    report = root / 'report.html'
    text = report.read_text().replace('所有预测只输入原排列的一个冻结表征', '所有预测只输入原排列的一个联合训练表征')
    text = text.replace('隐藏空间24次拟合为主实验，输出空间24次拟合为诊断，参数量不同。',
                        '编码器与算子共同学习，原数字读出器及其绑定的输入嵌入保持固定。24次拟合只提供e/C/I状态和关系；复合操作不参与任何训练或验证。')
    text = text.replace('<h1>两步关系正确性与未监督组合</h1>', '<h1>编码器与算子共同学习：两步关系正确性</h1>')
    report.write_text(text)


def run():
    plan, parent, config, root, sig = initialize(); _, _, tokens, _ = setup(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        for stage in ['prepare', 'teachers', 'train', 'evaluate']:
            atomic_json(root / 'state.json', {'status': 'running', 'stage': stage, 'updated_utc': core.now()})
            if stage == 'prepare': core.prepare_data(plan, parent, config, root, sig)
            elif stage == 'teachers': core.cache_supports(plan, parent, config, root, sig)
            elif stage == 'train':
                done = 0
                for index in range(6):
                    for condition in plan['conditions']:
                        train_one(plan, parent, config, root, sig, index, condition, tokens, device); done += 1
                        atomic_json(root / 'training_state.json', {'status': 'running', 'completed': done, 'planned': 24, 'updated_utc': core.now()})
                atomic_json(root / 'training_state.json', {'status': 'complete', 'completed': done, 'updated_utc': core.now()})
            else: evaluate(plan, parent, config, root, sig, tokens, device)
        atomic_json(root / 'state.json', {'status': 'evaluated_waiting_independent_verification', 'updated_utc': core.now()})
    except BaseException as error:
        atomic_json(root / 'state.json', {'status': 'failed', 'error': repr(error), 'updated_utc': core.now()})
        raise


if __name__ == '__main__':
    run()
