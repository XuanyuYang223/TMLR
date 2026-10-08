"""Prospective matched-parameter activation x correspondence intervention."""
import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import algebra_relation_campaign as core
from . import algebra_relation_worlds as worlds
from .algebra_relation_feasibility import apply_encoding
from .longrun_engine import atomic_json, atomic_torch
from .permworld_combinations import sha
from .two_step_relation_factorial import factorial_pairings, gram_cka, now

CONFIG = Path('configs/operator_capacity_confirmation.json')


class MatchedResidual(nn.Module):
    def __init__(self, a, bias, hidden_width, nonlinear):
        super().__init__()
        self.register_buffer('a', torch.as_tensor(a, dtype=torch.float32))
        self.register_buffer('bias', torch.as_tensor(bias, dtype=torch.float32))
        self.layers = nn.Sequential(nn.Linear(len(a), hidden_width),
                                   nn.GELU() if nonlinear else nn.Identity(),
                                   nn.Linear(hidden_width, len(a)))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, h):
        return h @ self.a + self.bias + self.layers(h)


def losses(pred, true_output, geometric_target, readout_weight, readout_bias,
           scale, temperature):
    logits = F.linear(pred, readout_weight, readout_bias)
    return core.kd(logits, true_output, temperature), (pred-geometric_target).square().mean()/scale


def setup():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    apply_encoding('numeric_gelu')
    config = json.loads(CONFIG.read_text())
    plan = json.loads(Path('configs/algebra_relation_common_v3.json').read_text())
    plan.update(source_seeds=config['source_seeds'], fit_repeats=3)
    root = Path(config['output'])
    for name in ['', 'dataset', 'sources', 'fits', 'models', 'predictors',
                 'predictor_fits', 'states', 'evaluations']:
        (root/name).mkdir(parents=True, exist_ok=True)
    paths = [__file__, str(CONFIG), core.__file__, worlds.__file__,
             'experiments/algebra_relation_feasibility.py',
             'experiments/two_step_relation_factorial.py']
    signature = {'config': config, 'source_plan': plan,
                 'source_code_sha256': {str(p): sha(p) for p in paths},
                 'excluded_data_sha256': {p: sha(p) for p in config['excluded_test_datasets']}}
    path = root/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())['signature'] == signature
    else:
        atomic_json(path, {'registered_utc': now(), 'new_compound_outcomes_observed': False,
                           'signature': signature})
        (root/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return config, plan, root


def prepare(config, plan, root):
    if (root/'data_audit.json').exists():
        return
    p, domain = 13, 'matrix'
    rng = np.random.default_rng(config['data_seed'])
    excluded = {worlds.key(tuple(x), domain, p)
                for file in config['excluded_test_datasets']
                for x in np.load(file)['x'][:, 0]}
    rows = [tuple(x) for x in product(range(p), repeat=4)
            if (x[0]*x[3]-x[1]*x[2]) % p and worlds.key(tuple(x), domain, p) not in excluded]
    used = set()
    collision = worlds.choose_collision_pairs(rows, config['collision_pairs'], rng, domain, p, used)
    rng.shuffle(rows)
    iid = []
    for x in rows:
        k = worlds.key(x, domain, p)
        if k not in used:
            iid.append(x); used.add(k)
            if len(iid) == config['iid_examples']:
                break
    available = defaultdict(list)
    for x in rows:
        if worlds.key(x, domain, p) not in used:
            available[worlds.key(x, domain, p)].append(x)
    keys = list(available); rng.shuffle(keys)
    nv, npv, ns = [config[k] for k in ['validation_anchors', 'pilot_validation_anchors', 'source_anchors']]
    vals = set(keys[:nv+npv]); srcs = set(keys[nv+npv:nv+npv+ns])
    parts = {
        'validation': worlds.encode([available[k][0] for k in keys[:nv]], domain, p),
        'pilot_validation': worlds.encode([available[k][0] for k in keys[nv:nv+npv]], domain, p),
        'source': worlds.encode([available[k][0] for k in keys[nv+npv:nv+npv+ns]], domain, p)}
    pool = [x for x in rows if worlds.key(x, domain, p) not in used | vals | srcs]
    for i in range(3):
        d = worlds.encode(worlds.paired_fit(pool, config['fit_anchors'],
                          np.random.default_rng(config['data_seed']+101+i), domain, p), domain, p)
        pc, pi, eligible = factorial_pairings(np.ones(len(d['x']), int), d['labels'],
                                             config['pairing_seed']+i, 3)
        assert eligible.all()
        d.update(pc=pc, pi=pi, eligible=eligible)
        parts[f'support{i}'] = d
    test = worlds.encode(iid+collision, domain, p, plan['words'])
    test.update(split=np.array([0]*len(iid)+[1]*len(collision)),
                pair_ids=np.array([-1]*len(iid)+[j for j in range(config['collision_pairs']) for _ in range(2)]))
    parts['test'] = test
    sets = {n: {worlds.key(tuple(x), domain, p) for x in d['x'][:, 0]} for n, d in parts.items()}
    train = set.union(sets['source'], *[sets[f'support{i}'] for i in range(3)])
    val = sets['validation'] | sets['pilot_validation']
    assert not train & val and not train & sets['test'] and not val & sets['test']
    assert not set.union(*sets.values()) & excluded
    for n, d in parts.items():
        np.savez_compressed(root/'dataset'/f'{n}.npz', **d)
    atomic_json(root/'data_audit.json', {'created_utc': now(), 'excluded_historical_test_orbits': len(excluded),
        'whole_orbit_splits_disjoint': True, 'wrong_all_three_gold_answers_matched': True,
        'wrong_complete_derangement': True, 'fit_pool_orbits': len({worlds.key(x, domain, p) for x in pool}),
        'dataset_sha256': {str(p): sha(p) for p in (root/'dataset').glob('*.npz')}})


def load_backbone(root, i, device):
    state = torch.load(root/'models'/f'n{i}_both_correct.pt', map_location='cpu', weights_only=True)
    model = core.Encoder(13, 128).to(device)
    ops = core.Operators(128).to(device)
    model.load_state_dict(state['model']); ops.load_state_dict(state['operators'])
    model.eval(); ops.eval()
    for param in list(model.parameters())+list(ops.parameters()):
        param.requires_grad_(False)
    return model, ops


@torch.no_grad()
def cache_known_states(root, i, device):
    file = root/'states'/f's{i}_known.npz'
    if file.exists():
        return dict(np.load(file))
    model, ops = load_backbone(root, i, device)
    _, x, _ = core.arrays(root/'dataset'/f'support{i}.npz', 13, device)
    _, vx, _ = core.arrays(root/'dataset/validation.npz', 13, device)
    z = dict(train=model(x).cpu().numpy(), validation=model(vx).cpu().numpy(),
             a=ops.maps[0].weight.cpu().numpy().T, bias_a=ops.maps[0].bias.cpu().numpy(),
             b=ops.maps[1].weight.cpu().numpy().T, bias_b=ops.maps[1].bias.cpu().numpy(),
             w=model.readout.weight.cpu().numpy(), bias_w=model.readout.bias.cpu().numpy())
    np.savez_compressed(file, **z)
    return z


def fit_predictor(config, root, i, condition, device):
    record_file = root/'predictor_fits'/f's{i}_{condition}.json'
    if record_file.exists():
        return json.loads(record_file.read_text())
    z = cache_known_states(root, i, device)
    train = torch.as_tensor(z['train'], device=device)
    val = torch.as_tensor(z['validation'], device=device)
    w, bw = [torch.as_tensor(z[k], device=device) for k in ['w', 'bias_w']]
    target_output = F.linear(train[:, 1], w, bw)
    scale = float(train[:, 1].var(0, unbiased=False).mean())
    nonlinear = condition.startswith('nonlinear_')
    wrong = condition.endswith('_wrong')
    data = dict(np.load(root/'dataset'/f'support{i}.npz'))
    pairing = data['pc'] if wrong else np.arange(len(train))
    pair = torch.as_tensor(pairing, device=device)
    torch.manual_seed(config['predictor_seed']+i)
    model = MatchedResidual(z['a'], z['bias_a'], config['predictor_hidden_width'], nonlinear).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == config['predictor_parameters']
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    rng = np.random.default_rng(config['schedule_seed']+i)
    digest = sha256(); counts = np.zeros(len(train), int); curve = []
    started = time.monotonic()
    for step in range(1, config['steps']+1):
        ids = rng.integers(0, len(train), config['batch_size'], dtype=np.int64)
        np.add.at(counts, ids, 1); digest.update(ids.tobytes())
        ix = torch.as_tensor(ids, device=device)
        pred = model(train[ix, 0])
        output, geom = losses(pred, target_output[ix], train[pair[ix], 1], w, bw, scale, config['temperature'])
        loss = config['output_kd_weight']*output+config['geometry_weight']*geom
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 500 == 0:
            with torch.no_grad():
                pred_val = model(val[:, 0])
                labels = np.load(root/'dataset/validation.npz')['labels']
                row = {'step': step, 'output_kd': float(output.detach()), 'geometry_loss': float(geom.detach()),
                    'validation_first_state_nmse': float((pred_val-val[:, 1]).square().mean()/scale),
                    'validation_first_accuracy': float((F.linear(pred_val, w, bw).argmax(-1).cpu().numpy()==labels[:, 1]).mean())}
                curve.append(row)
            print(json.dumps({'source': i, 'condition': condition, **row, 'elapsed_seconds': time.monotonic()-started}), flush=True)
    cp = root/'predictors'/f's{i}_{condition}.pt'
    atomic_torch(cp, {'state_dict': model.state_dict(), 'condition': condition, 'source': i,
                     'nonlinear': nonlinear, 'parameters': parameters, 'steps': config['steps']})
    record = {'status': 'complete', 'completed_utc': now(), 'source': i, 'condition': condition,
        'parameters': parameters, 'steps': config['steps'], 'batch_size': config['batch_size'],
        'anchor_exposures': int(counts.sum()), 'schedule_sha256': digest.hexdigest(),
        'exposure_count_sha256': sha256(counts.tobytes()).hexdigest(),
        'geometry_scale': scale, 'pairing_sha256': sha256(pairing.tobytes()).hexdigest(),
        'backbone_sha256': sha(root/'models'/f'n{i}_both_correct.pt'), 'known_states_sha256': sha(root/'states'/f's{i}_known.npz'),
        'checkpoint_sha256': sha(cp), 'curve': curve, 'frozen_backbone_not_in_optimizer': True,
        'fit_inputs': ['h(e x)', 'h(a x)', 'correct readout(h(a x))'], 'no_compound_fit_targets': True}
    atomic_json(record_file, record)
    return record


def crossed_interval(values, draws, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    si = rng.integers(0, values.shape[0], (draws, values.shape[0]))
    pi = rng.integers(0, values.shape[1], (draws, values.shape[1]))
    resampled = values[si[:, :, None], pi[:, None, :]].mean((1, 2))
    return np.quantile(resampled, [.025, .975]).tolist()


@torch.no_grad()
def evaluate(config, plan, root, device):
    files = [root/'predictor_fits'/f's{i}_{c}.json' for i in range(3) for c in config['conditions']]
    assert all(p.exists() for p in files)
    if (root/'test_opened.json').exists():
        assert (root/'evaluation.json').exists(), 'Do not silently reopen a partial test evaluation.'
        return
    atomic_json(root/'test_opened.json', {'opened_utc': now(), 'all12_fixed_budget_predictors_complete': True,
        'fit_record_sha256': {str(p): sha(p) for p in files}})
    data, x, _ = core.arrays(root/'dataset/test.npz', 13, device)
    rows = []; paired = {c: [] for c in config['conditions']}; paired_both = {c: [] for c in config['conditions']}
    for i in range(3):
        model, ops = load_backbone(root, i, device)
        h = model(x); w = model.readout.weight; bw = model.readout.bias
        hn = h.cpu().numpy(); z = dict(np.load(root/'states'/f's{i}_known.npz'))
        np.savez_compressed(root/'states'/f's{i}_test.npz', hidden=hn)
        scale = float(np.var(z['train'][:, 1], axis=0).mean())
        for condition in ['original']+config['conditions']:
            if condition == 'original':
                first = ops(h[:, 0], 0)
            else:
                predictor = MatchedResidual(z['a'], z['bias_a'], config['predictor_hidden_width'], condition.startswith('nonlinear_')).to(device)
                cp = torch.load(root/'predictors'/f's{i}_{condition}.pt', map_location=device, weights_only=True)
                predictor.load_state_dict(cp['state_dict']); predictor.eval()
                first = predictor(h[:, 0])
            compound = ops(first, 1)
            ans = model.readout(compound).argmax(-1).cpu().numpy()
            first_ans = model.readout(first).argmax(-1).cpu().numpy()
            hit = ans == data['labels'][:, 3]
            forced = model.readout(ops(h[:, 1], 1)).argmax(-1).cpu().numpy()
            np.savez_compressed(root/'evaluations'/f's{i}_{condition}.npz', first_pred=first.cpu().numpy(),
                                compound_pred=compound.cpu().numpy(), compound_answers=ans, first_answers=first_ans)
            for sid, split in [(0, 'iid'), (1, 'collisions')]:
                use = data['split'] == sid
                pairs = data['pair_ids'][use]
                row = {'source': i, 'source_seed': config['source_seeds'][i], 'condition': condition, 'split': split,
                    'accuracy': float(hit[use].mean()), 'pair_both_correct': None,
                    'first_accuracy': float((first_ans[use]==data['labels'][use, 1]).mean()),
                    'first_state_nmse': float(np.square(first.cpu().numpy()[use]-hn[use, 1]).mean()/scale),
                    'compound_state_nmse': float(np.square(compound.cpu().numpy()[use]-hn[use, 3]).mean()/max(float(np.var(hn[use, 3], axis=0).mean()), 1e-10)),
                    'cka': gram_cka(compound.cpu().numpy()[use], hn[use, 3]),
                    'true_intermediate_accuracy': float((forced[use]==data['labels'][use, 3]).mean()),
                    'direct_native_accuracy': float((model.readout(h[:, 3]).argmax(-1).cpu().numpy()[use]==data['labels'][use, 3]).mean())}
                if sid == 1:
                    phit = np.asarray([hit[use][pairs == pid] for pid in np.unique(pairs)])
                    assert phit.shape == (config['collision_pairs'], 2)
                    row['pair_both_correct'] = float(phit.all(1).mean())
                    if condition != 'original':
                        paired[condition].append(phit.mean(1)); paired_both[condition].append(phit.all(1).astype(float))
                rows.append(row)
    means = []
    for c in ['original']+config['conditions']:
        for split in ['iid', 'collisions']:
            select = [r for r in rows if r['condition']==c and r['split']==split]
            keys = ['accuracy', 'first_accuracy', 'first_state_nmse', 'compound_state_nmse', 'cka',
                    'true_intermediate_accuracy', 'direct_native_accuracy']
            if split == 'collisions':
                keys.append('pair_both_correct')
            means.append({'condition': c, 'split': split, **{k: float(np.mean([r[k] for r in select])) for k in keys}})
    contrasts = []
    for endpoint, values in [('accuracy', paired), ('pair_both_correct', paired_both)]:
        differences = {kind: np.asarray(values[kind+'_correct'])-np.asarray(values[kind+'_wrong']) for kind in ['linear', 'nonlinear']}
        differences['interaction'] = differences['nonlinear']-differences['linear']
        differences['nonlinear_gain_correct'] = np.asarray(values['nonlinear_correct'])-np.asarray(values['linear_correct'])
        differences['nonlinear_gain_wrong'] = np.asarray(values['nonlinear_wrong'])-np.asarray(values['linear_wrong'])
        for name, diff in differences.items():
            pp = 100*diff
            contrasts.append({'endpoint': endpoint, 'contrast': name, 'mean_pp': float(pp.mean()),
                'source_effects_pp': pp.mean(1).tolist(), 'two_way_bootstrap_95_pp': crossed_interval(pp, config['bootstrap_samples'], config['bootstrap_seed'])})
    gates = [json.loads((root/'fits'/f'n{i}_both_correct.json').read_text())['curve'][-1] for i in range(3)]
    atomic_json(root/'evaluation.json', {'status': 'complete', 'completed_utc': now(), 'formal_fits': 12,
        'independent_sources': 3, 'records': rows, 'means': means, 'contrasts': contrasts,
        'source_known_grades': gates, 'all_source_gates_pass': all(min(g['native'])>=config['source_gate']['native'] and min(g['generators'])>=config['source_gate']['generators'] for g in gates),
        'scope': config['scope'], 'inference': 'All compound predictions start from h(e x); learned residual first operator then the shared frozen second operator and readout. Gold composite states are used only for evaluation.'})


def run(stage):
    config, plan, root = setup()
    prepare(config, plan, root)
    if stage == 'prepare':
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        if stage in ['run', 'source']:
            for i in range(3):
                atomic_json(root/'state.json', {'status': 'running', 'stage': 'source_preparation', 'source': i, 'updated_utc': now()})
                core.fit(plan, root, i, 'both_correct', device)
        if stage in ['run', 'fit']:
            assert all((root/'models'/f'n{i}_both_correct.pt').exists() for i in range(3))
            for i in range(3):
                for condition in config['conditions']:
                    atomic_json(root/'state.json', {'status': 'running', 'stage': 'first_operator_fit', 'source': i, 'condition': condition, 'updated_utc': now()})
                    fit_predictor(config, root, i, condition, device)
        if stage in ['run', 'evaluate']:
            evaluate(config, plan, root, device)
            atomic_json(root/'state.json', {'status': 'trained_and_evaluated', 'updated_utc': now()})
    except BaseException as error:
        atomic_json(root/'state.json', {'status': 'failed', 'reason': repr(error), 'updated_utc': now()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'source', 'fit', 'evaluate', 'run'], default='run')
    run(parser.parse_args().stage)
