"""Exploratory geometry after known-only output-confidence nuisance regression."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .algebra_structure_direct import direct_probe
from .algebra_structure_replication import load_plan
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_source_subspace import contrast_basis
from .permworld_combinations import sha
from .representation_algebra import permutation_action_table, word_action


ROOT = Path('results/confidence_residual_geometry')
DEADLINE = datetime(2026, 10, 6, 17, tzinfo=timezone.utc).timestamp()
ALPHAS = [1e-6, 1e-4, .01, 1.]
WORDS = ['c', 'r', 'i', 'rc', 'ci', 'ri', 'rci']


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    ROOT.mkdir(exist_ok=True)
    for name in ['evaluations', 'maps', 'nuisance_fits', 'arrays']:
        (ROOT / name).mkdir(exist_ok=True)
    signature = {'code_sha256': sha(__file__), 'test_data_sha256': sha('results/joint_answer_matched_geometry/dataset.npz'),
        'calibration_data_sha256': sha('results/algebra_structure_replication/probe_dataset.npz'),
        'core_sha256': {p: sha(p) for p in ['experiments/algebra_structure_direct.py', 'experiments/algebra_structure_replication.py',
            'experiments/native_source_subspace.py', 'experiments/representation_algebra.py', 'experiments/longrun_transfer.py']},
        'nuisance_ridge_grid': ALPHAS,
        'scope': 'Exploratory analysis after all three new records query outcomes and eight source trainings were inspected. No residual-confidence geometry outcome has been inspected. Regress numeric-readout-null hidden vectors on four tasks centered numeric logits, softmax probabilities, entropy, maximum probability and length one-hot using calibration FIT; ridge selection uses only calibration VAL hidden reconstruction loss. Refit generator probes on these residual calibration vectors; compose without compound fitting and score all fixed joint answer-matched test pairs. All groups and trained/initialization models are reported. No true task labels or conditional-test hidden vectors enter nuisance fitting or selection. This removes only the specified linear dependence on the constructed confidence features, not all answer information or all confidence dependence, and is not a causal mechanism intervention. Retained pair energy is reported to detect near-zero residuals.'}
    path = ROOT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())['signature'] == signature
    else:
        atomic_json(path, {'registered_utc': now(), 'signature': signature, 'new_confidence_residual_evaluations': 0})


def weight_for(rec, root):
    if rec['status'] == 'trained':
        state = torch.load(root / 'source/checkpoints' / (rec['source'] + '.pt'), map_location='cpu', weights_only=True)['model']
        assert 'lm_head.bias' not in state
        return state['lm_head.weight'][:31].numpy().astype(np.float64)
    plan, config, _ = load_plan()
    model = make_model(config, plan['architecture'], rec['seed'], 'cpu')
    assert model.lm_head.bias is None
    return model.lm_head.weight[:31].detach().numpy().astype(np.float64)


def null_and_code(hidden, lengths, weight):
    block = hidden.reshape(*hidden.shape[:-1], 4, -1).astype(np.float64)
    basis, _ = contrast_basis(weight)
    null = (block - (block @ basis) @ basis.T).reshape(hidden.shape)
    logits = block @ weight.T
    logits -= logits.mean(-1, keepdims=True)
    exponential = np.exp(logits - logits.max(-1, keepdims=True))
    probability = exponential / exponential.sum(-1, keepdims=True)
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-300)), axis=-1, keepdims=True)
    maximum = probability.max(-1, keepdims=True)
    statistics = np.concatenate([logits, probability, entropy, maximum], axis=-1)
    size = np.eye(21)[lengths - 10]
    size = np.broadcast_to(size[:, None], (*hidden.shape[:2], 21))
    code = np.concatenate([statistics.reshape(*hidden.shape[:2], -1), size], axis=-1)
    return null, code


def nuisance_fit(code, null, split):
    x = code[split == 0].reshape(-1, code.shape[-1])
    y = null[split == 0].reshape(-1, null.shape[-1])
    xv = code[split == 1].reshape(-1, code.shape[-1])
    yv = null[split == 1].reshape(-1, null.shape[-1])
    mean, sigma = x.mean(0), np.maximum(x.std(0), 1e-4)
    design = np.column_stack([(x - mean) / sigma, np.ones(len(x))])
    validation = np.column_stack([(xv - mean) / sigma, np.ones(len(xv))])
    gram, rhs = design.T @ design, design.T @ y
    scale = np.trace(gram[:-1, :-1]) / x.shape[1]
    fits, losses = [], []
    for alpha in ALPHAS:
        fit = np.linalg.solve(gram + np.diag([alpha * max(scale, 1e-20)] * x.shape[1] + [0]), rhs)
        fits.append(fit)
        losses.append(float(np.square(validation @ fit - yv).sum()))
    selected = int(np.argmin(losses))
    return {'weight': fits[selected][:-1], 'bias': fits[selected][-1], 'code_mean': mean, 'code_sigma': sigma}, {
        'selected_alpha': ALPHAS[selected], 'calibration_validation_losses': losses,
        'nuisance_fit_rows': len(x), 'nuisance_validation_rows': len(xv), 'confidence_code_dimension': x.shape[1]}


def residual(null, code, fit):
    return null - ((code - fit['code_mean']) / fit['code_sigma']) @ fit['weight'] - fit['bias']


def evaluate(path):
    rec = json.loads(path.read_text())
    root = path.parent.parent
    dest = ROOT / 'evaluations' / (rec['source'] + '.json')
    if dest.exists():
        return
    plan, _, _ = load_plan()
    cp = root / 'features' / (rec['source'] + '_calibration.npz')
    tp = root / 'features' / (rec['source'] + '_joint_test.npz')
    cal = dict(np.load('results/algebra_structure_replication/probe_dataset.npz'))
    test = dict(np.load('results/joint_answer_matched_geometry/dataset.npz'))
    weight = weight_for(rec, root)
    cal_null, cal_code = null_and_code(np.load(cp)['source_query_concat'], cal['lengths'], weight)
    test_null, test_code = null_and_code(np.load(tp)['source_query_concat'], test['lengths'], weight)
    fitted, selection = nuisance_fit(cal_code, cal_null, cal['split'])
    nr = ROOT / 'nuisance_fits' / (rec['source'] + '.npz')
    np.savez_compressed(nr, **fitted, numeric_weight=weight)
    cal_residual = residual(cal_null, cal_code, fitted)
    test_residual = residual(test_null, test_code, fitted)
    fitting, maps = direct_probe(cal_residual, cal['split'], cal['lengths'], permutation_action_table(),
        {'c': 1, 'r': 2, 'i': 4}, [('r', 'c'), ('c', 'i'), ('r', 'i'), ('r', 'c', 'i')],
        plan['probe_dimension'], plan['ridge_grid'], rec['seed'] + 9200)
    mp = ROOT / 'maps' / (rec['source'] + '.npz')
    np.savez_compressed(mp, **maps)
    pairs = np.stack([np.flatnonzero(test['pair_ids'] == pair) for pair in np.unique(test['pair_ids'])])
    delta = test_residual[pairs[:, 0]] - test_residual[pairs[:, 1]]
    original_delta = test_null[pairs[:, 0]] - test_null[pairs[:, 1]]
    retained = float(np.square(delta).sum() / np.square(original_delta).sum())
    q = maps['basis'].astype(np.float64)
    source = delta.reshape(-1, delta.shape[-1])
    latent = source @ q
    table, letters = permutation_action_table(), {'c': 1, 'r': 2, 'i': 4}
    rows, arrays = [], {'pair_ids': np.unique(test['pair_ids'])}
    for word in WORDS:
        product = np.eye(q.shape[1])
        for g in word:
            product = product @ maps['map_' + g].astype(np.float64)
        predicted_latent = latent @ product
        prediction = source + (predicted_latent - latent) @ q.T
        target = delta[:, table[:, word_action(word, table, letters)]].reshape(source.shape)
        error = np.square(prediction - target).sum()
        rows.append({'word': word, 'pairs': len(pairs), 'pair_target_nmse': float(error / np.square(target).sum()),
            'pair_action_displacement_nmse': float(error / np.square(target - source).sum())})
        arrays[word + '_pair_latent_prediction'] = predicted_latent.astype(np.float32)
    ap = ROOT / 'arrays' / (rec['source'] + '.npz')
    np.savez_compressed(ap, **arrays)
    atomic_json(dest, {'source': rec['source'], 'seed': rec['seed'], 'group': rec['group'], 'status': rec['status'],
        'source_record_path': str(path), 'source_record_sha256': sha(path), 'source_feature_paths': [str(cp), str(tp)],
        'source_feature_sha256': [sha(cp), sha(tp)], 'nuisance_fit_path': str(nr), 'nuisance_fit_sha256': sha(nr),
        'selection': selection, 'map_path': str(mp), 'map_sha256': sha(mp), 'calibration_results': fitting,
        'prediction_archive_sha256': sha(ap), 'retained_test_pair_energy_fraction': retained,
        'projected_residual_pair_energy_fraction': float(np.square(latent).sum() / np.square(delta).sum()),
        'results': rows, 'completed_utc': now()})
    print({'confidence_residual_evaluated': rec['source'], 'retained_pair_energy': retained,
        'composite_target_nmse': np.mean([r['pair_target_nmse'] for r in rows if len(r['word']) > 1])}, flush=True)


if __name__ == '__main__':
    initialize()
    torch.set_num_threads(2)
    while time.time() < DEADLINE:
        for folder in ['ordinary_relation_seed_confirmation', 'ordinary_initialization_control']:
            for path in (Path('results') / folder / 'evaluations').glob('*.json'):
                if time.time() < DEADLINE:
                    evaluate(path)
        count = len(list((ROOT / 'evaluations').glob('*.json')))
        atomic_json(ROOT / 'state.json', {'status': 'complete' if count == 18 else 'waiting_or_evaluating', 'sources': count, 'updated_utc': now()})
        if count == 18:
            break
        time.sleep(min(45, max(0, DEADLINE-time.time())))
