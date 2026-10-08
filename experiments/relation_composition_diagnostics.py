"""Teacher-forced transitions localize, rather than claim, hidden inference."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .representation_algebra import permutation_action_table, word_action


PARENT = Path('results/algebra_hidden_relations')
ROOT = Path('results/frozen_relation_followup')
LETTERS = {'c': 1, 'i': 4}


def now():
    return datetime.now(timezone.utc).isoformat()


def numeric_basis(weights):
    _, singular, vt = np.linalg.svd(weights[1:] - weights[:1], full_matrices=False)
    keep = singular > max(singular[0] * 1e-10, 1e-12)
    return vt[keep].T


def measures(pred, source, target, labels, weights, bias, basis):
    error = pred - target
    delta = target - source
    denominator = float(np.square(delta).sum())
    logits = pred @ weights.T + bias
    oracle = target @ weights.T + bias
    correct = logits.argmax(-1) == labels
    direct = oracle.argmax(-1) == labels
    def margin(logits):
        changed = logits.copy()
        chosen = changed[np.arange(len(labels)), labels].copy()
        changed[np.arange(len(labels)), labels] = -np.inf
        return chosen - changed.max(-1)
    actual_margin, predicted_margin = margin(oracle), margin(logits)
    perturbation = np.max(np.abs(logits - oracle), axis=-1)
    projected_error = error @ basis
    projected_delta = delta @ basis
    numeric_denominator = float(np.square(projected_delta).sum())
    null_error = error - projected_error @ basis.T
    null_delta = delta - projected_delta @ basis.T
    null_denominator = float(np.square(null_delta).sum())
    return {
        'hidden_displacement_nmse': float(np.square(error).sum() / denominator) if denominator > 1e-20 else None,
        'numeric_contrast_displacement_nmse': float(np.square(projected_error).sum() / numeric_denominator) if numeric_denominator > 1e-20 else None,
        'numeric_null_displacement_nmse': float(np.square(null_error).sum() / null_denominator) if null_denominator > 1e-20 else None,
        'answer_accuracy': float(correct.mean()),
        'direct_input_answer_accuracy': float(direct.mean()),
        'median_direct_true_answer_margin': float(np.median(actual_margin)),
        'median_composed_true_answer_margin': float(np.median(predicted_margin)),
        'median_logit_Linf_error': float(np.median(perturbation)),
        'fraction_answer_protected_by_margin_bound': float(np.mean(direct & (actual_margin > 2 * perturbation))),
    }


def decompose_word(hidden, centers, maps, word, table):
    """Exact error = transported earlier errors + true-state transition errors."""
    source = hidden[:, 0] - centers
    current = source.copy()
    action = 0
    components = []
    stages = []
    for letter in word:
        next_action = int(table[action, LETTERS[letter]])
        rho, bias = maps[letter]
        actual_current = hidden[:, action] - centers
        actual_next = hidden[:, next_action] - centers
        local = actual_current @ rho + bias - actual_next
        components = [component @ rho for component in components] + [local]
        current = current @ rho + bias
        np.testing.assert_allclose(sum(components), current - actual_next, rtol=1e-7, atol=1e-7)
        stages.append({'input_action': action, 'output_action': next_action, 'letter': letter,
            'predicted_hidden': current + centers, 'teacher_forced_hidden': actual_current @ rho + bias + centers})
        action = next_action
    return current + centers, components, stages


def diagnose(hidden, data, maps, means, weights, bias):
    table = permutation_action_table()
    basis = numeric_basis(weights)
    rows, saved = [], {}
    known_edges = {(0, 1), (1, 0), (0, 4), (4, 0)}
    for split, tag in [(2, 'iid'), (3, 'answer_collisions')]:
        use = data['split'] == split
        h = hidden[use].astype(np.float64)
        centers = np.array([means.get(int(n), np.zeros(h.shape[-1])) for n in data['lengths'][use]])
        labels = data['labels'][use]
        for action in range(8):
            for letter in ['c', 'i']:
                changed = int(table[action, LETTERS[letter]])
                rho, offset = maps[letter]
                predicted = (h[:, action] - centers) @ rho + offset + centers
                row = {'kind': 'teacher_forced_transition', 'split': tag, 'input_action': action, 'output_action': changed,
                    'letter': letter, 'known_supervised_edge': (action, changed) in known_edges,
                    **measures(predicted, h[:, action], h[:, changed], labels[:, changed], weights, bias, basis)}
                rows.append(row)
                saved[f'{tag}_edge_{action}_{letter}'] = predicted.astype(np.float32)
        for word in ['ci', 'ici']:
            predicted, components, stages = decompose_word(h, centers, maps, word, table)
            action = word_action(word, table, LETTERS)
            target, source = h[:, action], h[:, 0]
            denominator = float(np.square(target - source).sum())
            assert denominator > 1e-20
            component_energies = [float(np.square(component).sum() / denominator) for component in components]
            total_error = float(np.square(predicted - target).sum() / denominator)
            cross = total_error - sum(component_energies)
            rows.append({'kind': 'composed_from_base', 'split': tag, 'word': word,
                'transported_local_error_energy_over_final_displacement': component_energies,
                'cross_terms_over_final_displacement': cross,
                **measures(predicted, source, target, labels[:, action], weights, bias, basis)})
            saved[f'{tag}_{word}_composed'] = predicted.astype(np.float32)
            for step in range(1, len(word)):
                intermediate = word_action(word[:step], table, LETTERS)
                forced = h[:, intermediate] - centers
                for letter in word[step:]:
                    rho, offset = maps[letter]
                    forced = forced @ rho + offset
                forced += centers
                rows.append({'kind': 'teacher_forced_intermediate', 'split': tag, 'word': word, 'forced_after_steps': step,
                    'true_intermediate_action': intermediate, 'diagnostic_only': True,
                    **measures(forced, source, target, labels[:, action], weights, bias, basis)})
                saved[f'{tag}_{word}_forced_after_{step}'] = forced.astype(np.float32)
    return rows, saved


def run():
    root = ROOT / 'diagnostics'
    root.mkdir(parents=True, exist_ok=True)
    signature = {'code_sha256': sha(__file__), 'frozen_protocol_sha256': sha(ROOT / 'protocol.json'),
        'data_sha256': sha(PARENT / 'probe_dataset.npz'),
        'scope': 'Post-parent, exploratory diagnosis. All true intermediate states are evaluation-only teacher-forcing diagnostics; hidden prediction remains base plus generator maps.'}
    protocol = root / 'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())['signature'] == signature
    else:
        atomic_json(protocol, {'registered_utc': now(), 'signature': signature, 'new_diagnostics': 0})
    plan = json.loads(Path('configs/frozen_relation_followup.json').read_text())
    data = dict(np.load(PARENT / 'probe_dataset.npz'))
    for seed in plan['source_seeds']:
        for condition in plan['source_conditions']:
            source = f'{condition}_s{seed}'
            hidden = np.load(PARENT / 'features' / f'{source}.npz')['source_query_concat'][:, :, -1]
            variants = [
                ('native_operators', PARENT / 'arrays' / f'{source}_query_native_operators.npz'),
                ('frozen_full_correct', ROOT / 'maps' / f'{source}_k256_correct.npz'),
                ('frozen_full_shuffled', ROOT / 'maps' / f'{source}_k256_shuffled.npz'),
            ]
            for method, path in variants:
                name = f'{source}_{method}'
                dest = root / f'{name}.json'
                if dest.exists():
                    continue
                archive = dict(np.load(path))
                maps = {letter: (archive[f'rho_{letter}'], archive[f'bias_{letter}']) for letter in ['c', 'i']}
                means = {int(n): v for n, v in zip(archive['mean_lengths'], archive['mean_vectors'])}
                rows, saved = diagnose(hidden, data, maps, means, archive['readout_weight'].astype(np.float64), archive['readout_bias'].astype(np.float64))
                output = root / f'{name}.npz'
                np.savez_compressed(output, **saved)
                atomic_json(dest, {'source': source, 'source_condition': condition, 'seed': seed, 'method': method,
                    'input_map_sha256': sha(path), 'output_archive_sha256': sha(output), 'rows': rows, 'completed_utc': now()})
                print(json.dumps({'diagnosed': name, 'primary': [row for row in rows if row['kind'] == 'composed_from_base' and row['split'] == 'answer_collisions']}), flush=True)


if __name__ == '__main__':
    run()
