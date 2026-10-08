"""Identical frozen-backbone controls for all additional source seeds."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from .frozen_relation_followup import fit_maps
from .hidden_relation_evaluate import score_method
from .longrun_engine import atomic_json
from .permworld_combinations import sha


ROOT = Path('results/relation_frozen_seed_confirmation')
DEADLINE = datetime(2026, 10, 6, 17, tzinfo=timezone.utc).timestamp()


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    plan = json.loads(Path('configs/relation_seed_extension.json').read_text())
    parent = json.loads(Path('configs/algebra_hidden_relations.json').read_text())
    ROOT.mkdir(exist_ok=True)
    for folder in ['maps', 'evaluations']:
        (ROOT / folder).mkdir(exist_ok=True)
    signature = {'code_sha256': sha(__file__), 'fit_helper_sha256': sha('experiments/frozen_relation_followup.py'),
        'score_helper_sha256': sha('experiments/hidden_relation_evaluate.py'),
        'source_extension_protocol_sha256': sha(Path(plan['output']) / 'protocol.json'),
        'known_source_sha256': sha(Path(plan['parent']) / 'source_data.npz'),
        'fit_anchors_per_length': 256, 'ridge_grid': parent['ridge_grid'], 'pairings': ['correct', 'shuffled'],
        'scope': 'Same frozen additional backbone, head, known input features and source anchor IDs for both relation pairings. Uses the full known source calibration rather than a larger training budget. Extension of the earlier fixed-backbone follow-up after five native additional-seed results were seen; exploratory corroboration, not a new primary forecast.'}
    path = ROOT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())['signature'] == signature
    else:
        atomic_json(path, {'registered_utc': now(), 'signature': signature, 'new_frozen_seed_maps': 0})
    return plan, parent


def evaluate(plan, parent, name):
    folder = Path(plan['output'])
    source_record = json.loads((folder / 'source' / (name + '.json')).read_text())
    ep = folder / 'evaluations' / (name + '.json')
    fp = Path('results/relation_operator_stability/features') / (name + '.npz')
    if not fp.exists() or not ep.exists():
        return
    cp = folder / 'checkpoints' / (name + '.pt')
    assert sha(cp) == source_record['checkpoint_sha256']
    known = dict(np.load(fp))
    probe_feature = folder / 'features' / (name + '.npz')
    hidden = np.load(probe_feature)['source_query_concat'][:, :, -1].astype(np.float64)
    source = dict(np.load(Path(plan['parent']) / 'source_data.npz'))
    data = dict(np.load(Path(plan['parent']) / 'probe_dataset.npz'))
    for pairing in ['correct', 'shuffled']:
        ident = name + '_' + pairing
        dest = ROOT / 'evaluations' / (ident + '.json')
        if dest.exists():
            continue
        maps, means, alphas, val, ids = fit_maps(known['train_hidden'], source['train_lengths'], known['validation_hidden'],
            source['validation_lengths'], 256, parent['ridge_grid'], source_record['seed'] + 2026100671, pairing == 'shuffled')
        rows, arrays = score_method(hidden, data, maps, means, known['readout_weight'].astype(np.float64), known['readout_bias'].astype(np.float64), 'query')
        arrays.update({f'rho_{g}': v[0] for g, v in maps.items()})
        arrays.update({f'bias_{g}': v[1] for g, v in maps.items()})
        arrays.update(mean_lengths=np.array(list(means)), mean_vectors=np.array(list(means.values())), source_anchor_ids=ids,
            readout_weight=known['readout_weight'], readout_bias=known['readout_bias'])
        ap = ROOT / 'maps' / (ident + '.npz')
        np.savez_compressed(ap, **arrays)
        atomic_json(dest, {'source': name, 'source_condition': source_record['condition'], 'seed': source_record['seed'],
            'fit_anchors_per_length': 256, 'pairing': pairing, 'source_checkpoint_sha256': sha(cp),
            'map_archive_sha256': sha(ap), 'known_feature_path': str(fp), 'known_feature_sha256': sha(fp),
            'probe_feature_path': str(probe_feature), 'probe_feature_sha256': sha(probe_feature),
            'visible_accuracy_e_C_I': source_record['observed_validation_accuracy'], 'selected_generator_alphas': alphas,
            'known_validation_displacement_nmse': val, 'metrics': rows, 'completed_utc': now()})
        print(json.dumps({'new_frozen_seed_control': ident}), flush=True)


if __name__ == '__main__':
    plan, parent = initialize()
    while time.time() < DEADLINE:
        for path in (Path(plan['output']) / 'source').glob('*.json'):
            if time.time() < DEADLINE:
                evaluate(plan, parent, path.stem)
        count = len(list((ROOT / 'evaluations').glob('*.json')))
        atomic_json(ROOT / 'state.json', {'status': 'complete' if count == 36 else 'waiting_or_evaluating', 'maps': count, 'updated_utc': now()})
        if count == 36:
            break
        time.sleep(min(45, max(0, DEADLINE-time.time())))
