"""Post-computation verification, never a retrospective preregistration."""
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from .algebra import composition_size, world
from .factor_kernel_readout import probabilities, ridge_readout
from .field_symmetry import group_sources
from .field_symmetry_transfer import splits
from .longrun_engine import atomic_json
from .matched_field import controlled_world
from .models import SourceModel
from .native_layer_diagnostic import layer_features
from .permworld_combinations import sha


def verify_kernels():
    torch.set_num_threads(4)
    root = Path('results/factor_kernel_readout')
    protocol = json.loads((root/'protocol.json').read_text())['signature']
    assert protocol['analysis_sha256'] == sha('experiments/factor_kernel_readout.py')
    count, parents = 0, {}
    for source_path, transfer_path, provider in (
        ('configs/field_symmetry.json', 'configs/field_symmetry_transfer.json', world),
        ('configs/field_matched_support.json', 'configs/field_matched_support_transfer.json', controlled_world),
    ):
        source = json.loads(Path(source_path).read_text()); transfer = json.loads(Path(transfer_path).read_text())
        source_root = Path(source['output']); cohort = source_root.name
        parents[source_path] = sha(source_path); parents[transfer_path] = sha(transfer_path)
        targets = np.array(transfer['targets']); p = source['p']
        for w, m in product(source['world_seeds'], source['model_seeds']):
            inputs, _, encoded, basis = provider(p, source['dimension'], w)
            labels = (inputs@basis.T % p)@targets.T % p
            torch.manual_seed(m)
            model = SourceModel(encoded.shape[1], source['hidden'], source['features'], p)
            with torch.no_grad(): logits = model(torch.tensor(encoded)).numpy().astype(np.float64)
            values = np.exp(logits-logits.max(-1, keepdims=True)); values /= values.sum(-1, keepdims=True)
            designs = [('random_heads', values), ('physical_factors', np.eye(p)[inputs])]
            for group in source['groups']:
                record = json.loads((source_root/f'{group}_w{w}_m{m}.json').read_text())
                checkpoint = source_root/'checkpoints'/f'{group}_w{w}_m{m}.pt'
                assert record['status'] == 'complete' and record['source_gate_passed']
                assert sha(checkpoint) == record['checkpoint_sha256']
                feature_path = source_root/f'{group}_w{w}_m{m}_step{source["steps"]}_features.npy'
                parents[str(checkpoint)] = sha(checkpoint); parents[str(feature_path)] = sha(feature_path)
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                model.load_state_dict(state)
                with torch.no_grad(): recomputed = model.encoder(torch.tensor(encoded)).numpy()
                h = np.load(feature_path)
                # GPU-saved source features and CPU recomputation have different roundoff.
                np.testing.assert_allclose(h, recomputed, rtol=5e-4, atol=5e-5)
                factors = probabilities(h, state['heads.weight'].numpy().astype(np.float64), state['heads.bias'].numpy().astype(np.float64), p)
                designs.append((group, factors))
            for group, factors in designs:
                path = root/f'{cohort}_{group}_w{w}_m{m}.json'
                record = json.loads(path.read_text()); assert record['protocol'] == protocol
                rows = record['rows']; assert record['status'] == 'complete'
                keys = {(r['target_id'], r['budget'], r['degree']) for r in rows}
                assert len(keys) == len(rows) == len(targets)*len(transfer['budgets'])*len(protocol['degrees'])
                assert keys == set(product(range(len(targets)), transfer['budgets'], protocol['degrees']))
                for t in range(len(targets)):
                    supports, test = splits(labels[:, t], p, w+90000+t, transfer['budgets'], transfer['test_per_class'])
                    for row in [r for r in rows if r['target_id'] == t]:
                        support = supports[row['budget']]
                        prediction = ridge_readout(factors, labels[support, t], support, test, row['degree'], p, protocol['alpha'])
                        assert float(np.mean(prediction == labels[test, t])) == row['test_accuracy']
                        order = composition_size(group_sources(group, p), targets[t], p) if group in source['groups'] else None
                        assert row['target_composition_size'] == order
                        assert row['target_physical_complexity'] == int(np.count_nonzero(targets[t]@basis % p))
                        count += 1
            print(json.dumps({'verified_kernel_world': w, 'seed': m, 'cohort': cohort, 'endpoints': count}), flush=True)
    assert count == 6912
    atomic_json(root/'verification.json', {
        'status': 'passed', 'verified_utc': datetime.now(timezone.utc).isoformat(),
        'all_kernel_endpoints_recomputed': count, 'source_features_recomputed_on_cpu': 72,
        'input_parent_sha256': parents, 'result_sha256': {p.name: sha(p) for p in root.glob('*.json') if p.name != 'verification.json'},
        'provenance_timing': 'post-computation verification of existing immutable source records, not preregistration',
        'verifier_sha256': sha(__file__),
    })


if __name__ == '__main__': verify_kernels()
