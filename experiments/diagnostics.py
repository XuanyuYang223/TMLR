"""Post-hoc symbolic-composition diagnostic; never a learned-transfer claim."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .algebra import scenarios, world
from .models import SourceModel


def symbolic_composition(source_answers, head_order, targets, p):
    native = source_answers[:, np.argsort(head_order)]
    return native[:, :3] @ targets.T % p


def diagnose(output):
    output = Path(output)
    config = json.loads((output / 'metadata.json').read_text())['config']
    targets = np.array(json.loads((output / 'algebra_audit.json').read_text())['targets'])
    groups = scenarios()
    groups = [g for pair in zip(groups[:4], groups[4:]) for g in pair][:config['scenario_limit']]
    rows = []
    torch.set_num_threads(4)
    for w in config['world_seeds']:
        _, latent, encoded, basis = world(config['p'], config['dimension'], w)
        x = torch.tensor(encoded)
        test = np.load(output / f'world_{w}.npz')['test']
        truth = latent @ targets.T % config['p']
        input_supports = np.count_nonzero(targets @ basis[:3] % config['p'], axis=1).tolist()
        for m in config['model_seeds']:
            for g in groups:
                rid = f"{g['id']}_w{w}_m{m}"
                result = json.loads((output / f'{rid}.json').read_text())
                model = SourceModel(encoded.shape[1], config['hidden'], config['features'], config['p'])
                state = torch.load(output / 'checkpoints' / f'{rid}.pt', map_location='cpu', weights_only=True)
                model.load_state_dict(state)
                with torch.no_grad():
                    source_answers = model(x).argmax(-1).numpy()
                prediction = symbolic_composition(source_answers, result['source_head_order'], targets, config['p'])
                for t in range(len(targets)):
                    rows.append({'scenario': g['id'], 'world_seed': w, 'model_seed': m, 'target_id': t,
                                 'oracle_test_accuracy': float(np.mean(prediction[test[t], t] == truth[test[t], t])),
                                 'target_input_support_size': input_supports[t]})
    summary = {'status': 'post_hoc_diagnostic', 'evaluation': 'same target test indices as the transfer experiment',
               'uses_target_support_labels': False,
               'uses_external_mathematics': True,
               'oracle': 'decode pretrained u/v/w heads, then compute the known target linear formula modulo p',
               'mean_oracle_accuracy': float(np.mean([r['oracle_test_accuracy'] for r in rows])),
               'minimum_oracle_accuracy': min(r['oracle_test_accuracy'] for r in rows),
               'interpretation': 'Perfect oracle composition establishes that trained representations retain the required target information on this exposed input domain. Poor learned readout/adaptation does not establish information loss.',
               'not_established': ['Spontaneously learned composition', 'Unseen-input generalization', 'A causal explanation of negative transfer'],
               'rows': rows}
    (output / 'symbolic_oracle.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/pilot')
    args = parser.parse_args()
    diagnose(args.output)
