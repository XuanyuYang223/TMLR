"""CPU-only paired initial features, independent of the active GPU trainer."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .longrun_attention import accelerate
from .longrun_engine import atomic_json
from .permworld_combinations import features, new_model, sha


def run(plan_path='configs/six_hour_session.json'):
    plan = json.loads(Path(plan_path).read_text())
    config = json.loads(Path(plan['base_config']).read_text())
    root = Path(plan['output'])
    arch = json.loads((root/'architecture_selection.json').read_text())['selected']
    config.update({k: arch[k] for k in ('d_model', 'layers', 'heads')})
    sys.path.insert(0, str(Path(config['repository'])/'src'))
    from neurips_permutations.passage import TOKEN_TO_ID
    torch.set_num_threads(4)
    with np.load(root/'dataset/data.npz') as archive:
        data = {k: torch.tensor(archive[k]) for k in ('representation_input', 'representation_lengths')}
    baseline = root/'initial'
    baseline.mkdir(exist_ok=True)
    signature = {'architecture': arch, 'seeds': plan['model_seeds'], 'data_sha256': sha(root/'dataset/data.npz'),
                 'training_code_sha256': sha('experiments/permworld_combinations.py'),
                 'attention_code_sha256': sha('experiments/longrun_attention.py'),
                 'execution': 'untrained, CPU FP32, eval mode, ONE_END task-free features'}
    for seed in plan['model_seeds']:
        model = accelerate(new_model(config, seed, 'cpu'))
        feature = features(model, data, 'representation', TOKEN_TO_ID)
        np.save(baseline/f's{seed}_features.npy', feature)
    signature['feature_sha256'] = {p.name: sha(p) for p in baseline.glob('*.npy')}
    atomic_json(baseline/'metadata.json', signature)
    print(json.dumps({'status': 'complete', 'initial_models': len(plan['model_seeds'])}))


if __name__ == '__main__':
    run()
