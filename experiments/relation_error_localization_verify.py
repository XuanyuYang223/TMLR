"""Independent fixed-readout invariance and per-input oracle-repair checks."""
import json
from pathlib import Path
import numpy as np
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import FLAGS, now
from .overnight_inverse_statistics_verify import bootstrap_interval

ROOT = Path('results/relation_error_localization_verification')


def run():
    ROOT.mkdir(exist_ok=True)
    old = Path('results/relation_error_localization')
    completion = json.loads((old / 'completion.json').read_text())
    for file, digest in completion['artifact_sha256'].items():
        assert sha(file) == digest
    rows = []
    max_change = 0.
    for endpoint, study in [(10, 'coverage'), (20, 'dose')]:
        folder = Path(f'results/two_step_relation_{study}_confirmation')
        data = dict(np.load(folder / 'dataset/test/dataset.npz'))
        for i in range(6):
            for condition in FLAGS:
                name = f'n{i}_hidden_{condition}'
                h = np.load(folder / 'features' / f'{name}_test.npz')['hidden'].astype(float)
                m = dict(np.load(folder / 'maps' / f'{name}.npz'))
                first = h[:, 0] @ m['rho_c'] + m['bias_c']
                error = first - h[:, 1]
                w = m['readout_weight'].astype(float)
                # Independent least-squares projection, rather than the original SVD projector.
                visible = np.linalg.lstsq(w.T, error.T, rcond=1e-10)[0].T @ w
                repaired = h[:, 1] + visible
                before_logits = first @ w.T + m['readout_bias']
                after_logits = repaired @ w.T + m['readout_bias']
                np.testing.assert_allclose(before_logits, after_logits, atol=1e-9, rtol=1e-9)
                assert np.array_equal(before_logits.argmax(-1), after_logits.argmax(-1))
                max_change = max(max_change, float(np.abs(before_logits-after_logits).max()))
                before = ((first @ m['rho_i'] + m['bias_i']) @ w.T + m['readout_bias']).argmax(-1)
                after = ((repaired @ m['rho_i'] + m['bias_i']) @ w.T + m['readout_bias']).argmax(-1)
                before_hit = before == data['labels'][:, 5]
                after_hit = after == data['labels'][:, 5]
                saved = dict(np.load(old / f'{endpoint}_{name}_diagnostic.npz'))
                assert np.array_equal(before_hit, saved['composed_correct'])
                assert np.array_equal(after_hit, saved['repair_C_readout_null_correct'])
                for sid, split in [(0, 'iid'), (1, 'collisions')]:
                    use = data['split'] == sid
                    rows.append({'epochs': endpoint, 'replicate': i, 'source_index': i%3,
                        'condition': condition, 'split': split,
                        'repaired_accuracy': float(after_hit[use].mean()),
                        'oracle_improvement_pp': float(100*(after_hit[use].mean()-before_hit[use].mean())),
                        'wrong_to_correct': int((use & ~before_hit & after_hit).sum()),
                        'correct_to_wrong': int((use & before_hit & ~after_hit).sum()),
                        'answers_changed': int((use & (before != after)).sum()),
                        'first_step_all_logits_preserved': True})
    contrasts=[]
    for epochs in [10,20]:
        for c in FLAGS:
            r=[x for x in rows if (x['epochs'],x['condition'],x['split'])==(epochs,c,'collisions')]
            values=[float(np.mean([x['oracle_improvement_pp'] for x in r if x['source_index']==i])) for i in range(3)]
            contrasts.append({'epochs':epochs,'condition':c,'oracle_improvement_pp':float(np.mean(values)),
                              'three_source_bootstrap_95_pp':bootstrap_interval(values)})
    atomic_json(ROOT/'verification.json',{'status':'complete','completed_utc':now(),
        'old_completion_artifacts_unchanged':len(completion['artifact_sha256']),
        'real_model_invariance_checks':48,'replayed_accuracy_vectors':96,
        'max_all31_first_step_logits_change':max_change,'records':rows,'contrasts':contrasts,
        'oracle_repair_scope':'Uses true intermediate hidden state; cannot count as hidden inference. Preserves all31 first-step numeric logits, not all possible answer information.',
        'energy_scope':'The original logit_error_null_energy_fraction divides null squared energy by null+row squared energy, excluding their cross term; it is not an additive causal fraction.',
        'uncertainty_scope':'Three-source bootstrap conditional on fixed evaluation samples; six fits reuse three sources.',
        'code_sha256':sha(__file__), 'original_completion_sha256':sha(old/'completion.json')})
    print(json.dumps({'status':'complete','checks':48,'max_logits_change':max_change}),flush=True)


if __name__=='__main__':run()
