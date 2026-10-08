"""Fresh visible-input coverage pilot, with compound predictions kept closed."""
import json
from pathlib import Path

import numpy as np
import torch

from . import two_step_relation_joint as joint
from .longrun_engine import atomic_json
from .native_confirmation import setup
from .permworld_combinations import sha


CONFIG = Path('configs/two_step_relation_coverage_pilot.json')


def run():
    joint.CONFIG = CONFIG
    plan, parent, config, root, sig = joint.initialize()
    registration = root / 'coverage_pilot_protocol.json'
    signature = {
        'code_sha256': sha(__file__), 'joint_code_sha256': sha(joint.__file__),
        'plan': plan,
        'selection_rule': 'At the fixed final epoch, native e/C/I validation all>=0.95 and generator C/I validation both>=0.90. A passing pilot permits a fresh four-way study; a failed pilot establishes no compound failure claim.',
        'selection_data': 'Only fresh e/C/I training and validation. No compound predictions or test grades are generated or used.',
        'rationale': 'The previous100-epoch visible-only fit had near-zero training operator KD but inverse validation accuracy0.7125. Increase unique visible input coverage16-fold at a similar update count, without changing relationship geometry or output supervision.',
    }
    if registration.exists():
        assert json.loads(registration.read_text())['signature'] == signature
    else:
        assert not (root / 'data_manifest.json').exists()
        atomic_json(registration, {'registered_utc': joint.core.now(), 'new_outcomes_observed': False,
                                  'signature': signature})
        (root / 'coverage_pilot_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(root / 'state.json', {'status': 'running', 'stage': 'prepare_visible_coverage', 'updated_utc': joint.core.now()})
    joint.core.prepare_data(plan, parent, config, root, sig)
    atomic_json(root / 'state.json', {'status': 'running', 'stage': 'visible_teacher_cache', 'updated_utc': joint.core.now()})
    joint.core.cache_supports(plan, parent, config, root, sig)
    _, _, tokens, _ = setup(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    atomic_json(root / 'state.json', {'status': 'running', 'stage': 'visible_fit', 'updated_utc': joint.core.now()})
    joint.train_one(plan, parent, config, root, sig, 0, 'both_correct', tokens, device)
    record_path = root / 'fits/n0_hidden_both_correct.json'
    record = json.loads(record_path.read_text())
    checkpoint = root / 'maps' / f'n0_hidden_both_correct_e{plan["epochs"]}.pt'
    # Independent original-model replay uses only the three known states.
    model = joint.core.source_model(sig['sources'][0], parent, config, device, accelerated=False)
    original_readout = model.lm_head.weight.detach().cpu().clone()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    torch.testing.assert_close(model.lm_head.weight.detach().cpu(), original_readout, atol=0, rtol=0)
    ops = joint.AffineOperators(parent['architecture']['d_model']).to(device)
    ops.load_state_dict(state['operators'])
    raw = dict(np.load(root / 'dataset/n0/support/dataset.npz'))
    assert raw['input'].shape[1] == 3
    validation = {k: raw[k][raw['split'] == 1] for k in ['input', 'lengths', 'labels']}
    grade = joint.validate(model, ops, validation, parent, tokens)
    for key in grade:
        np.testing.assert_allclose(grade[key], record['curve'][-1][key], atol=1e-8, rtol=1e-8)
    # Fixed first512 eligible anchors per length, declared before fitting,
    # diagnose training fit. They do not select settings or epochs.
    fit_ids = np.flatnonzero(raw['split'] == 0)
    pair = np.load(root / 'dataset/n0/support/pairings.npz')
    sample = np.concatenate([fit_ids[np.flatnonzero(pair['eligible'] & (raw['lengths'][fit_ids] == n))[:512]]
                             for n in plan['lengths']])
    training_grade = joint.validate(model, ops, {k: raw[k][sample] for k in ['input', 'lengths', 'labels']}, parent, tokens)
    passed = min(grade['native_e_C_I_accuracy']) >= .95 and min(grade['generator_C_I_accuracy']) >= .90
    result = {'completed_utc': joint.core.now(), 'feasible': passed, 'fixed_epochs': plan['epochs'],
              'source_seed': sig['sources'][0]['seed'], 'visible_validation': grade,
              'fixed_training_subset_grade': training_grade, 'training_subset_examples': len(sample),
              'validation_examples': len(validation['lengths']), 'fit_record_sha256': sha(record_path),
              'checkpoint_sha256': sha(checkpoint), 'unaccelerated_validation_replayed': True,
              'numeric_readout_preserved': True, 'compound_predictions_generated': False,
              'compound_test_used_for_selection': False,
              'limitation': 'One visible-only pilot source. A pass is feasibility, not a replicated structural result.'}
    assert not (root / 'test_opened.json').exists()
    atomic_json(root / 'visible_coverage_results.json', result)
    atomic_json(root / 'state.json', {'status': 'complete', 'compound_predictions_generated': False, 'updated_utc': joint.core.now()})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
