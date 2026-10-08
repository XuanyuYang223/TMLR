"""Visible-state feasibility tuning; never opens a compound test archive."""
from datetime import datetime, timezone
import json
from pathlib import Path

from . import two_step_relation_joint as joint
from .longrun_engine import atomic_json
from .native_confirmation import setup
from .permworld_combinations import sha


def run():
    import torch
    base = json.loads(Path('configs/two_step_relation_joint.json').read_text())
    parent_root = Path(base['output']).resolve()
    registration = parent_root / 'visible_feasibility_protocol.json'
    signature = {'code_sha256': sha(__file__), 'parent_joint_code_sha256': sha(joint.__file__),
                 'candidate_source_learning_rates': [.0001, .0003], 'fixed_epochs': 100,
                 'operator_learning_rate': .001, 'only_condition': 'both_correct', 'only_replicate': 'n0',
                 'selection_rule': 'Native e/C/I validation all>=0.95 and generator C/I validation both>=0.90. Choose smaller sourceLR if both pass. If neither passes, no successful-generator claim or confirmatory compound interpretation.',
                 'selection_data': 'Existing joint n0 visible training/validation only. All compound test files and scores are excluded.'}
    if registration.exists():
        assert json.loads(registration.read_text())['signature'] == signature
    else:
        atomic_json(registration, {'registered_utc': datetime.now(timezone.utc).isoformat(),
                    'joint_compound_test_opened_at_registration': (parent_root / 'test_opened.json').exists(), 'signature': signature})
        (parent_root / 'visible_feasibility_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    results = []
    for rate, identifier in [(.0001, 'lr1e4'), (.0003, 'lr3e4')]:
        plan = {**base, 'output': f'results/two_step_joint_visible_{identifier}', 'epochs': 100,
                'primary_epoch': 100, 'checkpoint_epochs': [25, 50, 100], 'source_learning_rate': rate,
                'scope': 'Visible-only feasibility pilot after the frozen assay and early joint visible grades; no new compound evaluation. Same n0 source/data across two registered rates.'}
        config_path = Path(f'configs/two_step_joint_visible_{identifier}.json')
        if config_path.exists():
            assert json.loads(config_path.read_text()) == plan
        else:
            config_path.write_text(json.dumps(plan, indent=2) + '\n')
        joint.CONFIG = config_path
        plan, parent, config, root, sig = joint.initialize()
        for name in ['dataset/n0/support', 'features/n0_support.npz', 'features/n0_support.json']:
            dest = root / name; dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(parent_root / name, target_is_directory=name.endswith('support'))
        atomic_json(root / 'state.json', {'status': 'running', 'stage': 'visible_only_fit', 'updated_utc': joint.core.now()})
        _, _, tokens, _ = setup(config)
        joint.train_one(plan, parent, config, root, sig, 0, 'both_correct', tokens,
                        'cuda' if torch.cuda.is_available() else 'cpu')
        record = json.loads((root / 'fits/n0_hidden_both_correct.json').read_text())
        grade = record['curve'][-1]
        passed = min(grade['native_e_C_I_accuracy']) >= .95 and min(grade['generator_C_I_accuracy']) >= .90
        results.append({'source_learning_rate': rate, 'epochs': 100, 'visible_grade': grade, 'feasible': passed,
                        'fit_record_sha256': sha(root / 'fits/n0_hidden_both_correct.json'), 'compound_test_loaded': False})
        atomic_json(root / 'state.json', {'status': 'complete', 'compound_test_loaded': False, 'updated_utc': joint.core.now()})
    passing = [row for row in results if row['feasible']]
    selected = min([row['source_learning_rate'] for row in passing]) if passing else None
    atomic_json(parent_root / 'visible_feasibility_results.json', {'completed_utc': joint.core.now(), 'records': results,
                'selected_source_learning_rate': selected, 'selected_epochs': 100, 'compound_test_used_for_selection': False})


if __name__ == '__main__':
    run()
