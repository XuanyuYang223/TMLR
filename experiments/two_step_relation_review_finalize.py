"""Assemble the review after fitting, replay and uncertainty audits finish."""
import json
from pathlib import Path
import time

from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now
from . import two_step_relation_report as report


def run():
    root = report.ROOT
    root.mkdir(parents=True, exist_ok=True)
    confirmation = Path('results/two_step_relation_coverage_confirmation')
    required = ['completion.json', 'uncertainty.json', 'factorial_effects.json']
    while not all((confirmation / f).exists() for f in required):
        state = json.loads((confirmation / 'state.json').read_text()) if (confirmation / 'state.json').exists() else {}
        if state.get('status') in ['failed', 'not_started_visible_feasibility_failed']:
            atomic_json(root / 'state.json', {'status': 'confirmation_not_complete', 'confirmation_state': state, 'updated_utc': now()})
            return
        atomic_json(root / 'state.json', {'status': 'waiting_confirmation_and_audits', 'updated_utc': now(),
                    'completed_fits': len(list((confirmation / 'fits').glob('*.json'))), 'planned_fits': 24})
        time.sleep(10)
    for f in required:
        assert json.loads((confirmation / f).read_text())['status'] == 'complete'
    assert json.loads((confirmation / 'verification.json').read_text())['all24_readouts_preserved']
    assert json.loads((confirmation / 'statistics_verification.json').read_text())['all24_fits_completed_before_test_opened']
    report.run()
    summaries = [(folder / 'summary.json') for _, _, folder in report.STUDIES] + [confirmation / 'summary.json']
    files = [root / 'report.html', root / 'results.json', root / 'project_test_verification.json',
             Path('results/two_step_relation_review_project_tests.log')]
    for folder in [p.parent for p in summaries]:
        files.extend(folder / f for f in ['protocol.json', 'completion.json', 'summary.json',
                     'verification.json', 'statistics_verification.json', 'data_verification.json'])
    files.extend(confirmation / f for f in ['uncertainty.json', 'uncertainty_draws.npz', 'factorial_effects.json',
                 'uncertainty_protocol.json', 'factorial_effects_protocol.json', 'conditional_confirmation_protocol.json'])
    pilot = Path('results/two_step_relation_coverage_pilot')
    files.extend(pilot / f for f in ['visible_coverage_results.json', 'visible_coverage_verification.json', 'coverage_pilot_protocol.json'])
    files.extend(Path(f) for f in ['experiments/two_step_relation_report.py', 'experiments/two_step_relation_review_finalize.py',
                                  'experiments/two_step_relation_coverage_pilot.py',
                                  'experiments/two_step_relation_coverage_confirmation.py',
                                  'experiments/two_step_relation_coverage_uncertainty.py',
                                  'experiments/two_step_relation_coverage_factorial_effects.py'])
    atomic_json(root / 'completion.json', {'status': 'complete', 'completed_utc': now(),
                'completed_factorial_studies': 3, 'factorial_fits': sum(json.loads(p.read_text())['operator_fits'] for p in summaries),
                'visible_only_feasibility_fits': 4, 'independent_replay_and_budget_audits_complete': True,
                'source_and_input_uncertainty_complete': True, 'artifact_sha256': {str(p): sha(p) for p in files}})
    atomic_json(root / 'state.json', {'status': 'complete', 'updated_utc': now()})
    print(json.dumps({'status': 'complete', 'report': str(root / 'report.html'), 'completed_utc': now()}), flush=True)


if __name__ == '__main__':
    run()
