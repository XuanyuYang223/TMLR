"""Conditionally run a fresh fixed-budget factorial after visible feasibility."""
import argparse
import json
from pathlib import Path
import time

from . import two_step_relation_joint as joint
from . import two_step_relation_joint_verify as replay
from .longrun_engine import atomic_json
from .permworld_combinations import sha


CONFIG = Path('configs/two_step_relation_coverage_confirmation.json')
PILOT = Path('results/two_step_relation_coverage_pilot')


def register():
    joint.CONFIG = CONFIG
    plan, parent, config, root, sig = joint.initialize()
    file = root / 'conditional_confirmation_protocol.json'
    signature = {'code_sha256': sha(__file__), 'joint_code_sha256': sha(joint.__file__),
                 'replay_code_sha256': sha(replay.__file__), 'plan': plan,
                 'pilot_protocol_sha256': sha(PILOT / 'coverage_pilot_protocol.json'),
                 'start_gate': 'Run all24 fresh fixed10-epoch fits only if the registered single-source visible-coverage pilot passes native>=0.95 and generators>=0.90. Otherwise retain the failed feasibility result and generate no compound predictions.',
                 'setting_selection': 'SourceLR0.0003 chosen using earlier visible-only grades; increased coverage and10-epoch endpoint fixed before this pilot finishes. No compound test outcome selects these settings.'}
    if file.exists():
        assert json.loads(file.read_text())['signature'] == signature
    else:
        assert not (root / 'data_manifest.json').exists()
        atomic_json(file, {'registered_utc': joint.core.now(), 'pilot_result_observed': (PILOT / 'visible_coverage_results.json').exists(),
                          'new_confirmation_outcomes_observed': False, 'signature': signature})
        (root / 'conditional_confirmation_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return plan, parent, config, root, sig


def run(phase):
    plan, parent, config, root, sig = register()
    if phase == 'register':
        return
    if phase == 'run':
        while not (PILOT / 'visible_coverage_results.json').exists():
            if time.time() >= joint.datetime.fromisoformat(plan['deadline_utc']).timestamp():
                raise TimeoutError('Visible feasibility did not finish before10AM')
            time.sleep(10)
        result = json.loads((PILOT / 'visible_coverage_results.json').read_text())
        if not result['feasible']:
            atomic_json(root / 'state.json', {'status': 'not_started_visible_feasibility_failed',
                        'pilot_result_sha256': sha(PILOT / 'visible_coverage_results.json'), 'updated_utc': joint.core.now(),
                        'compound_test_opened': False, 'new_confirmation_fits': 0})
            return
        joint.run()
        replay.run()
        from . import two_step_relation_coverage_statistics_verify as statistics
        statistics.run()
    elif phase == 'verify':
        replay.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['register', 'run', 'verify'])
    run(parser.parse_args().phase)
