"""Finish unchanged schedule verification with eager archive loading."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from . import overnight_inverse_budget_check as budget
from . import overnight_inverse_campaign as campaign
from .longrun_engine import atomic_json
from .permworld_combinations import sha

ROOT = Path('results/overnight_inverse_functional')


def run():
    original_load = np.load
    checks = 0
    for folder in [ROOT, Path('results/overnight_inverse_residual_geometry')]:
        for file in sorted((folder / 'training').glob('*_schedule.npz')):
            with original_load(file) as archive:
                eager = {key: archive[key] for key in archive.files}
                for key, value in eager.items():
                    # Compare real row reads from the unchanged lazy API.
                    for index in [0, len(value) // 2, len(value) - 1]:
                        assert np.array_equal(value[index], archive[key][index])
                        checks += 1
    atomic_json(ROOT / 'budget_cache_protocol.json', {
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'wrapper_sha256': sha(__file__),
        'original_verifier_sha256': sha(budget.__file__),
        'real_lazy_and_eager_schedule_row_checks': checks,
        'change': 'Load each numeric NPZ once into arrays instead of repeatedly decompressing whole members on row access. Invoke the original verifier, RNG schedule reconstruction, assertions and output unchanged.',
        'scope': 'Verification performance only. No training, fitting choices, data, model parameters, metrics or test selection change.'})

    def eager_load(*args, **kwargs):
        value = original_load(*args, **kwargs)
        if isinstance(value, np.lib.npyio.NpzFile):
            try:
                return {key: value[key] for key in value.files}
            finally:
                value.close()
        return value

    atomic_json(ROOT / 'campaign_state.json', {'status': 'running', 'stage': 'verify_budgets_cached',
                                              'updated_utc': datetime.now(timezone.utc).isoformat()})
    np.load = eager_load
    try:
        budget.run()
    finally:
        np.load = original_load
    plan = json.loads(Path('configs/overnight_inverse_functional.json').read_text())
    deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
    atomic_json(ROOT / 'campaign_state.json', {'status': 'fits_and_checks_complete_timed_goal_active',
                'updated_utc': datetime.now(timezone.utc).isoformat(), 'deadline_utc': plan['deadline_utc']})
    while time.time() < deadline:
        time.sleep(min(10, max(.1, deadline - time.time())))
    campaign.final_manifest()
    atomic_json(ROOT / 'campaign_state.json', {'status': 'complete',
                                              'updated_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'status': 'complete', 'verification_only_cache_wrapper': True}), flush=True)


if __name__ == '__main__':
    run()
