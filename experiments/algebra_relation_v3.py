"""Registered scalar encoding; independent workers, then a sealed-test evaluation."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import multiprocessing
from pathlib import Path
import shutil

import torch

from . import algebra_relation_campaign as campaign
from . import algebra_relation_feasibility as feasibility
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now

ROOT = Path('results/algebra_relation_v3')
CONFIG = Path('configs/algebra_relation_common_v3.json')


def configure():
    campaign.ROOT = ROOT
    campaign.CONFIG = CONFIG
    feasibility.apply_encoding('numeric_gelu')
    torch.set_num_threads(1)


def worker(domain, index, condition):
    configure()
    plan = json.loads(CONFIG.read_text())
    return campaign.fit(plan, ROOT / domain, index, condition,
                        'cuda' if torch.cuda.is_available() else 'cpu')


def register_runtime(domain):
    configure()
    root = ROOT / domain
    root.mkdir(parents=True, exist_ok=True)
    evidence = Path('results/algebra_relation_feasibility/results.json')
    assert json.loads(evidence.read_text())['selected'] == 'numeric_gelu'
    signature = {
        'wrapper_sha256': sha(__file__), 'encoding_code_sha256': sha(feasibility.__file__),
        'config_sha256': sha(CONFIG), 'core_code_sha256': sha(campaign.__file__),
        'encoding': 'Four centered scalar field coordinates; ordinary GELU MLP. No mathematical operations in encoder.',
        'architecture_selection_evidence_sha256': sha(evidence),
        'architecture_selection_used_known_matrix_validation_only': True,
        'compound_results_observed': False,
        'workers': 2,
        'selection_scope': 'Matrix pilot and heldout known confirmation selected encoding. New compound tests remain sealed. Polynomial uses the same selected encoding without domain-specific compound tuning.',
        'pilot_reuse': 'Matrix source0 and known-only pilot copied exactly from encoding feasibility run; sources1/2 and all36 formal fits are new. Polynomial source and pilot new.',
    }
    file = root / 'runtime_protocol.json'
    if file.exists():
        assert json.loads(file.read_text())['signature'] == signature
    else:
        atomic_json(file, {'registered_utc': now(), 'signature': signature})
        (root / 'wrapper_snapshot.py').write_bytes(Path(__file__).read_bytes())
        (root / 'encoding_snapshot.py').write_bytes(Path(feasibility.__file__).read_bytes())
    plan, root = campaign.register(domain)
    if domain == 'matrix' and not (root / 'data_audit.json').exists():
        old = Path('results/algebra_relation_followup/matrix').resolve()
        (root / 'dataset').rmdir()
        (root / 'dataset').symlink_to(old / 'dataset', target_is_directory=True)
        shutil.copyfile(old / 'data_audit.json', root / 'data_audit.json')
        chosen = Path('results/algebra_relation_feasibility/numeric_gelu')
        for name in ['sources/s0.pt', 'sources/s0.json', 'models/pilot.pt', 'fits/pilot.json']:
            shutil.copyfile(chosen / name, root / name)
        atomic_json(root / 'reuse_provenance.json', {
            'created_utc': now(), 'immutable_original_dataset': str(old / 'dataset'),
            'original_dataset_audit_sha256': sha(old / 'data_audit.json'),
            'copied_known_only_artifacts': {name: sha(chosen / name) for name in
                ['sources/s0.pt', 'sources/s0.json', 'models/pilot.pt', 'fits/pilot.json']},
            'matrix_compound_test_previously_opened': False,
        })
    return plan, root


def run(domain):
    plan, root = register_runtime(domain)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        atomic_json(root / 'state.json', {'status': 'running', 'stage': 'known_pilot', 'updated_utc': now()})
        campaign.prepare(plan, root, domain)
        pilot = campaign.fit(plan, root, 0, 'both_correct', device, True)
        grade = pilot['curve'][-1]
        feasible = (min(grade['native']) >= plan['pilot_known_gate']['native']
                    and min(grade['generators']) >= plan['pilot_known_gate']['generators'])
        atomic_json(root / 'pilot_gate.json', {'status': 'complete', 'feasible': feasible,
                    'grade': grade, 'compound_test_not_opened': True})
        if not feasible:
            atomic_json(root / 'state.json', {'status': 'known_pilot_failed',
                        'updated_utc': now(), 'compound_test_closed': True})
            return
        for index in range(3):
            campaign.source_model(plan, root, index, device)
        # Training processes have their own seeded RNG and model/optimizer state.
        # The evaluator cannot run until every future and fixed-budget record exists.
        jobs = [(i, c) for i in range(6) for c in plan['conditions']]
        atomic_json(root / 'state.json', {'status': 'running', 'stage': 'all36_fixed_formal_fits',
                    'workers': 2, 'updated_utc': now()})
        context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=2, mp_context=context) as executor:
            pending = [executor.submit(worker, domain, i, c) for i, c in jobs]
            for future in pending:
                future.result()
        campaign.evaluate(plan, root, device)
        atomic_json(root / 'completion.json', {'status': 'complete', 'completed_utc': now(),
            'artifact_sha256': {str(p): sha(p) for p in root.rglob('*') if p.is_file()
                and p.name not in ['state.json', 'current_job.json', 'completion.json']}})
        atomic_json(root / 'state.json', {'status': 'complete', 'updated_utc': now()})
    except BaseException as error:
        atomic_json(root / 'state.json', {'status': 'failed', 'reason': repr(error), 'updated_utc': now()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('domain', choices=['matrix', 'polynomial', 'both'])
    args = parser.parse_args()
    if args.domain == 'both':
        run('matrix')
        if json.loads((ROOT / 'matrix/state.json').read_text())['status'] == 'complete':
            run('polynomial')
    else:
        run(args.domain)
