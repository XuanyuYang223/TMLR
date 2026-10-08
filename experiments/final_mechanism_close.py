"""Seal the completed fixed confirmation after scientific and document audits."""
import json
from pathlib import Path

from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now

ROOT = Path('results/final_mechanism_confirmation')


def run():
    result = json.loads((ROOT / 'results.json').read_text())
    verification = json.loads((ROOT / 'independent_verification.json').read_text())
    document = json.loads((ROOT / 'paper_audit.json').read_text())
    assert result['all_source_gates_pass'] and result['all_convex_certificates_pass']
    assert verification['status'] == 'passed' and document['status'] == 'passed'
    assert document['pdf_sha256'] == sha('paper/main_final.pdf')
    assert result['source_models'] == 5 and result['same_budget_models'] == 20
    assert result['convex_fits'] == 20
    assert len(list((ROOT / 'predictor_fits').glob('*.json'))) == 20
    assert len(list((ROOT / 'affine_fits').glob('*.json'))) == 20
    tests = (ROOT / 'pytest.log').read_text().strip()
    assert '174 passed' in tests and 'failed' not in tests.lower()

    previous = {}
    for study in ['algebra_relation_review', 'readout_null_confirmation',
                  'operator_capacity_confirmation', 'operator_mechanism_diagnostic']:
        manifest = Path('results') / study / 'delivery.json'
        entries = json.loads(manifest.read_text())['artifact_sha256']
        for filename, expected in entries.items():
            assert sha(filename) == expected, f'Prior artifact changed: {filename}'
        previous[study] = {'artifacts_preserved': len(entries), 'delivery_sha256': sha(manifest)}

    effects = {r['contrast']: r for r in result['contrasts'] if r['endpoint'] == 'accuracy'}
    conclusion = (
        'Five new sources confirm the same-budget correctness benefit and predicted-null '
        'component transfer. Adequately converged affine maps under the same single-step '
        'objective recover most of the relation-specific composition benefit. An additional '
        'functional expressivity advantage was not confirmed; ill-conditioning, precision '
        'sensitivity and state-fitting differences remain limitations.'
    )
    completed = now()
    atomic_json(ROOT / 'state.json', {
        'status': 'complete', 'updated_utc': completed, 'experimental_scope_closed': True,
        'source_models': 5, 'same_budget_models': 20, 'same_loss_convex_solves': 20,
        'remaining_stage': 'Manuscript refinement and submission preparation.',
    })
    atomic_json(ROOT / 'closure.json', {
        'closed_utc': completed, 'predeclared_checks_completed': True,
        'ending_rule': result['ending_rule'], 'no_postoutcome_experiment_expansion': True,
        'conclusion': conclusion, 'previous_artifacts': previous,
        'remaining_stage': 'Manuscript refinement and submission preparation.',
    })
    files = [p for p in ROOT.rglob('*') if p.is_file() and p.name != 'delivery.json']
    files += [Path(p) for p in [
        'configs/final_mechanism_confirmation.json',
        'experiments/final_mechanism_train.py', 'experiments/final_mechanism_solver.py',
        'experiments/final_mechanism_evaluate.py', 'experiments/final_mechanism_verify.py',
        'experiments/final_mechanism_report.py', 'experiments/final_mechanism_close.py',
        'tests/test_final_mechanism.py', 'paper/main_final.tex',
        'paper/generated_final_mechanism.tex', 'paper/main_final.pdf',
        'paper/main_final.log', 'results/final_mechanism_paper_build.log',
        'results/final_mechanism_sources.log', 'results/final_mechanism_fits.log',
        'results/final_mechanism_evaluate.log',
    ]]
    hashes = {str(p): sha(p) for p in sorted(set(files))}
    atomic_json(ROOT / 'delivery.json', {
        'status': 'complete', 'completed_utc': completed, 'experimental_scope_closed': True,
        'source_models': 5, 'same_budget_models': 20, 'same_loss_convex_solves': 20,
        'independent_verification_checks': verification['checks'],
        'pytest': tests.splitlines()[-1], 'previous_artifacts': previous,
        'primary_contrasts': {k: effects[k] for k in [
            'same_budget_interaction', 'correct_null_swap_gain',
            'correct_vs_wrong_null_donor', 'remaining_relation_interaction']},
        'conclusion': conclusion, 'report': str(ROOT / 'report.html'),
        'manuscript_source': 'paper/main_final.tex', 'manuscript_pdf': 'paper/main_final.pdf',
        'latex_compiled': True, 'paper_audit': str(ROOT / 'paper_audit.json'),
        'artifact_sha256': hashes,
    })
    for filename, expected in hashes.items():
        assert sha(filename) == expected
    print(json.dumps({'status': 'complete', 'experimental_scope_closed': True,
                      'sealed_artifacts': len(hashes), 'manuscript_pdf': 'paper/main_final.pdf'}), flush=True)


if __name__ == '__main__':
    run()
