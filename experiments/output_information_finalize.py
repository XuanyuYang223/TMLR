"""Finalize finite saved-model analyses without rewriting the completed campaign."""
from datetime import datetime, timezone
from pathlib import Path

from .until_10_verify import read, digest, save
from .until_10_finalize import Links
from .output_information_report import ROOT, run as report


def run():
    studies = ['budget_matched_geometry', 'collision_pair_followup']
    counts = {}
    for name, expected in [('budget_matched_geometry', 18), ('collision_pair_followup', 504)]:
        folder = Path('results') / name
        protocol = read(folder / 'protocol.json')
        assert digest('experiments/' + name + '.py') == protocol['signature']['code_sha256']
        if 'config_sha256' in protocol['signature']:
            assert digest('configs/' + name + '.json') == protocol['signature']['config_sha256']
        verified = read(folder / 'verification.json')
        assert verified['status'].startswith('passed') and verified['checks_completed'] == expected and not verified['failures']
        assert verified['verifier_sha256'] == digest('experiments/output_information_verify.py')
        assert read(folder / 'state.json')['status'] == 'complete'
        counts[name] = expected
    previous = read('results/until_10_followup/completion.json')
    for path, expected in previous['artifact_sha256'].items():
        assert digest(path) == expected, path
    for path, expected in previous['final_analysis_and_verifier_code_sha256'].items():
        assert digest(path) == expected, path
    checkpoints = {}
    for name, pattern in [('relation_seed_extension', 'source/*.json'), ('relation_world_confirmation', 'world*/source/*.json'),
            ('ordinary_relation_seed_confirmation', 'source/*.json')]:
        for p in (Path('results')/name).glob(pattern):
            r = read(p)
            cp = p.parent / 'checkpoints' / (p.stem + '.pt') if name == 'ordinary_relation_seed_confirmation' else p.parent.parent / 'checkpoints' / (p.stem + '.pt')
            assert digest(cp) == r['checkpoint_sha256']
            checkpoints[str(cp)] = r['checkpoint_sha256']
    assert len(checkpoints) == 36
    assert '113 passed' in Path('results/output_information_tests.log').read_text()
    summary = report()
    assert summary['budget_verification']['checks_completed'] == 18 and summary['pair_verification']['checks_completed'] == 504
    parser = Links()
    parser.feed((ROOT/'report.html').read_text())
    for p in parser.paths:
        assert (ROOT/p).is_file(), p
    for name in ['matched_budget', 'collision_pairs']:
        assert (ROOT/(name+'.png')).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert (ROOT/(name+'.pdf')).read_bytes().startswith(b'%PDF')
    artifacts = {}
    for folder in [ROOT, *[Path('results')/n for n in studies]]:
        for p in folder.rglob('*'):
            if p.is_file() and p.name != 'completion.json':
                artifacts[str(p)] = digest(p)
    for p in ['configs/budget_matched_geometry.json', 'experiments/budget_matched_geometry.py',
            'experiments/collision_pair_followup.py', 'experiments/output_information_verify.py',
            'experiments/output_information_report.py', 'experiments/output_information_finalize.py',
            'tests/test_budget_pair_followup.py', 'results/output_information_tests.log']:
        artifacts[p] = digest(p)
    save(ROOT/'completion.json', {'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'source_models_retrained': 0, 'existing_budget_models_including_initializations': 18,
        'verified_record_counts': counts, 'project_tests_passed': 113, 'test_command': '.venv/bin/python -m pytest -q tests',
        'test_cwd': str(Path.cwd()), 'previous_completed_artifact_hashes_preserved': len(previous['artifact_sha256']),
        'previous_completed_source_checkpoints_rechecked': checkpoints,
        'matched_budget_primary_fixed_input_loadings': True,
        'fitted_matrix_size_matching_equals_causal_mechanism_matching': False,
        'stable_above_50_hidden_inference_confirmed': False,
        'partial_hidden_answer_orientation_information_supported': True,
        'output_independent_algebraic_mechanism_confirmed': False,
        'interpretation': 'Under the declared matched fitting matrices, calibration rows and search budgets, the full-feature output surrogate remains better than the hidden-input reconstruction on records sources. Source learned operators exhibit partial collision-pair target orientation, but neither-correct pairs outweigh both-correct pairs and aggregate inference stays below 50% on new six seeds. These exploratory, externally probed results do not establish a unique internal mechanism or spontaneous discovery of algebraic rules.',
        'artifact_sha256': artifacts, 'report_local_links_verified': len(parser.paths)})
    print({'status': 'complete', 'models_retrained': 0, 'verified_records': counts, 'tests_passed': 113,
        'preserved_previous_artifacts': len(previous['artifact_sha256']), 'report_links': len(parser.paths)}, flush=True)


if __name__ == '__main__':
    run()
