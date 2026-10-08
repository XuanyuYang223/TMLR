"""Finalize the fixed-deadline study after workers stop, preserving prior studies."""
import csv
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import time

from .until_10_verify import ROOT, DEADLINE, read, digest, save
from .until_10_report import run as report, source_relation_rows, ordinary_rows, confidence_rows, prefix_rows, proxy_rows
from .until_10_figures import run as figures


STUDIES = ['frozen_relation_followup', 'relation_seed_extension', 'relation_operator_stability', 'relation_observed_start',
    'paired_relation_geometry', 'answer_matched_geometry', 'relation_world_confirmation', 'joint_answer_matched_geometry',
    'ordinary_relation_seed_confirmation', 'relation_readout_diagnostics', 'task_block_geometry_control',
    'conditional_relation_specificity', 'ordinary_initialization_control', 'relation_frozen_seed_confirmation',
    'relation_error_transport_confirmation', 'known_route_readout_control', 'matched_prediction_conditioning',
    'ordinary_prefix_confirmation', 'confidence_residual_geometry', 'confidence_proxy_geometry']
VERIFY_COUNTS = {'verification': 179, 'control_verification': 216, 'transport_verification': 36,
    'route_verification': 36, 'world_fit_verification': 9, 'specificity_verification': 36,
    'conditioning_verification': 18, 'prefix_verification': 18, 'confidence_verification': 18,
    'proxy_verification': 18, 'data_provenance_verification': 28}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ['href', 'src'] and value and not value.startswith(('https:', 'http:', '#')):
                self.paths.append(value.split('#')[0])


def scientific_workers():
    modules = {('experiments.' + name).encode() for name in STUDIES}
    active = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if (p / 'cwd').resolve() == Path.cwd() and modules.intersection(args):
                active.append({'pid': int(p.name), 'module': next(m.decode() for m in modules.intersection(args))})
        except (FileNotFoundError, PermissionError):
            continue
    return active


def source_csv():
    rows = []
    for r in source_relation_rows():
        if r['condition'] == 'correct_relations' and r['method'] == 'native_operators' and r['word'] in ['ci', 'ici']:
            rows.append({'study': 'native_hidden_answer', 'cohort': r['cohort'], 'seed': r['seed'], 'group': r['condition'],
                'status': 'trained', 'case': r['word'], 'metric': 'answer_accuracy', 'value': r['answer_accuracy']})
    for stage, values in [('ordinary', ordinary_rows()), ('confidence_residual', confidence_rows()), ('prefix', prefix_rows()), ('output_proxy', proxy_rows())]:
        for r in values:
            if r['kind'] != '未拟合复合' or (stage == 'ordinary' and r['view'] != 'source_query_numeric_null'):
                continue
            rows.append({'study': stage, 'cohort': r.get('cohort', '新增普通源种子'), 'seed': r['seed'], 'group': r['group'],
                'status': r['status'], 'case': 'within_source_mean_4_composites', 'metric': 'pair_target_nmse', 'value': r['pair_target_nmse']})
            if stage == 'confidence_residual':
                rows.append({**rows[-1], 'metric': 'retained_pair_energy_fraction', 'value': r['retained_pair_energy_fraction']})
    with (ROOT / 'per_source_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['study', 'cohort', 'seed', 'group', 'status', 'case', 'metric', 'value'])
        writer.writeheader()
        writer.writerows(rows)


def run():
    assert time.time() >= DEADLINE, 'Only finalize after the agreed deadline.'
    assert not scientific_workers(), scientific_workers()
    for name in ['deadline_stop.json', 'deadline_stop_additional.json']:
        assert (ROOT / name).exists()
        assert read(ROOT / name)['deadline_utc'] == '2026-10-06T17:00:00+00:00'
    checks = {}
    for name, expected in VERIFY_COUNTS.items():
        verified = read(ROOT / (name + '.json'))
        assert verified['status'].startswith('passed'), (name, verified['status'])
        assert not verified['failures'], (name, verified['failures'])
        assert verified['checks_completed'] == expected, (name, verified['checks_completed'], expected)
        checks[name] = {'checks_completed': expected, 'independently_replayed_endpoints': verified.get('independently_replayed_endpoints'),
            'scope': verified.get('scope', 'See individual checks in the linked verification record.')}
    counts = {}
    for name, expected in [('relation_seed_extension', 18), ('relation_world_confirmation', 9), ('ordinary_relation_seed_confirmation', 9)]:
        root = Path('results') / name
        pattern = 'world*/source/*.json' if name == 'relation_world_confirmation' else 'source/*.json'
        records = [read(p) for p in root.glob(pattern)]
        assert len(records) == expected and all(r['status'] == 'complete' for r in records)
        assert all(r.get('steps', r.get('job', {}).get('steps')) == 20000 for r in records)
        counts[name] = len(records)
    assert read('results/native_ablation/state.json')['status'] == 'paused_for_research_focus_change'
    test_log = Path('results/until_10_tests_final.log')
    assert '107 passed' in test_log.read_text()
    test_scopes = read(ROOT / 'tests_scopes.json')
    assert test_scopes['current_project']['exit_code'] == 0
    assert test_scopes['external_repository_own_root']['exit_code'] == 0
    for entry in test_scopes.values():
        assert digest(entry['log_path']) == entry['log_sha256']
    figures()
    source_csv()
    report()
    parser = Links()
    parser.feed((ROOT / 'report.html').read_text())
    for name in parser.paths:
        assert (ROOT / name).is_file(), name
    for stem in ['relation_accuracy', 'ordinary_geometry', 'confidence_geometry', 'confidence_proxy_geometry']:
        assert (ROOT / (stem + '.pdf')).read_bytes().startswith(b'%PDF')
        assert (ROOT / (stem + '.png')).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    artifacts = {}
    for p in ROOT.iterdir():
        if p.is_file() and p.name != 'completion.json' and ('cache' not in p.name) and p.suffix in ['.json', '.html', '.csv', '.png', '.pdf']:
            artifacts[str(p)] = digest(p)
    for study in STUDIES:
        root = Path('results') / study
        for p in [root / 'protocol.json', *root.glob('source/*.json'), *root.glob('world*/source/*.json'),
                *root.glob('evaluations/*.json'), *root.glob('world*/evaluations/*.json'), *root.glob('world*/observed_start/*.json')]:
            artifacts[str(p)] = digest(p)
    artifacts[str(test_log)] = digest(test_log)
    for entry in test_scopes.values():
        artifacts[entry['log_path']] = entry['log_sha256']
    completion = {'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'deadline_local': '2026-10-06 10:00 America/Los_Angeles', 'deadline_utc': '2026-10-06T17:00:00+00:00',
        'new_source_models': sum(counts.values()), 'source_model_counts': counts, 'source_steps_each': 20000,
        'total_direct_source_label_exposures': 27 * 1920000 + 9 * 2560000,
        'extra_relation_edges_are_separate_from_direct_label_count': True, 'matched_untrained_ordinary_controls': 9,
        'partial_sources_counted_as_completed': 0, 'scientific_workers_remaining': [], 'project_tests_passed': 107,
        'test_command_scopes': test_scopes,
        'independent_verification': checks, 'local_report_links_checked': len(parser.paths),
        'prior_completed_studies_preserved': True, 'LIS_branch_still_paused': True,
        'stable_hidden_relation_inference_confirmed': False, 'output_confidence_independent_algebra_mechanism_confirmed': False,
        'interpretation': 'Ordinary records-task query geometry replicates on new source seeds, but its advantage does not survive the declared output-confidence association regression. An output-only-at-prediction proxy also reproduces the geometric prediction against the same original hidden targets, with an additional decoder fitted to calibration hidden vectors. These exploratory controls do not establish confidence as the sole causal mechanism. Explicit true relations beat wrong pairings on identical frozen backbones, while new source seeds and worlds do not establish stable above-ceiling hidden-answer composition.',
        'artifact_sha256': artifacts,
        'final_analysis_and_verifier_code_sha256': {str(p): digest(p) for p in sorted(Path('experiments').glob('until_10_*.py'))}}
    save(ROOT / 'completion.json', completion)
    print({'status': completion['status'], 'new_source_models': completion['new_source_models'],
        'checks': sum(v['checks_completed'] for v in checks.values()), 'report_links': len(parser.paths)}, flush=True)
    return completion


if __name__ == '__main__':
    run()
