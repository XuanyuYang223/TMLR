"""Preserve prior completed work and authenticate the finite functional assay."""
import json
from pathlib import Path
import re

from .inverse_functional_alignment import initialize, now
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .specialist_cka_finalize import Links


def run():
    plan, root, sig = initialize()
    assert sha(Path(plan['previous_study']) / 'completion.json') == sig['prior_completion_sha256']
    previous = json.loads((Path(plan['previous_study']) / 'completion.json').read_text())
    older = json.loads(Path('results/output_information_followup/completion.json').read_text())
    for completed in [previous, older]:
        for filename, digest in completed['artifact_sha256'].items(): assert sha(filename) == digest, filename
    for filename, digest in sig['dependencies_sha256'].items(): assert sha(filename) == digest, filename
    for filename, digest in sig['upstream_sha256'].items(): assert sha(Path(plan['upstream_python']) / 'neurips_permutations' / filename) == digest
    prior_sources = []
    for folder in [plan['previous_study'], 'results/frozen_operator_constraint_ablation']:
        prior_sources.extend(json.loads((Path(folder) / 'protocol.json').read_text())['signature']['sources'])
    for source in prior_sources: assert sha(source['checkpoint']) == source['checkpoint_sha256']
    auxiliary = json.loads((root / 'auxiliary_protocol.json').read_text())
    assert sha('experiments/inverse_functional_auxiliary.py') == auxiliary['code_sha256']
    analysis = json.loads((root / 'analysis_protocol.json').read_text())
    assert sha('experiments/inverse_functional_report.py') == analysis['analysis_code_sha256']
    assert sha('experiments/inverse_functional_verify.py') == analysis['verification_code_sha256']
    assert analysis['registered_utc'] < json.loads((root / 'test_opened.json').read_text())['opened_utc']
    verification = json.loads((root / 'verification.json').read_text()); assert verification['status'] == 'passed'
    assert verification['accuracy_endpoints_checked'] == 60 and verification['validation_selections_replayed'] == 30
    assert verification['all_30_fits_complete_before_test_opening']
    diagnostic_protocol = json.loads((root / 'diagnostics/protocol.json').read_text())
    assert diagnostic_protocol['rules']['exploratory_post_test']
    assert sha('experiments/inverse_functional_diagnostics.py') == diagnostic_protocol['code_sha256']
    diagnostic = json.loads((root / 'diagnostics/summary.json').read_text())
    assert diagnostic['independent_gram_covariance_checks'] == 1650 and diagnostic['target_models_changed'] == 0
    for entry in diagnostic['records']:
        file = root / 'diagnostics' / f"{entry['replicate']}_{entry['condition']}_{entry['endpoint']}.npz"
        assert sha(file) == entry['support_features_sha256']
    report = (root / 'report.html').read_text()
    start, end = '<!-- post-test fit diagnostic start -->', '<!-- post-test fit diagnostic end -->'
    if start in report: report = report[:report.index(start)] + report[report.index(end) + len(end):]
    values = diagnostic['means']['correct_alignment']['selected']
    ordinary = diagnostic['means']['ordinary']['selected']
    section = start + '<h2>拟合与新输入泛化：事后诊断</h2><p>本节在主实验测试成绩揭晓后追加；没有重训、调参或改变检查点。控制偏置线性CKA的样本数依赖后，每长度相同12例：正确对齐的训练CKA为 ' + f"{values['matched_count_train_cka']:.3f}" + '，验证CKA为 ' + f"{values['matched_count_validation_cka']:.3f}" + '，新测试CKA为 ' + f"{values['matched_count_test_cka']:.3f}" + '；普通训练的新测试CKA为 ' + f"{ordinary['matched_count_test_cka']:.3f}" + '。这描述了训练集对齐拟合与新输入泛化之间的差距，不能据此声称高泛化CKA没有功能价值。</p><p><a href="diagnostics/report.html">完整事后诊断</a> · <a href="diagnostics/protocol.json">诊断规则与时间</a> · <a href="diagnostics/summary.json">逐种子逐长度核验结果</a></p>' + end
    assert '<h2>数据与核验</h2>' in report
    report = report.replace('<h2>数据与核验</h2>', section + '<h2>数据与核验</h2>')
    (root / 'report.html').write_text(report)
    assert len(list((root / 'evaluations').glob('r*_*.json'))) == 30
    assert len(list((root / 'training').glob('r*_*.json'))) == 30
    assert len(list((root / 'checkpoints').glob('*_selected.pt'))) == 30
    assert len(list((root / 'checkpoints').glob('*_latest.pt'))) == 30
    log = Path('results/inverse_functional_alignment_tests.log'); text = log.read_text()
    tests = int(re.search(r'(\d+) passed', text)[1]); assert tests >= 128 and 'failed' not in text
    links = Links(); links.feed((root / 'report.html').read_text())
    for rel in links.paths:
        if rel != 'completion.json': assert (root / rel).is_file(), rel
    summary = json.loads((root / 'summary.json').read_text())
    files = [p for p in root.rglob('*') if p.is_file() and p.name != 'completion.json']
    files += [Path(p) for p in ['configs/inverse_functional_alignment.json', 'experiments/inverse_functional_alignment.py',
        'experiments/inverse_functional_auxiliary.py', 'experiments/inverse_functional_verify.py',
        'experiments/inverse_functional_report.py', 'experiments/inverse_functional_finalize.py',
        'experiments/inverse_functional_diagnostics.py',
        'tests/test_inverse_functional_alignment.py', str(log),
        'results/inverse_functional_alignment_data.log', 'results/inverse_functional_alignment_teachers.log',
        'results/inverse_functional_alignment_train.log', 'results/inverse_functional_alignment_evaluate.log',
        'results/inverse_functional_alignment_auxiliary.log', 'results/inverse_functional_alignment_verify.log',
        'results/inverse_functional_alignment_diagnostics.log', 'results/inverse_functional_alignment_report.log']]
    completion = {'status': 'complete', 'completed_utc': now(), 'source_models_retrained': 0,
        'target_models_trained': 30, 'target_labels_including_validation_per_replicate': 256,
        'target_training_labels_per_replicate': 192, 'fresh_target_test_examples': 2560,
        'accuracy_endpoints_independently_checked': 60, 'validation_selected_models_independently_replayed': 30,
        'original_inputs_independently_rescanned': verification['data']['original_inputs_rescanned'],
        'secondary_cka_gram_scores_independently_checked': verification['gram_scores_independently_checked'],
        'post_test_diagnostic_gram_scores_independently_checked': diagnostic['independent_gram_covariance_checks'],
        'post_test_diagnostic_exploratory': True,
        'project_tests_passed': tests, 'prior_specialist_completed_artifact_hashes_preserved': len(previous['artifact_sha256']),
        'older_output_followup_artifact_hashes_preserved': len(older['artifact_sha256']),
        'original_source_checkpoints_unchanged': len(prior_sources),
        'spontaneous_algebra_discovery_confirmed': False,
        'mean_primary_accuracy': {c: v['mean_accuracy'] for c, v in summary['conditions']['selected'].items()},
        'primary_contrast_mean_pp': {c: v['mean_pp'] for c, v in summary['primary_and_final_contrasts']['selected'].items()},
        'primary_contrasts_positive_at_all_three_teacher_clusters': summary['positive_at_all_three_teacher_clusters'],
        'report_local_links_verified': len(links.paths), 'artifact_sha256': {str(p): sha(p) for p in sorted(set(files))}}
    atomic_json(root / 'completion.json', completion)
    for rel in links.paths: assert (root / rel).is_file(), rel
    print(json.dumps({k: v for k, v in completion.items() if k != 'artifact_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__': run()
