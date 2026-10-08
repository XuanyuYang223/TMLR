"""Frozen-prediction modal/non-modal audit; no fitting or test-derived modes."""
import argparse
import json
from pathlib import Path

import numpy as np

from .inverse_functional_alignment import now
from .inverse_functional_report import paired_interval, table
from .longrun_engine import atomic_json
from .permworld_combinations import sha

OLD = Path('results/inverse_alignment_coverage')
ROOT = Path('results/overnight_inverse_functional')


def stratified(hit, modal, lengths):
    """Keep the common mask; decomposition uses actual sample proportions."""
    result = {'accuracy': float(hit.mean()), 'modal_fraction': float(modal.mean())}
    for label, mask in [('modal', modal), ('nonmodal', ~modal)]:
        result[label] = {'count': int(mask.sum()), 'accuracy': float(hit[mask].mean()) if mask.any() else None}
    result['per_length'] = {str(int(n)): stratified(hit[lengths == n], modal[lengths == n], np.array([], dtype=int))
                            for n in np.unique(lengths)} if len(lengths) else {}
    expected = sum(result[k]['count'] * result[k]['accuracy'] for k in ['modal', 'nonmodal'] if result[k]['count']) / len(hit)
    assert abs(expected - result['accuracy']) < 1e-12
    return result


def analyze(data_path, prediction_root, output, fresh=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    protocol = output/'protocol.json'
    if not protocol.exists():
        atomic_json(protocol, {'registered_utc': now(), 'code_sha256': sha(__file__),
            'fresh_confirmation': fresh, 'existing_outcomes_exploratory': not fresh,
            'modes_sha256': sha(OLD/'length_priors/summary.json'),
            'definition': 'Primary masks use each repeat teacher-argmax length modes fitted on its original4096 U rows; secondary masks use192 train-label modes. Both mappings are frozen before new holdout generation. All methods within a repeat share a mask. No test-derived modes.',
            'contrasts': ['coverage_alignment-coverage_matched_mismatch', 'coverage_alignment-ordinary_exposure', 'coverage_distillation-ordinary_exposure', 'coverage_distillation_alignment-coverage_distillation'],
            'interpretation': 'Length prior has zero nonmodal accuracy by definition; positive nonmodal accuracy alone is not a relation-specific effect. Report paired differences and weighted contributions to overall accuracy, not only subgroup improvements.'})
    registered = json.loads(protocol.read_text()); assert registered['code_sha256'] == sha(__file__)
    assert registered['modes_sha256'] == sha(OLD/'length_priors/summary.json')
    data = dict(np.load(data_path)); n = data['lengths']; y = data['labels']
    sig = json.loads((OLD/'protocol.json').read_text())['signature']; conditions = sig['plan']['conditions']
    modes = json.loads((OLD/'length_priors/summary.json').read_text())['records']
    records = []; paired = []
    for prior in ['teacher_length_mode', 'labeled_length_mode']:
        for rep in sig['replicates']:
            mapping = next(r['modes'] for r in modes if r['replicate'] == rep['id'] and r['condition'] == prior)
            modal = np.array([mapping[str(int(nn))] == int(yy) for nn, yy in zip(n, y)])
            predictions = {}
            teacher_path = Path(prediction_root)/'teacher'/f"test_s{rep['source_seed']}.npz"
            predictions['teacher'] = np.load(teacher_path)['logits'].argmax(-1)
            for condition in conditions:
                file = Path(prediction_root)/'evaluations'/f"{rep['id']}_{condition}.npz"
                predictions[condition] = np.load(file)['final_logits'].argmax(-1)
            for condition, prediction in predictions.items():
                hit = prediction == y
                score = stratified(hit, modal, n)
                # Independently count all cells without NumPy boolean means.
                for group, flag in [('modal', True), ('nonmodal', False)]:
                    counted = [(int(a) == int(b)) for a, b, m in zip(prediction, y, modal) if bool(m) == flag]
                    assert sum(counted) == round(score[group]['accuracy'] * len(counted))
                records.append({'replicate': rep['id'], 'source_seed': rep['source_seed'], 'prior': prior,
                                'condition': condition, **score})
            for contrast in registered['contrasts']:
                left, right = contrast.split('-'); delta = (predictions[left] == y).astype(int) - (predictions[right] == y).astype(int)
                contributions = {group: float(delta[mask].sum() / len(delta)) for group, mask in [('modal', modal), ('nonmodal', ~modal)]}
                assert abs(sum(contributions.values()) - delta.mean()) < 1e-12
                paired.append({'replicate': rep['id'], 'source_seed': rep['source_seed'], 'prior': prior, 'contrast': contrast,
                    'overall_difference': float(delta.mean()),
                    'modal_difference': float(delta[modal].mean()), 'nonmodal_difference': float(delta[~modal].mean()),
                    'modal_contribution': contributions['modal'], 'nonmodal_contribution': contributions['nonmodal'],
                    'benefited_examples': int((delta > 0).sum()), 'harmed_examples': int((delta < 0).sum())})
    means = []
    for prior in ['teacher_length_mode', 'labeled_length_mode']:
        for condition in ['teacher'] + conditions:
            rows = [r for r in records if r['prior'] == prior and r['condition'] == condition]
            means.append({'prior': prior, 'condition': condition, 'accuracy': float(np.mean([r['accuracy'] for r in rows])),
                'modal_accuracy': float(np.mean([r['modal']['accuracy'] for r in rows])),
                'nonmodal_accuracy': float(np.mean([r['nonmodal']['accuracy'] for r in rows])),
                'modal_fraction': float(np.mean([r['modal_fraction'] for r in rows]))})
    contrasts = []
    for prior in ['teacher_length_mode', 'labeled_length_mode']:
        for contrast in registered['contrasts']:
            rows = [r for r in paired if r['prior'] == prior and r['contrast'] == contrast]
            values = {key: [r[key] for r in rows] for key in ['overall_difference', 'modal_difference', 'nonmodal_difference', 'modal_contribution', 'nonmodal_contribution']}
            clusters = {key: [float(np.mean([r[key] for r in rows if r['source_seed'] == s])) for s in [17, 42, 101]] for key in values}
            def stats(v):
                return {'mean_pp': float(100*np.mean(v)), 'deltas_pp': list(map(float,100*np.asarray(v))),
                        'positive': int(np.sum(np.asarray(v)>0)), 'bootstrap_95_pp': paired_interval(100*np.asarray(v))}
            contrasts.append({'prior': prior, 'contrast': contrast,
                              'paired_repeats': {k: stats(v) for k,v in values.items()},
                              'three_teacher_clusters': {k: stats(v) for k,v in clusters.items()}})
    summary = {'created_utc': now(), 'fresh_confirmation': fresh, 'means': means, 'records': records, 'paired': paired,
               'contrasts': contrasts, 'dataset_sha256': sha(data_path), 'modes_sha256': registered['modes_sha256'],
               'teachers': 3, 'paired_target_support_repeats': 6, 'shared_test_examples': len(y),
               'teacher_rows_repeated_for_target_pairing_not_independent_sources': True}
    atomic_json(output/'summary.json', summary)
    display = [[r['condition'], f"{100*r['accuracy']:.2f}%", f"{100*r['modal_accuracy']:.2f}%", f"{100*r['nonmodal_accuracy']:.2f}%"]
               for r in means if r['prior'] == 'teacher_length_mode']
    title = '新留出集确认' if fresh else '既有预测的事后诊断'
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+title+'</title><style>body{font-family:system-ui;max-width:1100px;margin:30px auto;padding:20px;line-height:1.7}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}</style><body><h1>'+title+'</h1><p>主要分层使用原4096无标签样本的教师长度众数，映射提前固定。非众数指真实答案不同于该映射；长度基线在这些样本上按定义为零，不能把超过零本身解释为正确关系优势。</p>'
    html += table(['方法', '整体', '众数样本', '非众数样本'], display)
    html += '<p>六个重复共用三个教师；逐重复、逐长度成绩及整体差异的加权分解见完整结果。事后诊断与新测试确认分开保存。</p><p><a href="summary.json">完整结果</a> · <a href="protocol.json">固定规则</a></p></body></html>'
    (output/'report.html').write_text(html)
    print(json.dumps({'fresh': fresh, 'primary_means': [r for r in means if r['prior'] == 'teacher_length_mode']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--fresh', action='store_true'); args = parser.parse_args()
    if args.fresh:
        analyze(ROOT/'confirmation/dataset/test/dataset.npz', ROOT/'confirmation', ROOT/'confirmation/diagnostic', True)
    else:
        analyze(OLD/'dataset/test/dataset.npz', OLD, ROOT/'existing_diagnostic')
