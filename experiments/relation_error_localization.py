"""Exploratory, exact affine-error accounting on completed model caches."""
import json
from pathlib import Path

import numpy as np

from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now, FLAGS
from .two_step_relation_final_report import table, NAMES

ROOT = Path('results/relation_error_localization')
STUDIES = {'10': Path('results/two_step_relation_coverage_confirmation'),
           '20': Path('results/two_step_relation_dose_confirmation')}


def row_projection(weight):
    _, singular, vt = np.linalg.svd(weight, full_matrices=False)
    keep = singular > singular.max() * 1e-10
    return vt[keep].T @ vt[keep]


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    signature = {'code_sha256': sha(__file__), 'completed_study_summary_sha256': {
        key: sha(path / 'summary.json') for key, path in STUDIES.items()},
        'methods': ['exact hidden error = propagated first error + true-intermediate second error',
                    'readout-rowspace and readout-nullspace decomposition of first error',
                    'oracle intermediate component repairs: diagnostics only',
                    'all four conditions, six fitting repeats, both fixed endpoints and both test splits'],
        'scope': 'Exploratory analysis registered after aggregate test results. No fitting, intervention training, checkpoint selection, or evidence of output-independent mechanism. Numeric readout nullspace removes linear numeric logits, not all answer information.'}
    protocol = ROOT / 'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())['signature'] == signature
    else:
        atomic_json(protocol, {'registered_utc': now(), 'aggregate_test_results_previously_observed': True,
                               'signature': signature})
        (ROOT / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    records = []
    identities = 0
    maximum_error = 0.
    for endpoint, folder in STUDIES.items():
        data = dict(np.load(folder / 'dataset/test/dataset.npz'))
        previous = json.loads((folder / 'evaluation_records.json').read_text())['records']
        for replicate in range(6):
            for condition in FLAGS:
                name = f'n{replicate}_hidden_{condition}'
                h = np.load(folder / 'features' / (name + '_test.npz'))['hidden'].astype(float)
                maps = dict(np.load(folder / 'maps' / (name + '.npz')))
                a, b = maps['rho_c'], maps['rho_i']
                w, bias = maps['readout_weight'].astype(float), maps['readout_bias'].astype(float)
                first = h[:, 0] @ a + maps['bias_c']
                predicted = first @ b + maps['bias_i']
                forced = h[:, 1] @ b + maps['bias_i']
                delta = first - h[:, 1]
                propagated = delta @ b
                second = forced - h[:, 5]
                total = predicted - h[:, 5]
                p = row_projection(w)
                visible_error, null_error = delta @ p, delta @ (np.eye(len(p)) - p)
                repaired_visible = (first - visible_error) @ b + maps['bias_i']
                repaired_null = (first - null_error) @ b + maps['bias_i']
                np.testing.assert_allclose(total, propagated + second, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(null_error @ w.T, 0, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(predicted @ w.T - forced @ w.T,
                                           visible_error @ b @ w.T + null_error @ b @ w.T,
                                           atol=1e-9, rtol=1e-9)
                identities += 3
                maximum_error = max(maximum_error, float(np.abs(total - propagated - second).max()))
                outputs = {'composed': predicted, 'true_intermediate': forced,
                           'repair_C_readout_rows': repaired_visible, 'repair_C_readout_null': repaired_null}
                hits = {k: (v @ w.T + bias).argmax(-1) == data['labels'][:, 5] for k, v in outputs.items()}
                c_correct = (first @ w.T + bias).argmax(-1) == data['labels'][:, 1]
                i_correct = ((h[:, 0] @ b + maps['bias_i']) @ w.T + bias).argmax(-1) == data['labels'][:, 4]
                per_row = {'total_hidden_sq': np.square(total).sum(1),
                           'propagated_first_sq': np.square(propagated).sum(1),
                           'second_at_true_C_sq': np.square(second).sum(1),
                           'cross_term': 2 * (propagated * second).sum(1),
                           'visible_propagated_logit_sq': np.square(visible_error @ b @ w.T).sum(1),
                           'null_propagated_logit_sq': np.square(null_error @ b @ w.T).sum(1)}
                np.testing.assert_allclose(per_row['total_hidden_sq'], per_row['propagated_first_sq'] +
                                           per_row['second_at_true_C_sq'] + per_row['cross_term'], atol=1e-8, rtol=1e-10)
                identities += 1
                np.savez_compressed(ROOT / f'{endpoint}_{name}_diagnostic.npz', **per_row, **{
                    k + '_correct': v for k, v in hits.items()}, c_correct=c_correct, i_correct=i_correct)
                for split_id, split in [(0, 'iid'), (1, 'collisions')]:
                    use = data['split'] == split_id
                    denominator = np.square(h[use, 5] - h[use, 0]).sum()
                    row = {'endpoint_epochs': int(endpoint), 'replicate': f'n{replicate}',
                           'source_seed': [5081, 6091, 7103][replicate % 3],
                           'condition': condition, 'split': split,
                           'C_accuracy': float(c_correct[use].mean()), 'I_accuracy': float(i_correct[use].mean()),
                           **{key + '_accuracy': float(value[use].mean()) for key, value in hits.items()},
                           'true_C_correct_but_composed_wrong': float((hits['true_intermediate'][use] & ~hits['composed'][use]).mean()),
                           'composed_correct_but_true_C_wrong': float((hits['composed'][use] & ~hits['true_intermediate'][use]).mean()),
                           'C_answer_correct_but_composed_wrong': float((c_correct[use] & ~hits['composed'][use]).mean()),
                           **{key + '_normalized': float(value[use].sum() / denominator) for key, value in per_row.items()
                              if key in ['total_hidden_sq', 'propagated_first_sq', 'second_at_true_C_sq', 'cross_term']},
                           'logit_error_null_energy_fraction': float(per_row['null_propagated_logit_sq'][use].sum() /
                               max(per_row['null_propagated_logit_sq'][use].sum() + per_row['visible_propagated_logit_sq'][use].sum(), 1e-30))}
                    prior = next(r for r in previous if (r['replicate'], r['condition'], r['split'], r['word']) ==
                                 (f'n{replicate}', condition, split, 'ci'))
                    assert abs(row['composed_accuracy'] - prior['accuracy']) < 1e-12
                    assert abs(row['true_intermediate_accuracy'] - prior['teacher_forced_accuracy']) < 1e-12
                    records.append(row)
    means = []
    numeric = [k for k in records[0] if k not in ['endpoint_epochs', 'replicate', 'source_seed', 'condition', 'split']]
    for endpoint in [10, 20]:
        for condition in FLAGS:
            for split in ['iid', 'collisions']:
                selected = [r for r in records if (r['endpoint_epochs'], r['condition'], r['split']) == (endpoint, condition, split)]
                assert len(selected) == 6
                means.append({'endpoint_epochs': endpoint, 'condition': condition, 'split': split,
                              **{k: float(np.mean([r[k] for r in selected])) for k in numeric}})
    atomic_json(ROOT / 'results.json', {'status': 'complete', 'completed_utc': now(), 'records': records, 'means': means,
               'exact_affine_identities_checked': identities, 'maximum_hidden_identity_error': maximum_error,
               'original_accuracy_checks': len(records) * 2, 'source_initializations': 3,
               'oracle_repairs_are_diagnostics': True, 'output_independent_mechanism_established': False})
    pieces = ['<h1>现有模型的复合误差定位</h1><p>事后诊断；不训练、不选择模型。修复方法使用真实中间表征，只用于定位，不能计为隐藏推断成功。</p>']
    for endpoint in [10, 20]:
        selected = [r for r in means if r['endpoint_epochs'] == endpoint and r['split'] == 'collisions']
        pieces.append('<h2>' + str(endpoint) + '轮：碰撞测试</h2>' + table(
            ['几何配对', '单步C', '单步I', '预测C后接I', '真实C后接I', '仅修复C读出方向', '仅修复C读出零空间'],
            [[NAMES[r['condition']]] + [f'{100*r[k]:.2f}%' for k in ['C_accuracy', 'I_accuracy', 'composed_accuracy',
              'true_intermediate_accuracy', 'repair_C_readout_rows_accuracy', 'repair_C_readout_null_accuracy']] for r in selected]))
        pieces.append(table(['几何配对', '总隐藏误差', '第一步传播误差', '真实C上第二步误差', '交叉项'],
            [[NAMES[r['condition']]] + [f'{r[k]:.4f}' for k in ['total_hidden_sq_normalized', 'propagated_first_sq_normalized',
              'second_at_true_C_sq_normalized', 'cross_term_normalized']] for r in selected]))
    pieces.append('<p>平方误差包含交叉项，不能把前两项相加当作百分比贡献。零空间只针对当前固定数字读出的全部线性logits，不排除非线性答案信息，也不证明独立于输出的机制。六次拟合复用三个来源。输出空间旧对照采用冻结编码器，与联合训练对照的优化、参数量不同，保留为信息诊断。</p><p><a href="results.json">逐模型数据与恒等式复核</a></p>')
    (ROOT / 'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:16px/1.7 system-ui;max-width:1250px;margin:35px auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}</style>' + ''.join(pieces))
    atomic_json(ROOT / 'completion.json', {'status': 'complete', 'completed_utc': now(),
                'artifact_sha256': {str(p): sha(p) for p in ROOT.iterdir() if p.is_file() and p.name != 'completion.json'}})
    print(json.dumps({'status': 'complete', 'report': str(ROOT / 'report.html'), 'means': [r for r in means if r['condition'] == 'both_correct' and r['split'] == 'collisions']}), flush=True)


if __name__ == '__main__':
    run()
