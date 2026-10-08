"""Group-heldout exploratory gain predictions, fixed before native testing."""
import argparse
import csv
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

from .algebra import categorical_mi
from .longrun_engine import atomic_json
from .native_transform_audit import certificates
from .permworld_combinations import select_groups, sha
from .six_hour_report import write_rows


def read_rows(path):
    with Path(path).open() as handle: return list(csv.DictReader(handle))


def entropy(values):
    _, counts = np.unique(values, axis=0, return_counts=True)
    p = counts/counts.sum()
    return float(-(p*np.log(p)).sum())


def label_statistics(source, target, lengths):
    marginal, joint, correlation, maximum_mi, mean_mi = [], [], [], [], []
    for n in np.unique(lengths):
        left, right = source[lengths == n], target[lengths == n]
        marginal.append(np.mean([entropy(left[:, j]) for j in range(left.shape[1])]))
        joint.append(entropy(left))
        matrix = np.corrcoef(left.T)
        correlation.append(np.mean(np.abs(np.nan_to_num(matrix[np.triu_indices(left.shape[1], 1)]))))
        values = [categorical_mi(left[:, j], right) for j in range(left.shape[1])]
        maximum_mi.append(max(values)); mean_mi.append(np.mean(values))
    return [float(np.mean(v)) for v in (marginal, joint, correlation, maximum_mi, mean_mi)]


def ridge_predict(x, y, test, alpha=1.):
    mean, sigma = x.mean(0), np.maximum(x.std(0), 1e-8)
    train, test = (x-mean)/sigma, (test-mean)/sigma
    ymean = y.mean()
    coefficients = np.linalg.solve(train.T@train+alpha*np.eye(train.shape[1]), train.T@(y-ymean))
    return test@coefficients+ymean


def build_features(plan, config, groups):
    root = Path(plan['output'])
    names = json.loads((root/'dataset/metadata.json').read_text())['names']
    with np.load(root/'dataset/data.npz') as archive:
        labels, lengths = archive['train_labels'], archive['train_lengths']
    code = read_rows(root/'native_transform_audit/categorical_kernel.csv')
    cross_seed = read_rows(root/'landmarks/cross_seed.csv')
    transformed = read_rows(root/'landmarks/transformed_cka.csv')
    arch = json.loads((root/'architecture_selection.json').read_text())['selected']
    rows = []
    for group, seed, target in product(groups, plan['model_seeds'], config['target_tasks']):
        run_id = f"{arch['id']}_{group['id']}_s{seed}"
        record = json.loads((root/'multi'/f'{run_id}.json').read_text())
        assert record['status'] == 'complete'
        audit = np.mean([r['accuracy'] for r in record['source_audit']])
        curve = np.mean([r['accuracy'] for r in record['curve']])
        statistics = label_statistics(labels[:, [names.index(t) for t in group['tasks']]], labels[:, names.index(target)], lengths)
        baseline = [audit, curve]+statistics+[float(target == t) for t in config['target_tasks'][1:]]+[float(seed == s) for s in plan['model_seeds'][1:]]
        code_cka = [float(r['categorical_code_cka']) for r in code if r['group'] == group['id'] and r['centering'] == 'within_length']
        assert len(code_cka) == 4
        actions = certificates(group)
        unary = sum(bool(np.isin(matrix, [0, 1]).all() and (matrix.sum(0) == 1).all() and (matrix.sum(1) == 1).all()) for matrix, _ in actions.values())
        minimum = group['minimum_constraint_size']
        mathematical = [minimum or 5, float(minimum is None), np.mean(code_cka), len(actions)/4, unary/4]
        seed_cells = [float(r['trained_cka']) for r in cross_seed if r['group'] == group['id'] and r['landmark'] == 'source_query_concat' and r['centering'] == 'within_length' and seed in (int(r['seed_a']), int(r['seed_b']))]
        operator_cells = [float(r['trained_cka']) for r in transformed if r['group'] == group['id'] and int(r['seed']) == seed and r['landmark'] == 'source_query_concat' and r['centering'] == 'within_length']
        assert len(seed_cells) == 2 and len(operator_cells) == 4
        geometry = [np.mean(seed_cells), np.mean(operator_cells)]
        rows.append({'group': group['id'], 'seed': seed, 'target': target,
                     'baseline': [float(v) for v in baseline], 'math': [float(v) for v in mathematical],
                     'geometry': [float(v) for v in geometry]})
    return rows


def complete_records(root):
    records = [json.loads(p.read_text()) for p in (root/'transfer').glob('*.json')]
    return [r for r in records if r['status'] == 'complete']


def run(wait=False):
    plan = json.loads(Path('configs/six_hour_session.json').read_text())
    config = json.loads(Path(plan['base_config']).read_text()); groups = select_groups(config)
    root = Path(plan['output']); output = root/'transfer_prediction'; output.mkdir(exist_ok=True)
    protocol_path = root/'native_transfer_prediction_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    signature = {'code_sha256': sha(__file__), 'protocol_sha256': sha(protocol_path),
                 'input_sha256': {str(p): sha(p) for p in [root/'dataset/data.npz', root/'landmarks/cross_seed.csv', root/'landmarks/transformed_cka.csv', root/'native_transform_audit/categorical_kernel.csv']}}
    metadata_path = output/'metadata.json'
    if metadata_path.exists(): assert json.loads(metadata_path.read_text())['signature'] == signature
    else:
        assert not list((root/'transfer').glob('*.json')), 'First-time feature freeze requires no target test records'
        atomic_json(metadata_path, {'signature': signature, 'feature_code_frozen_utc': datetime.now(timezone.utc).isoformat(), 'target_test_records_at_feature_freeze': 0})
    features_path = output/'features.json'
    if not features_path.exists(): atomic_json(features_path, build_features(plan, config, groups))
    while len(complete_records(root)) != 27:
        if not wait: raise RuntimeError('All 27 native target evaluations must finish first')
        if time.time() >= datetime.fromisoformat(plan['deadline_utc']).timestamp()-300:
            atomic_json(output/'state.json', {'status': 'not_run_incomplete_primary_targets', 'completed_conditions': len(complete_records(root))}); return
        atomic_json(output/'state.json', {'status': 'waiting_for_all_primary_test_conditions', 'completed_conditions': len(complete_records(root))})
        time.sleep(20)
    features = json.loads(features_path.read_text())
    endpoints = [{**row, 'group': record['group'], 'seed': record['seed']} for record in complete_records(root) for row in record['rows']]
    random = {(r['seed'], r['target'], r['budget'], r['mode']): r['test_accuracy'] for r in endpoints if r['group'] == 'random'}
    lookup = {(r['group'], r['seed'], r['target']): r for r in features}
    predictions, scores, folds = [], [], []
    for mode, budget in product(('linear_task_free', 'linear_query', 'mlp_task_free', 'finetune', 'finetune_shared'), config['target_budgets']):
        rows = [r for r in endpoints if r['group'] != 'random' and (r['mode'], r['budget']) == (mode, budget)]
        assert len(rows) == 96
        truth = np.array([r['test_accuracy']-random[r['seed'], r['target'], budget, mode] for r in rows])
        for variant in protocol['variants']:
            matrix = []
            for row in rows:
                f = lookup[row['group'], row['seed'], row['target']]
                matrix.append(f['baseline']+(f['math'] if variant in ('plus_mathematical_relations', 'plus_both') else [])+(f['geometry'] if variant in ('plus_measured_geometry', 'plus_both') else []))
            matrix = np.array(matrix); predicted = np.zeros(len(rows))
            for group in groups:
                test = np.array([r['group'] == group['id'] for r in rows])
                assert test.sum() == 12
                predicted[test] = ridge_predict(matrix[~test], truth[~test], matrix[test], protocol['ridge_alpha'])
                folds.append({'mode': mode, 'budget': budget, 'variant': variant, 'heldout_group': group['id'],
                              'mae': float(np.abs(predicted[test]-truth[test]).mean()), 'test_rows': int(test.sum())})
            denominator = ((truth-truth.mean())**2).sum()
            score = {'mode': mode, 'budget': budget, 'variant': variant,
                     'r2': float(1-((truth-predicted)**2).sum()/denominator),
                     'mae': float(np.abs(truth-predicted).mean()), 'rows': len(rows), 'independent_group_folds': 8}
            baseline_score = next((r for r in scores if (r['mode'], r['budget'], r['variant']) == (mode, budget, 'baseline')), score)
            score['delta_r2'] = score['r2']-baseline_score['r2']; scores.append(score)
            predictions.extend({'group': r['group'], 'seed': r['seed'], 'target': r['target'], 'mode': mode, 'budget': budget, 'variant': variant,
                                'actual_paired_gain': float(y), 'predicted_paired_gain': float(p)} for r, y, p in zip(rows, truth, predicted))
    write_rows(output/'predictions.csv', predictions); write_rows(output/'scores.csv', scores); write_rows(output/'folds.csv', folds)
    metadata = json.loads(metadata_path.read_text())
    assert datetime.fromisoformat(metadata['feature_code_frozen_utc']).timestamp() < min(p.stat().st_mtime for p in (root/'transfer').glob('*.json'))
    atomic_json(output/'summary.json', {'status': 'complete', 'protocol': protocol, 'scores': scores,
                                      'predictions': len(predictions), 'feature_code_frozen_utc': metadata['feature_code_frozen_utc'],
                                      'all_27_test_models_complete': True, 'all_variants_retained': True})
    atomic_json(output/'state.json', {'status': 'complete', 'prediction_rows': len(predictions), 'scores': len(scores)})
    report(output, scores)
    print(json.dumps(scores, indent=2), flush=True)


def report(output, scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes = list(dict.fromkeys(r['mode'] for r in scores))
    variants = list(dict.fromkeys(r['variant'] for r in scores))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    for ax, budget in zip(axes, (64, 256)):
        for i, variant in enumerate(variants[1:]):
            values = [next(r['delta_r2'] for r in scores if (r['mode'], r['budget'], r['variant']) == (mode, budget, variant)) for mode in modes]
            ax.bar(np.arange(len(modes))+(i-1)*.24, values, width=.24, label=variant)
        ax.axhline(0, color='gray'); ax.set_xticks(range(len(modes)), modes, rotation=30)
        ax.set_title(f'{budget} support labels'); ax.set_ylabel('Whole-group-heldout delta R2'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output/'prediction.png', dpi=160); fig.savefig(output/'prediction.pdf'); plt.close(fig)
    table = ''.join(f"<tr><td>{r['mode']}</td><td>{r['budget']}</td><td>{r['variant']}</td><td>{r['r2']:.3f}</td><td>{r['delta_r2']:+.3f}</td><td>{100*r['mae']:.2f} pp</td></tr>" for r in scores)
    (output/'report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>整组合留出的迁移预测</title><style>body{{max-width:1200px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui}}table{{border-collapse:collapse;width:100%}}td,th{{padding:6px;border-bottom:1px solid #ddd}}img{{width:100%}}aside{{background:#f2f4f7;padding:16px}}</style>
<h1>整组留出：数学关系与表征几何能否预测迁移收益</h1><p>在本轮目标测试之前固定特征与规则，等所有 27 条件测试结束后运行。先导迁移结果及本轮源表征已知，因此仍是追加的探索，不称为整个项目的前瞻主检验。每次留出一个任务组合，包括它的全部种子及目标；五种读出、两种预算、四个特征版本全部报告，岭惩罚固定为 1。</p>
<p>预测目标是对同种子、目标、预算和读出随机模型的准确率差值。基线包括源学习程度、源输出熵与相关性、源—目标按长度条件互信息，以及目标和初始化固定效应。数学特征包括已知恒等式阶数、精确源答案变换核及可证明向量/单因子置换的比例。几何特征包括源查询拼接的跨种子 CKA 和输入变换 CKA，均逐长度去均值。特征标准化只用训练折。</p>
<img src="prediction.png"><table><tr><th>读出</th><th>标签</th><th>预测版本</th><th>留组 R²</th><th>相对基线 ΔR²</th><th>平均绝对误差</th></tr>{table}</table>
<aside>一个输入世界、八个组合、三个初始化，样本行彼此相关。只留组并不能确认跨领域有效或证明数学关系的因果作用。标签相似性分析使用源训练输入上未用于源优化的真目标标签，属于知道任务定义的分析参照。完整联合输出统计及源难度没有严格匹配。负 R² 表示预测比整体均值还差；仅 ΔR² 为正并不意味着预测可靠。</aside><p><a href="../native_transfer_prediction_protocol.json">测试前规则</a> · <a href="metadata.json">冻结时间与指纹</a> · <a href="features.json">全量特征</a> · <a href="predictions.csv">全部预测</a> · <a href="scores.csv">全部得分</a> · <a href="folds.csv">每个留出组</a></p></html>''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--wait', action='store_true')
    run(parser.parse_args().wait)
