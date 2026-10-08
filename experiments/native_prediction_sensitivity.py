"""Transparent sensitivity for one constant indicator in frozen predictor.

The original code and results remain intact. Correction identified from source
features after partial testing, before complete test or any prediction fitting.
"""
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

from .longrun_engine import atomic_json
from .native_transfer_prediction import ridge_predict
from .permworld_combinations import select_groups, sha
from .six_hour_report import write_rows


def corrected_math(features):
    result = list(features['math'])
    result[1] = float(result[0] == 5)
    return result


def run():
    plan = json.loads(Path('configs/six_hour_session.json').read_text())
    config = json.loads(Path(plan['base_config']).read_text()); root = Path(plan['output'])
    groups = select_groups(config); output = root/'transfer_prediction'
    original_meta = json.loads((output/'metadata.json').read_text())
    assert original_meta['signature']['code_sha256'] == sha('experiments/native_transfer_prediction.py')
    features = json.loads((output/'features.json').read_text())
    assert all(r['math'][1] == 0 for r in features)
    signature = {'code_sha256': sha(__file__), 'original_feature_sha256': sha(output/'features.json'),
                 'original_code_sha256': sha('experiments/native_transfer_prediction.py'),
                 'change': 'source group null sentinel is 0 rather than None; intended no-known-circuit indicator is 1 exactly when minimum-size feature is 5',
                 'scope': 'bug identified from source features during partial native testing, before complete testing or predictor fitting; original frozen results preserved',
                 'policy': 'same folds, baseline, ridge, variants and all readouts/budgets; report original and corrected scores together, never choose the better'}
    path = output/'indicator_correction.json'
    if path.exists(): assert json.loads(path.read_text())['signature'] == signature
    else:
        assert not (output/'summary.json').exists()
        atomic_json(path, {'registered_utc': datetime.now(timezone.utc).isoformat(), 'signature': signature,
                           'corrected_feature_rows': sum(r['math'][0] == 5 for r in features)})
    while not (output/'summary.json').exists():
        if time.time() >= datetime.fromisoformat(plan['deadline_utc']).timestamp()-240: return
        time.sleep(20)
    old = json.loads((output/'summary.json').read_text())
    endpoints = [{**row, 'group': record['group'], 'seed': record['seed']} for p in (root/'transfer').glob('*.json')
                 for record in [json.loads(p.read_text())] if record['status'] == 'complete' for row in record['rows']]
    random = {(r['seed'], r['target'], r['budget'], r['mode']): r['test_accuracy'] for r in endpoints if r['group'] == 'random'}
    lookup = {(r['group'], r['seed'], r['target']): r for r in features}
    scores, predictions = [], []
    for mode, budget in product(('linear_task_free', 'linear_query', 'mlp_task_free', 'finetune', 'finetune_shared'), (64, 256)):
        rows = [r for r in endpoints if r['group'] != 'random' and (r['mode'], r['budget']) == (mode, budget)]
        assert len(rows) == 96
        truth = np.array([r['test_accuracy']-random[r['seed'], r['target'], budget, mode] for r in rows])
        baseline = next(r['r2'] for r in old['scores'] if (r['mode'], r['budget'], r['variant']) == (mode, budget, 'baseline'))
        for variant in ('plus_mathematical_relations', 'plus_both'):
            matrix = np.array([lookup[r['group'], r['seed'], r['target']]['baseline']+corrected_math(lookup[r['group'], r['seed'], r['target']])
                               +(lookup[r['group'], r['seed'], r['target']]['geometry'] if variant == 'plus_both' else []) for r in rows])
            predicted = np.zeros(len(rows))
            for group in groups:
                mask = np.array([r['group'] == group['id'] for r in rows])
                predicted[mask] = ridge_predict(matrix[~mask], truth[~mask], matrix[mask], 1.)
            r2 = float(1-((truth-predicted)**2).sum()/((truth-truth.mean())**2).sum())
            original = next(r for r in old['scores'] if (r['mode'], r['budget'], r['variant']) == (mode, budget, variant))
            scores.append({'mode': mode, 'budget': budget, 'variant': variant, 'original_r2': original['r2'],
                           'corrected_r2': r2, 'corrected_delta_r2': r2-baseline,
                           'original_delta_r2': original['delta_r2'], 'corrected_mae': float(np.abs(truth-predicted).mean())})
            predictions.extend({'group': r['group'], 'seed': r['seed'], 'target': r['target'], 'mode': mode, 'budget': budget,
                                'variant': variant, 'actual_paired_gain': float(y), 'corrected_prediction': float(p)} for r, y, p in zip(rows, truth, predicted))
    write_rows(output/'indicator_corrected_scores.csv', scores)
    write_rows(output/'indicator_corrected_predictions.csv', predictions)
    atomic_json(output/'indicator_corrected_summary.json', {'status': 'complete', 'signature': signature, 'scores': scores,
                                                           'original_results_retained': True, 'no_score_based_version_selection': True})
    table = ''.join(f"<tr><td>{r['mode']}</td><td>{r['budget']}</td><td>{r['variant']}</td><td>{r['original_r2']:.3f}</td><td>{r['corrected_r2']:.3f}</td><td>{r['corrected_delta_r2']:+.3f}</td></tr>" for r in scores)
    (output/'indicator_correction.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>无约束指示变量修正</title><style>body{{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}</style><h1>冻结分析的一个常量特征：原版与修正版并列</h1><p>原实现把无约束标记 0 当作 None 检查，使单独的无约束指示变量恒为零；最小阶数 3/4/5 编码正确，因此原版仍含无约束信息。此错误在部分原生测试已有结果、但全部测试及预测拟合尚未完成时从源特征中发现。修正版将阶数为 5 的指示变量置为 1，其他模型、特征、折与岭惩罚保持一致。原冻结代码与结果没有覆盖，两个版本全部保留，不按得分选择。</p><table><tr><th>读出</th><th>标签</th><th>版本</th><th>原冻结 R²</th><th>修正 R²</th><th>修正相对基线 ΔR²</th></tr>{table}</table><p><a href="report.html">原冻结结果</a> · <a href="indicator_correction.json">发现与修正规则</a> · <a href="indicator_corrected_scores.csv">所有版本对照</a> · <a href="indicator_corrected_predictions.csv">修正预测</a></p></html>''')
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == '__main__': run()
