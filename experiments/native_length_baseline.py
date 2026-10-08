"""Post hoc practical support-only length baselines; validation-only tuning."""
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import numpy as np

from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .six_hour_report import write_rows


def smooth_prediction(train_n, labels, query_n, bandwidth, classes):
    logweights = -.5*((query_n[:, None]-train_n[None, :])/bandwidth)**2
    weights = np.exp(logweights-logweights.max(1, keepdims=True))
    return (weights@np.eye(classes)[labels]).argmax(1)


def majority_predictions(train_n, labels, query_n):
    majority = int(np.bincount(labels).argmax())
    global_prediction = np.full(len(query_n), majority)
    conditional = global_prediction.copy()
    for n in np.unique(train_n):
        conditional[query_n == n] = np.bincount(labels[train_n == n]).argmax()
    return global_prediction, conditional


def run():
    plan = json.loads(Path('configs/six_hour_session.json').read_text())
    config = json.loads(Path(plan['base_config']).read_text()); root = Path(plan['output'])
    output = root/'length_baselines'; output.mkdir(exist_ok=True)
    names = json.loads((root/'dataset/metadata.json').read_text())['names']
    support = json.loads((root/'dataset/support_indices.json').read_text())
    protocol = {'code_sha256': sha(__file__), 'data_sha256': sha(root/'dataset/data.npz'),
                'scope': 'post hoc after partial native test results; practical support-only length reference; source-training target labels never supplied',
                'bandwidths': [.5, 1., 2., 4., 8., 16.],
                'selection': 'one bandwidth selected by mean target-validation accuracy over all four targets and two budgets; smaller bandwidth breaks ties',
                'controls': ['support global majority', 'support exact-length majority with global fallback', 'Gaussian-smoothed support length category counts'],
                'no_test_based_rule_selection': True}
    path = output/'protocol.json'
    if path.exists(): assert json.loads(path.read_text())['protocol'] == protocol
    else: atomic_json(path, {'registered_utc': datetime.now(timezone.utc).isoformat(), 'protocol': protocol})
    with np.load(root/'dataset/data.npz') as archive:
        # No source training labels are read.
        data = {split: {'lengths': archive[split+'_lengths'], 'labels': archive[split+'_labels']}
                for split in ('support_pool', 'target_validation', 'target_test')}
    validation = []
    for target, budget, bandwidth in product(config['target_tasks'], config['target_budgets'], protocol['bandwidths']):
        ids = support[str(budget)]; t = names.index(target)
        predicted = smooth_prediction(data['support_pool']['lengths'][ids], data['support_pool']['labels'][ids, t],
                                      data['target_validation']['lengths'], bandwidth, config['lengths'][1]+1)
        validation.append({'target': target, 'budget': budget, 'bandwidth': bandwidth,
                           'validation_accuracy': float(np.mean(predicted == data['target_validation']['labels'][:, t]))})
    scores = [{'bandwidth': b, 'validation_macro': float(np.mean([r['validation_accuracy'] for r in validation if r['bandwidth'] == b]))} for b in protocol['bandwidths']]
    selected = max(scores, key=lambda r: (r['validation_macro'], -r['bandwidth']))
    atomic_json(output/'selection.json', {'selected': selected, 'scores': scores,
                                        'frozen_before_baseline_test_predictions_utc': datetime.now(timezone.utc).isoformat()})
    rows = []
    for target, budget in product(config['target_tasks'], config['target_budgets']):
        ids = support[str(budget)]; t = names.index(target)
        train_n = data['support_pool']['lengths'][ids]; train_y = data['support_pool']['labels'][ids, t]
        query_n = data['target_test']['lengths']; truth = data['target_test']['labels'][:, t]
        predictions = list(majority_predictions(train_n, train_y, query_n))
        predictions.append(smooth_prediction(train_n, train_y, query_n, selected['bandwidth'], config['lengths'][1]+1))
        for mode, prediction in zip(('global_majority', 'exact_length_majority', 'smooth_length'), predictions):
            rows.append({'target': target, 'budget': budget, 'mode': mode,
                         'test_accuracy': float(np.mean(prediction == truth))})
    write_rows(output/'validation.csv', validation); write_rows(output/'test.csv', rows)
    summaries = [{'budget': budget, 'mode': mode, 'test_macro': float(np.mean([r['test_accuracy'] for r in rows if (r['budget'], r['mode']) == (budget, mode)]))}
                 for budget, mode in product(config['target_budgets'], ('global_majority', 'exact_length_majority', 'smooth_length'))]
    atomic_json(output/'summary.json', {'status': 'complete', 'protocol': protocol, 'selected': selected,
                                      'deterministic_baseline_not_independent_seed_replications': True, 'endpoints': rows, 'macro': summaries})
    table = ''.join(f"<tr><td>{r['target']}</td><td>{r['budget']}</td><td>{r['mode']}</td><td>{100*r['test_accuracy']:.2f}%</td></tr>" for r in rows)
    (output/'report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>只看长度的支持集基线</title><style>body{{max-width:1000px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}</style><h1>只看排列长度：模型成绩是否超出简单统计</h1><p>这是在部分原生测试已有结果后追加的实用对照。仅使用相同 64/256 个支持输入的长度与目标标签；不读取源训练目标标签，不看排列内容，不训练源模型。三个基线均保留：全局多数类、同长度多数类（无支持时回退全局），以及按长度高斯平滑的支持标签计数。一个统一带宽在四目标两预算的平均验证准确率上从六个候选选择，之后才计算这些基线的测试预测。选出带宽 {selected['bandwidth']}。</p><p>确定性结果在三个初始化种子间完全相同，不能算三次独立复制。超过长度基线也不足以证明具体代数算法，低于该基线则不宜把绝对准确率描述为有用的排列理解。</p><table><tr><th>目标</th><th>标签</th><th>基线</th><th>测试准确率</th></tr>{table}</table><p><a href="protocol.json">追加范围</a> · <a href="selection.json">全部验证与冻结规则</a> · <a href="test.csv">全部测试</a> · <a href="summary.json">汇总</a></p></html>''')
    print(json.dumps({'selected': selected, 'macro': summaries}, indent=2))


if __name__ == '__main__': run()
