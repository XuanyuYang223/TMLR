"""Explicitly post-test check that fitted alignment generalizes to fresh inputs."""
import json
from pathlib import Path

import numpy as np
import torch

from .inverse_functional_alignment import initialize, configure, now
from .inverse_functional_verify import direct, gram
from .inverse_functional_report import LABELS, table
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .specialist_cka_controls import api


RULES = {
    'exploratory_post_test': True,
    'reason': 'Primary accuracy is not improved and predeclared fresh-test CKA is weak despite low minibatch geometry loss. Check fit versus input generalization without further training.',
    'models': 'All six exact initializations and all thirty selected/final models; no selection or parameter changes.',
    'features': 'Task-free ONE_END, frozen inverse teacher. All conditions use the identical geometry-eligible training rows; validation uses every row.',
    'comparison': 'Report full train/validation CKA and a matched-count comparison using the first 12 eligible train, first 12 validation and first 12 fresh test rows per length. Equal count avoids treating the sample-size bias of linear CKA as fit generalization.',
    'aggregate': 'Five lengths averaged equally; retain every replicate and endpoint. Fit accuracy uses all 192 train labels; validation uses 64.',
    'scope': 'Implementation and overfitting diagnostic, not a new confirmatory hypothesis or hyperparameter search.',
}


def measured(hidden, teacher, support, eligible, test_h, test_teacher, test, sizes):
    rows = []; errors = []
    for n in sizes:
        train = np.flatnonzero((support['split'] == 0) & (support['lengths'] == n) & eligible)
        val = np.flatnonzero((support['split'] == 1) & (support['lengths'] == n))
        new = np.flatnonzero(test['lengths'] == n)
        assert min(len(train), len(val), len(new)) >= 12
        pairs = {'full_train_cka': (hidden[train], teacher[train]), 'full_validation_cka': (hidden[val], teacher[val]),
                 'matched_count_train_cka': (hidden[train[:12]], teacher[train[:12]]),
                 'matched_count_validation_cka': (hidden[val[:12]], teacher[val[:12]]),
                 'matched_count_test_cka': (test_h[new[:12]], test_teacher[new[:12]])}
        scores = {}
        for name, (x, y) in pairs.items():
            scores[name] = gram(x, y)
            a = torch.tensor(x, dtype=torch.float64); b = torch.tensor(y, dtype=torch.float64)
            a -= a.mean(0); b -= b.mean(0)
            covariance = float((a.T @ b).square().sum() / torch.sqrt((a.T @ a).square().sum() * (b.T @ b).square().sum()))
            np.testing.assert_allclose(covariance, scores[name], atol=2e-12, rtol=2e-12)
            errors.append(abs(covariance - scores[name]))
        rows.append({'length': n, 'train_geometry_rows': len(train), 'validation_rows': len(val), 'matched_count': 12, **scores})
    return rows, errors


def run():
    plan, root, sig = initialize(); device = configure(); _, tokens, _, TrainConfig, factory = api(plan)
    assert json.loads((root / 'verification.json').read_text())['status'] == 'passed'
    folder = root / 'diagnostics'; folder.mkdir(exist_ok=True); protocol = folder / 'protocol.json'
    if not protocol.exists():
        atomic_json(protocol, {'registered_utc': now(), 'rules': RULES, 'code_sha256': sha(__file__),
                              'primary_summary_sha256_before_diagnostic': sha(root / 'summary.json'), 'all_target_training_complete': True})
    else:
        registered = json.loads(protocol.read_text()); assert registered['rules'] == RULES and registered['code_sha256'] == sha(__file__)
    test = dict(np.load(root / 'dataset/test/dataset.npz')); records = []; errors = []
    for rep in plan['replicates']:
        support = dict(np.load(root / 'dataset' / rep['id'] / 'dataset.npz'))
        eligibility = np.load(root / 'dataset' / rep['id'] / 'mismatch.npz')['eligible']
        teacher = np.load(root / 'teacher' / f"{rep['id']}.npz")['hidden']
        test_teacher = np.load(root / 'teacher' / f"test_s{rep['source_seed']}.npz")['hidden']
        source = next(s for s in sig['sources'] if s['seed'] == rep['source_seed'])
        cp = torch.load(source['checkpoint'], weights_only=True, map_location='cpu'); cfg = TrainConfig.from_value(cp['config']); del cp
        for condition in ['initialization'] + plan['conditions']:
            endpoints = ['initialization'] if condition == 'initialization' else ['selected', 'latest']
            for endpoint in endpoints:
                file = folder / f"{rep['id']}_{condition}_{endpoint}.npz"
                if not file.exists():
                    model = factory(cfg)
                    path = root / 'initializations' / f"{rep['id']}.pt" if condition == 'initialization' else root / 'checkpoints' / f"{rep['id']}_{condition}_{endpoint}.pt"
                    cp = torch.load(path, weights_only=True, map_location='cpu'); model.load_state_dict(cp if condition == 'initialization' else cp['model']); del cp
                    model.to(device); out = direct(model, support, plan['target_task'], tokens, 0, np.arange(len(support['lengths'])))
                    np.savez_compressed(file, **out); del model
                out = dict(np.load(file))
                test_path = root / 'evaluations' / (f"{rep['id']}_initialization.npz" if condition == 'initialization' else f"{rep['id']}_{condition}.npz")
                test_key = 'hidden' if condition == 'initialization' else endpoint + '_hidden'
                test_h = np.load(test_path)[test_key]
                scores, check = measured(out['hidden'], teacher, support, eligibility, test_h, test_teacher, test, plan['lengths']); errors.extend(check)
                hit = out['logits'].argmax(-1) == support['labels']
                records.append({'replicate': rep['id'], 'condition': condition, 'endpoint': endpoint,
                    'train_accuracy': float(hit[support['split'] == 0].mean()), 'validation_accuracy': float(hit[support['split'] == 1].mean()),
                    'per_length': scores, 'support_features_sha256': sha(file)})
    means = {}
    for condition in ['initialization'] + plan['conditions']:
        means[condition] = {}
        for endpoint in (['initialization'] if condition == 'initialization' else ['selected', 'latest']):
            subset = [r for r in records if r['condition'] == condition and r['endpoint'] == endpoint]
            means[condition][endpoint] = {k: float(np.mean([np.mean([s[k] for s in r['per_length']]) for r in subset])) for k in records[0]['per_length'][0] if k.endswith('_cka')}
            for k in ['train_accuracy', 'validation_accuracy']: means[condition][endpoint][k] = float(np.mean([r[k] for r in subset]))
    result = {'completed_utc': now(), 'exploratory_post_test': True, 'means': means, 'records': records,
        'independent_gram_covariance_checks': len(errors), 'maximum_gram_covariance_error': max(errors), 'target_models_changed': 0}
    atomic_json(folder / 'summary.json', result)
    rows = []
    for condition in means:
        for endpoint, values in means[condition].items():
            rows.append(['目标初始化' if condition == 'initialization' else LABELS[condition], endpoint,
                         f"{100*values['train_accuracy']:.2f}%", f"{100*values['validation_accuracy']:.2f}%"] + [f"{values[k]:.4f}" for k in ['full_train_cka', 'matched_count_train_cka', 'matched_count_validation_cka', 'matched_count_test_cka']])
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>取逆对齐的拟合与泛化诊断</title><style>body{font-family:system-ui;max-width:1300px;margin:30px auto;padding:15px;line-height:1.7}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:7px}th{background:#eef2f6}</style><body><h1>拟合与泛化：事后诊断</h1><p>看到主实验阴性准确率和弱测试CKA后，固定本诊断。没有重训、改变检查点或调参；不能当成新的确认性证据。全训练CKA使用同一几何合格训练行；另以每长度12例比较训练、验证和测试，控制偏置线性CKA对样本数的依赖。各长度等权，六次重复等权。</p>'
    html += table(['条件', '端点', '训练准确率', '验证准确率', '全训练CKA', '12例训练CKA', '12例验证CKA', '12例测试CKA'], rows)
    html += '<p>如果训练CKA很高而同样12例的新输入CKA很低，只能说明本次约束拟合缺乏输入泛化；不能写成“高泛化CKA无助于准确率”。小样本随机初始化目标的任务学习与关系学习都可能过拟合。</p><p><a href="../report.html">返回主实验报告</a> · <a href="protocol.json">事后分析登记</a> · <a href="summary.json">逐重复逐长度结果及核验</a></p></body></html>'
    (folder / 'report.html').write_text(html)
    print(json.dumps({'means': means, 'independent_scores_checked': len(errors)}, ensure_ascii=False, indent=2))


if __name__ == '__main__': run()
