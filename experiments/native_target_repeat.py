"""Repeat support/test sampling with frozen sources, policies and forecasts.

Same task sets and source checkpoints: target-data robustness, not a new source
world, unseen task combination, or independent model replication.
"""
import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .longrun_engine import atomic_json
from .longrun_transfer import evaluate_model, make_model
from .native_prediction_sensitivity import corrected_math
from .native_transfer_prediction import label_statistics
from .permutation_audit import IDENTITIES
from .permworld_combinations import select_groups, sha
from .six_hour_report import write_rows, transfer_summaries


def fit_weights(x, y):
    mean, sigma = x.mean(0), np.maximum(x.std(0), 1e-8)
    train = (x-mean)/sigma; ymean = y.mean()
    beta = np.linalg.solve(train.T@train+np.eye(x.shape[1]), train.T@(y-ymean))
    return {'mean': mean.tolist(), 'sigma': sigma.tolist(), 'beta': beta.tolist(), 'intercept': float(ymean)}


def apply_weights(x, weights):
    return ((x-np.array(weights['mean']))/np.array(weights['sigma']))@np.array(weights['beta'])+weights['intercept']


def feature_matrix(features, variant, corrected=False):
    return np.array([r['baseline']+(corrected_math(r) if corrected else r['math'] if variant in ('plus_mathematical_relations', 'plus_both') else [])
                     +(r['geometry'] if variant in ('plus_measured_geometry', 'plus_both') else []) for r in features])


def paths():
    plan = json.loads(Path('configs/six_hour_session.json').read_text())
    config = json.loads(Path(plan['base_config']).read_text())
    return plan, config, Path(plan['output']), Path('results/native_target_repeat')


def prepare():
    plan, config, source_root, root = paths(); root.mkdir(exist_ok=True)
    groups = select_groups(config)
    signature = {'code_sha256': sha(__file__), 'source_data_sha256': sha(source_root/'dataset/data.npz'),
                 'source_feature_sha256': sha(source_root/'transfer_prediction/features.json'),
                 'target_tuning_sha256': sha(source_root/'target_tuning.json'),
                 'transfer_code_sha256': sha('experiments/longrun_transfer.py'),
                 'seed': 202610056, 'split_counts_per_length': {'support_pool': 100, 'target_validation': 50, 'target_test': 100},
                 'scope': 'added after original native test outcomes; same source models/tasks/policies, globally new support/validation/test inputs; all groups retained',
                 'forecast_policy': 'fit all four frozen predictor variants plus both corrected-indicator variants to first target-data gains; freeze weights and forecasts before generating new input labels',
                 'not_claimed': 'not unseen source task sets, new source training world, independent model runs, or an original primary endpoint'}
    protocol_path = root/'protocol.json'
    if protocol_path.exists(): assert json.loads(protocol_path.read_text())['signature'] == signature
    else:
        assert not (root/'dataset/data.npz').exists()
        atomic_json(protocol_path, {'registered_utc': datetime.now(timezone.utc).isoformat(), 'signature': signature})
    features = json.loads((source_root/'transfer_prediction/features.json').read_text())
    old_records = [json.loads(p.read_text()) for p in (source_root/'transfer').glob('*.json')]
    assert len(old_records) == 27 and all(r['status'] == 'complete' for r in old_records)
    old_rows = [{**row, 'group': record['group'], 'seed': record['seed']} for record in old_records for row in record['rows']]
    random = {(r['seed'], r['target'], r['budget'], r['mode']): r['test_accuracy'] for r in old_rows if r['group'] == 'random'}
    source_lookup = {(r['group'], r['seed'], r['target'], r['budget'], r['mode']): r for r in old_rows}
    weights, forecasts = [], []
    forecast_path = root/'forecasts.json'
    if not forecast_path.exists():
        assert not (root/'dataset/data.npz').exists()
        for mode, budget in product(('linear_task_free', 'linear_query', 'mlp_task_free', 'finetune', 'finetune_shared'), (64, 256)):
            y = np.array([source_lookup[r['group'], r['seed'], r['target'], budget, mode]['test_accuracy']-random[r['seed'], r['target'], budget, mode] for r in features])
            for variant, corrected in [('baseline', False), ('plus_mathematical_relations', False), ('plus_measured_geometry', False), ('plus_both', False), ('plus_mathematical_relations', True), ('plus_both', True)]:
                matrix = feature_matrix(features, variant, corrected)
                fitted = fit_weights(matrix, y); version = variant+('_corrected' if corrected else '')
                weights.append({'mode': mode, 'budget': budget, 'variant': version, 'weights': fitted})
                predicted = apply_weights(matrix, fitted)
                forecasts.extend({'group': r['group'], 'seed': r['seed'], 'target': r['target'], 'mode': mode,
                                  'budget': budget, 'variant': version, 'predicted_paired_gain': float(p)} for r, p in zip(features, predicted))
        atomic_json(root/'weights.json', weights)
        atomic_json(forecast_path, {'frozen_utc': datetime.now(timezone.utc).isoformat(),
                                   'new_input_labels_not_generated': True, 'forecasts': forecasts,
                                   'old_behavior_sha256': {p.name: sha(p) for p in (source_root/'transfer').glob('*.json')}})
    dataset = root/'dataset'; dataset.mkdir(exist_ok=True)
    sys.path.insert(0, str(Path(config['repository'])/'src'))
    from neurips_permutations.math_ops import PROPERTY32_TASK_NAMES, PROPERTY_FUNCTIONS
    from neurips_permutations.passage import TOKEN_TO_ID, one_line_tokens
    names = list(PROPERTY32_TASK_NAMES)
    if not (dataset/'data.npz').exists():
        with np.load(source_root/'dataset/data.npz') as archive:
            seen = {tuple(row) for key in archive.files if key.endswith('_input') for row in archive[key]}
        forbidden_count = len(seen); data = {}; rng = np.random.default_rng(signature['seed'])
        lo, hi = config['lengths']
        for split, count in signature['split_counts_per_length'].items():
            inputs, lengths, labels = [], [], []
            for n in range(lo, hi+1):
                for _ in range(count):
                    while True:
                        permutation = tuple(map(int, rng.permutation(n)+1))
                        prefix = [TOKEN_TO_ID['<BOS>'], TOKEN_TO_ID['<SIZE>'], n]+[TOKEN_TO_ID[t] for t in one_line_tokens(permutation)]
                        padded = prefix+[TOKEN_TO_ID['<PAD>']]*(2*hi+4-len(prefix))
                        if tuple(padded) not in seen: seen.add(tuple(padded)); break
                    inputs.append(padded); lengths.append(n)
                    labels.append([PROPERTY_FUNCTIONS[name](permutation) for name in names])
            data[split+'_input'] = np.array(inputs); data[split+'_lengths'] = np.array(lengths); data[split+'_labels'] = np.array(labels)
            for identity in IDENTITIES:
                value = sum(c*data[split+'_labels'][:, names.index(t)] for t, c in identity['terms'].items())
                assert np.array_equal(value, identity['n']*data[split+'_lengths']+identity['constant'])
        np.savez_compressed(dataset/'data.npz', **data)
        atomic_json(dataset/'metadata.json', {'names': names, 'source_original_inputs_excluded': forbidden_count,
                                            'new_inputs': len(seen)-forbidden_count, 'data_sha256': sha(dataset/'data.npz')})
    support_order = np.random.default_rng(signature['seed']+50000).permutation(2100)
    atomic_json(dataset/'support_indices.json', {str(b): support_order[:b].tolist() for b in config['target_budgets']})
    print(json.dumps({'status': 'prepared', 'new_inputs': 5250, 'frozen_forecasts': 5760}), flush=True)


def evaluate():
    prepare()
    plan, config, source_root, root = paths()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(Path(config['repository'])/'src'))
    from neurips_permutations.passage import TOKEN_TO_ID
    names = json.loads((root/'dataset/metadata.json').read_text())['names']
    with np.load(root/'dataset/data.npz') as archive:
        data = {k: torch.tensor(archive[k], device='cuda') for k in archive.files}
    arch = json.loads((source_root/'architecture_selection.json').read_text())['selected']
    tuning = json.loads((source_root/'target_tuning.json').read_text()); assert tuning['status'] == 'complete'
    support = json.loads((root/'dataset/support_indices.json').read_text())
    for seed in plan['model_seeds']:
        for group in ['random']+[g['id'] for g in select_groups(config)]:
            model = make_model(config, arch, seed, 'cuda'); record = None
            if group != 'random':
                run_id = f"{arch['id']}_{group}_s{seed}"
                record = json.loads((source_root/'multi'/f'{run_id}.json').read_text())
                checkpoint = source_root/'multi/checkpoints'/f'{run_id}.pt'
                assert record['status'] == 'complete' and sha(checkpoint) == record['checkpoint_sha256']
                model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cuda')['model'])
            evaluate_model(model, record, plan, config, data, names, TOKEN_TO_ID, root, support, tuning, group, seed)
    report()


def report():
    _, _, source_root, root = paths()
    records = [json.loads(p.read_text()) for p in (root/'transfer').glob('*.json')]
    assert len(records) == 27 and all(r['status'] == 'complete' for r in records)
    endpoints = [{**row, 'group': record['group'], 'seed': record['seed']} for record in records for row in record['rows']]
    gains, summaries = transfer_summaries(endpoints)
    saved = json.loads((root/'forecasts.json').read_text()); forecasts = saved['forecasts']
    lookup = {(r['group'], r['seed'], r['target'], r['budget'], r['mode']): r['gain'] for r in gains}
    assert len(lookup) == 960
    scores = []
    for mode, budget, variant in sorted({(r['mode'], r['budget'], r['variant']) for r in forecasts}):
        selected = [r for r in forecasts if (r['mode'], r['budget'], r['variant']) == (mode, budget, variant)]
        y = np.array([lookup[r['group'], r['seed'], r['target'], budget, mode] for r in selected])
        predicted = np.array([r['predicted_paired_gain'] for r in selected])
        scores.append({'mode': mode, 'budget': budget, 'variant': variant,
                       'r2': float(1-((y-predicted)**2).sum()/((y-y.mean())**2).sum()),
                       'mae': float(np.abs(y-predicted).mean()), 'rows': len(y)})
    for name, rows in (('endpoints.csv', endpoints), ('paired_gains.csv', gains), ('group_summary.csv', summaries), ('forecast_scores.csv', scores)):
        write_rows(root/name, rows)
    atomic_json(root/'summary.json', {'status': 'complete', 'source_models_reused': 24, 'random_models_recreated': 3,
                                    'target_endpoints': len(endpoints), 'new_inputs': 5250, 'forecast_rows': len(forecasts),
                                    'source_tasks_and_models_not_new': True, 'group_summary': summaries, 'forecast_scores': scores})
    table = ''.join(f"<tr><td>{r['group']}</td><td>{r['mode']}</td><td>{r['budget']}</td><td>{100*r['gain']:+.2f} pp</td><td>{r['positive_seed_means']}/3</td></tr>" for r in summaries)
    prediction_table = ''.join(f"<tr><td>{r['mode']}</td><td>{r['budget']}</td><td>{r['variant']}</td><td>{r['r2']:.3f}</td><td>{100*r['mae']:.2f} pp</td></tr>" for r in scores)
    (root/'report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>独立目标抽样复核</title><style>body{{max-width:1200px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd}}aside{{background:#f2f4f7;padding:16px}}</style><h1>新支持集与测试集：冻结源模型、参数及预测</h1><p>本复核是在第一套原生测试之后追加。同一批 24 个编码器与 3 个随机初始化，同样八个源任务集、四目标、两预算、五读出，沿用第一套验证选择的所有参数，不重新调参。新生成 5,250 个输入，全局排除原先 52,290 个输入；支持、验证、测试互不重复。复用目标验证调用仅作记录，不选择策略。</p><aside>新的只是目标数据抽样。没有新源训练世界、新源模型或新任务组合，不能称为独立模型/任务集复制。第一套预测器在所有八组上拟合，四个原版本及两个指示变量修正版的全部 5,760 条预测先于新输入标签生成保存；全版本并列，未挑最好得分。</aside><table><tr><th>组</th><th>读出</th><th>标签</th><th>对随机收益</th><th>种子均值为正</th></tr>{table}</table><h2>冻结预测器在新目标抽样上的表现</h2><table><tr><th>读出</th><th>标签</th><th>版本</th><th>新抽样 R²</th><th>平均绝对误差</th></tr>{prediction_table}</table><p><a href="protocol.json">追加规则</a> · <a href="forecasts.json">标签生成前冻结预测</a> · <a href="weights.json">全部拟合权重</a> · <a href="paired_gains.csv">全部配对行为</a> · <a href="forecast_scores.csv">全部预测得分</a> · <a href="dataset/metadata.json">输入范围</a></p></html>''')
    print(json.dumps({'status': 'complete', 'endpoints': len(endpoints), 'forecasts': len(forecasts)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=('prepare', 'evaluate', 'report'), default='prepare')
    {'prepare': prepare, 'evaluate': evaluate, 'report': report}[parser.parse_args().phase]()
