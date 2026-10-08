"""Assess saved forecasts without fitting on the independent readout cohort."""
from datetime import datetime, timezone
import hashlib
from html import escape
from itertools import product
import json
from pathlib import Path

import numpy as np

from .field_forecast import VARIANTS, design, predict
from .six_hour_report import write_rows


KEY = ('group', 'world_seed', 'model_seed', 'target_id', 'budget', 'mode')


def match_outcomes(forecasts, rows):
    """Require exactly matched random baselines and preserve signed gains."""
    lookup = {}
    for row in rows:
        key = tuple(row[k] for k in KEY)
        assert key not in lookup, f'Duplicate readout endpoint: {key}'
        lookup[key] = row
    evaluated = []
    keys = set()
    for forecast in forecasts:
        key = tuple(forecast[k] for k in KEY)
        unique = key+(forecast['variant'],)
        assert unique not in keys, f'Duplicate forecast: {unique}'
        keys.add(unique)
        if key not in lookup:
            continue
        row = lookup[key]
        if row['mode'] in ('linear', 'mlp'):
            baseline_key = ('random',)+key[1:]
            assert baseline_key in lookup, f'Missing paired random readout: {key}'
            random_accuracy = lookup[baseline_key]['accuracy']
            truth = row['accuracy']-random_accuracy
        else:
            random_accuracy = None
            truth = row['accuracy']
        evaluated.append({**forecast, 'target_accuracy': row['accuracy'],
                          'random_accuracy': random_accuracy, 'actual_outcome': truth,
                          'error': forecast['predicted_outcome']-truth})
    return evaluated


def metrics(rows):
    truth = np.array([r['actual_outcome'] for r in rows])
    error = np.array([r['error'] for r in rows])
    total = float(np.square(truth-truth.mean()).sum())
    return {'cells': len(rows), 'r2': 1-float(np.square(error).sum())/total if total > 0 else None,
            'mae': float(np.mean(abs(error))), 'rmse': float(np.sqrt(np.square(error).mean())),
            'mean_actual': float(truth.mean()), 'mean_predicted': float((truth+error).mean())}


def verify(root, saved, rows, evaluated):
    protocol = saved['protocol']
    assert protocol == json.loads(Path('results/six_hour_session/independent_forecast_preregistration.json').read_text())
    assert protocol['code_sha256'] == hashlib.sha256(Path('experiments/field_forecast.py').read_bytes()).hexdigest()
    old_transfer = json.loads(Path('configs/field_symmetry_transfer.json').read_text())
    old_source = json.loads(Path(old_transfer['source_config']).read_text())
    new_transfer = json.loads(Path('configs/field_matched_support_transfer.json').read_text())
    new_source = json.loads(Path(new_transfer['source_config']).read_text())
    assert saved['original_behavior_fingerprint'] == json.loads(Path(old_transfer['output']).joinpath('metadata.json').read_text())['fingerprint']
    assert saved['independent_source_fingerprint'] == json.loads(Path(new_source['output']).joinpath('metadata.json').read_text())['fingerprint']
    expected = len(new_source['groups'])*len(new_source['world_seeds'])*len(new_source['model_seeds'])*len(new_transfer['targets'])*len(new_transfer['budgets'])*3*len(VARIANTS)
    assert len(saved['forecasts']) == expected
    assert set(new_source['world_seeds']).isdisjoint(old_source['world_seeds'])
    assert set(new_source['groups']) <= set(old_source['groups'])
    for mode, budget, variant in product(('linear', 'mlp', 'categorical_subset'), new_transfer['budgets'], VARIANTS):
        selected = [r for r in saved['forecasts'] if (r['mode'], r['budget'], r['variant']) == (mode, budget, variant)]
        result = predict(saved['fitted_models'][f'{mode}_{budget}_{variant}'], design(selected, variant, old_transfer, old_source))
        np.testing.assert_allclose(result, [r['predicted_outcome'] for r in selected], rtol=0, atol=1e-12)
    assert saved['independent_behavior_not_started'] is True
    frozen = datetime.fromisoformat(saved['frozen_utc']).timestamp()
    metadata_path = Path(new_transfer['output'])/'metadata.json'
    assert frozen < metadata_path.stat().st_mtime
    observed_keys = {tuple(r[k] for k in KEY) for r in rows if r['group'] != 'random' and r['mode'] != 'full_tuple_lookup'}
    evaluated_keys = {tuple(r[k] for k in KEY) for r in evaluated}
    assert observed_keys == evaluated_keys
    verification = {'status': 'passed_for_completed_artifacts', 'frozen_forecasts': expected,
        'evaluated_forecasts': len(evaluated), 'unique_behavior_outcomes': len(evaluated_keys),
        'exact_paired_random_gains': True, 'saved_fitted_predictions_recomputed': True,
        'fit_worlds_and_replication_worlds_disjoint': True, 'source_formulas_overlap_fit_cohort': True,
        'forecast_code_matches_pretraining_registration': True,
        'forecast_before_behavior_metadata_file_time': True,
        'temporal_evidence': 'Frozen runner asserts no independent behavior metadata exists before saving; the sequential queue starts independent readouts afterwards. File modification times provide an additional local check.',
        'forecasts_sha256': hashlib.sha256((root/'forecasts.json').read_bytes()).hexdigest()}
    (root/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    return verification


def report(root='results/field_forecast'):
    root = Path(root)
    saved = json.loads((root/'forecasts.json').read_text())
    config = json.loads(Path('configs/field_matched_support_transfer.json').read_text())
    source = json.loads(Path(config['source_config']).read_text())
    behavior = Path(config['output'])
    metadata = json.loads((behavior/'metadata.json').read_text())
    rows, conditions = [], 0
    for path in behavior.glob('*_w*_m*.json'):
        record = json.loads(path.read_text())
        assert record['fingerprint'] == metadata['fingerprint']
        if record['status'] == 'complete':
            rows.extend(record['rows']); conditions += 1
    evaluated = match_outcomes(saved['forecasts'], rows)
    if not evaluated:
        raise ValueError('No completed independent behavior outcomes')
    scores, blocks = [], []
    for mode, budget in product(('linear', 'mlp', 'categorical_subset'), config['budgets']):
        baseline_rows = [r for r in evaluated if (r['mode'], r['budget'], r['variant']) == (mode, budget, 'baseline')]
        if not baseline_rows: continue
        baseline_score = metrics(baseline_rows)
        for variant in VARIANTS:
            selected = [r for r in evaluated if (r['mode'], r['budget'], r['variant']) == (mode, budget, variant)]
            assert {tuple(r[k] for k in KEY) for r in selected} == {tuple(r[k] for k in KEY) for r in baseline_rows}
            value = metrics(selected)
            original = saved['original_whole_combination_cv'][f'{mode}_{budget}_{variant}']
            scores.append({'mode': mode, 'budget': budget, 'variant': variant, **value,
                           'delta_r2': value['r2']-baseline_score['r2'] if value['r2'] is not None else None,
                           'mae_reduction': baseline_score['mae']-value['mae'],
                           'original_combination_holdout_r2': original['r2'],
                           'original_combination_holdout_mae': original['mae']})
            for w, m in product(source['world_seeds'], source['model_seeds']):
                cell = [r for r in selected if (r['world_seed'], r['model_seed']) == (w, m)]
                base = [r for r in baseline_rows if (r['world_seed'], r['model_seed']) == (w, m)]
                if cell:
                    blocks.append({'mode': mode, 'budget': budget, 'variant': variant,
                                   'world_seed': w, 'model_seed': m, **metrics(cell),
                                   'mae_reduction': metrics(base)['mae']-metrics(cell)['mae']})
    summary = {'reported_utc': datetime.now(timezone.utc).isoformat(), 'forecast_frozen_utc': saved['frozen_utc'],
               'completed_behavior_conditions': conditions, 'planned_behavior_conditions': 36,
               'frozen_forecasts': len(saved['forecasts']), 'evaluated_forecasts': len(evaluated),
               'scores': scores, 'scope': 'Independent input bases and model runs; source formulas appear in fitting cohort.',
               'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    write_rows(root/'evaluated_forecasts.csv', evaluated)
    write_rows(root/'scores.csv', scores)
    write_rows(root/'paired_world_initialization_scores.csv', blocks)
    verify(root, saved, rows, evaluated)
    plot(root, scores, config['budgets'])
    table = ''.join(f"<tr><td>{r['mode']}</td><td>{r['budget']}</td><td>{r['variant']}</td><td>{fmt(r['original_combination_holdout_r2'])}</td><td>{fmt(r['r2'])}</td><td>{fmt(r['delta_r2'])}</td><td>{r['mae']:.4f}</td><td>{r['cells']}</td></tr>" for r in scores)
    (root/'report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>事先冻结的独立输入基预测</title>
<style>body{{max-width:1200px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui;color:#18202b}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}aside{{background:#f2f4f7;padding:16px}}</style>
<h1>先保存预测，再运行独立输入基的目标读出</h1>
<p>预测于 {escape(saved['frozen_utc'])} 冻结，共 {len(saved['forecasts'])} 条。独立行为条件已完成 {conditions}/36，当前可核对 {len(evaluated)} 条预测；每个实际结果对应五个预测规则。完整目标、条件与预算均保留，负 R² 和负迁移收益照常报告。</p>
<p>预测器只在原始五组、三个输入基、三个初始化的行为结果上拟合。复核使用 P、M1、M2、三个新的输入基及三个初始化；每个源任务在四个物理输入坐标上均非零。这三个源公式已在拟合集中出现，所以检验的是新输入基与新模型运行，尚未检验新的抽象任务组合。</p>
<p>基线包含源学习准确率、学习曲线面积、源与目标物理系数支撑大小，以及目标、初始化和原输入基固定效应。未知输入基使用训练基效应的平均值。岭回归惩罚固定为 1，不用本次行为结果重新拟合、选参数或选择预测规则。</p>
<img src="forecast_scores.png"><table><tr><th>读出</th><th>标签</th><th>预测规则</th><th>原组合留出 R²</th><th>独立基 R²</th><th>相对基线 ΔR²</th><th>独立基 MAE</th><th>结果数</th></tr>{table}</table>
<p>linear、mlp 的预测目标是测试准确率减去相同输入基、初始化、目标、预算和读出规则的随机编码器准确率；categorical_subset 的预测目标是通用类别子集查表的准确率。五种规则分别加入最小源答案组合阶数、理想已知子集的查表覆盖率、目标字符方向的隐藏方差比例，或同时加入组合阶数和字符方差比例。</p>
<aside>已知真实目标公式的数学分析属于额外信息；隐藏字符方差也只作为分析诊断。组合阶数同时是类别条件依赖阶数，成功不能独立证明模型学会了特定代数。理想查表覆盖率假定知道正确子集，通用子集搜索未必达到该值。完整 625 输入均用于源训练，本试验是新任务标签的转导迁移。三个新输入基与三个初始化交叉形成相关重复，不报告独立数据集显著性。</aside>
<p><a href="../six_hour_session/independent_forecast_preregistration.json">训练前登记</a> · <a href="forecasts.json">事先保存的预测及拟合权重</a> · <a href="evaluated_forecasts.csv">逐结果核对</a> · <a href="scores.csv">全部评分</a> · <a href="paired_world_initialization_scores.csv">逐输入基与初始化评分</a> · <a href="verification.json">核验</a> · <a href="forecast_scores.pdf">PDF 图</a> · <a href="../field_matched_support_transfer/report.html">完整独立行为报告</a></p></html>''')
    print(json.dumps({'completed_behavior_conditions': conditions, 'evaluated_forecasts': len(evaluated), 'scores': scores}, indent=2))
    return summary


def fmt(value):
    return '未定义' if value is None else f'{value:.3f}'


def plot(root, scores, budgets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    labels = ['base', '+order', '+oracle', '+spectrum', '+both']
    for i, mode in enumerate(('linear', 'mlp', 'categorical_subset')):
        for j, budget in enumerate(budgets):
            ax = axes[i, j]
            cell = {r['variant']: r for r in scores if (r['mode'], r['budget']) == (mode, budget)}
            ax.bar(np.arange(len(VARIANTS)), [cell[v]['r2'] if cell.get(v, {}).get('r2') is not None else np.nan for v in VARIANTS])
            ax.axhline(0, color='gray', linestyle=':')
            ax.set(xticks=np.arange(len(VARIANTS)), xticklabels=labels, ylabel='Independent-world R²', title=f'{mode}; {budget} labels')
            ax.tick_params(axis='x', labelsize=8)
    for suffix in ('png', 'pdf'): fig.savefig(root/f'forecast_scores.{suffix}', dpi=160)
    plt.close(fig)


if __name__ == '__main__': report()
