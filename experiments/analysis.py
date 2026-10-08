"""Descriptive plots and predictions with entire source combinations held out."""
import csv
import html
import json
from pathlib import Path
import numpy as np


def linear_cka(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean(0)
    right = right - right.mean(0)
    numerator = np.square(left.T @ right).sum()
    denominator = np.sqrt(np.square(left.T @ left).sum() * np.square(right.T @ right).sum())
    return float(numerator / denominator) if denominator > 0 else None


def grouped_predictions(rows, config, group_key='scenario'):
    target_ids = sorted({r['target_id'] for r in rows})
    world_ids = sorted({r['world_seed'] for r in rows})
    baseline = np.array([[r['source_auc'], r['source_accuracy'], r['mean_single_auc']]
                         + [float(r['target_id'] == t) for t in target_ids[1:]]
                         + [float(r['world_seed'] == w) for w in world_ids[1:]] for r in rows])
    structure = np.array([[r['minimum_circuit_size'], r['target_composition_size']] for r in rows])
    truth = np.array([r['transfer_gain'] for r in rows])
    labels = np.array([r[group_key] for r in rows])
    if len(set(labels)) < 2:
        return {'status': 'insufficient_holdout_groups'}
    results = {}
    design = baseline[:, 3:]
    for name, x in [('design_only', design), ('design_plus_structure', np.column_stack([design, structure])),
                    ('baseline', baseline), ('plus_circuit', np.column_stack([baseline, structure[:, 0]])),
                    ('plus_composition', np.column_stack([baseline, structure[:, 1]])),
                    ('plus_both', np.column_stack([baseline, structure]))]:
        predictions = np.zeros(len(truth))
        folds = []
        for label in sorted(set(labels)):
            test = labels == label
            train = ~test
            mean, std = x[train].mean(0), x[train].std(0)
            std = np.where(std > 1e-8, std, 1)
            xt = np.column_stack([np.ones(train.sum()), (x[train] - mean) / std])
            xv = np.column_stack([np.ones(test.sum()), (x[test] - mean) / std])
            penalty = np.eye(xt.shape[1]) * config['prediction_alpha']
            penalty[0, 0] = 0
            weights = np.linalg.solve(xt.T @ xt + penalty, xt.T @ truth[train])
            predictions[test] = xv @ weights
            folds.append({'held_out': str(label), 'mae': float(np.abs(predictions[test] - truth[test]).mean()),
                          'count': int(test.sum())})
        residual = np.square(predictions - truth).sum()
        total = np.square(truth - truth.mean()).sum()
        results[name] = {'r2': float(1 - residual / total) if total > 0 else None,
                         'mae': float(np.abs(predictions - truth).mean()),
                         'rmse': float(np.sqrt(np.mean(np.square(predictions - truth)))),
                         'folds': folds, 'predictions': predictions.tolist()}
    results['incremental'] = {'delta_r2': results['plus_both']['r2'] - results['baseline']['r2'],
                              'mae_reduction': results['baseline']['mae'] - results['plus_both']['mae'],
                              'folds_with_lower_mae': sum(a['mae'] > b['mae'] for a, b in zip(results['baseline']['folds'], results['plus_both']['folds'])),
                              'fold_count': len(set(labels))}
    results['design_incremental'] = {'delta_r2': results['design_plus_structure']['r2'] - results['design_only']['r2'],
                                     'mae_reduction': results['design_only']['mae'] - results['design_plus_structure']['mae']}
    return results


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output, metrics, groups, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    write_csv(output / 'metrics.csv', metrics)
    primary = [r for r in metrics if r['mode'] == config['primary_mode'] and r['budget'] == config['primary_budget']]
    import hashlib
    analysis_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    predictions = {'analysis_sha256': analysis_hash,
                   'combination_holdout': grouped_predictions(primary, config),
                   'input_basis_holdout': grouped_predictions(primary, config, 'world_seed'),
                   'endpoint': 'target accuracy gain over matched random initialization',
                   'primary_budget': config['primary_budget'], 'primary_mode': config['primary_mode'],
                   'baseline_features': ['source learning-curve AUC', 'source final accuracy', 'mean single-source learning-curve AUC', 'target fixed effects', 'input-basis fixed effects'],
                   'structural_features': ['minimum circuit size', 'minimum number of source tasks whose linear span contains the target'],
                   'notes': ['Ridge penalty fixed before transfer evaluation; feature scaling fit inside each training fold.',
                             'Design-only predictions use no held-out source learning measurements; learning-controlled predictions condition on source fitting.',
                             'Combination holdout removes every seed and target row of the held-out source combination.',
                             'Both circuit families remain in training; this is not prediction of an unseen abstract relation family.']}
    (output / 'prediction_scores.json').write_text(json.dumps(predictions, indent=2))
    run_rows = []
    curves = []
    for g in groups:
        for w in config['world_seeds']:
            for m in config['model_seeds']:
                rid = f"{g['id']}_w{w}_m{m}"
                result = json.loads((output / f'{rid}.json').read_text())
                run_rows.append({'scenario': g['id'], 'family': g['family'], 'world_seed': w, 'model_seed': m,
                                 'source_accuracy': result['source_accuracy'], 'source_min_accuracy': result['source_min_accuracy']})
                curves.extend({'scenario': g['id'], 'world_seed': w, 'model_seed': m, 'step': c['step'],
                               'accuracy': float(np.mean(c['accuracy'])), 'loss': c['loss']} for c in result['source_curve'])
    write_csv(output / 'source_endpoints.csv', run_rows)
    write_csv(output / 'source_learning_curves.csv', curves)
    cka = np.zeros((len(groups), len(groups)))
    initial_cka, cka_rows = [], []
    for w in config['world_seeds']:
        for m in config['model_seeds']:
            representations = [np.load(output / f"{g['id']}_w{w}_m{m}_features.npy") for g in groups]
            initial = np.load(output / f'initial_w{w}_m{m}.npy')
            for i, rep in enumerate(representations):
                initial_cka.append(linear_cka(rep, initial))
                for j, other in enumerate(representations):
                    value = linear_cka(rep, other)
                    cka[i, j] += value / (len(config['world_seeds']) * len(config['model_seeds']))
                    cka_rows.append({'world_seed': w, 'model_seed': m, 'left': groups[i]['id'], 'right': groups[j]['id'], 'linear_cka': value})
    write_csv(output / 'cka.csv', cka_rows)
    summary_rows = []
    for mode in ['probe', 'finetune']:
        for budget in config['budgets']:
            for family in ['A', 'B']:
                rows = [r for r in metrics if r['mode'] == mode and r['budget'] == budget and r['family'] == family]
                if not rows:
                    continue
                world_accuracy = [np.mean([r['test_accuracy'] for r in rows if r['world_seed'] == w]) for w in config['world_seeds']]
                world_gain = [np.mean([r['transfer_gain'] for r in rows if r['world_seed'] == w]) for w in config['world_seeds']]
                summary_rows.append({'mode': mode, 'budget': budget, 'family': family,
                                     'accuracy_mean': float(np.mean(world_accuracy)),
                                     'accuracy_world_sd': float(np.std(world_accuracy, ddof=1)) if len(world_accuracy) > 1 else 0.,
                                     'gain_mean': float(np.mean(world_gain)),
                                     'gain_world_sd': float(np.std(world_gain, ddof=1)) if len(world_gain) > 1 else 0.})
    write_csv(output / 'transfer_summary.csv', summary_rows)
    gate = all(r['source_min_accuracy'] >= config['source_accuracy_gate'] for r in run_rows)
    summary = {'analysis_sha256': analysis_hash, 'pretrained_models': len(run_rows), 'source_combinations': len(groups),
               'input_bases': len(config['world_seeds']), 'initialization_seeds': len(config['model_seeds']),
               'target_count': config['target_count'], 'target_endpoints': len(metrics),
               'source_accuracy_min': min(r['source_min_accuracy'] for r in run_rows),
               'all_sources_pass_accuracy_gate': gate,
               'source_accuracy_partition': 'full source-training domain; this gate measures training fit, not unseen-input generalization',
               'source_examples_per_task': config['p'] ** config['dimension'] * config['pretrain_steps'],
               'mean_cka_to_initialization': float(np.mean(initial_cka)),
               'primary_prediction_incremental': predictions['combination_holdout'].get('incremental'),
               'transfer_summary': summary_rows,
               'limitations': ['This is transductive new-task transfer; every input appears in source training.',
                               'The source accuracy gate measures training fit, not generalization to new inputs.',
                               'Three random input bases give little statistical power; target rows are not independent replications.',
                               'One MLP architecture and one finite-field domain do not establish cross-domain generality.',
                               'No direction of the A-versus-B transfer effect was assumed.',
                               'Held-out combinations are coefficient variants of two known circuit templates.',
                               'Equal marginal output statistics do not guarantee equal computational difficulty.']}
    if not gate:
        summary['limitations'].append('Source-learning gate failed: transfer differences cannot be interpreted as matched learned tasks.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'A': '#2166ac', 'B': '#d95f02'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, mode in zip(axes, ['probe', 'finetune']):
        for family in ['A', 'B']:
            rows = [r for r in summary_rows if r['mode'] == mode and r['family'] == family]
            ax.errorbar([r['budget'] for r in rows], [r['accuracy_mean'] for r in rows],
                        yerr=[r['accuracy_world_sd'] for r in rows], marker='o', color=colors[family], label=f'{family}: {3 if family == "A" else 4}-task circuit')
        random_rows = [r for r in metrics if r['mode'] == mode]
        ax.plot(config['budgets'], [np.mean([r['random_accuracy'] for r in random_rows if r['budget'] == b]) for b in config['budgets']], '--', color='gray', label='Matched random initialization')
        ax.axhline(1/config['p'], color='black', linewidth=.7, alpha=.4)
        ax.set(title='Frozen linear probe' if mode == 'probe' else 'Full encoder fine-tuning', xlabel='Labeled target examples', ylabel='Held-out label accuracy', ylim=(0, 1))
        ax.legend(fontsize=8)
    fig.savefig(figures / 'transfer.png', dpi=180)
    fig.savefig(figures / 'transfer.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for family in ['A', 'B']:
        ids = {g['id'] for g in groups if g['family'] == family}
        steps = sorted({r['step'] for r in curves})
        ax.plot(steps, [np.mean([r['accuracy'] for r in curves if r['scenario'] in ids and r['step'] == step]) for step in steps], label=family, color=colors[family])
    ax.set(xlabel='Source training updates', ylabel='Source training accuracy', ylim=(0, 1.02))
    ax.legend()
    fig.savefig(figures / 'source_learning.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(cka, vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(len(groups)), [g['id'] for g in groups], rotation=45)
    ax.set_yticks(range(len(groups)), [g['id'] for g in groups])
    ax.set_title('CKA on identical input rows, averaged over bases')
    fig.colorbar(im, ax=ax)
    fig.savefig(figures / 'cka.png', dpi=180)
    plt.close(fig)
    pred = predictions['combination_holdout']
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    labels = ['Baseline', '+ circuit', '+ composition', '+ both']
    scores = [pred[name]['r2'] for name in ['baseline', 'plus_circuit', 'plus_composition', 'plus_both']]
    ax.bar(labels, scores, color=['#777777', '#2166ac', '#d95f02', '#39824a'])
    ax.axhline(0, color='black', linewidth=.7)
    ax.set(ylabel='Grouped out-of-fold R squared', title=f"Transfer-gain prediction, {config['primary_budget']} labels")
    fig.savefig(figures / 'prediction.png', dpi=180)
    plt.close(fig)
    increment = pred['incremental']
    table = ''.join(f"<tr><td>{r['mode']}</td><td>{r['budget']}</td><td>{r['family']}</td><td>{100*r['accuracy_mean']:.2f}%</td><td>{100*r['gain_mean']:+.2f} pp</td></tr>" for r in summary_rows)
    conclusions = ('结构特征的改善很小；当前证据不足以支持稳定的增量预测能力。' if increment['delta_r2'] > 0 and increment['mae_reduction'] > 0 and increment['delta_r2'] < .02 else ('结构特征提高了本轮留组合预测的平均表现；仍需要独立组合复验。' if increment['delta_r2'] > 0 and increment['mae_reduction'] > 0 else '本轮结构特征未同时改善 R² 和 MAE；当前结果不足以支持增量预测能力。'))
    oracle_text = ''
    if (output / 'symbolic_oracle.json').exists():
        oracle = json.loads((output / 'symbolic_oracle.json').read_text())
        oracle_text = f"<p>事后诊断：已训练的 u/v/w 输出经外部已知数学公式组合，目标测试准确率为 {100*oracle['mean_oracle_accuracy']:.2f}%。这证明目标信息在这些已见输入的表征中可恢复；普通读出差不等于信息缺失。该 oracle 使用外部公式，不是模型自发学会组合。<a href='symbolic_oracle.json'>诊断与每个端点</a>。</p>"
    negative_text = ('两组在所有标签预算与普通读出设定下的平均迁移收益均为负。' if all(r['gain_mean'] < 0 for r in summary_rows) else '不同设定的迁移收益见下表。')
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>代数关系与迁移：第一轮试验</title>
<style>body{{max-width:1000px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui;color:#20242b}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}code{{background:#eee;padding:2px 4px}}</style>
<h1>代数关系与迁移：第一轮受控试验</h1>
<p>完成 {len(run_rows)} 个源模型、{len(groups)} 种组合、{len(config['world_seeds'])} 套随机输入基与 {config['target_count']} 个共同未见目标。源任务最低训练准确率 {100*summary['source_accuracy_min']:.2f}%；{100*config['source_accuracy_gate']:.0f}% 拟合程度门槛：{'通过' if gate else '未通过'}。门槛不检验未见输入上的源任务泛化。</p>
<p>A 的最小依赖涉及三个任务，B 涉及四个任务。两组均有四任务、秩三、均匀边际、两两独立标签和相同联合熵。每个源任务训练曝光为 {summary['source_examples_per_task']:,} 次。</p>
<p><strong>范围：</strong>源训练覆盖所有输入，支持集与测试集只在目标标签上隔离。本轮检验同实体的新任务迁移，不是新实体或分布外泛化。</p>
<p>主结果为 {config['primary_budget']} 个目标标签下的完整微调收益。按整个源任务组合留出所有种子与目标行；基线控制源学习曲线、单任务学习曲线、目标和输入基效应。加入结构后 ΔR²={increment['delta_r2']:+.4f}，MAE 减少={100*increment['mae_reduction']:+.3f} 个百分点，{increment['folds_with_lower_mae']}/{increment['fold_count']} 折 MAE 改善。</p>
<p>{negative_text} {conclusions}</p>{oracle_text}<img src="figures/transfer.png"><img src="figures/prediction.png">
<table><tr><th>评估</th><th>标签数</th><th>组</th><th>准确率</th><th>相对随机初始化收益</th></tr>{table}</table>
<img src="figures/source_learning.png"><img src="figures/cka.png">
<p>误差条是随机输入基之间的样本标准差。{len(metrics)} 个目标端点不构成 {len(metrics)} 个独立重复。单一 MLP 与有限域不足以证明跨领域普适性；留出的是系数组合，训练仍包含两种依赖模板。</p>
<p>原始数据：<a href="metrics.csv">metrics.csv</a>；<a href="prediction_scores.json">分组预测与各折结果</a>；<a href="algebra_audit.json">精确代数审计</a>；<a href="metadata.json">配置与环境</a>。</p>
</html>'''
    (output / 'report.html').write_text(document)
    print(json.dumps({k: v for k, v in summary.items() if k != 'transfer_summary'}, indent=2), flush=True)


if __name__ == '__main__':
    import argparse
    from .algebra import scenarios
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/pilot')
    args = parser.parse_args()
    output = Path(args.output)
    metadata = json.loads((output / 'metadata.json').read_text())
    config = metadata['config']
    groups = scenarios()
    groups = [g for pair in zip(groups[:4], groups[4:]) for g in pair][:config['scenario_limit']]
    rows = []
    for g in groups:
        for w in config['world_seeds']:
            for m in config['model_seeds']:
                rows.extend(json.loads((output / f"{g['id']}_w{w}_m{m}.json").read_text())['metrics'])
    summarize(output, rows, groups, config)
