"""Evaluate a registered geometric prediction, retaining paired repetitions."""
import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
from itertools import product
import json
from pathlib import Path

import numpy as np

from .analysis import linear_cka, write_csv
from .field_symmetry import audit_group, group_sources, spectral_power
from .algebra import world


def summarize(output='results/field_symmetry', verify=False):
    output = Path(output)
    metadata = json.loads((output/'metadata.json').read_text())
    config = metadata['config']
    records, curves, finals = [], [], []
    for w, m, g in product(config['world_seeds'], config['model_seeds'], config['groups']):
        path = output/f'{g}_w{w}_m{m}.json'
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        records.append(record)
        curves.extend(record['curve'])
        if record['status'] == 'complete':
            initial, last = record['curve'][0], record['curve'][-1]
            finals.append({**last, 'source_gate_passed': record['source_gate_passed'],
                           'initial_hidden_cka': initial['transformed_hidden_cka'],
                           'change_from_initial': last['transformed_hidden_cka']-initial['transformed_hidden_cka']})
    summaries = []
    for g in config['groups']:
        cells = [r for r in finals if r['group'] == g]
        if cells:
            summaries.append({'group': g, 'models': len(cells), 'gate_passed': sum(r['source_gate_passed'] for r in cells),
                              'ideal_code_prediction': audit_group(g, config['p'])['predicted_categorical_cka'],
                              'hidden_cka': float(np.mean([r['transformed_hidden_cka'] for r in cells])),
                              'initial_hidden_cka': float(np.mean([r['initial_hidden_cka'] for r in cells])),
                              'source_min_accuracy': float(min(r['source_min_accuracy'] for r in cells)),
                              'source_character_energy_fraction': float(np.mean([r['source_character_energy_fraction'] for r in cells]))})
    paired = []
    for w, m in product(config['world_seeds'], config['model_seeds']):
        cell = {r['group']: r for r in finals if (r['world_seed'], r['model_seed']) == (w, m)}
        if set(cell) == set(config['groups']):
            assert max(r['initial_hidden_cka'] for r in cell.values())-min(r['initial_hidden_cka'] for r in cell.values()) < 1e-10
            high = cell['P']['transformed_hidden_cka']
            mid = np.mean([cell[g]['transformed_hidden_cka'] for g in ('M1', 'M4') if g in cell])
            low = np.mean([cell[g]['transformed_hidden_cka'] for g in ('M2', 'M3') if g in cell])
            paired.append({'world_seed': w, 'model_seed': m, 'P': high, 'middle_cka': float(mid), 'lower_cka': float(low),
                           'high_minus_mid': float(high-mid), 'mid_minus_low': float(mid-low),
                           'ordered': bool(high > mid > low), 'all_source_gates_passed': all(r['source_gate_passed'] for r in cell.values())})
    summary = {'reported_utc': datetime.now(timezone.utc).isoformat(), 'complete_models': len(finals),
               'planned_models': len(config['groups'])*len(config['world_seeds'])*len(config['model_seeds']),
               'source_gate_passed': sum(r['source_gate_passed'] for r in finals),
               'group_summary': summaries, 'paired_settings': len(paired),
               'ordered_paired_settings': sum(r['ordered'] for r in paired),
               'mean_high_minus_mid': float(np.mean([r['high_minus_mid'] for r in paired])) if paired else None,
               'mean_mid_minus_low': float(np.mean([r['mid_minus_low'] for r in paired])) if paired else None,
               'prediction_scope': 'ideal source categorical kernel CKA is exact; learned hidden CKA ordering is a falsifiable registered hypothesis',
               'source_statistics': 'full rank 4; uniform joint distribution over all 625 codes; all marginals uniform and pairwise MI zero',
               'generalization_scope': 'all input points used for source training; no unseen-input or transfer claim',
               'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.joinpath('summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for filename, rows in (('curves.csv', curves), ('final_models.csv', finals), ('group_summary.csv', summaries), ('paired_contrasts.csv', paired)):
        if rows:
            write_csv(output/filename, rows)
    plot(output, curves, summaries, config)
    table = ''.join(f"<tr><td>{r['group']}</td><td>{r['ideal_code_prediction']:.2f}</td><td>{r['hidden_cka']:.4f}</td><td>{r['source_character_energy_fraction']:.3f}</td><td>{r['gate_passed']}/{r['models']}</td></tr>" for r in summaries)
    coefficients = ', '.join(g[1:] for g in config['groups'] if g != 'P')
    preregistration = 'matched_field_preregistration.json' if 'world_provider_sha256' in config else 'secondary_preregistration.json'
    physical_note = '<p>本独立复核进一步要求每个源任务的四个物理输入系数全部非零；每个单任务都可通过各输入坐标的类别置换化为四输入模加法。函数复杂度匹配不保证多任务训练轨迹相同。<a href="controlled_worlds.json">实际输入基与函数等价检查</a>已保存。</p>' if 'world_provider_sha256' in config else ''
    output.joinpath('report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>有限域：精确匹配统计的变换试验</title>
<style>body{{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui;color:#18202b}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}aside{{background:#f2f4f7;padding:16px}}</style>
<h1>相同任务数量与完整输出统计下，变换结构能否预测隐藏几何</h1>
<p>模型已完成 {len(finals)}/{summary['planned_models']}，源任务门槛通过 {summary['source_gate_passed']}/{len(finals)}；报告更新于 {escape(summary['reported_utc'])}。三个可逆输入基变换与三个初始化构成九个配对设置，{len(config['groups'])} 种源组合共享输入、初始化、训练更新数与每任务曝光。</p>
<p>输入为完整 F5⁴ 的 625 点。P 组合输出 z1、z2、z3、z4；Mk 输出 z1、z2、z3、z3+k·z4，本轮 k={coefficients}。所有组合都是四个独立任务，秩均为 4，完整源答案联合分布均为 625 个等概率状态，每列均匀、每对互信息为零。完整输出分布也严格匹配；所有源答案联合起来都保留全部输入信息。</p>{physical_note}
<p>同一输入变换 T 交换 z3、z4。在单独编码每个源答案的独热特征中，P 保留四个任务方向；M1、M4 保留三个；M2、M3 保留两个。由有限域字符方向正交性可精确推出独热核 CKA 为 1、0.75、0.5。训练前登记的待检验预测是：隐藏表征是否也出现 P &gt; 平均(M1,M4) &gt; 平均(M2,M3)。精确公式仅适用于指定的理想源答案编码，不能直接当成学习后隐藏层的定理。</p>
<img src="symmetry.png"><table><tr><th>源组合</th><th>理想答案核预测</th><th>隐藏 CKA</th><th>源字符方向方差比例</th><th>源学习门槛</th></tr>{table}</table>
<p>完整配对设置 {summary['paired_settings']}/9，其中满足所登记三层排序 {summary['ordered_paired_settings']}/{summary['paired_settings']}。每个设置内所有源组合的训练前隐藏表征完全相同，可排除不同初始表征造成的组间差异。逐设置差值和全部中间检查点均保留。</p>
<aside>这项试验测量训练输入全集上的表征几何，不检验新实体泛化或迁移。输出统计严格匹配仍不能保证各任务的优化难度与学习轨迹相同，因而同时报告源准确率门槛和学习过程。九个设置由三个输入基与三个初始化交叉而成，并不是九个互不相关的新数据集；不据此报告普适显著性结论。</aside>
<p>训练前协议记录：<a href="../six_hour_session/{preregistration}">登记时间及冻结协议</a>。<a href="metadata.json">数据生成与严格统计审计</a> · <a href="summary.json">结果汇总</a> · <a href="final_models.csv">逐模型结果</a> · <a href="paired_contrasts.csv">配对差值</a> · <a href="curves.csv">全部曲线</a> · <a href="verification.json">核验</a> · <a href="symmetry.pdf">PDF 图</a></p></html>''')
    if verify:
        verification(output, metadata, records)
    print(json.dumps({k: summary[k] for k in ('complete_models', 'source_gate_passed', 'paired_settings', 'ordered_paired_settings', 'mean_high_minus_mid', 'mean_mid_minus_low')}))
    return summary


def plot(output, curves, summaries, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    colors = plt.get_cmap('tab10').colors
    for i, g in enumerate(config['groups']):
        selected = [r for r in curves if r['group'] == g]
        steps = sorted({r['step'] for r in selected})
        for ax, key in zip(axes[:2], ('source_min_accuracy', 'transformed_hidden_cka')):
            ax.plot(steps, [np.mean([r[key] for r in selected if r['step'] == s]) for s in steps], 'o-', color=colors[i], label=g)
    axes[0].set(xlabel='Source updates', ylabel='Minimum source-task accuracy', ylim=(0, 1.04))
    axes[1].set(xlabel='Source updates', ylabel='Hidden CKA under swap T', ylim=(0, 1.04))
    axes[1].legend()
    for i, r in enumerate(summaries):
        axes[2].scatter(r['ideal_code_prediction'], r['hidden_cka'], color=colors[config['groups'].index(r['group'])], label=r['group'])
        axes[2].annotate(r['group'], (r['ideal_code_prediction'], r['hidden_cka']), xytext=(5, 5), textcoords='offset points')
    axes[2].plot([.45, 1.03], [.45, 1.03], ':', color='gray', label='Identity reference')
    axes[2].set(xlabel='Exact categorical-code CKA', ylabel='Learned hidden CKA', xlim=(.45, 1.03), ylim=(0, 1.04))
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'symmetry.{suffix}', dpi=160)
    plt.close(fig)


def verification(output, metadata, records):
    expected = {k: v for k, v in metadata.items() if k not in ('fingerprint', 'registered_unix', 'audit')}
    assert hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest() == metadata['fingerprint']
    for key, path in (('code_sha256', 'experiments/field_symmetry.py'), ('model_sha256', 'experiments/models.py'), ('algebra_sha256', 'experiments/algebra.py')):
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == metadata[key]
    config = metadata['config']
    if 'world_provider_sha256' in config:
        assert hashlib.sha256(Path('experiments/matched_field.py').read_bytes()).hexdigest() == config['world_provider_sha256']
    checked = 0
    for record in records:
        assert record['fingerprint'] == metadata['fingerprint']
        if record['status'] != 'complete':
            continue
        last = record['curve'][-1]
        g, w, m = last['group'], last['world_seed'], last['model_seed']
        checkpoint = output/'checkpoints'/f'{g}_w{w}_m{m}.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == record['checkpoint_sha256']
        inputs, _, _, basis = world(config['p'], config['dimension'], w)
        latent = inputs@basis.T % config['p']
        lookup = {tuple(z): i for i, z in enumerate(latent)}
        transformed = latent.copy(); transformed[:, [2, 3]] = transformed[:, [3, 2]]
        ids = np.array([lookup[tuple(z)] for z in transformed])
        feature = np.load(output/f'{g}_w{w}_m{m}_step{config["steps"]}_features.npy')
        assert feature.shape == (625, config['features'])
        assert np.isclose(linear_cka(feature, feature[ids]), last['transformed_hidden_cka'], rtol=0, atol=1e-12)
        assert record['source_gate_passed'] == (last['source_min_accuracy'] >= config['source_accuracy_gate'])
        checked += 1
    output.joinpath('verification.json').write_text(json.dumps({'status': 'passed', 'complete_models_checked': checked,
        'frozen_training_code_and_checkpoints': True, 'transformed_cka_recomputed_from_saved_features': True,
        'exact_joint_statistics_and_ideal_kernel': True}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default='results/field_symmetry'); parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(); summarize(args.output, args.verify)
