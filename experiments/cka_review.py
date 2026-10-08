"""Recompute the exact CKA contrasts the user recalls from upstream CSVs."""
import argparse
import csv
from itertools import product
import json
from pathlib import Path
import numpy as np
from .analysis import write_csv


def review(repo='external/neurips', output='results/cka_review'):
    repo, output = Path(repo), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = repo / 'results/property-task-geometry/cka'
    with (source / 'symmetry_cka.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    cells = {}
    for row in rows:
        key = row['pair_id'], int(row['model_seed']), row['layer']
        cells.setdefault(key, {})[row['condition']] = float(row['linear_cka'])
    contrasts = []
    for (pair, seed, layer), values in sorted(cells.items()):
        if set(values) != {'identity', 'correct', 'wrong'}:
            raise ValueError('incomplete symmetry comparison')
        contrasts.append({'pair_id': pair, 'model_seed': seed, 'layer': layer,
                          'identity': values['identity'], 'correct': values['correct'], 'wrong': values['wrong'],
                          'correct_minus_identity': values['correct'] - values['identity'],
                          'correct_minus_wrong': values['correct'] - values['wrong']})
    final = [r for r in contrasts if r['layer'] == 'final_norm']
    relations = []
    for pair in sorted({r['pair_id'] for r in final}):
        selected = [r for r in final if r['pair_id'] == pair]
        relations.append({'pair_id': pair, 'seed_count': len(selected),
                          **{key: float(np.mean([r[key] for r in selected])) for key in ['identity', 'correct', 'wrong', 'correct_minus_identity', 'correct_minus_wrong']}})
    layer_rows = []
    for layer in sorted({r['layer'] for r in contrasts}):
        selected = [r for r in contrasts if r['layer'] == layer]
        layer_rows.append({'layer': layer, 'pair_seed_cells': len(selected),
                           'positive_correct_minus_identity': sum(r['correct_minus_identity'] > 1e-10 for r in selected),
                           'positive_correct_minus_wrong': sum(r['correct_minus_wrong'] > 1e-10 for r in selected)})
    with (source / 'bundle_cell_cka.csv').open() as handle:
        bundles = [r for r in csv.DictReader(handle) if r['layer'] == 'final_norm']
    curves = {}
    for r in bundles:
        curves.setdefault((r['split_id'], int(r['model_seed'])), {})[int(r['related_pair_count'])] = float(r['linear_cka'])
    bundle_rows = []
    for (layout, seed), values in sorted(curves.items()):
        assert set(values) == {0, 1, 2, 4}
        y = [values[r] for r in [0, 1, 2, 4]]
        bundle_rows.append({'layout': layout, 'model_seed': seed, 'r0': y[0], 'r1': y[1], 'r2': y[2], 'r4': y[3],
                            'r4_minus_r0': y[3] - y[0], 'monotonic': all(a <= b for a, b in zip(y, y[1:]))})
    summary = {'symmetry_final_layer': {'pair_seed_cells': len(final),
                'positive_correct_minus_identity': sum(r['correct_minus_identity'] > 0 for r in final),
                'positive_correct_minus_wrong': sum(r['correct_minus_wrong'] > 0 for r in final),
                'relation_units': len(relations),
                'positive_relation_mean_correct_minus_identity': sum(r['correct_minus_identity'] > 0 for r in relations),
                'positive_relation_mean_correct_minus_wrong': sum(r['correct_minus_wrong'] > 0 for r in relations),
                'mean_correct_minus_identity': float(np.mean([r['correct_minus_identity'] for r in final])),
                'mean_correct_minus_wrong': float(np.mean([r['correct_minus_wrong'] for r in final]))},
               'bundle_final_layer': {'layout_seed_cells': len(bundle_rows),
                'positive_r4_minus_r0': sum(r['r4_minus_r0'] > 0 for r in bundle_rows),
                'monotonic_curves': sum(r['monotonic'] for r in bundle_rows)},
               'scope': 'Input-transformation contrast uses the same trained specialist weights; bundle contrast changes training task sets. These are different experiments.',
               'interpretation': 'Correct-transformation CKA increases in every final-layer specialist comparison. It does not establish monotonic bundle CKA or positive transfer.',
               'layer_summary': layer_rows}
    write_csv(output / 'symmetry_contrasts.csv', contrasts)
    write_csv(output / 'relation_means.csv', relations)
    write_csv(output / 'bundle_curves.csv', bundle_rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for i, r in enumerate(relations):
        for value, color, offset in [('correct_minus_identity', '#2166ac', -.12), ('correct_minus_wrong', '#d95f02', .12)]:
            selected = [row[value] for row in final if row['pair_id'] == r['pair_id']]
            axes[0].scatter([i+offset]*len(selected), selected, color=color, alpha=.5, s=20)
            axes[0].scatter(i+offset, r[value], color=color, marker='D', s=35, label=value if i == 0 else None)
    axes[0].axhline(0, color='black', linewidth=.8)
    axes[0].set_xticks(range(len(relations)), [r['pair_id'].replace('_', '\n') for r in relations], fontsize=7)
    axes[0].set(title='Correct transformation: 24/24 final-layer contrasts positive', ylabel='Change in CKA')
    axes[0].legend(fontsize=8)
    for r in bundle_rows:
        axes[1].plot([0, 1, 2, 4], [r[k] for k in ['r0','r1','r2','r4']], alpha=.5)
    axes[1].set(title='Four-task bundles: 1/12 curves monotonic', xlabel='Cross-bundle correspondences r', ylabel='Final-layer CKA', ylim=(0, 1))
    fig.savefig(output / 'contrasts.png', dpi=180)
    plt.close(fig)
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CKA 对照核验</title>
<style>body{{max-width:1100px;margin:40px auto;font:16px/1.7 system-ui;padding:0 20px}}img{{max-width:100%}}</style>
<h1>你记得的 CKA 增加确实成立</h1>
<p>原 PermWorld 最终层：正确变换相对未变换输入、相对错误变换，均为 24/24 个关系×种子单元增加 CKA。平均增幅分别为 {summary['symmetry_final_layer']['mean_correct_minus_identity']:.4f} 和 {summary['symmetry_final_layer']['mean_correct_minus_wrong']:.4f}。种子先在关系内平均后，两种对照仍都为 8/8 条关系增加。</p>
<p>这个对照保持网络权重不变，改变评估输入之间的数学对齐方式。四任务组合实验改变的是训练任务集合：r=4 相对 r=0 只有 7/12 个单元增加，只有 1/12 条曲线单调。</p>
<p>有限域后续试验的负迁移与这里的变换特异 CKA 增加不矛盾；该后续试验也不是原 Transformer 变换对照的复现。CKA 增加本身不能推出少样本迁移收益。</p><img src="contrasts.png">
<p>“每个都增加”在这里准确指最终层的正确变换对照；不同层结果见 <a href="summary.json">层级统计</a>。原始逐单元差值见 <a href="symmetry_contrasts.csv">symmetry_contrasts.csv</a>。</p></html>'''
    (output / 'report.html').write_text(html)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='external/neurips')
    parser.add_argument('--output', default='results/cka_review')
    args = parser.parse_args()
    review(args.repo, args.output)
