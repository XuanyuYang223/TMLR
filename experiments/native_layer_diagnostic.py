"""Post hoc causal activation-path and all-layer prefix CKA diagnostic.

CPU only. No optimizer, target labels, target-result selection, or source edits.
The activation derivative is distinct from gradients of shared parameters.
"""
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from .analysis import linear_cka
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .permworld_combinations import prompts, select_groups, sha
from .permworld_combinations_report import within_length_center
from .six_hour_report import write_rows


def activation_gradients(model, ids, mask, positions, labels):
    """Derivatives of answer loss with respect to actual block outputs."""
    captured, handles = {}, []
    def hook(name):
        def capture(module, inputs, output):
            output.retain_grad()
            captured[name] = output
        return capture
    for i, block in enumerate(model.blocks, 1):
        handles.append(block.register_forward_hook(hook(f'block_{i}')))
    handles.append(model.final_norm.register_forward_hook(hook('final_norm')))
    model.eval(); model.zero_grad(set_to_none=True)
    try:
        logits = model(ids, mask)[torch.arange(len(ids)), positions]
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        result = {name: tensor.grad.detach().clone() for name, tensor in captured.items()}
        parameter_norm = float(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None).sqrt())
    finally:
        for handle in handles: handle.remove()
        model.zero_grad(set_to_none=True)
    return result, float(loss.detach()), parameter_norm


def gradient_rows(gradients, positions):
    rows = []
    index = torch.arange(len(positions))
    for layer, gradient in gradients.items():
        # Query is '='; ONE_END occurs two positions earlier.
        for position, value in (
            ('prefix_all', gradient[torch.arange(gradient.shape[1])[None, :] <= (positions-2)[:, None]]),
            ('ONE_END', gradient[index, positions-2]),
            ('task_token', gradient[index, positions-1]),
            ('equals_query', gradient[index, positions]),
        ):
            rows.append({'layer': layer, 'position': position,
                         'gradient_l2': float(value.square().sum().sqrt()),
                         'gradient_max_abs': float(value.abs().max()),
                         'nonzero_coordinates': int((value != 0).sum()), 'coordinates': value.numel()})
    return rows


@torch.no_grad()
def layer_features(model, prefixes, lengths, pad_id):
    """All block outputs, final norm and initial embedding; no query appended."""
    model.eval()
    sites = ['embedding']+[f'block_{i}' for i in range(1, len(model.blocks)+1)]+['final_norm']
    accum = {(site, position): [] for site in sites for position in ('ONE_END', 'prefix_mean')}
    for start in range(0, len(prefixes), 64):
        x = torch.tensor(prefixes[start:start+64]); n = torch.tensor(lengths[start:start+64])
        mask = x != pad_id
        h, valid = model._embed_inputs(x, mask)
        def save(site, value):
            accum[site, 'ONE_END'].append(value[torch.arange(len(x)), 2*n+3].numpy())
            accum[site, 'prefix_mean'].append(((value*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)).numpy())
        save('embedding', h)
        for i, block in enumerate(model.blocks, 1):
            h = block(h, valid); save(f'block_{i}', h)
        save('final_norm', model.final_norm(h))
    return {f'{site}__{position}': np.concatenate(value) for (site, position), value in accum.items()}


def run():
    torch.set_num_threads(4)
    plan = json.loads(Path('configs/six_hour_session.json').read_text())
    config = json.loads(Path(plan['base_config']).read_text()); root = Path(plan['output'])
    output = root/'layer_diagnostic'; output.mkdir(exist_ok=True)
    sys.path.insert(0, str(Path(config['repository'])/'src'))
    from neurips_permutations.passage import TOKEN_TO_ID
    groups = select_groups(config)
    arch = json.loads((root/'architecture_selection.json').read_text())['selected']
    names = json.loads((root/'dataset/metadata.json').read_text())['names']
    checkpoint_hashes = {p.stem: sha(p) for p in sorted((root/'multi'/'checkpoints').glob('*.pt')) if '_step' not in p.stem}
    signature = {'code_sha256': sha(__file__), 'data_sha256': sha(root/'dataset/data.npz'),
                 'checkpoint_sha256': checkpoint_hashes, 'architecture': arch,
                 'scope': 'post hoc after landmark sensitivity; all layers retained, no target labels or adaptation results used',
                 'inference': 'FP32 CPU', 'gradient_probe': 'first 16 unseen n=10 representation inputs; all four source tasks; seed 17',
                 'interpretation': 'zero activation derivative does not imply zero shared-parameter gradients or untrained prefix representations'}
    if (output/'metadata.json').exists():
        saved = json.loads((output/'metadata.json').read_text()); assert saved['signature'] == signature
    else:
        atomic_json(output/'metadata.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'signature': signature})
    with np.load(root/'dataset/data.npz') as data:
        x, lengths, y = data['representation_input'], data['representation_lengths'], data['representation_labels']
    sites = ['embedding']+[f'block_{i}' for i in range(1, arch['layers']+1)]+['final_norm']
    initial, trained, gradients = {}, {}, []
    for seed in plan['model_seeds']:
        model = make_model(config, arch, seed, 'cpu')
        path = output/f'initial_s{seed}.npz'
        if not path.exists(): np.savez_compressed(path, **layer_features(model, x, lengths, TOKEN_TO_ID['<PAD>']))
        with np.load(path) as cache: initial[seed] = {k: cache[k] for k in cache.files}
        for group in groups:
            run_id = f"{arch['id']}_{group['id']}_s{seed}"
            record = json.loads((root/'multi'/f'{run_id}.json').read_text())
            checkpoint = root/'multi'/'checkpoints'/f'{run_id}.pt'
            assert record['status'] == 'complete' and checkpoint_hashes[run_id] == record['checkpoint_sha256']
            model = make_model(config, arch, seed, 'cpu')
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu')['model'])
            path = output/f'{run_id}.npz'
            if not path.exists(): np.savez_compressed(path, **layer_features(model, x, lengths, TOKEN_TO_ID['<PAD>']))
            with np.load(path) as cache: trained[group['id'], seed] = {k: cache[k] for k in cache.files}
            if seed == 17:
                selected = np.flatnonzero(lengths == 10)[:16]
                ids, mask, position = prompts(torch.tensor(x[selected]), torch.tensor(lengths[selected]), group['tasks'], TOKEN_TO_ID)
                labels = torch.tensor(y[selected][:, [names.index(t) for t in group['tasks']]].T.reshape(-1))
                derivative, loss, parameter_norm = activation_gradients(model, ids, mask, position, labels)
                gradients.extend({'group': group['id'], 'seed': seed, 'source_loss': loss,
                                  'shared_parameter_gradient_l2': parameter_norm, **r} for r in gradient_rows(derivative, position))
                assert all(r['gradient_max_abs'] == 0 for r in gradients if r['group'] == group['id'] and r['layer'] in (f"block_{arch['layers']}", 'final_norm') and r['position'] != 'equals_query')
                assert derivative[f"block_{arch['layers']-1}"][:, :24].abs().max() > 0
            print(json.dumps({'layer_features_complete': run_id}), flush=True)
    rows, summaries = [], []
    for site in sites:
        for position in ('ONE_END', 'prefix_mean'):
            key = site+'__'+position
            for centering in ('raw', 'within_length'):
                cells = []
                for group in groups:
                    for a, b in combinations(plan['model_seeds'], 2):
                        left = [trained[group['id'], seed][key] for seed in (a, b)]
                        right = [initial[seed][key] for seed in (a, b)]
                        if centering == 'within_length':
                            left = [within_length_center(v, lengths) for v in left]
                            right = [within_length_center(v, lengths) for v in right]
                        start, end = linear_cka(*right), linear_cka(*left)
                        row = {'layer': site, 'position': position, 'centering': centering,
                               'group': group['id'], 'seed_a': a, 'seed_b': b,
                               'initial_cka': start, 'trained_cka': end,
                               'change': end-start if start is not None and end is not None else None}
                        rows.append(row); cells.append(row)
                defined = [r for r in cells if r['change'] is not None]
                summaries.append({'layer': site, 'position': position, 'centering': centering,
                                  'cells': len(cells), 'defined_cells': len(defined),
                                  'initial_cka': float(np.mean([r['initial_cka'] for r in defined])) if defined else None,
                                  'trained_cka': float(np.mean([r['trained_cka'] for r in defined])) if defined else None,
                                  'positive_changes': sum(r['change'] > 0 for r in defined)})
    write_rows(output/'activation_gradients.csv', gradients)
    write_rows(output/'cross_seed.csv', rows); write_rows(output/'summary.csv', summaries)
    atomic_json(output/'summary.json', {'source_models': len(trained), 'initial_models': len(initial),
                                      'gradient_models': 8, 'all_layer_cross_seed_endpoints': len(rows),
                                      'gradient_endpoints': len(gradients), 'cross_seed': summaries,
                                      'final_layer_prefix_activation_derivative_zero_in_all_eight_groups': True})
    report(output, summaries, gradients)


def report(output, summaries, gradients):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sites = list(dict.fromkeys(r['layer'] for r in summaries))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, position in zip(axes, ('ONE_END', 'prefix_mean')):
        for mode, style in (('raw', '-o'), ('within_length', '--s')):
            cells = [r for r in summaries if r['position'] == position and r['centering'] == mode]
            ax.plot(range(len(sites)), [r['trained_cka']-r['initial_cka'] if r['defined_cells'] else np.nan for r in cells], style, label=mode)
        ax.axhline(0, color='gray', linewidth=1); ax.set_title(position)
        ax.set_xticks(range(len(sites)), sites, rotation=30); ax.set_ylabel('Mean trained CKA - initial CKA'); ax.legend()
    fig.tight_layout(); fig.savefig(output/'all_layers.png', dpi=160); fig.savefig(output/'all_layers.pdf'); plt.close(fig)
    def fmt(value): return '未定义（零方差）' if value is None else f'{value:.3f}'
    table = ''.join(f"<tr><td>{r['layer']}</td><td>{r['position']}</td><td>{r['centering']}</td><td>{fmt(r['initial_cka'])}</td><td>{fmt(r['trained_cka'])}</td><td>{r['positive_changes']}/{r['defined_cells']}</td></tr>" for r in summaries)
    derivative = ''.join(f"<tr><td>{r['group']}</td><td>{r['layer']}</td><td>{r['position']}</td><td>{r['gradient_l2']:.3g}</td><td>{r['nonzero_coordinates']}</td></tr>" for r in gradients)
    (output/'report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>因果梯度路径与各层 CKA</title><style>body{{max-width:1150px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui}}table{{border-collapse:collapse;width:100%}}td,th{{padding:6px;border-bottom:1px solid #ddd}}img{{width:100%}}aside{{background:#f2f4f7;padding:16px}}</style><h1>因果梯度路径与各层前缀表征</h1>
<p>这是观察到表征位置差异后追加的事后诊断。24 个源模型、3 个初始化、全部 6 个层级位置均报告。8 个组的种子 17 模型另做真实答案损失的激活梯度检查；没有更新模型，也没有使用目标任务标签。</p>
<aside>答案损失在最终查询“=”处计算。最后一层前缀输出和最终归一化后的前缀输出，对该损失的激活导数精确为零；前一层的前缀参与最后一层注意力，导数可以非零。共享参数仍由查询损失训练，因此不能称前缀“没有训练”。此结构事实提示比较表征应报告层与位置，不能独自解释全部 CKA 差异，也不是干预实验。</aside>
<img src="all_layers.png" alt="All layer CKA changes"><table><tr><th>层</th><th>位置</th><th>中心化</th><th>初始化 CKA</th><th>训练后 CKA</th><th>上升</th></tr>{table}</table>
<details><summary>全部实际激活梯度</summary><table><tr><th>组</th><th>层</th><th>位置</th><th>梯度 L2</th><th>非零坐标</th></tr>{derivative}</table></details><p>同长度去均值后，初始化 ONE_END 只由长度确定，残差理论上为零，CKA 会受浮点舍入支配；该初始化层的条件 CKA 不应作科学解释。后续层的结果均原样保留。</p>
<p><a href="metadata.json">事后范围与输入指纹</a> · <a href="activation_gradients.csv">梯度数据</a> · <a href="cross_seed.csv">所有 CKA 端点</a> · <a href="summary.json">汇总</a> · <a href="all_layers.pdf">PDF 图</a></p></html>''')


if __name__ == '__main__': run()
