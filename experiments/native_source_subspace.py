"""Post hoc numeric-answer contrast space versus its orthogonal complement."""
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .permworld_combinations import select_groups, sha
from .permworld_combinations_report import within_length_center
from .six_hour_report import write_rows


def contrast_basis(weights):
    weights = np.asarray(weights, dtype=np.float64)
    centered = weights-weights.mean(0)
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular > max(singular[0], 1e-20)*1e-10))
    return vectors[:rank].T, centered


def kernels(hidden, basis, centered_weights, lengths, centering):
    hidden = np.asarray(hidden, dtype=np.float64)
    if centering == 'within_length': hidden = within_length_center(hidden, lengths)
    hidden = hidden-hidden.mean(0)
    width = basis.shape[0]; blocks = hidden.reshape(len(hidden), -1, width)
    projected = (blocks@basis).reshape(len(hidden), -1)
    contrast = (blocks@centered_weights.T).reshape(len(hidden), -1)
    full = hidden@hidden.T; row = projected@projected.T
    null = full-row
    assert np.trace(null) >= -1e-7
    return {'full': full, 'numeric_contrast_space': row, 'numeric_null_space': null,
            'centered_numeric_logits': contrast@contrast.T}


def cka(left, right):
    denominator = np.sqrt(np.square(left).sum()*np.square(right).sum())
    return float((left*right).sum()/denominator) if denominator else None


def run():
    torch.set_num_threads(4)
    plan = json.loads(Path('configs/six_hour_session.json').read_text())
    config = json.loads(Path(plan['base_config']).read_text()); root = Path(plan['output'])
    output = root/'numeric_subspace'; output.mkdir(exist_ok=True)
    arch = json.loads((root/'architecture_selection.json').read_text())['selected']
    groups = select_groups(config)
    sys.path.insert(0, str(Path(config['repository'])/'src'))
    from neurips_permutations.passage import TOKEN_TO_ID
    numeric = [TOKEN_TO_ID[f'{i:02d}'] for i in range(config['lengths'][1]+1)]
    with np.load(root/'dataset/data.npz') as archive: lengths = archive['representation_lengths']
    checkpoint_hashes = {p.name: sha(p) for p in (root/'multi/checkpoints').glob('*.pt') if '_step' not in p.name}
    signature = {'code_sha256': sha(__file__), 'data_sha256': sha(root/'dataset/data.npz'),
                 'checkpoint_sha256': checkpoint_hashes, 'classes': list(range(config['lengths'][1]+1)),
                 'scope': 'post hoc after all first native adaptation outcomes; same 24 models, all two landmarks and all four components',
                 'interpretation': 'null preserves numeric class logit differences, not other-vocabulary decisions, future-task information, or fine-tuning behavior'}
    if (output/'metadata.json').exists(): assert json.loads((output/'metadata.json').read_text())['signature'] == signature
    else: atomic_json(output/'metadata.json', {'registered_utc': datetime.now(timezone.utc).isoformat(), 'signature': signature})
    initial_weights = {}
    for seed in plan['model_seeds']:
        model = make_model(config, arch, seed, 'cpu')
        initial_weights[seed] = model.lm_head.weight.detach().numpy()[numeric].astype(np.float64)
    rows, energies = [], []
    feature_hashes = {}
    for group in groups:
        values = {}
        for seed in plan['model_seeds']:
            run_id = f"{arch['id']}_{group['id']}_s{seed}"
            checkpoint = root/'multi/checkpoints'/f'{run_id}.pt'
            record = json.loads((root/'multi'/f'{run_id}.json').read_text())
            assert checkpoint_hashes[checkpoint.name] == record['checkpoint_sha256']
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
            weights = state['lm_head.weight'].numpy()[numeric].astype(np.float64)
            for status, w in (('initial', initial_weights[seed]), ('trained', weights)):
                basis, centered = contrast_basis(w)
                assert basis.shape[1] == len(numeric)-1
                for landmark in ('ONE_END', 'source_query_concat'):
                    if status == 'initial':
                        paths = [root/'initial'/f's{seed}_features.npy'] if landmark == 'ONE_END' else [root/'landmarks'/f'initial_s{seed}_query_{t}.npy' for t in group['tasks']]
                    else:
                        paths = [root/'multi'/f'{run_id}_features.npy'] if landmark == 'ONE_END' else [root/'landmarks'/f'{run_id}_source_query_concat.npy']
                    for path in paths: feature_hashes[str(path)] = sha(path)
                    hidden = np.concatenate([np.load(path) for path in paths], axis=1)
                    for mode in ('raw', 'within_length'):
                        result = kernels(hidden, basis, centered, lengths, mode)
                        for component, kernel in result.items(): values[seed, status, landmark, mode, component] = kernel
                        energies.append({'group': group['id'], 'seed': seed, 'status': status, 'landmark': landmark,
                                         'centering': mode, 'numeric_contrast_rank': basis.shape[1],
                                         'numeric_contrast_variance_fraction': float(np.trace(result['numeric_contrast_space'])/np.trace(result['full']))})
                    # The actual source-number logit differences are unchanged by projection.
                    block = hidden.reshape(len(hidden), -1, arch['d_model']).astype(np.float64)
                    projection = (block@basis)@basis.T
                    np.testing.assert_allclose(projection@centered.T, block@centered.T, rtol=0, atol=1e-10)
        for landmark in ('ONE_END', 'source_query_concat'):
            for mode in ('raw', 'within_length'):
                for component in ('full', 'numeric_contrast_space', 'numeric_null_space', 'centered_numeric_logits'):
                    for a, b in combinations(plan['model_seeds'], 2):
                        initial = cka(values[a, 'initial', landmark, mode, component], values[b, 'initial', landmark, mode, component])
                        trained = cka(values[a, 'trained', landmark, mode, component], values[b, 'trained', landmark, mode, component])
                        rows.append({'group': group['id'], 'seed_a': a, 'seed_b': b, 'landmark': landmark,
                                     'centering': mode, 'component': component, 'initial_cka': initial,
                                     'trained_cka': trained, 'change': trained-initial})
        print(json.dumps({'numeric_subspace_complete': group['id']}), flush=True)
    summaries = []
    for landmark in ('ONE_END', 'source_query_concat'):
        for mode in ('raw', 'within_length'):
            for component in ('full', 'numeric_contrast_space', 'numeric_null_space', 'centered_numeric_logits'):
                cells = [r for r in rows if (r['landmark'], r['centering'], r['component']) == (landmark, mode, component)]
                summaries.append({'landmark': landmark, 'centering': mode, 'component': component,
                                  'initial_cka': float(np.mean([r['initial_cka'] for r in cells])),
                                  'trained_cka': float(np.mean([r['trained_cka'] for r in cells])),
                                  'positive_changes': sum(r['change'] > 0 for r in cells), 'cells': len(cells)})
    write_rows(output/'cross_seed.csv', rows); write_rows(output/'variance.csv', energies)
    atomic_json(output/'summary.json', {'status': 'complete', 'models': 24, 'cross_seed_endpoints': len(rows),
                                      'all_numeric_margin_projection_checks_passed': True,
                                      'cross_seed': summaries, 'variance': energies, 'feature_sha256': feature_hashes})
    table = ''.join(f"<tr><td>{r['landmark']}</td><td>{r['centering']}</td><td>{r['component']}</td><td>{r['initial_cka']:.3f}</td><td>{r['trained_cka']:.3f}</td><td>{r['positive_changes']}/{r['cells']}</td></tr>" for r in summaries)
    (output/'report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>数值答案读出子空间</title><style>body{{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd}}aside{{background:#f2f4f7;padding:16px}}</style><h1>查询 CKA 与数值答案子空间</h1><p>这是第一套迁移测试之后追加的源表征诊断。对同一批 24 个模型，分析 ONE_END 与四个训练源查询拼接，初始与训练状态、原始与去长度两种中心化，全部四种分量均报告。数值类别 0–30 的源输出权重去类别均值后秩为 30；其行空间的正交投影保持全部数值类别的 logit 差值。另报告该空间的正交补及中心化数值 logits。</p><table><tr><th>位置</th><th>中心化</th><th>分量</th><th>初始 CKA</th><th>训练 CKA</th><th>上升</th></tr>{table}</table><aside>数值类正交补对这 31 个类别之间的 logit 差值没有贡献；它仍可能影响其他词表类别、将来任务或微调。不能称其为无用、未训练或非语义空间。投影没有修改模型，也不是因果干预。拼接查询只比较相同任务集合，三个种子配对彼此相关；不同空间本身使用不同度量，高 CKA 不可代替迁移证据。</aside><p><a href="metadata.json">范围与模型指纹</a> · <a href="cross_seed.csv">全部端点</a> · <a href="variance.csv">分量方差占比</a> · <a href="summary.json">汇总与特征指纹</a></p></html>''')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__': run()
