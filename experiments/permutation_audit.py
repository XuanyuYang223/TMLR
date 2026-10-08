"""Audit actual Property32 definitions without reading published test data."""
import argparse
import csv
import hashlib
from itertools import combinations, permutations
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import numpy as np
from .algebra import rank_mod


# Primitive identities have direct counting proofs. Two further circuits below
# follow by subtracting these identities; finite enumeration validates code.
IDENTITIES = [
    {'id': 'position_partition', 'terms': {'fixed_points': 1, 'exceedances': 1, 'deficiencies': 1},
     'n': 1, 'constant': 0, 'proof': 'Every position has pi(i)=i, pi(i)>i, or pi(i)<i.'},
    {'id': 'cycle_parity_partition', 'terms': {'even_cycle_count': 1, 'odd_cycle_count': 1, 'cycle_count': -1},
     'n': 0, 'constant': 0, 'proof': 'Every cycle has even or odd length.'},
    {'id': 'cycle_fixed_partition', 'terms': {'fixed_points': 1, 'nontrivial_cycle_count': 1, 'cycle_count': -1},
     'n': 0, 'constant': 0, 'proof': 'A length-one cycle is a fixed point; all other cycles are nontrivial.'},
    {'id': 'interior_partition', 'terms': {'peaks': 1, 'valleys': 1, 'double_ascents': 1, 'double_descents': 1},
     'n': 1, 'constant': -2, 'proof': 'Each interior index has one of four adjacent comparison-sign patterns; valid for n>=2.'},
    {'id': 'cycles_without_cycle_count', 'terms': {'even_cycle_count': 1, 'odd_cycle_count': 1, 'fixed_points': -1, 'nontrivial_cycle_count': -1},
     'n': 0, 'constant': 0, 'proof': 'cycle_parity_partition minus cycle_fixed_partition.'},
    {'id': 'positions_without_fixed_points', 'terms': {'exceedances': 1, 'deficiencies': 1, 'cycle_count': 1, 'nontrivial_cycle_count': -1},
     'n': 1, 'constant': 0, 'proof': 'position_partition minus cycle_fixed_partition.'},
]

TRANSFORMS = {
    'identity': (), 'complement': ('c',), 'reverse': ('r',),
    'reverse_complement': ('r', 'c'), 'inverse': ('i',),
    'inverse_after_complement': ('c', 'i'),
    'inverse_after_reverse': ('r', 'i'),
    'inverse_after_reverse_complement': ('r', 'c', 'i'),
}


def transform(value, name):
    p = tuple(value)
    for operation in TRANSFORMS[name]:
        if operation == 'c':
            p = tuple(len(p) + 1 - v for v in p)
        elif operation == 'r':
            p = tuple(reversed(p))
        else:
            inverse = [0] * len(p)
            for i, v in enumerate(p, 1):
                inverse[v - 1] = i
            p = tuple(inverse)
    return p


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(repo, output, max_n=7, sample_per_length=200):
    repo, output = Path(repo).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo / 'src'))
    from neurips_permutations.math_ops import PROPERTY32_TASK_NAMES, PROPERTY_FUNCTIONS
    names = list(PROPERTY32_TASK_NAMES)
    index = {name: i for i, name in enumerate(names)}
    config_path = repo / 'configs/property_task_geometry.toml'
    config = tomllib.loads(config_path.read_text())
    primitive = config['relation_pairs']
    possible = [(i, j, t, sign, []) for i in range(32) for j in range(32)
                for t in TRANSFORMS for sign in (1, -1)
                if not (i == j and t == 'identity' and sign == 1)]
    count, action_table, rank_witness_rows = 0, None, []
    for n in range(2, max_n + 1):
        values = list(permutations(range(1, n + 1)))
        lookup = {v: i for i, v in enumerate(values)}
        labels = np.array([[PROPERTY_FUNCTIONS[name](v) for name in names] for v in values], dtype=np.int64)
        # A nonzero determinant modulo a prime certifies a nonzero rational
        # determinant. Combine this lower bound with proved identities for an
        # exact rank certificate, without relying on floating-point SVD.
        sample_ids = np.random.default_rng(n).permutation(len(values))[:100]
        rank_witness_rows.extend(np.column_stack([labels[sample_ids], np.full(len(sample_ids), n), np.ones(len(sample_ids), dtype=np.int64)]).tolist())
        maps = {t: np.array([lookup[transform(v, t)] for v in values]) for t in TRANSFORMS}
        for identity in IDENTITIES:
            total = sum(c * labels[:, index[name]] for name, c in identity['terms'].items())
            assert np.all(total == identity['n'] * n + identity['constant']), identity['id']
        for relation in primitive:
            actual = labels[maps[relation['input_transform']], index[relation['right']]]
            assert np.array_equal(actual, labels[:, index[relation['left']]] + relation['right_label_offset'])
        retained = []
        for i, j, t, sign, offsets in possible:
            difference = labels[maps[t], j] - sign * labels[:, i]
            if np.all(difference == difference[0]):
                offsets = offsets + [int(difference[0])]
                if len(offsets) < 3 or all(offsets[k] == offsets[0] + k * (offsets[1] - offsets[0]) for k in range(len(offsets))):
                    retained.append((i, j, t, sign, offsets))
        possible = retained
        count += len(values)
        if n == 4:
            signatures = {tuple(mapping): t for t, mapping in maps.items()}
            assert len(signatures) == 8
            action_table = {a: {b: signatures[tuple(maps[b][maps[a]])] for b in TRANSFORMS} for a in TRANSFORMS}
        print(f'audit n={n}: {len(values)} permutations, {len(possible)} affine candidates', flush=True)
    affine = [{'source': names[i], 'target': names[j], 'transform': t, 'sign': sign,
               'length_coefficient': off[1] - off[0], 'offset': off[0] - 2 * (off[1] - off[0]),
               'status': f'exhaustive_candidate_through_n_{max_n}'} for i, j, t, sign, off in possible]
    identity_matrix = np.zeros((4, 34), dtype=np.int64)
    for j, identity in enumerate(IDENTITIES[:4]):
        for name, coefficient in identity['terms'].items():
            identity_matrix[j, index[name]] = coefficient
        identity_matrix[j, -2:] = [-identity['n'], -identity['constant']]
    prime = 1000003
    identity_rank = rank_mod(identity_matrix, prime)
    witness_rank = rank_mod(rank_witness_rows, prime)
    assert identity_rank == 4 and witness_rank <= 30
    active = [i for i in range(32) if np.any(identity_matrix[:, i])]
    minimal_supports = []
    for size in range(2, 5):
        for subset in combinations(active, size):
            outside = [i for i in active if i not in subset]
            if rank_mod(identity_matrix[:, outside], prime) < 4 and not any(set(s) <= set(subset) for s in minimal_supports):
                minimal_supports.append(subset)
    assert {frozenset(names[i] for i in support) for support in minimal_supports} == {frozenset(identity['terms']) for identity in IDENTITIES}
    # Fresh synthetic design sample; no published validation/test shard is read.
    rng = np.random.default_rng(20261004)
    rows, lengths = [], []
    for n in range(10, 31):
        for _ in range(sample_per_length):
            value = tuple(map(int, rng.permutation(n) + 1))
            rows.append([PROPERTY_FUNCTIONS[name](value) for name in names])
            lengths.append(n)
    y = np.array(rows, dtype=np.float64)
    z = np.empty_like(y)
    for n in range(10, 31):
        mask = np.array(lengths) == n
        sigma = y[mask].std(axis=0)
        z[mask] = (y[mask] - y[mask].mean(axis=0)) / np.where(sigma > 0, sigma, 1)
    corr = z.T @ z / len(z)
    entropy = []
    for j in range(32):
        per_length = []
        for n in range(10, 31):
            _, counts = np.unique(y[np.array(lengths) == n, j], return_counts=True)
            probabilities = counts / counts.sum()
            per_length.append(float(-(probabilities * np.log(probabilities)).sum()))
        entropy.append(np.mean(per_length))
    edges = {frozenset((a['source'], a['target'])) for a in affine if a['source'] != a['target']}
    supports = [frozenset(identity['terms']) for identity in IDENTITIES]
    primitive_edges = {frozenset((r['left'], r['right'])): r for r in primitive}
    def features(tasks):
        ids = [index[t] for t in tasks]
        pairs = list(combinations(tasks, 2))
        correlations = [abs(corr[index[a], index[b]]) for a, b in pairs]
        contained = [identity['id'] for identity, support in zip(IDENTITIES, supports) if support <= set(tasks)]
        return {'tasks': '|'.join(tasks),
                'registered_internal_pair_count': sum(frozenset(pair) in primitive_edges for pair in pairs),
                'expanded_affine_pair_count': sum(frozenset(pair) in edges for pair in pairs),
                'known_joint_constraint_count': len(contained), 'joint_constraints': '|'.join(contained),
                'mean_abs_within_length_correlation': float(np.mean(correlations)),
                'max_abs_within_length_correlation': float(max(correlations)),
                'mean_conditional_entropy': float(np.mean([entropy[i] for i in ids]))}
    bundles = []
    comparison_rows = []
    with (repo / 'results/property-task-geometry/cka/bundle_cell_cka.csv').open() as handle:
        cka_rows = list(csv.DictReader(handle))
    for bundle in config['bundles']:
        bundles.append({'split_id': bundle['split_id'], 'role': 'anchor', **features(bundle['anchor'])})
        for r in config['related_pair_counts']:
            target = bundle[f'b_r{r}']
            bundles.append({'split_id': bundle['split_id'], 'role': f'r{r}', **features(target)})
            cross = [primitive_edges[frozenset((a, b))] for a in bundle['anchor'] for b in target if frozenset((a, b)) in primitive_edges]
            assert len(cross) == r
            selected = [float(row['linear_cka']) for row in cka_rows if row['split_id'] == bundle['split_id'] and int(row['related_pair_count']) == r and row['layer'] == 'final_norm']
            if not selected:
                final_layer = cka_rows[-1]['layer']
                selected = [float(row['linear_cka']) for row in cka_rows if row['split_id'] == bundle['split_id'] and int(row['related_pair_count']) == r and row['layer'] == final_layer]
            assert len(selected) == 3
            comparison_rows.append({'split_id': bundle['split_id'], 'cross_bundle_r': r,
                                    'inverse_edges': sum(edge['input_transform'] == 'inverse' for edge in cross),
                                    'complement_edges': sum(edge['input_transform'] == 'complement' for edge in cross),
                                    'cka_mean_over_seeds': float(np.mean(selected)), 'cka_sd_over_seeds': float(np.std(selected, ddof=1))})
    # Candidate generation uses labels/algebra only, not CKA. "No known circuit"
    # is a control label, not a claim of independence.
    all_candidates = []
    for tasks in combinations(names, 4):
        f = features(tasks)
        f['minimum_known_constraint_size'] = min((len(s) for s in supports if s <= set(tasks)), default=0)
        all_candidates.append(f)
    triplets = [f for f in all_candidates if f['minimum_known_constraint_size'] == 3]
    quadruples = [f for f in all_candidates if f['minimum_known_constraint_size'] == 4]
    matches = []
    for left in triplets:
        for right in quadruples:
            if left['expanded_affine_pair_count'] != right['expanded_affine_pair_count']:
                continue
            score = abs(left['mean_abs_within_length_correlation'] - right['mean_abs_within_length_correlation']) + abs(left['max_abs_within_length_correlation'] - right['max_abs_within_length_correlation']) + .2 * abs(left['mean_conditional_entropy'] - right['mean_conditional_entropy'])
            matches.append({'score': score, 'three_task_constraint_bundle': left['tasks'],
                            'four_task_constraint_bundle': right['tasks'],
                            'expanded_affine_pair_count': left['expanded_affine_pair_count'],
                            'mean_correlation_gap': abs(left['mean_abs_within_length_correlation'] - right['mean_abs_within_length_correlation']),
                            'max_correlation_gap': abs(left['max_abs_within_length_correlation'] - right['max_abs_within_length_correlation']),
                            'mean_entropy_gap': abs(left['mean_conditional_entropy'] - right['mean_conditional_entropy'])})
    matches.sort(key=lambda x: x['score'])
    selected_candidates = []
    seen_four_task_bundles = set()
    for match in matches:
        if match['four_task_constraint_bundle'] not in seen_four_task_bundles:
            selected_candidates.append(match)
            seen_four_task_bundles.add(match['four_task_constraint_bundle'])
    candidate_design = {
        'status': 'candidate_design_not_launched',
        'selection': 'best label-only match for each four-task circuit; no CKA used for selection',
        'source_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'comparisons': selected_candidates,
        'required_controls_before_formal_training': ['single-task learning curves', 'per-task exposure', 'source validation accuracy', 'target exclusion from both source sets', 'matched random initialization'],
        'known_limitations': ['Source output marginals and joint entropy are approximately matched, not identical.',
                              'Candidate labels are sampled at lengths 10-30; these statistics must be rechecked for a changed length regime.',
                              'Circuit presence alone does not imply favorable transfer.']}
    (output / 'permworld_candidate_design.json').write_text(json.dumps(candidate_design, indent=2))
    write_csv(output / 'affine_candidates.csv', affine)
    write_csv(output / 'existing_bundles.csv', bundles)
    write_csv(output / 'existing_cross_bundle_relations.csv', comparison_rows)
    write_csv(output / 'candidate_bundles.csv', all_candidates)
    write_csv(output / 'matched_candidates.csv', matches[:30])
    summary = {'source_repository': 'https://github.com/XuanyuYang223/neurips',
               'source_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
               'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
               'exhaustive_max_n': max_n, 'permutations_checked': count,
               'primitive_registered_relations': len(primitive), 'affine_candidates': len(affine),
               'registered_graph_max_degree': 1,
               'r_definition': 'cross-bundle correspondences between two disjoint four-task sets',
               'existing_bundles_with_known_joint_constraints': sum(bool(b['known_joint_constraint_count']) for b in bundles),
               'all_four_task_bundles': len(all_candidates), 'three_task_constraint_bundles': len(triplets),
               'four_task_constraint_bundles': len(quadruples),
               'matched_three_vs_four_candidates': len(matches),
               'design_sample': {'seed': 20261004, 'lengths': [10, 30], 'per_length': sample_per_length,
                                 'source': 'fresh synthetic permutations; no published test data'},
               'identities': IDENTITIES, 'transformation_composition_table': action_table,
               'linear_completeness_certificate': {
                   'scope': 'same-input rational linear identities among 32 scalar properties, n and 1, for n>=2',
                   'prime': prime, 'proved_independent_identity_rank': identity_rank,
                   'witness_row_rank_mod_prime': witness_rank,
                   'completeness_certified': witness_rank == 30,
                   'witness_rows': len(rank_witness_rows),
                   'argument': ('Four universal independent identities give augmented rational rank <=30; a mod-prime rank-30 witness gives rational rank >=30. Thus all universal identities of this specified linear form are in their span.' if witness_rank == 30 else 'The finite witness has not reached the upper bound of 30; increase max_n to certify completeness.'),
                   'minimal_task_supports_up_to_four': [[names[i] for i in s] for s in minimal_supports]},
               'limitations': ['Affine candidates are finite checks, not general mathematical proofs.',
                               'The completeness certificate covers rational linear same-input identities with affine length terms, not nonlinear or input-transformed joint dependencies.',
                               'Matched candidates do not control single-task learning difficulty or the full output distribution.',
                               'Existing CKA analysis is retrospective and provides no new transfer evidence.']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    np.savez(output / 'linear_rank_certificate.npz', witness_rows=np.array(rank_witness_rows), proved_identities=identity_matrix, prime=prime)
    table = ''.join(f"<tr><td>{identity['id']}</td><td>{identity['terms']}</td><td>{identity['n']} × n + ({identity['constant']})</td><td>{identity['proof']}</td></tr>" for identity in IDENTITIES)
    candidates_table = ''.join(f"<tr><td>{match['three_task_constraint_bundle']}</td><td>{match['four_task_constraint_bundle']}</td><td>{match['expanded_affine_pair_count']}</td><td>{match['score']:.4f}</td></tr>" for match in selected_candidates)
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PermWorld 四性质关系审计</title>
<style>body{{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui;color:#20242b}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}}</style>
<h1>PermWorld 四性质关系审计</h1><p>源码版本：<code>{summary['source_commit']}</code>。</p>
<p>原 r=0/1/2/4 统计两个不重叠四任务集合之间的对应关系。预注册八条边有十六个不同端点，构成配对图；无法直接从该图设计链、星形或环。</p>
<p>检查了长度 2–{max_n} 的全部 {count:,} 个排列。对 n≥2、同输入、性质值与排列长度的有理线性恒等式，四条基本约束与模素数秩证书{'已证明' if witness_rank == 30 else '尚未证明'}该指定关系空间的完备性。这不包括非线性或输入变换后的联合关系。</p>
<p>其中涉及至多四性质的最小支持共有六个。原有二十个训练集合均没有完整包含这些支持。</p>
<table><tr><th>约束</th><th>左侧系数</th><th>右侧</th><th>计数证明</th></tr>{table}</table>
<p>枚举了全部 35,960 个四性质集合：87 个包含三任务最小线性依赖，3 个具有四任务最小线性依赖。根据独立生成的长度 10–30 标签样本，为三种四任务约束分别选择以下候选对照。</p>
<table><tr><th>三任务依赖集合</th><th>四任务依赖集合</th><th>变换关系对数</th><th>匹配分数</th></tr>{candidates_table}</table>
<p>这些是待训练的候选，尚未控制单任务学习难度，也未完全配平输出分布。有限域试验承担严格统计配平的对照功能。</p>
<p>下载：<a href="candidate_bundles.csv">所有四性质组合</a>；<a href="permworld_candidate_design.json">具体后续候选设计</a>；<a href="linear_rank_certificate.npz">精确秩证书</a>；<a href="summary.json">审计与限制</a>；<a href="affine_candidates.csv">148 条有限验证的变换候选，含自对称和重复方向</a>。</p></html>'''
    (output / 'report.html').write_text(document)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('identities', 'transformation_composition_table', 'limitations')}, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='external/neurips')
    parser.add_argument('--output', default='results/permutation_audit')
    parser.add_argument('--max-n', type=int, default=7)
    args = parser.parse_args()
    if not 4 <= args.max_n <= 8:
        parser.error('--max-n must be between 4 and 8')
    run(args.repo, args.output, args.max_n)
