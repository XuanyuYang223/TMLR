"""Summarize actual held-out action errors without claiming universal structure."""
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np

from .longrun_engine import atomic_json


def write_csv(path, rows):
    if not rows: return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def average(rows, key): return float(np.mean([r[key] for r in rows])) if rows else None


def direct_mean(result, kind):
    values = [r['full_space_displacement_nmse'] for r in result[kind] if r['full_space_displacement_nmse'] is not None]
    return float(np.mean(values)) if values else None


def endpoints(result):
    if result['status'] != 'complete': return {}
    return {'generator_nmse': average(result['generators'], 'test_nmse'),
            'generator_shuffled_nmse': average(result['generators'], 'shuffled_fit_nmse'),
            'generator_identity_nmse': average(result['generators'], 'identity_nmse'),
            'composite_nmse': average(result['composites'], 'test_nmse'),
            'composite_shuffled_nmse': average(result['composites'], 'shuffled_fit_nmse'),
            'law_consistency_nmse': average(result['laws'], 'consistency_nmse'),
            'law_prediction_nmse': average(result['laws'], 'left_prediction_nmse'),
            'wrong_order_gap': average(result['wrong_order'], 'gap_wrong_minus_correct'),
            'wrong_order_target_separation': average(result['wrong_order'], 'target_separation_nmse'),
            'orbit_sensitive_energy_fraction': result['orbit_sensitive_energy_fraction'],
            'test_pca_energy_fraction': result['test_pca_energy_fraction'],
            'probe_dimension': result['probe_dimension']}


def tables(rows, columns, digits=4):
    out = '<table><thead><tr>'+''.join(f'<th>{html.escape(label)}</th>' for _, label in columns)+'</tr></thead><tbody>'
    for r in rows:
        out += '<tr>'+''.join('<td>'+html.escape(f'{r[key]:.{digits}f}' if isinstance(r[key], float) else str(r[key]))+'</td>' for key, _ in columns)+'</tr>'
    return out+'</tbody></table>'


def run():
    plan = json.loads(Path('configs/algebra_structure.json').read_text()); root = Path(plan['output'])
    native_records = [json.loads(p.read_text()) for p in sorted((root/'native/probes').glob('*.json')) if not p.name.startswith('teacher_')]
    field_records = [json.loads(p.read_text()) for p in sorted((root/'field/probes').glob('*.json'))]
    assert len(native_records) == 48 and len(field_records) == 54
    null_records = { (r['group'], r['seed']): r for p in (root/'orbit_null').glob('*_trained.json') for r in [json.loads(p.read_text())] }
    assert len(null_records) == 24
    native_rows, action_rows = [], []
    for r in native_records:
        for view, result in r['results'].items():
            native_rows.append({'group': r['group'], 'seed': r['seed'], 'model_status': r['model_status'], 'view': view,
                'source_audit_accuracy': r['source_audit_accuracy'], **endpoints(result)})
            for kind in ('generators', 'composites'):
                for action in result[kind]:
                    action_rows.append({'group': r['group'], 'seed': r['seed'], 'model_status': r['model_status'],
                        'view': view, 'kind': kind, 'word': action.get('generator', action.get('word')),
                        'test_nmse': action['test_nmse'], 'identity_nmse': action['identity_nmse'],
                        'shuffled_fit_nmse': action['shuffled_fit_nmse'], 'full_space_nmse': action['full_space_nmse']})
    write_csv(root/'native_endpoints.csv', native_rows); write_csv(root/'native_actions.csv', action_rows)
    lookup = {(r['group'], r['seed'], r['view'], r['model_status']): r for r in native_rows}
    native_summary = []
    for group in sorted({r['group'] for r in native_rows}):
        for landmark in plan['native_landmarks']:
            view = f'{landmark}_layer4'
            trained = [lookup[group, s, view, 'trained'] for s in plan['native_seeds']]
            random = [lookup[group, s, view, 'random'] for s in plan['native_seeds']]
            null = [endpoints(null_records[group, s]['results'][landmark]) for s in plan['native_seeds']]
            nullspace = [lookup[group, s, f'{landmark}_numeric_null', 'trained'] for s in plan['native_seeds']]
            teacher = json.loads((root/'native/probes'/f'teacher_{group}.json').read_text())['results']
            native_summary.append({'group': group, 'landmark': landmark,
                'trained_generator_nmse': average(trained, 'generator_nmse'),
                'random_generator_nmse': average(random, 'generator_nmse'),
                'paired_generator_improvement': average(random, 'generator_nmse')-average(trained, 'generator_nmse'),
                'seeds_generator_better_than_random': sum(a['generator_nmse'] < b['generator_nmse'] for a, b in zip(trained, random)),
                'orbit_relabel_generator_nmse': average(null, 'generator_nmse'),
                'trained_composite_nmse': average(trained, 'composite_nmse'),
                'random_composite_nmse': average(random, 'composite_nmse'),
                'seeds_composite_better_than_random': sum(a['composite_nmse'] < b['composite_nmse'] for a, b in zip(trained, random)),
                'orbit_relabel_composite_nmse': average(null, 'composite_nmse'),
                'law_consistency_nmse': average(trained, 'law_consistency_nmse'),
                'law_prediction_nmse': average(trained, 'law_prediction_nmse'),
                'wrong_order_gap': average(trained, 'wrong_order_gap'),
                'wrong_order_target_separation': average(trained, 'wrong_order_target_separation'),
                'orbit_sensitive_energy_fraction': average(trained, 'orbit_sensitive_energy_fraction'),
                'test_pca_energy_fraction': average(trained, 'test_pca_energy_fraction'),
                'numeric_null_generator_nmse': average(nullspace, 'generator_nmse'),
                'numeric_null_variance_fraction': float(np.mean([r['results'][f'{landmark}_numeric_null']['variance_fraction_of_orbit_residual'] for r in native_records if r['group'] == group and r['model_status'] == 'trained'])),
                'teacher_numeric_generator_nmse': endpoints(teacher['numeric']).get('generator_nmse'),
                'teacher_onehot_generator_nmse': endpoints(teacher['onehot']).get('generator_nmse'),
                'source_audit_accuracy': average(trained, 'source_audit_accuracy')})
    write_csv(root/'native_group_summary.csv', native_summary)
    field_rows = []
    for r in field_records:
        result = r['results']['hidden_dihedral']; projection = r['results']['hidden_projection']
        rotation = next(a for a in result['generators'] if a['generator'] == 'a')
        cyclic = [a for a in result['composites'] if a['word'] in ('aa', 'aaa')]
        field_rows.append({'group': r['group'], 'world_seed': r['world_seed'], 'seed': r['seed'], 'model_status': r['model_status'],
            'rotation_nmse': rotation['test_nmse'], 'cyclic_composite_nmse': average(cyclic, 'test_nmse'),
            **endpoints(result), 'projection_nmse': projection['test_nmse'],
            'projection_shuffled_nmse': projection['shuffled_fit_nmse'],
            'projection_identity_nmse': projection['identity_nmse'],
            'projection_double_nmse': projection['double_prediction_nmse'],
            'projection_idempotence_nmse': projection['idempotence_consistency_nmse'],
            'projection_correct_versus_identity_gap': projection['correct_versus_identity_gap']})
    write_csv(root/'field_endpoints.csv', field_rows)
    field_summary = []
    flookup = {(r['group'], r['world_seed'], r['seed'], r['model_status']): r for r in field_rows}
    fconfig = json.loads(Path(plan['field_config']).read_text())
    for group in fconfig['groups']:
        trained = [r for r in field_rows if r['group'] == group and r['model_status'] == 'trained']
        random = [flookup[group, r['world_seed'], r['seed'], 'random'] for r in trained]
        row = {'group': group, 'crossed_world_seed_settings': len(trained)}
        for key in ('rotation_nmse', 'cyclic_composite_nmse', 'generator_nmse', 'composite_nmse',
                    'law_consistency_nmse', 'wrong_order_gap', 'projection_nmse',
                    'projection_double_nmse', 'projection_idempotence_nmse'):
            row['trained_'+key] = average(trained, key); row['random_'+key] = average(random, key)
        row['rotation_better_than_random_settings'] = sum(a['rotation_nmse'] < b['rotation_nmse'] for a, b in zip(trained, random))
        row['projection_better_than_random_settings'] = sum(a['projection_nmse'] < b['projection_nmse'] for a, b in zip(trained, random))
        teacher = [r['results']['exact_onehot_dihedral'] for r in field_records if r['group'] == group and r['model_status'] == 'trained']
        row['onehot_generator_nmse'] = float(np.mean([endpoints(r)['generator_nmse'] for r in teacher]))
        teacher_p = [r['results']['exact_onehot_projection'] for r in field_records if r['group'] == group and r['model_status'] == 'trained']
        row['onehot_projection_nmse'] = average(teacher_p, 'test_nmse')
        field_summary.append(row)
    write_csv(root/'field_group_summary.csv', field_summary)
    direct_native = [json.loads(p.read_text()) for p in (root/'direct/native').glob('*.json') if not p.name.startswith('teacher_')]
    direct_field = [json.loads(p.read_text()) for p in (root/'direct/field').glob('*.json')]
    assert len(direct_native) == 48 and len(direct_field) == 54
    direct_native_summary, direct_action_rows = [], []
    dlookup = {(r['group'], r['seed'], r['model_status']): r for r in direct_native}
    for r in direct_native:
        for view, result in r['results'].items():
            for kind in ('generators', 'composites'):
                for action in result[kind]:
                    direct_action_rows.append({'group': r['group'], 'seed': r['seed'], 'model_status': r['model_status'],
                        'view': view, 'kind': kind, 'word': action.get('generator', action.get('word')),
                        'action_status': action['status'], 'action_displacement_energy': action['action_displacement_energy'],
                        'full_space_displacement_nmse': action['full_space_displacement_nmse'],
                        'shuffled_fit_displacement_nmse': action['shuffled_fit_displacement_nmse'], 'raw_target_nmse': action['raw_target_nmse']})
    for group in sorted({r['group'] for r in direct_native}):
        for landmark in plan['native_landmarks']:
            trained = [dlookup[group, s, 'trained']['results'][landmark] for s in plan['native_seeds']]
            random = [dlookup[group, s, 'random']['results'][landmark] for s in plan['native_seeds']]
            null = [dlookup[group, s, 'trained']['results'][f'{landmark}_numeric_null'] for s in plan['native_seeds']]
            direct_native_summary.append({'group': group, 'landmark': landmark,
                'trained_generator_displacement_nmse': float(np.mean([direct_mean(r, 'generators') for r in trained])),
                'random_generator_displacement_nmse': float(np.mean([direct_mean(r, 'generators') for r in random])),
                'seeds_better_than_random': sum(direct_mean(a, 'generators') < direct_mean(b, 'generators') for a, b in zip(trained, random)),
                'trained_composite_displacement_nmse': float(np.mean([direct_mean(r, 'composites') for r in trained])),
                'random_composite_displacement_nmse': float(np.mean([direct_mean(r, 'composites') for r in random])),
                'numeric_null_generator_displacement_nmse': float(np.mean([direct_mean(r, 'generators') for r in null])),
                'law_consistency_nmse': float(np.mean([average(r['laws'], 'consistency_nmse') for r in trained])),
                'wrong_order_gap': float(np.mean([average(r['wrong_order'], 'gap_wrong_minus_correct') for r in trained]))})
    direct_field_summary = []
    for group in fconfig['groups']:
        trained = [r['results']['hidden'] for r in direct_field if r['group'] == group and r['model_status'] == 'trained']
        random = [r['results']['hidden'] for r in direct_field if r['group'] == group and r['model_status'] == 'random']
        teacher = [r['results']['onehot'] for r in direct_field if r['group'] == group and r['model_status'] == 'trained']
        direct_field_summary.append({'group': group,
            'trained_generator_displacement_nmse': float(np.mean([direct_mean(r, 'generators') for r in trained])),
            'random_generator_displacement_nmse': float(np.mean([direct_mean(r, 'generators') for r in random])),
            'trained_composite_displacement_nmse': float(np.mean([direct_mean(r, 'composites') for r in trained])),
            'random_composite_displacement_nmse': float(np.mean([direct_mean(r, 'composites') for r in random])),
            'onehot_generator_displacement_nmse': float(np.mean([direct_mean(r, 'generators') for r in teacher])),
            'law_consistency_nmse': float(np.mean([average(r['laws'], 'consistency_nmse') for r in trained]))})
    write_csv(root/'direct_native_summary.csv', direct_native_summary); write_csv(root/'direct_native_actions.csv', direct_action_rows)
    write_csv(root/'direct_field_summary.csv', direct_field_summary)
    rank_native = [json.loads(p.read_text()) for p in (root/'rank_sensitivity/native').glob('*.json')]
    rank_field = [json.loads(p.read_text()) for p in (root/'rank_sensitivity/field').glob('*.json')]
    assert len(rank_native) == 48 and len(rank_field) == 54
    rank_summary = []
    for group in sorted({r['group'] for r in rank_native}):
        for landmark in plan['native_landmarks']:
            trained = [r['results'][landmark] for r in rank_native if r['group'] == group and r['model_status'] == 'trained']
            random = [r['results'][landmark] for r in rank_native if r['group'] == group and r['model_status'] == 'random']
            rank_summary.append({'group':group, 'landmark':landmark,
                'trained_generator_displacement_nmse':float(np.mean([direct_mean(r,'generators') for r in trained])),
                'random_generator_displacement_nmse':float(np.mean([direct_mean(r,'generators') for r in random])),
                'trained_composite_displacement_nmse':float(np.mean([direct_mean(r,'composites') for r in trained])),
                'random_composite_displacement_nmse':float(np.mean([direct_mean(r,'composites') for r in random]))})
    rank_field_summary = []
    for group in fconfig['groups']:
        trained=[r['results']['hidden'] for r in rank_field if r['group']==group and r['model_status']=='trained']
        random=[r['results']['hidden'] for r in rank_field if r['group']==group and r['model_status']=='random']
        projection=[r['results']['projection'] for r in rank_field if r['group']==group and r['model_status']=='trained']
        rank_field_summary.append({'group':group,
            'trained_generator_displacement_nmse':float(np.mean([direct_mean(r,'generators') for r in trained])),
            'random_generator_displacement_nmse':float(np.mean([direct_mean(r,'generators') for r in random])),
            'trained_composite_displacement_nmse':float(np.mean([direct_mean(r,'composites') for r in trained])),
            'projection_nmse':average(projection,'test_nmse'), 'projection_idempotence_nmse':average(projection,'idempotence_consistency_nmse')})
    new_combinations=json.loads((root/'new_combinations/summary.json').read_text())
    write_csv(root/'rank_native_summary.csv',rank_summary);write_csv(root/'rank_field_summary.csv',rank_field_summary)
    summary = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'native_trained_models': 24,
        'native_random_initializations': 3, 'native_groups': 8, 'native_probe_orbits': 588, 'native_probe_states': 4704,
        'field_trained_models': 27, 'field_random_initializations': 3, 'field_worlds': 3,
        'native_summary': native_summary, 'field_summary': field_summary,
        'direct_native_summary': direct_native_summary, 'direct_field_summary': direct_field_summary,
        'rank_native_summary':rank_summary, 'rank_field_summary':rank_field_summary,
        'new_combinations':new_combinations,
        'limitations': ['One fresh permutation probe world; three source seeds per existing task group.',
            'Field source models saw the complete 625 inputs; only action probes have held-out inputs.',
            'Known source-label structure may explain query results; numeric-null-space results are diagnostic, not proof of new semantics.',
            'Independent orbit relabeling control was added after early primary endpoints.',
            'Layer/action rows are not independent repeats; task types and source learning differ across existing groups.',
            'PCA probes approximate the actual hidden action; full-space errors and retained variance must be checked.',
            'Orbit mean preprocessing can create sign actions; interpret only with the separately registered direct raw-h action-displacement probes. Raw probes were added after early orbit-centered outputs.',
            'Three new structural combinations were tested at one previously trained source seed; this is a pilot, not independent source-seed replication.',
            'No new learning rule, theorem, or claim of first discovery is established.']}
    atomic_json(root/'summary.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, landmark in zip(axes, plan['native_landmarks']):
        selected = [r for r in native_summary if r['landmark'] == landmark]
        y = np.arange(len(selected)); ax.barh(y-.2, [r['random_generator_nmse'] for r in selected], .4, label='Random initialization', color='#a4b1c1')
        ax.barh(y+.2, [r['trained_generator_nmse'] for r in selected], .4, label='Trained', color='#2563a8')
        ax.set_yticks(y, [r['group'] for r in selected]); ax.set_xlabel('Held-out generator NMSE (lower is better)'); ax.set_title(landmark)
    axes[1].legend(); fig.savefig(root/'native_actions.png', dpi=180); fig.savefig(root/'native_actions.pdf'); plt.close(fig)
    groups=sorted({r['group'] for r in direct_native});actions=('c','r','i')
    heat=np.zeros((len(groups),3))
    for i,group in enumerate(groups):
        trained=[dlookup[group,s,'trained']['results']['source_query_concat']['generators'] for s in plan['native_seeds']]
        random=[dlookup[group,s,'random']['results']['source_query_concat']['generators'] for s in plan['native_seeds']]
        heat[i]=np.mean([[a['full_space_displacement_nmse'] for a in r] for r in random],axis=0)-np.mean([[a['full_space_displacement_nmse'] for a in r] for r in trained],axis=0)
    fig,ax=plt.subplots(figsize=(8.5,5.5),constrained_layout=True)
    maximum=max(.1,float(np.max(np.abs(heat))));im=ax.imshow(heat,cmap='RdBu',vmin=-maximum,vmax=maximum,aspect='auto')
    ax.set_xticks(range(3),['C: complement','R: reverse','I: inverse']);ax.set_yticks(range(len(groups)),groups)
    ax.tick_params(labelsize=10)
    for i in range(len(groups)):
        for j in range(3):ax.text(j,i,f'{heat[i,j]:+.3f}',ha='center',va='center',fontsize=10,color='white' if abs(heat[i,j])>.6*maximum else 'black')
    ax.set_title('Operator prediction at source queries\nPositive values: training improves displacement NMSE',fontsize=11)
    fig.colorbar(im,ax=ax,label='NMSE improvement (random minus trained)')
    fig.savefig(root/'operator_effects.png',dpi=180);fig.savefig(root/'operator_effects.pdf');plt.close(fig)
    css = 'body{font:16px system-ui;max-width:1200px;margin:36px auto;padding:0 20px;color:#172334}table{border-collapse:collapse;font-size:14px;margin:20px 0}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}img{max-width:100%}p{line-height:1.6}.note{background:#eef4fb;padding:14px}'
    body = '<h1>代数变换与隐藏表征：第一轮结构检验</h1><p>研究主线是任务的精确数学关系如何体现在普通训练的表征中。LIS 迁移消融保持暂停。</p>'
    body += '<p class="note">主要结果：现有八组模型没有整体优于随机模型的完整群结构证据；position_four 的取逆出现三个源种子一致的局部可预测性。三个新组合的单种子预先预测检验中，四方向记录数组合的生成元误差、取逆排序和非交换顺序判别均符合预测。该信号集中在任务查询位置，不能推广为全部隐藏层或任务自由表征的群表示。原轨道去均值的强结果有指标构造的影响。</p>'
    body += '<h2>新任务组合：预测先于几何测量</h2><p>复用此前完成、此前未测这些几何指标的三个源模型，每组四任务、每任务每步 32 输入、20,000 步，共 256 万标签。数学预测在提取三个模型表征前保存：四方向记录数对 C/R/I 任务置换闭合，生成元平均误差和取逆误差应低于另外两组；正确 CI 应优于错误 IC。三项结果均符合预测。只测了源种子 1009，三项不是三个独立重复。</p>'
    body += tables(new_combinations['rows'], [('group','组合'), ('source_accuracy','源学习正确率'),
        ('generator_mean_nmse','生成元变化误差'), ('composite_mean_nmse','未拟合复合变化误差'),
        ('wrong_order_gap','错误顺序 − 正确顺序'), ('full_width_generator_mean_nmse','完整维度生成元误差'),
        ('full_width_composite_mean_nmse','完整维度复合误差')])
    body += '<p>新记录组的查询表征误差 0.141，对照随机初始化 0.352；复合误差 0.214 对照 0.503。ONE_END 位置为 0.421 对照 0.353，未优于随机。查询数字读出零空间的生成元误差为 0.248，正确顺序仍优于错误顺序；这排除了仅限于同一数字线性读出方向的解释，但不排除其他输出相关编码。源学习成绩和联合标签统计仍有差异。</p>'
    body += '<h2>判读依据：直接预测原始隐藏向量的变化</h2><p class="note">原轨道去均值方案可能人为构造变号规律。因此在看到早期辅助结果后，单独登记并补做直接检验：仅输入 h(x)，不输入轨道均值或目标隐藏向量；长度均值、PCA 和生成元均仅在拟合划分估计。以下数值是完整隐藏空间的变化量归一化误差，恒等预测基线为 1；越低越好。ρ=I+W，以源 h(x) 保留未投影成分，并预测真实变换造成的变化。复合词未用于拟合。</p>'
    body += tables(direct_native_summary, [('group','任务组'), ('landmark','位置'),
        ('trained_generator_displacement_nmse','训练：生成元变化误差'), ('random_generator_displacement_nmse','随机：变化误差'),
        ('seeds_better_than_random','优于随机 /3'), ('trained_composite_displacement_nmse','训练：未拟合复合变化误差'),
        ('random_composite_displacement_nmse','随机：复合变化误差'), ('numeric_null_generator_displacement_nmse','读出零空间变化误差'),
        ('law_consistency_nmse','复合律一致性误差')])
    body += '<p>相应有限域循环/二面体直接检验：</p>'+tables(direct_field_summary, [('group','任务组'),
        ('trained_generator_displacement_nmse','训练：生成元变化误差'), ('random_generator_displacement_nmse','随机：变化误差'),
        ('trained_composite_displacement_nmse','训练：复合变化误差'), ('onehot_generator_displacement_nmse','精确标签编码变化误差'),
        ('law_consistency_nmse','复合律一致性误差')])
    body += '<img src="operator_effects.png" alt="operator-specific action prediction improvements">'
    body += '<h2>完整维度敏感性检查</h2><p>在看到低维结果后，额外固定一次完整维度复核，对全部任务组与种子执行，同样只用验证集选正则化。PermWorld 的前缀保留最多 256 维，四任务查询最多 1,024 维；实际受每个任务向量的归一化约束，查询有效秩为 1,020。有限域保留全部 64 维。结论仍是操作特定信号，完整群复合尚未成立。该复核没有按测试结果搜索维度。</p>'
    body += tables(rank_field_summary,[('group','有限域任务组'),('trained_generator_displacement_nmse','完整维度生成元变化误差'),
        ('random_generator_displacement_nmse','随机变化误差'),('trained_composite_displacement_nmse','完整维度复合变化误差'),
        ('projection_nmse','完整维度投影误差'),('projection_idempotence_nmse','投影幂等一致性误差')])
    body += '<h2>辅助分析：轨道去均值表征</h2>'
    body += '<p class="note">生成元单独拟合，复合操作不参与拟合；验证集只选择岭回归强度。测试误差越低越好。表征按完整操作轨道留出，并去除轨道不变成分。复合一致性必须结合实际预测误差解释。指标不是 CKA，也不对应此前跨模型的 24/24 CKA 结果。</p>'
    body += '<p>PermWorld：24 个已有四任务训练模型、3 个匹配随机初始化；588 个新完整轨道，共 4,704 个排列，与记录的源/目标数据全部不重叠。拟合/验证/测试轨道为 336/84/168。全部四层和两个提取位置均报告。64 维 PCA 只使用拟合轨道。</p><img src="native_actions.png" alt="held-out action errors">'
    for landmark in plan['native_landmarks']:
        body += f'<h2>{landmark}：最终层</h2>'+tables([r for r in native_summary if r['landmark'] == landmark], [
            ('group','任务组'), ('trained_generator_nmse','训练：生成元误差'), ('random_generator_nmse','随机：生成元误差'),
            ('seeds_generator_better_than_random','胜过随机的种子 /3'), ('trained_composite_nmse','训练：未拟合复合误差'),
            ('random_composite_nmse','随机：复合误差'), ('orbit_relabel_generator_nmse','轨道重标记控制误差'),
            ('numeric_null_generator_nmse','数字读出零空间误差'), ('wrong_order_gap','错误顺序 − 正确顺序')])
    body += '<h2>循环、二面体与非群投影：有限域扩展</h2><p>F5⁴ 上的四坐标循环移位生成 C4，加反射生成 D4。不可逆投影 P 将第三坐标置零，严格满足 P²=P。使用此前 27 个统计匹配的模型（3 任务组 ×3 世界 ×3 源种子），各源任务正确率均为 100%。旋转/反射按完整轨道留出；投影按保留的三个坐标划分，变换后的图像也跨划分不重叠。源模型曾见过全部 625 输入，因此仅检验探针推广。32 维 PCA 与变换拟合只使用探针拟合集。</p>'
    body += tables(field_summary, [('group','任务组'), ('trained_rotation_nmse','循环生成元误差'),
        ('random_rotation_nmse','随机循环误差'), ('trained_composite_nmse','D4 复合误差'),
        ('trained_projection_nmse','投影误差'), ('random_projection_nmse','随机投影误差'),
        ('trained_projection_double_nmse','双投影预测误差'), ('trained_projection_idempotence_nmse','P² 与 P 的一致性误差'),
        ('onehot_projection_nmse','精确标签编码投影误差')])
    body += '<h2>解释范围</h2><ul>'+''.join(f'<li>{html.escape(x)}</li>' for x in summary['limitations'])+'</ul>'
    body += '<p>完整结果：<a href="direct_native_actions.csv">直接逐操作误差</a> · <a href="native_endpoints.csv">轨道去均值：全部层与子空间</a> · <a href="rank_native_summary.csv">完整维度汇总</a> · <a href="field_endpoints.csv">有限域逐设置</a> · <a href="summary.json">机器可读汇总</a> · <a href="verification.json">主核验</a> · <a href="supplementary_verification.json">补充核验</a>。</p>'
    body += '<p>相关原始研究：<a href="https://proceedings.mlr.press/v97/kornblith19a.html">Kornblith 等，CKA，2019</a>；<a href="https://proceedings.mlr.press/v202/chughtai23a.html">Chughtai 等，有限群计算机制，2023</a>。本实验尚不能确认新颖性。</p>'
    (root/'report.html').write_text(f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>Algebra structure</title><style>{css}</style>{body}</html>')
    print(json.dumps({'direct_native': direct_native_summary, 'direct_field': direct_field_summary, 'field': field_summary}, indent=2))


if __name__ == '__main__': run()
