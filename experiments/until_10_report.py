"""Live, clearly scoped report for the requested until-10 follow-up."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import statistics
import time

from .longrun_engine import atomic_json


ROOT = Path('results/until_10_followup')
DEADLINE = datetime(2026, 10, 6, 17, tzinfo=timezone.utc).timestamp()


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def average(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(values) if values else None


def summarize(rows, fields, metrics):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[field] for field in fields)].append(row)
    return [{**dict(zip(fields, key)), 'models': len(values), **{field: average(values, field) for field in metrics}}
        for key, values in sorted(buckets.items())]


def source_relation_rows():
    rows = []
    sources = []
    for location, cohort in [('results/algebra_hidden_relations', '原三种子'), ('results/relation_seed_extension', '新增六种子')]:
        sources.extend((path, cohort) for path in (Path(location) / 'evaluations').glob('*.json'))
    for world in Path('results/relation_world_confirmation').glob('world*'):
        sources.extend((path, '三个新世界') for path in (world / 'evaluations').glob('*.json'))
    for path, cohort in sources:
        rec = read(path)
        source = rec.get('source', f"{rec.get('condition')}_s{rec['seed']}")
        condition = rec['condition']
        grades = rec['source_validation_accuracy']
        for method in rec['methods']:
            if method.get('view', 'query') != 'query':
                continue
            for row in method.get('results', method.get('metrics', [])):
                if row['split'] == 'answer_collisions' and row['word'] in ['c', 'i', 'ci', 'ici']:
                    rows.append({'cohort': cohort, 'source': source, 'condition': condition, 'seed': rec['seed'],
                        'method': method['method'], 'word': row['word'], 'e_accuracy': grades[0], 'C_accuracy': grades[1], 'I_accuracy': grades[2],
                        **{key: row.get(key) for key in ['answer_accuracy', 'displacement_nmse', 'pair_both_correct']}})
    return rows


def observed_rows():
    rows = []
    for path in Path('results/relation_observed_start/evaluations').glob('*.json'):
        rec = read(path)
        cohort = '原三种子' if rec['scope'] == 'original_exploratory' else '新增六种子'
        for method in rec['methods']:
            for row in method['rows']:
                if row['split'] == 'answer_collisions':
                    rows.append({'cohort': cohort, 'condition': rec['source_condition'], 'seed': rec['seed'], 'method': method['method'],
                        **{key: row.get(key) for key in ['case', 'answer_accuracy', 'displacement_nmse', 'pair_both_correct']}})
    for world in Path('results/relation_world_confirmation').glob('world*'):
        for path in (world / 'observed_start').glob('*.json'):
            rec = read(path)
            for method in rec['methods']:
                for row in method['metrics']:
                    if row['split'] == 'answer_collisions':
                        rows.append({'cohort': '三个新世界', 'condition': rec['condition'], 'seed': rec['world']['source_seed'], 'method': method['method'],
                            **{key: row.get(key) for key in ['case', 'answer_accuracy', 'displacement_nmse', 'pair_both_correct']}})
    return rows


def frozen_rows():
    rows = []
    paths = [(p, cohort) for location, cohort in [('frozen_relation_followup', '原三种子'), ('relation_frozen_seed_confirmation', '新增六种子')]
        for p in (Path('results') / location / 'evaluations').glob('*.json')]
    for path, cohort in paths:
        rec = read(path)
        if rec['fit_anchors_per_length'] != 256:
            continue
        for row in rec['metrics']:
            if row['split'] == 'answer_collisions' and row['word'] in ['ci', 'ici']:
                rows.append({'cohort': cohort, 'source': rec['source'], 'seed': rec['seed'], 'condition': rec['source_condition'],
                    'pairing': rec['pairing'], 'word': row['word'], 'visible_accuracy': statistics.mean(rec['visible_accuracy_e_C_I']),
                    **{key: row[key] for key in ['answer_accuracy', 'displacement_nmse', 'pair_both_correct']}})
    for world in Path('results/relation_world_confirmation').glob('world*'):
        for path in (world / 'observed_start').glob('*.json'):
            rec = read(path)
            source = read(world / 'source' / (rec['source'] + '.json'))
            for method in rec['methods']:
                if method['method'] not in ['frozen_full_correct', 'frozen_full_shuffled']:
                    continue
                for row in method['metrics']:
                    if row['split'] == 'answer_collisions' and row['case'] in ['base_CI', 'base_ICI']:
                        rows.append({'cohort': '三个新世界', 'source': rec['source'], 'seed': rec['world']['source_seed'],
                            'condition': rec['condition'], 'pairing': method['method'].removeprefix('frozen_full_'),
                            'word': 'ci' if row['case'] == 'base_CI' else 'ici', 'visible_accuracy': statistics.mean(source['observed_validation_accuracy']),
                            **{key: row[key] for key in ['answer_accuracy', 'displacement_nmse', 'pair_both_correct']}})
    return rows


def ordinary_rows():
    rows = []
    for location, cohort in [('results/joint_answer_matched_geometry', '旧源模型／新联合匹配测试'), ('results/ordinary_relation_seed_confirmation', '新增普通源种子'),
            ('results/ordinary_initialization_control', '新增普通源种子')]:
        for path in (Path(location) / 'evaluations').glob('*.json'):
            rec = read(path)
            for row in rec['results']:
                if row['subset'] not in ['all_joint_answer_matched_pairs']:
                    continue
                rows.append({'cohort': cohort, 'group': rec['group'], 'status': rec['status'], 'seed': rec['seed'],
                    'query_accuracy': rec.get('model_query_accuracy', rec.get('conditional_query_accuracy')),
                    'view': row['view'], 'kind': '生成元' if len(row['word']) == 1 else '未拟合复合', 'word': row['word'],
                    **{key: row[key] for key in ['pair_action_displacement_nmse', 'pair_target_nmse', 'pair_assignment_accuracy', 'paired_projection_energy_fraction']}})
    # Summarize within each model first: words are correlated endpoints.
    return summarize(rows, ['cohort', 'group', 'status', 'seed', 'view', 'kind'],
        ['query_accuracy', 'pair_action_displacement_nmse', 'pair_target_nmse', 'pair_assignment_accuracy', 'paired_projection_energy_fraction'])


def readout_rows():
    rows = []
    for path in Path('results/relation_readout_diagnostics/evaluations').glob('*.json'):
        rec = read(path)
        for cal in rec['calibrations']:
            for row in cal['rows']:
                if row['split'] == 'answer_collisions' and row['case'] in ['base_CI', 'base_ICI', 'observed_C_then_I']:
                    rows.append({'cohort': rec['scope'], 'condition': rec['source_condition'], 'seed': rec['seed'], 'readout': cal['readout'],
                        'case': row['case'], 'known_validation_accuracy': cal['known_validation_accuracy'], 'answer_accuracy': row['answer_accuracy']})
    return rows


def transport_rows():
    rows=[]
    for path in Path('results/relation_error_transport_confirmation/evaluations').glob('*.json'):
        rec=read(path)
        cohort={'original_exploratory':'原三种子','additional_six_seeds':'新增六种子'}.get(rec['cohort'],'三个新世界')
        for row in rec['rows']:
            if row['kind']=='composed_from_base' and row['split']=='answer_collisions' and row['word']=='ci':
                components=row['transported_local_error_energy_over_final_displacement']
                rows.append({'cohort':cohort,'condition':rec['source_condition'],'seed':rec['seed'],
                    'C_error_gain_squared':rec['C_error_effective_I_gain_squared'],
                    'transported_C_error_energy':components[0],'final_I_local_error_energy':components[1],
                    'cross_terms':row['cross_terms_over_final_displacement'],'total_error':row['hidden_displacement_nmse']})
    return rows


def known_route_rows():
    rows = []
    for path in Path('results/known_route_readout_control/evaluations').glob('*.json'):
        rec = read(path)
        cohort = {'original_exploratory': '原三种子', 'additional_six_seeds': '新增六种子'}.get(rec['scope'], '三个新世界')
        for method in rec['methods']:
            for row in method['rows']:
                if row['split'] == 'answer_collisions' and row['case'] in ['base_CI', 'base_ICI']:
                    rows.append({'cohort': cohort, 'condition': rec['source_condition'], 'seed': rec['seed'],
                        'readout': method['method'], 'case': row['case'], 'answer_accuracy': row['answer_accuracy'],
                        'known_validation_accuracy': method['true_known_validation_accuracy']})
    return rows


def ordinary_learning_rows():
    rows = []
    for folder, cohort in [('algebra_structure_replication', '旧源模型／新联合匹配测试'),
            ('ordinary_relation_seed_confirmation', '新增普通源种子')]:
        for path in (Path('results') / folder / 'source').glob('*.json'):
            rec = read(path)
            if rec['job']['id'] not in ['novel_records', 'novel_minima_pair', 'novel_peak_mixed']:
                continue
            final_step = rec['steps']
            final = [r for r in rec['curve'] if r['step'] == final_step]
            rows.append({'cohort': cohort, 'group': rec['job']['id'], 'seed': rec['job']['seed'],
                'steps': final_step, 'total_labels': rec['total_labels'], 'per_task_exposures': rec['per_task_exposures'],
                'source_audit_accuracy': average(rec['source_audit'], 'accuracy'),
                'source_validation_accuracy': average(final, 'accuracy'),
                'task_accuracies': rec['source_audit']})
    return rows


def specificity_rows():
    rows = []
    for path in Path('results/conditional_relation_specificity/evaluations').glob('*.json'):
        rec = read(path)
        cohort = '新增普通源种子' if rec['cohort'] == 'additional_ordinary_seeds' else '旧源模型／新联合匹配测试'
        for row in rec['results']:
            if len(row['word']) > 1:
                rows.append({'cohort': cohort, 'group': rec['group'], 'status': rec['status'], 'seed': rec['seed'],
                    'view': row['view'], 'best_wrong_minus_correct_nmse': row['best_wrong_minus_correct_nmse'],
                    'pair_derangement_minus_correct_nmse': row['pair_derangement_minus_correct_nmse'],
                    'correct_action_first_fraction': float(row['correct_action_rank_among_8'] == 1)})
    return summarize(rows, ['cohort', 'group', 'status', 'seed', 'view'],
        ['best_wrong_minus_correct_nmse', 'pair_derangement_minus_correct_nmse', 'correct_action_first_fraction'])


def block_rows():
    rows = []
    for path in Path('results/task_block_geometry_control/evaluations').glob('*.json'):
        rec = read(path)
        cohort = '新增普通源种子' if rec['seed'] in [17203, 18211, 19213] else '旧源模型／新联合匹配测试'
        for view, values in rec['results'].items():
            for row in values:
                if len(row['word']) > 1:
                    rows.append({'cohort': cohort, 'status': rec['status'], 'seed': rec['seed'], 'view': view,
                        'block_pair_action_displacement_nmse': row['block_pair_action_displacement_nmse'],
                        'block_pair_target_nmse': row['block_pair_target_nmse'],
                        'correct_block_first_fraction': float(row['correct_block_permutation_rank_among_24'] == 1)})
    return summarize(rows, ['cohort', 'status', 'seed', 'view'],
        ['block_pair_action_displacement_nmse', 'block_pair_target_nmse', 'correct_block_first_fraction'])


def conditioning_rows():
    rows = []
    for path in Path('results/matched_prediction_conditioning/evaluations').glob('*.json'):
        rec = read(path)
        for selector, selection in rec['selections'].items():
            values = [r for r in rec['results'] if r['selector'] == selector and r['view'] == 'source_query_numeric_null' and len(r['word']) > 1]
            trained = average([r for r in values if r['status'] == 'trained'], 'pair_target_nmse')
            random = average([r for r in values if r['status'] == 'random'], 'pair_target_nmse')
            rows.append({'cohort': rec['cohort'], 'group': rec['group'], 'seed': rec['seed'], 'selector': selector,
                'pairs': selection['pairs'], 'trained_composite_target_nmse': trained, 'random_composite_target_nmse': random,
                'paired_training_improvement': random - trained if trained is not None else None,
                'source_path': str(path)})
    return sorted(rows, key=lambda r: (r['cohort'], r['group'], r['seed'], r['selector']))


def prefix_rows():
    rows = []
    for path in Path('results/joint_answer_matched_geometry/evaluations').glob('*.json'):
        rec = read(path)
        for row in rec['results']:
            if row['view'] == 'ONE_END' and row['subset'] == 'all_joint_answer_matched_pairs':
                rows.append({'cohort': '旧源模型／新联合匹配测试', 'group': rec['group'], 'seed': rec['seed'], 'status': rec['status'],
                    'kind': '生成元' if len(row['word']) == 1 else '未拟合复合',
                    'pair_target_nmse': row['pair_target_nmse'], 'pair_action_displacement_nmse': row['pair_action_displacement_nmse']})
    for path in Path('results/ordinary_prefix_confirmation/evaluations').glob('*.json'):
        rec = read(path)
        for row in rec['results']:
            rows.append({'cohort': '新增普通源种子', 'group': rec['group'], 'seed': rec['seed'], 'status': rec['status'],
                'kind': '生成元' if len(row['word']) == 1 else '未拟合复合',
                'pair_target_nmse': row['pair_target_nmse'], 'pair_action_displacement_nmse': row['pair_action_displacement_nmse']})
    return summarize(rows, ['cohort', 'group', 'seed', 'status', 'kind'], ['pair_target_nmse', 'pair_action_displacement_nmse'])


def confidence_rows():
    rows = []
    for path in Path('results/confidence_residual_geometry/evaluations').glob('*.json'):
        rec = read(path)
        for row in rec['results']:
            rows.append({'group': rec['group'], 'seed': rec['seed'], 'status': rec['status'],
                'kind': '生成元' if len(row['word']) == 1 else '未拟合复合',
                'pair_target_nmse': row['pair_target_nmse'], 'pair_action_displacement_nmse': row['pair_action_displacement_nmse'],
                'retained_pair_energy_fraction': rec['retained_test_pair_energy_fraction'],
                'projected_residual_pair_energy_fraction': rec['projected_residual_pair_energy_fraction']})
    return summarize(rows, ['group', 'seed', 'status', 'kind'],
        ['pair_target_nmse', 'pair_action_displacement_nmse', 'retained_pair_energy_fraction', 'projected_residual_pair_energy_fraction'])


def proxy_rows():
    rows = []
    for path in Path('results/confidence_proxy_geometry/evaluations').glob('*.json'):
        rec = read(path)
        for row in rec['results']:
            rows.append({'group': rec['group'], 'seed': rec['seed'], 'status': rec['status'],
                'kind': '生成元' if len(row['word']) == 1 else '未拟合复合',
                'pair_target_nmse': row['pair_target_nmse'], 'pair_action_displacement_nmse': row['pair_action_displacement_nmse']})
    return summarize(rows, ['group', 'seed', 'status', 'kind'], ['pair_target_nmse', 'pair_action_displacement_nmse'])


def table(rows, columns):
    def value(obj):
        if obj is None:
            return '—'
        if isinstance(obj, float):
            return f'{obj:.4f}'
        return html.escape(str(obj))
    return '<table><tr>' + ''.join('<th>' + label + '</th>' for _, label in columns) + '</tr>' + ''.join(
        '<tr>' + ''.join('<td>' + value(row.get(key)) + '</td>' for key, _ in columns) + '</tr>' for row in rows) + '</table>'


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    relation = source_relation_rows()
    relation_summary = summarize(relation, ['cohort', 'condition', 'method', 'word'],
        ['e_accuracy', 'C_accuracy', 'I_accuracy', 'answer_accuracy', 'displacement_nmse', 'pair_both_correct'])
    observed = summarize(observed_rows(), ['cohort', 'condition', 'method', 'case'], ['answer_accuracy', 'displacement_nmse', 'pair_both_correct'])
    frozen = summarize(frozen_rows(), ['cohort', 'condition', 'pairing', 'word'], ['visible_accuracy', 'answer_accuracy', 'displacement_nmse', 'pair_both_correct'])
    ordinary = summarize(ordinary_rows(), ['cohort', 'group', 'status', 'view', 'kind'],
        ['query_accuracy', 'pair_action_displacement_nmse', 'pair_target_nmse', 'pair_assignment_accuracy', 'paired_projection_energy_fraction'])
    readout = summarize(readout_rows(), ['cohort', 'condition', 'readout', 'case'], ['known_validation_accuracy', 'answer_accuracy'])
    routes = summarize(known_route_rows(), ['cohort', 'condition', 'readout', 'case'], ['known_validation_accuracy', 'answer_accuracy'])
    specificity = summarize(specificity_rows(), ['cohort', 'group', 'status', 'view'],
        ['best_wrong_minus_correct_nmse', 'pair_derangement_minus_correct_nmse', 'correct_action_first_fraction'])
    blocks = summarize(block_rows(), ['cohort', 'status', 'view'],
        ['block_pair_action_displacement_nmse', 'block_pair_target_nmse', 'correct_block_first_fraction'])
    learning = summarize(ordinary_learning_rows(), ['cohort', 'group'],
        ['steps', 'total_labels', 'per_task_exposures', 'source_audit_accuracy', 'source_validation_accuracy'])
    prefix = summarize(prefix_rows(), ['cohort', 'group', 'status', 'kind'], ['pair_target_nmse', 'pair_action_displacement_nmse'])
    confidence = summarize(confidence_rows(), ['group', 'status', 'kind'],
        ['pair_target_nmse', 'pair_action_displacement_nmse', 'retained_pair_energy_fraction', 'projected_residual_pair_energy_fraction'])
    proxy = summarize(proxy_rows(), ['group', 'status', 'kind'], ['pair_target_nmse', 'pair_action_displacement_nmse'])
    transport=summarize(transport_rows(),['cohort','condition'],['C_error_gain_squared','transported_C_error_energy','final_I_local_error_energy','cross_terms','total_error'])
    jobs = []
    for location in ['relation_seed_extension', 'relation_world_confirmation', 'ordinary_relation_seed_confirmation']:
        folder = Path('results') / location
        sources = list(folder.glob('source/*.json')) if location != 'relation_world_confirmation' else list(folder.glob('world*/source/*.json'))
        jobs.append({'study': location, 'complete_sources': len(sources), 'state': read(folder / 'state.json')})
    summary = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'deadline_local': '2026-10-06 10:00 America/Los_Angeles',
        'deadline_reached': time.time() >= DEADLINE, 'source_jobs': jobs,
        'collision_general_scalar_answer_ceiling': .5, 'relation_word_source_and_accuracy': relation_summary,
        'observed_start_separate_scope': observed, 'identical_frozen_backbone_controls': frozen,
        'ordinary_joint_answer_matched_geometry': ordinary, 'known_only_readout_calibrations': readout,
        'ordinary_source_learning_and_budgets': learning, 'ordinary_source_learning_per_seed': ordinary_learning_rows(),
        'known_return_route_readout_controls': routes, 'conditional_correct_wrong_action_diagnostics': specificity,
        'direct_task_block_permutation_controls': blocks,
        'same_selected_pairs_answer_prediction_diagnostics': conditioning_rows(),
        'task_free_prefix_boundary_checks': prefix, 'task_free_prefix_per_seed': prefix_rows(),
        'confidence_association_residual_geometry': confidence, 'confidence_residual_per_seed': confidence_rows(),
        'output_only_hidden_proxy_geometry': proxy, 'output_only_proxy_per_seed': proxy_rows(),
        'native_CI_error_transport':transport,
        'empirical_scalar_lookup_ceiling': read('results/paired_relation_geometry/answer_code_ceiling.json'),
        'verification': read(ROOT / 'verification.json'), 'control_verification': read(ROOT / 'control_verification.json'),
        'transport_verification':read(ROOT/'transport_verification.json'),'data_provenance_verification':read(ROOT/'data_provenance_verification.json'),
        'route_verification': read(ROOT/'route_verification.json'),
        'world_fit_verification': read(ROOT/'world_fit_verification.json'),
        'specificity_verification': read(ROOT/'specificity_verification.json'),
        'conditioning_verification': read(ROOT/'conditioning_verification.json'),
        'prefix_verification': read(ROOT/'prefix_verification.json'),
        'confidence_verification': read(ROOT/'confidence_verification.json'),
        'proxy_verification': read(ROOT/'proxy_verification.json'),
        'seed_level_uncertainty': read(ROOT / 'seed_statistics.json')}
    atomic_json(ROOT / 'summary.json', summary)
    now_local = datetime.now(timezone.utc).isoformat()
    body = f'<h1>代数关系检验：持续至洛杉矶上午10:00</h1><p class="note">更新于 {now_local}。只计入已完成源模型；未完成种子不参与均值。旧主张仍限于关系监督改善留出复合预测，稳定的原始表征连续组合推断需另检验。</p>'
    new_native=[r for r in relation if r['cohort']=='新增六种子' and r['condition']=='correct_relations' and r['method']=='native_operators']
    if confidence:
        body += '<p class="note">普通四记录任务的数字读出零空间几何在三个新种子复现，但进一步回归输出置信度关联后，残差复合误差约1.278，随机为1.148，优势消失。残差只保留原配对变化能量的13%–18%。目前支持可复现的任务输出相关结构，不能排除置信度解释，也不能确认答案与置信度之外的独立代数计算机制。该回归是探索性关联控制，不是因果干预。</p>'
    ci=[r for r in new_native if r['word']=='ci'];ici=[r for r in new_native if r['word']=='ici']
    if ci and ici:
        body+=f'<p class="note">新增已完成种子：CI平均 {100*average(ci,"answer_accuracy"):.1f}%（{sum(r["answer_accuracy"]>.5 for r in ci)}/{len(ci)} 超过50%）；ICI平均 {100*average(ici,"answer_accuracy"):.1f}%（{sum(r["answer_accuracy"]>.5 for r in ici)}/{len(ici)} 超过50%）。目前不支持稳定超过标量答案上限。正确配对与错误配对的冻结源模型对照另列，不能把对照优势等同于突破该上限。</p>'
    body += '<h2>计分口径与源任务成绩</h2><p>CI、ICI分开，下面均为同一条件碰撞测试的逐排列准确率。每对三个真实已知答案及长度相同、隐藏答案不同，任意只用这些标量的确定性预测器准确率至多50%。该上限不用于可见生成元成绩或未筛选iid测试。CI与ICI预测同一个缺失属性，不能算作独立重复。各世界使用自己的新碰撞对。</p>'
    selected = [r for r in relation_summary if r['method'] in ['native_operators', 'posthoc_correct_generators'] and r['word'] in ['ci', 'ici']]
    body += table(selected, [('cohort', '队列'), ('condition', '训练条件'), ('method', '算子来源'), ('word', '隐藏词'), ('models', '源模型数'),
        ('e_accuracy', '可见e'), ('C_accuracy', '可见C'), ('I_accuracy', '可见I'), ('answer_accuracy', '隐藏答案准确率'), ('displacement_nmse', '隐藏位移NMSE')])
    body += '<h2>冻结完全相同源模型的关系对照</h2><p>每个源模型权重及数字读出固定。正确与错误配对使用相同已知输入特征、相同拟合量和验证规则；这一比较的可见任务准确率完全相同。每长度256个源锚点；不在隐藏关系上选参数。</p>'
    body += table(frozen, [('cohort', '队列'), ('condition', '冻结的源模型类型'), ('pairing', '关系配对'), ('word', '隐藏词'), ('models', '源模型数'), ('visible_accuracy', '相同可见准确率'), ('answer_accuracy', '隐藏答案准确率'), ('displacement_nmse', '隐藏位移NMSE')])
    body += '<h2>已知状态起步与连续组合</h2><p>base_CI、base_ICI只从原始h(x)连续组合。observed_C_then_I则先对真实的C输入做一次网络前向，再应用学得的I，测试训练未监督的C→CI边；它没有访问CI输入，却有额外的已知状态输入。I这一生成元类型已经受过监督，未见的是具体轨道边，因此这个成绩检验生成元在新输入上的泛化，不能解释为发现了新的关系类型。observed_I_then_CI同理，从真实已知I状态做两步组合。它们是不同输入条件，不能用单边成绩替代自主连续组合成绩。原三种子的中间状态结果先被诊断看见；新增六种子的该端点另行保存预测后检验。</p>'
    body += table([r for r in observed if r['method'] == 'native_operators'], [('cohort', '队列'), ('condition', '训练条件'), ('case', '输入与推断条件'), ('models', '源模型数'), ('answer_accuracy', '答案准确率'), ('displacement_nmse', '从该起点算的位移NMSE')])
    body += '<h2>普通训练：匹配所有任务组的答案轨道</h2><p>配对时匹配三个任务组所用六种性质在全部八个变换状态的真实答案。组间使用相同排列对，且不在新测试上重拟合生成元。配对差消除了这些真实答案、任务及长度的任意确定性编码；仍不能排除输入相关置信度等机制。数字读出零空间为这轮的新主口径，选择发生在联合匹配数据与结果出现之前；旧普通源模型尚非新的训练重复，新增普通种子及同种子随机初始化另列。原始空间、完整宽度与前缀全部保存。投影能量比例说明64维探针对配对变化覆盖多少。</p><p>这一分支直接训练四个性质任务，检验的是普通训练之后能否识别答案条件化的变换几何。它与只监督e/C/I的隐藏关系推断分支采用不同的任务与数据定义；不能把四任务几何结果当成未监督第四个性质的推断成绩。生成元由分析者在校准轨道上拟合，复合规则由框架给定，网络没有被要求自行发现关系类型。</p>'
    body += table([r for r in ordinary if r['view'] == 'source_query_numeric_null'], [('cohort', '队列'), ('group', '任务组合'), ('status', '模型'), ('kind', '变换'), ('models', '源模型数'),
        ('query_accuracy', '此条件任务准确率'), ('pair_action_displacement_nmse', '配对位移NMSE'), ('pair_target_nmse', '配对目标NMSE'), ('paired_projection_energy_fraction', '投影覆盖')])
    body += '<h2>普通源模型的学习成绩与预算</h2><p>三个任务组使用相同初始化、采样输入、更新次数和每任务监督量。源审计与验证成绩另列，且逐任务成绩保存在汇总文件。源任务难度、完整联合标签分布和最终学习成绩没有严格匹配，因此任务组之间的几何差异不能直接归因于关系闭合；同种子随机初始化及错误操作对照能检验部分解释，不能消除这一限制。</p>'
    body += table(learning, [('cohort', '队列'), ('group', '组合'), ('models', '源模型数'), ('steps', '更新次数'),
        ('total_labels', '总源标签曝光'), ('per_task_exposures', '每任务曝光'), ('source_audit_accuracy', '源审计集准确率'), ('source_validation_accuracy', '末次源验证准确率')])
    body += '<h2>无任务提示前缀：结构范围的边界</h2><p>在输入结束、任务提示尚未出现的ONE_END位置，使用相同的完整轨道校准及64维生成元拟合，然后在相同联合答案匹配测试上检验未拟合复合。新前缀分析在五个新查询结果出现后登记，属于次要边界检验，不替代查询位置主结果。新记录组前缀三个种子的配对目标误差为1.046、0.956、0.977，随机为1.019、1.011、1.006；优势明显较弱，且一个种子失败。不能把查询位置的信号推广为整个输入表征形成了同样结构。</p>'
    body += table(prefix, [('cohort', '队列'), ('group', '组合'), ('status', '模型'), ('kind', '变换'), ('models', '源模型数'),
        ('pair_target_nmse', '配对目标NMSE'), ('pair_action_displacement_nmse', '配对位移NMSE')])
    body += '<h2>条件几何：正确操作与错误操作</h2><p>固定已经保存的复合预测，比较八种真实操作的目标配对差，以及同长度中打乱排列对的目标。表中先对每个源模型的四个复合词平均，再对源种子平均；正的差值表示正确目标误差更小。正确操作排名及最佳错误操作只作诊断，不用测试结果选择变换。旧记录数组合的三个种子均只有四个复合词中的三个排名第一。新增记录组三个种子的首位比例为3/4、2/4、3/4：CI与RI均排第一，RC均未排第一，第二个种子的RCI也失败。不能写成所有复合关系都被区分。</p>'
    body += table([r for r in specificity if r['view'] == 'source_query_numeric_null'], [('cohort', '队列'), ('group', '组合'), ('status', '模型'), ('models', '源模型数'),
        ('best_wrong_minus_correct_nmse', '最佳错误操作减正确操作'), ('pair_derangement_minus_correct_nmse', '打乱配对减正确配对'), ('correct_action_first_fraction', '正确操作排名第一比例')])
    body += '<h2>离散预测答案条件化：两种模型评估同一批排列对</h2><p>新增探索性对照用训练模型筛选两种子集：两条轨道在所有状态的离散预测答案相同，或两条轨道在所有状态的答案都正确。每个训练模型及其同种子随机模型使用完全相同的选中排列对，算子及已经保存的预测均不改变。表中是读出零空间的四个复合词均值；所有组及种子都列出。该筛选依赖训练模型及其表现，不是随机测试抽样；正确子集还使用测试标签。因此它只诊断离散预测答案解释，置信度及输入答案交互仍未排除，不替代全样本主结果。组间选中子集不同，不能用本表直接比较任务组因果效应。小子集尤其需要谨慎。</p>'
    body += table(conditioning_rows(), [('cohort', '队列'), ('group', '组合'), ('seed', '源种子'), ('selector', '选样规则'), ('pairs', '同一批排列对数'),
        ('trained_composite_target_nmse', '训练模型误差'), ('random_composite_target_nmse', '同子集随机误差'), ('paired_training_improvement', '随机减训练误差')])
    body += '<h2>进一步控制输出置信度关联</h2><p>在校准FIT输入上，用四任务的中心化数字logit、softmax概率、熵、最大概率和长度编码回归数字读出零空间隐藏向量；只看校准VAL隐藏重构损失选择岭参数。条件测试的隐藏向量及真实任务标签都不参与该拟合或参数选择。然后在残差校准表征中重新拟合生成元，用其乘积预测相同条件测试的复合变化。</p><p>新记录组三个种子的残差只保留原配对变化能量的13%–18%，残差上的复合误差全部超过随机模型及零预测基线。两个混合组也没有残差优势。这说明当前控制仍不能排除输出置信度关联的解释；不能把数字读出零空间中的优势直接写成答案之外的独立机制。减法也会改变表征空间，可能移除与置信度相关的真实计算特征，因此这个负结果不能证明原几何全部由置信度造成。所有模型按同一事先固定的特征、FIT/VAL划分和参数网格分析；表中同时报告保留能量，避免把小残差当成完整表征。</p>'
    body += table(confidence, [('group', '任务组合'), ('status', '模型'), ('kind', '变换'), ('models', '源模型数'),
        ('pair_target_nmse', '残差配对目标NMSE'), ('pair_action_displacement_nmse', '残差配对位移NMSE'),
        ('retained_pair_energy_fraction', '原配对变化保留比例'), ('projected_residual_pair_energy_fraction', '64维残差覆盖')])
    body += '<h2>只用任务输出的隐藏代理：预测原隐藏向量</h2><p>另一个探索性直接对照复用已知校准阶段的输出重构映射，把当前输入的四任务数字输出分布、置信度及长度转换成隐藏代理；只在代理的校准FIT/VAL上拟合生成元。预测时使用当前起点的任务输出及固定算子，原条件测试目标隐藏向量只用于评分。新记录组三个种子的代理复合误差为0.867、0.857、0.847，平均0.857，原表征平均0.911；这里两者都针对原数字读出零空间目标，因此有相同的误差分母。</p><p>这一代理在校准阶段需要真实隐藏向量训练重构映射，并增加了一步拟合；它不是完全无隐藏信息的手工编码，也不是参数量匹配的新网络比较。它说明只凭预测时的任务输出信息就能复现这个几何预测优势，为输出层面的解释提供了更直接的替代基线；仍不能证明输出信息是原网络的唯一因果机制。所有三组、种子及初始化对照均报告。</p>'
    body += table(proxy, [('group', '任务组合'), ('status', '模型'), ('kind', '变换'), ('models', '源模型数'),
        ('pair_target_nmse', '对原隐藏目标NMSE'), ('pair_action_displacement_nmse', '对原隐藏位移NMSE')])
    body += '<h2>记录数组合：直接任务块置换</h2><p>把数学上已知的四任务重排直接作用于隐藏向量，不拟合任何矩阵。这是一个低容量的机制诊断；它使用完整宽度，而拟合算子使用64维探针，因此误差不是同容量方法之间的比较。置换关系由研究者给定，不能称为网络发现关系。</p>'
    body += table([r for r in blocks if r['view'] == 'source_query_numeric_null'], [('cohort', '队列'), ('status', '模型'), ('models', '源模型数'),
        ('block_pair_action_displacement_nmse', '配对位移NMSE'), ('block_pair_target_nmse', '配对目标NMSE'), ('correct_block_first_fraction', '正确块顺序排名第一比例')])
    body += '<h2>已知标签校准读出</h2><p>保持源网络与原算子不变，仅用源训练中的三个已知状态及其标签拟合读出。所有超参数只看已知验证集，所有读出都报告；隐藏测试不决定胜者。这是外部诊断，不是原网络未经辅助的能力。</p>'
    body += table([r for r in readout if r['condition'] == 'correct_relations'], [('cohort', '队列'), ('readout', '读出'), ('case', '推断条件'), ('models', '源模型数'), ('known_validation_accuracy', '可见验证准确率'), ('answer_accuracy', '隐藏准确率')])
    body += '<h2>读出诊断：仅回到已知状态的生成路径</h2><p>另一个探索性对照在源网络与原算子固定后，用13条仅从e/C/I出发、终点仍为e/C/I的路径校准读出，包括研究者提供的CC和II恒等关系。没有使用真实复合状态或其标签；干净对照重复相同终点的真实隐藏向量，锚点、路径数与标签完全一致。新六种子的CI中，生成路径校准为42.1%，同数量干净校准为42.6%，没有显示额外收益。该分析在新六种子结果和前两个新世界结果出现后提出，全部方法均报告，不能算预先预测的正结果。</p>'
    body += table([r for r in routes if r['condition'] == 'correct_relations'], [('cohort', '队列'), ('readout', '读出'), ('case', '推断条件'), ('models', '源模型数'), ('known_validation_accuracy', '可见验证准确率'), ('answer_accuracy', '隐藏准确率')])
    body += '<h2>误差定位与限制</h2><p>原三种子中，连续CI为50.7%，真实C状态起步后应用I为87.8%；ICI在真实IC中间状态后应用I为88.6%，但IC是隐藏状态，这个数只作教师强制诊断。逐步误差的精确分解包含前序误差运输和交叉项，不能把能量简单当作独立因果贡献。原始I算子最大奇异值约6.8–7.7。正交反射与增益限制是另外施加的先验对照，不能证明网络自行发现群规律。</p><p>新增源种子共享训练世界；三个新世界同时更换源初始化与训练输入，不能分离两种方差。词、层、投影和多个端点不是独立重复。额外读出与算子拟合不使用新隐藏标签，仍是模型外部辅助。LIS迁移分支继续暂停。</p>'
    body+=table([r for r in transport if r['condition']=='correct_relations'],[('cohort','队列'),('models','源模型数'),('C_error_gain_squared','C误差经I后的平方增益'),
        ('transported_C_error_energy','传递的C误差'),('final_I_local_error_energy','最后I局部误差'),('cross_terms','交叉项'),('total_error','总误差')])
    body+='<p>后三项与总误差均用同一个最终CI位移能量归一化；交叉项可正可负。平方增益是这批C误差方向上的实际增益，不是I矩阵最大奇异值的平方，也不是因果效应。新世界与新种子的分解协议及独立闭式重算均另存。</p>'
    body+='<p>这些增益与奇异值还依赖当前隐藏坐标及欧氏度量；它们能定位当前预测误差，但不能单独判定群表示是否成立。额外施加欧氏正交反射的对照失败，也不能当作不存在其他坐标下的可组合结构。</p>'
    for name, label in [('relation_accuracy.png', '关系监督：连续组合、已知状态起步和可见成绩'), ('ordinary_geometry.png', '普通训练：每个源种子的条件几何误差'),
            ('confidence_geometry.png', '置信度关联控制前后的几何及保留能量'),
            ('confidence_proxy_geometry.png', '只用输出信息预测原隐藏空间的代理基线')]:
        if (ROOT / name).exists():
            body += f'<h2>{label}</h2><img src="{name}" style="width:100%;height:auto"><p>点表示源种子，短线表示均值；同一模型的多个变换词先平均，避免把相关端点当作独立重复。各输入条件及投影分母在图中分开。</p>'
    if summary['verification']:
        verified = summary['verification']
        body += '<h2>独立核验</h2>' + table([verified], [('status', '状态'), ('checks_completed', '已完成检查'), ('independently_replayed_endpoints', '独立重算端点'), ('paired_initialization_and_sampling_groups_checked', '成对初始化与采样组')])
        body += '<p>核验使用独立写出的排列、性质标签及齐次仿射矩阵乘积，重新计算预测、答案与分母；冻结对照另核对只用已知数据的岭回归方程及参数选择。检查覆盖范围写入核验文件，未完成源模型不计入。</p>'
    if summary['control_verification']:
        body += table([summary['control_verification']], [('status', '对照核验状态'), ('checks_completed', '对照检查数'), ('independently_replayed_endpoints', '对照独立重算端点')])
    for key,label in [('transport_verification','误差运输核验'),('route_verification','已知路径读出独立拟合核验'),
            ('world_fit_verification','新世界冻结对照独立拟合核验'),('specificity_verification','正确错误操作诊断核验'),
            ('conditioning_verification','相同答案子集配对核验'),
            ('prefix_verification','无任务提示前缀预测核验'),
            ('confidence_verification','置信度回归与残差组合独立核验'),
            ('proxy_verification','输出代理对原隐藏目标独立核验'),
            ('data_provenance_verification','跨研究输入排除与代码协议核验')]:
        if summary[key]:body+=table([summary[key]],[('status',label),('checks_completed','检查数')])
    links = [('summary.json', '完整动态汇总'), ('verification.json', '独立核验记录'), ('control_verification.json', '独立对照核验'),
        ('seed_statistics.json', '按源种子列出的结果与区间'), ('readout_path_provenance.json', '读出校验记录修正'), ('../frozen_relation_followup/protocol.json', '冻结对照协议'),
        ('transport_verification.json','误差运输独立核验'),('data_provenance_verification.json','输入与代码来源核验'),
        ('route_verification.json', '已知路径校准独立核验'), ('../known_route_readout_control/protocol.json', '已知路径校准协议'),
        ('world_fit_verification.json', '新世界对照独立拟合'), ('specificity_verification.json', '操作目标及直接块置换核验'),
        ('conditioning_verification.json', '同子集答案控制独立核验'), ('../matched_prediction_conditioning/protocol.json', '同子集答案控制协议'),
        ('prefix_verification.json', '前缀预测独立核验'), ('../ordinary_prefix_confirmation/protocol.json', '新源前缀检验协议'),
        ('confidence_verification.json', '置信度控制独立核验'), ('../confidence_residual_geometry/protocol.json', '置信度关联控制协议'),
        ('confidence_geometry.pdf', '置信度控制图PDF'),
        ('proxy_verification.json', '输出代理独立核验'), ('../confidence_proxy_geometry/protocol.json', '输出代理几何协议'),
        ('confidence_proxy_geometry.pdf', '输出代理图PDF'), ('per_source_results.csv', '逐源模型结果CSV'),
        ('../relation_error_transport_confirmation/protocol.json','新源种子的误差分解协议'),
        ('relation_accuracy.pdf','关系预测图PDF'),('ordinary_geometry.pdf','条件几何图PDF'),
        ('../relation_frozen_seed_confirmation/protocol.json', '新增源种子冻结对照'), ('../ordinary_initialization_control/protocol.json', '同种子初始化对照'),
        ('../conditional_relation_specificity/protocol.json', '正确／错误关系诊断'), ('../task_block_geometry_control/protocol.json', '直接任务块置换基线'),
        ('../relation_seed_extension/protocol.json', '新增种子协议'), ('../relation_observed_start/protocol.json', '已知状态起步预测'),
        ('../relation_world_confirmation/protocol.json', '新训练世界协议'), ('../joint_answer_matched_geometry/protocol.json', '联合答案匹配协议'),
        ('../ordinary_relation_seed_confirmation/protocol.json', '新增普通训练种子协议'), ('../relation_readout_diagnostics/protocol.json', '读出协议'),
        ('../paired_relation_geometry/answer_code_ceiling.json', '更紧的有限样本答案上界'), ('../until_10_tests_final.log', '当前项目107项测试日志'),
        ('tests_scopes.json', '测试命令及工作目录范围'), ('../until_10_external_tests_own_root.log', '原仓库单独测试日志'),
        ('../until_10_repository_discovery_tests.log', '混合收集失败记录')]
    body += '<p>' + ' · '.join(f'<a href="{path}">{label}</a>' for path, label in links if (ROOT / path).exists()) + '</p>'
    body += '<h2>运行进度</h2>' + table(jobs, [('study', '研究'), ('complete_sources', '完成源模型数'), ('state', '当前状态')])
    css = 'body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:0 18px;color:#172334}p{line-height:1.65}table{border-collapse:collapse;font-size:12px;margin:18px 0}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.note{background:#eef4fb;padding:16px}'
    (ROOT / 'report.html').write_text(f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>代数关系后续检验</title><style>{css}</style>{body}</html>')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        summary = run()
        if not args.watch or time.time() >= DEADLINE:
            break
        time.sleep(min(45, max(0, DEADLINE - time.time())))
