"""Present completed factorials with their generator-competence limits."""
import json
from html import escape
from pathlib import Path

from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now


ROOT = Path('results/two_step_relation_review')
STUDIES = [('frozen', '冻结编码器', Path('results/two_step_relation_factorial')),
           ('joint', '编码器与算子联合学习', Path('results/two_step_relation_joint'))]
NAMES = {'both_correct': '双正确', 'c_correct_i_wrong': '仅补排列正确',
         'c_wrong_i_correct': '仅取逆正确', 'both_wrong': '双错配'}


def table(headers, rows):
    return '<table><thead><tr>' + ''.join('<th>' + escape(x) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    data = {'generated_utc': now(), 'completed_studies': [], 'visible_only_pilots': [],
            'primary_order': 'complement then inverse (ci)',
            'secondary_order_limit': 'For the LR-max statistic, inverse then complement (ic) has the same answer as visible complement. It is not evidence for predicting an unknown answer.',
            'procedure_order_supplied': True, 'spontaneous_discovery_established': False,
            'stable_missing_answer_inference_established': False,
            'source_replication': 'Six fitting repeats reuse three existing source initializations; they are not six independent source trainings.'}
    sections = []
    studies = list(STUDIES)
    confirmation = Path('results/two_step_relation_coverage_confirmation')
    dose = Path('results/two_step_relation_dose_confirmation')
    if (confirmation / 'completion.json').exists():
        studies.append(('coverage', '扩大输入覆盖后的新四组复核', confirmation))
    elif (confirmation / 'state.json').exists():
        state = json.loads((confirmation / 'state.json').read_text())
        data['pending_confirmation'] = {'state': state, 'completed_fits': len(list((confirmation / 'fits').glob('*.json'))),
                                        'planned_fits': 24, 'compound_test_opened': (confirmation / 'test_opened.json').exists()}
        sections.append('<h2>新四组复核进度</h2><p>状态：' + escape(state['status']) + '；已完成' +
                        str(data['pending_confirmation']['completed_fits']) + '/24。完整结果会在全部固定预算拟合及独立验证完成后加入。</p>')
    if (dose / 'completion.json').exists():
        studies.append(('dose', '20轮预算敏感性：延续原有模型', dose))
    for key, title, folder in studies:
        completion = json.loads((folder / 'completion.json').read_text())
        assert completion['status'] == 'complete'
        summary = json.loads((folder / 'summary.json').read_text())
        assert completion['artifact_sha256'][str(folder / 'summary.json')] == sha(folder / 'summary.json')
        # Reconstruct all displayed primary averages directly from raw records.
        records = json.loads((folder / 'evaluation_records.json').read_text())['records']
        primary = [r for r in summary['primary_means'] if r['view'] == 'hidden']
        rows = []
        for condition, name in NAMES.items():
            displayed = []
            for split in ['iid', 'collisions']:
                raw = [r for r in records if (r['view'], r['word'], r['split'], r['condition']) == ('hidden', 'ci', split, condition)]
                assert len(raw) == 6
                row = next(r for r in primary if r['condition'] == condition and r['split'] == split)
                assert abs(sum(r['accuracy'] for r in raw) / 6 - row['accuracy']) < 1e-12
                displayed.append(row)
            rows.append([name, f'{100*displayed[0]["accuracy"]:.3f}%', f'{100*displayed[1]["accuracy"]:.3f}%',
                         f'{100*displayed[1]["pair_both_correct"]:.3f}%', f'{displayed[1]["cka"]:.4f}'])
        contrasts = [r for r in summary['contrasts'] if r['primary'] and r['split'] == 'collisions']
        study = {'id': key, 'title': title, 'root': str(folder), 'operator_fits': summary['operator_fits'],
                 'primary_means': primary, 'primary_collision_contrasts': contrasts,
                 'fixed_prediction_gates': summary['fixed_prediction_gates'],
                 'single_generator_interpretation_gate': summary.get('single_generator_interpretation_gate', False),
                 'summary_sha256': sha(folder / 'summary.json'),
                 'completion_sha256': sha(folder / 'completion.json')}
        data['completed_studies'].append(study)
        sections.append('<h2>' + escape(title) + '</h2>' + table(['条件', 'IID CI准确率', '碰撞 CI准确率', '碰撞配对双正确', '碰撞 CI CKA'], rows))
        if key == 'joint':
            sections.append('<p>双正确减双错配的碰撞准确率差为1.328个百分点，六次配对差均为正；按三个来源聚合的bootstrap区间为[1.044,1.626]个百分点。两个关系正确性的交互项为0.182个百分点，区间跨零，尚不能声称两条关系产生超加性收益。少量来源的bootstrap区间不能代表广泛外推。</p>')
            sections.append('<p>预设的单步解释门槛未通过。三个来源的双正确验证成绩（补排列／取逆）如下；因此复合失败不能归因于仅有组合误差。</p>' +
                table(['来源', '补排列', '取逆'], [[str(seed), f'{100*g[0]:.2f}%', f'{100*g[1]:.2f}%']
                      for seed, g in zip([5081,6091,7103], summary['correct_generator_three_source_validation_accuracy'])]))
        elif key == 'frozen':
            sections.append('<p>冻结编码器上的正确单步映射也没有可靠泛化。这一轮主要定位仿射可读性的限制；四组复合成绩接近，不能单独否定关系组合假设。</p>')
        if key in ['joint', 'coverage', 'dose']:
            grades = []
            for condition, label in NAMES.items():
                fits = [json.loads(p.read_text()) for p in (folder / 'fits').glob('*.json')
                        if json.loads(p.read_text())['condition'] == condition]
                assert len(fits) == 6
                native = [sum(f['curve'][-1]['native_e_C_I_accuracy'][j] for f in fits) / 6 for j in range(3)]
                generators = [sum(f['curve'][-1]['generator_C_I_accuracy'][j] for f in fits) / 6 for j in range(2)]
                grades.append([label] + [f'{100*v:.2f}%' for v in native + generators])
            sections.append('<p>各条件已教过的任务与算子验证成绩，避免把基本映射学习差异直接称为组合能力差异：</p>' +
                            table(['条件', '原输入任务', '补排列后任务', '取逆后任务', '补排列算子', '取逆算子'], grades))
        if key in ['coverage', 'dose']:
            study['correct_generator_three_source_validation_accuracy'] = summary['correct_generator_three_source_validation_accuracy']
            established = summary['single_generator_interpretation_gate'] and all(summary['fixed_prediction_gates'].values())
            prefix = '' if key == 'coverage' else 'dose_'
            data[prefix + 'stable_missing_answer_inference_established'] = established
            data[prefix + 'missing_answer_accuracy_above_answer_code_ceiling_all_sources'] = summary['fixed_prediction_gates']['both_correct_collision_above_half_in_all3_sources']
            sections.append('<p>单步解释门槛：' + ('通过' if summary['single_generator_interpretation_gate'] else '未通过') +
                            '。固定复合预测门槛：<code>' + escape(json.dumps(summary['fixed_prediction_gates'], ensure_ascii=False)) + '</code>。</p>')
            sections.append('<p>本轮训练只提供e/C/I。预测由原输入表征依次经过学到的补排列和取逆算子得到；不输入正确复合终点。所有24次固定预算训练完成后才打开新复合测试。覆盖检查与旧配置还存在批量和曝光差异，跨轮变化不能单独归因于输入覆盖；四组内的预算与监督保持匹配。</p>')
            contrast_rows = []
            for contrast in contrasts:
                stats = contrast['accuracy_pp']
                contrast_rows.append([contrast['contrast'], f'{stats["mean"]:.3f}', str(stats['positive_repeats']) + '/6',
                                      '[' + ', '.join(f'{v:.3f}' for v in stats['three_source_bootstrap_95']) + ']'])
            sections.append(table(['碰撞准确率对比', '差值（百分点）', '正配对次数', '三个来源bootstrap区间'], contrast_rows))
            for filename, title in [('uncertainty.json', '同时重采样来源和测试输入的不确定性'),
                                    ('factorial_effects.json', '预先登记的关系类型主效应')]:
                file = folder / filename
                if file.exists():
                    result = json.loads(file.read_text())
                    data[prefix + filename[:-5]] = {'path': str(file), 'sha256': sha(file), 'result': result}
                    selected = [r for r in result['results'] if r['split'] == 'collisions' and
                                (filename == 'uncertainty.json' or (r['word'] == 'ci' and r['metric'] == 'accuracy'))]
                    rows = []
                    for r in selected:
                        interval = r['source_and_input_bootstrap_95_pp'] if filename == 'uncertainty.json' else r['three_source_bootstrap_95']
                        rows.append([r.get('contrast', r.get('effect')), f'{r.get("mean_accuracy_pp", r.get("mean")):.3f}',
                                     '[' + ', '.join(f'{v:.3f}' for v in interval) + ']'])
                    sections.append('<h3>' + title + '</h3>' + table(['对比／主效应', '碰撞准确率差（百分点）', '95% bootstrap区间'], rows))
            sections.append('<p>主效应分析承接先前登记的精确四统计量作用：补排列将缺失的第四个统计量带入前三个统计量的空间，取逆保持前三个的空间。它预期两类关系可能发挥不对称的作用，而非预设两步都正确一定优于单正确。此补充预测没有替换原主检验。</p>')
            if key == 'dose':
                sections.append('<p><strong>20轮是相关的预算敏感性分析。</strong>它延续原四组模型、优化器、输入及监督；前10轮的轨迹完整保留。后10轮按照提前登记的20轮余弦轨迹后半段运行。延长方案依据第三个来源的可见验证检查登记，早于两批新复合测试。它没有替换10轮主结果，也不是新的来源或数据重复。</p>')
                file = folder / 'paired_dose.json'
                if file.exists():
                    paired = json.loads(file.read_text())
                    data['paired_dose'] = {'path': str(file), 'sha256': sha(file), 'result': paired}
                    selected = [r for r in paired['means'] if r['word'] == 'ci' and r['split'] == 'collisions' and r['metric'] == 'accuracy_delta_pp']
                    sections.append('<h3>相同新测试输入上的20轮减10轮</h3>' + table(['条件', '碰撞 CI准确率变化（百分点）', '三个来源bootstrap区间'],
                        [[NAMES[r['condition']], f'{r["mean"]:.3f}', '[' + ', '.join(f'{v:.3f}' for v in r['three_source_bootstrap_95']) + ']'] for r in selected]))
        sections.append('<p><a href="../' + folder.name + '/report.html">完整报告</a> · <a href="../' + folder.name + '/summary.json">原始汇总</a> · <a href="../' + folder.name + '/verification.json">独立验证</a></p>')
    for file, label in [(Path('results/two_step_relation_joint/visible_feasibility_results.json'), '学习速率可见任务检查'),
                        (Path('results/two_step_relation_coverage_pilot/visible_coverage_results.json'), '增加输入覆盖的可见任务检查'),
                        (Path('results/two_step_relation_dose_pilot/visible_dose_results.json'), '第三个来源的可见任务延长检查')]:
        if file.exists():
            pilot = json.loads(file.read_text())
            data['visible_only_pilots'].append({'label': label, 'path': str(file), 'sha256': sha(file), 'result': pilot})
            sections.append('<h2>' + escape(label) + '</h2><pre>' + escape(json.dumps(pilot, ensure_ascii=False, indent=2)) + '</pre>')
    finding = '固定检验支持当前设置下的关系监督组合推断；尚未建立普通训练的自发发现或独立于输出的机制。' if data['stable_missing_answer_inference_established'] else '原定的整套正结果判据尚未全部通过，分别报告单步学习、隐藏答案推断和关系正确性的效果。'
    sections.insert(0, '<p><strong>已完成四组关系正确性实验。' + finding + '</strong></p><p>主目标是先补排列再取逆（CI）。取逆再补排列在当前统计量上与一个可见答案相同，只作次要诊断。训练和预测使用的步骤顺序由实验提供，不声称自行发现关系或推导程序。</p><p>四组共享正确的单步输出监督、训练输入边际、预算与来源，只改变几何对应。错配按长度及三个可见答案分层；离散错配排除了两步恰好回到同一样本。测试预测只接收原输入的表征和学到的算子。</p><p>碰撞配对的50%准确率上限仅适用于只使用三个真实可见答案码的确定性预测器，不是所有输出概率或所有模型的理论上限。配对双正确大于零在对照中也出现，不能单独证明正确关系的特异作用。</p>')
    css = 'body{font:16px/1.65 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#17212b}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #d7dce2;padding:9px;text-align:left}th{background:#edf3f8}pre{background:#f4f6f8;padding:16px;overflow:auto;font-size:13px}a{color:#075baf}'
    (ROOT / 'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>两步关系实验：结果与解释边界</title><style>' + css + '</style><h1>两步关系实验：结果与解释边界</h1>' + ''.join(sections) + '<p>生成时间：' + escape(data['generated_utc']) + '</p></html>')
    atomic_json(ROOT / 'results.json', data)
    print(json.dumps({'report': str(ROOT / 'report.html'), 'studies': len(data['completed_studies']), 'pilots': len(data['visible_only_pilots'])}))


if __name__ == '__main__':
    run()
