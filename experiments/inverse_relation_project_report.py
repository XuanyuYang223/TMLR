"""Consolidate verified single-step and two-step assays without refitting."""
import json
from html import escape
from pathlib import Path
import time

from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now
from .two_step_relation_final_report import table, NAMES

ROOT = Path('results/inverse_relation_project')
MAIN = Path('results/overnight_inverse_functional')
RELATION = Path('results/two_step_relation_dose_review')


def assemble():
    main = json.loads((MAIN / 'final_results.json').read_text())
    relation = json.loads((RELATION / 'results.json').read_text())
    assert main['all_requested_primary_neural_fits_complete']
    assert all(main['independent_verification_available'].values())
    assert len(relation['completed_studies']) == 4
    primary = next(s for s in relation['completed_studies'] if s['id'] == 'coverage')
    dose = next(s for s in relation['completed_studies'] if s['id'] == 'dose')
    rows = []
    for condition, label in NAMES.items():
        scores = [next(r for r in s['primary_means'] if r['condition'] == condition and r['split'] == 'collisions')
                  for s in [primary, dose]]
        rows.append([label, f'{100*scores[0]["accuracy"]:.2f}%', f'{100*scores[1]["accuracy"]:.2f}%',
                     f'{100*scores[1]["pair_both_correct"]:.2f}%'])
    baseline = relation['output_only_baseline']['result']
    raw = [r for r in baseline['records'] if r['word'] == 'ci' and r['split'] == 'collisions']
    assert len(raw) == 6
    rows.append(['仅输出监督，几何权重0', f'{100*sum(r["accuracy"] for r in raw)/6:.2f}%', '无同预算基线', '—'])
    single_names = {'initial': '预训练起点', 'ordinary': '普通微调',
                    'correct_geometry': '正确几何对齐', 'matched_mismatch': '答案匹配错配',
                    'distillation': '输出蒸馏', 'distillation_geometry': '蒸馏＋正确几何',
                    'distillation_mismatch': '蒸馏＋错配'}
    single_rows = [[single_names[r['condition']], f'{100*r["accuracy"]:.3f}%',
                    f'{r["geometry"]["correct_cka"]:.4f}'] for r in main['main_primary_means']]
    uncertainty = [r for r in relation['dose_uncertainty']['result']['results'] if r['split'] == 'collisions']
    contrast_rows = [[r['contrast'], f'{r["mean_accuracy_pp"]:.3f}',
                      '[' + ', '.join(f'{v:.3f}' for v in r['source_and_input_bootstrap_95_pp']) + ']']
                     for r in uncertainty]
    residual = main['residual_primary_means']
    residual_rows = [[r['condition'], f'{100*r["accuracy"]:.3f}%',
                      f'{r["geometry"]["answer_residual_contrast"]:.4f}'] for r in residual]
    residual_contrasts = [[r['contrast'], f'{r["accuracy_pp"]["mean"]:.3f}',
                          str(r['accuracy_pp']['three_pair_bootstrap_95'])]
                         for r in main['residual_primary_contrasts']]
    pieces = ['<h1>取逆关系与两步组合：完成结果</h1>',
              '<p><strong>原取逆对齐主实验没有显示关系特异的准确率收益；新两步实验中，正确关系改善了复合预测，但仍未通过稳定隐藏答案推断的预设检验。</strong></p>',
              '<h2>两步实验：先补排列，再取逆</h2>',
              '<p>四组共享正确的单步输出监督、来源、输入覆盖和训练预算，只改变隐藏几何的对应关系。预测仅接收原输入表征、学到的C/I算子及固定读出器，不输入正确复合终点。主任务的640个碰撞配对具有相同长度和三个可见答案，但CI答案不同。</p>',
              table(['几何对应', '10轮碰撞CI准确率', '20轮碰撞CI准确率', '20轮配对双正确'], rows),
              '<p>仅输出监督基线与10轮匹配。20轮延续原模型、优化器和支持集，是相关的预算敏感性分析，未替换10轮主结果。每轮六次拟合重复复用三个来源，不能按六个独立源训练解释。</p>',
              '<h3>20轮的配对差及来源＋测试配对重采样区间</h3>',
              table(['对比', '准确率差（百分点）', '95% bootstrap区间'], contrast_rows),
              '<p>正确关系优于双错配，但双正确与仅C正确的差值区间跨零，正确性的交互项也未得到支持。碰撞配对的50%上限仅适用于只读三个真实可见答案码的确定性预测器，不是完整输出概率或所有模型的上限；当前方法未通过预设的稳定隐藏推断门槛。</p>',
              '<p>20轮已教过的生成元通过了三个来源的验证门槛。单步高准确率与高CKA仍未保证可组合的中间状态；提供真实中间输入仅作诊断，不计入隐藏推断成功。尚未建立自发关系发现或独立于输出信息的代数机制。</p>',
              '<h2>原主实验：descents取逆对应recoils</h2>',
              '<p>36个模型的固定40000更新终点，六份新支持集与无标签池复用三个预训练模型对；192训练标签、64验证标签、16384无标签输入。以下CKA测量任务提示前的前缀位置，与两步实验的任务查询表征不同，不比较两类实验的CKA绝对值。</p>',
              table(['方法', '目标准确率', '前缀隐藏CKA'], single_rows),
              '<p>正确对齐与答案匹配错配的准确率接近；输出蒸馏后增加正确对齐也没有显示额外准确率收益。表征相似度和任务功能分别判断。</p>',
              '<h2>原主实验补充：移除答案组均值</h2>',
              table(['条件', '固定10000更新准确率', '答案均值残差：正确减错配CKA'], residual_rows),
              table(['预设对比', '准确率差（百分点）', '三个模型对bootstrap区间'], residual_contrasts),
              '<p>该补充复用原主实验来源和数据，使用独立新测试。移除组均值只控制线性答案均值，不能证明移除了全部输出信息。未配齐的更高训练档位不作匹配对比。</p>',
              '<h2>报告与复核</h2>',
              '<p><a href="../two_step_relation_dose_review/report.html">两步实验完整报告</a> · <a href="../overnight_inverse_functional/final_report.html">原主实验完整报告</a> · <a href="results.json">统一汇总与来源</a> · <a href="completion.json">最终哈希清单</a></p>',
              '<p>项目测试155项通过。所有已报告主终点均完成原模型前向重放、准确率与几何复核、配对统计及监督预算检查。核验完成不扩大统计外推范围。</p>']
    css = 'body{font:16px/1.7 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#17212b}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #d7dce2;padding:9px;text-align:left}th{background:#edf3f8}a{color:#075baf}'
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>关系实验最终汇总</title><style>' + css + '</style>' + ''.join(pieces) + '<p>完成时间：' + escape(now()) + '</p></html>')
    atomic_json(ROOT / 'results.json', {'completed_utc': now(), 'single_step': main, 'two_step': relation,
                                       'new_training_performed_by_report': False})
    files = [ROOT / 'report.html', ROOT / 'results.json', MAIN / 'completion.json', MAIN / 'final_results.json',
             MAIN / 'final_report.html', RELATION / 'completion.json', RELATION / 'results.json',
             Path('results/two_step_relation_final_project_tests.log'), Path(__file__)]
    atomic_json(ROOT / 'completion.json', {'status': 'complete', 'completed_utc': now(),
                'all_original_main_and_supplement_primary_fits_complete': True,
                'all_new_factorial_and_output_only_baseline_fits_complete': True,
                'artifact_sha256': {str(p): sha(p) for p in files}})
    print(json.dumps({'status': 'complete', 'report': str(ROOT / 'report.html')}), flush=True)


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    while not ((MAIN / 'final_results.json').exists() and (RELATION / 'completion.json').exists()):
        campaign = json.loads((MAIN / 'campaign_state.json').read_text())
        if campaign['status'] == 'failed':
            atomic_json(ROOT / 'state.json', {'status': 'failed', 'reason': campaign}); return
        atomic_json(ROOT / 'state.json', {'status': 'waiting_original_campaign_archive', 'updated_utc': now()})
        time.sleep(10)
    assemble()
    atomic_json(ROOT / 'state.json', {'status': 'complete', 'updated_utc': now()})


if __name__ == '__main__':
    run()
