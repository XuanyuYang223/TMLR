"""Assemble the timed run's results after its existing completion audit.

Reporting only: no model fitting, choice of checkpoint, or outcome-based change
to a registered assay. Missing supplementary checks stay visibly missing.
"""
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import time

from .inverse_functional_report import table
from .longrun_engine import atomic_json
from .permworld_combinations import sha


ROOT = Path('results/overnight_inverse_functional')
SUPP = Path('results/overnight_inverse_residual_geometry')
NAMES = {
    'initial': '预训练起点', 'ordinary': '普通微调',
    'correct_geometry': '正确几何对齐', 'matched_mismatch': '答案匹配错配',
    'distillation': '输出蒸馏', 'distillation_geometry': '蒸馏＋正确几何',
    'distillation_mismatch': '蒸馏＋错配', 'raw_correct': '蒸馏＋原始几何',
    'residual_correct': '蒸馏＋答案均值残差正确几何',
    'residual_mismatch': '蒸馏＋答案均值残差错配',
    'ONE_END_ridge': '教师前缀线性读出',
    'teacher_query_output': '教师原答案输出',
    'native_ONE_END_ridge': '冻结目标前缀线性读出',
    'native_query_output': '冻结目标原答案输出',
    'inverse_teacher_query': '取逆教师原答案输出',
    'teacher_length_mode': '教师预测拟合的长度众数',
}


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def percent(value):
    return '未计算' if value is None else f'{100 * value:.2f}%'


def interval(value):
    return '[' + ', '.join(f'{v:+.3f}' for v in value) + ']'


def means_table(rows):
    output = []
    for row in rows:
        geom = row.get('geometry', {})
        cka = geom.get('correct_cka', row.get('cka'))
        residual = geom.get('answer_residual_contrast')
        output.append([NAMES.get(row['condition'], row['condition']),
                       percent(row['accuracy']), percent(row.get('nonmodal_accuracy')),
                       f'{cka:.5f}' if cka is not None else '—',
                       f'{residual:+.5f}' if residual is not None else '—'])
    return table(['方法', '整体准确率', '非众数准确率', '逐长度隐藏CKA',
                  '答案均值残差：正确−错配'], output)


def contrasts_table(rows):
    output = []
    for row in rows:
        left, right = row['contrast'].split('-')
        value = row['accuracy_pp']
        output.append([NAMES.get(left, left) + ' − ' + NAMES.get(right, right),
                       f"{value['mean']:+.3f}",
                       interval(value['paired_bootstrap_95']),
                       interval(value['three_pair_bootstrap_95']),
                       f"{value['positive_repeats']}/6"])
    return table(['固定对比', '准确率差（百分点）', '六重复经验bootstrap95%',
                  '三模型对经验bootstrap95%', '正差重复'], output)


def assemble():
    completion = read(ROOT / 'completion.json')
    assert completion and completion['status'] == 'complete'
    main = read(ROOT / 'summary.json')
    residual = read(SUPP / 'summary.json')
    lengths = read(ROOT / 'heldout_lengths/summary.json')
    source_probe = read(ROOT / 'readout_probe/summary.json')
    target_probe = read(ROOT / 'frozen_target_probe/summary.json')
    extra_verification = read(ROOT / 'addendum_verification/verification.json')
    base_verification = read(ROOT / 'verification.json')
    supplementary_verification = read(SUPP / 'verification.json')
    budget_verification = read(ROOT / 'budget_verification.json')
    primary = [r for r in main['means'] if r['step'] in [0, 40000]]
    contrasts = [r for r in main['contrasts'] if r['primary']]
    stamp = datetime.now(timezone.utc).isoformat()
    checks = {
        'main': base_verification is not None,
        'residual': supplementary_verification is not None,
        'matched_training_budgets': budget_verification is not None,
        'readouts_and_extra_lengths': bool(extra_verification and extra_verification.get('status') == 'complete'),
    }
    summary = {
        'assembled_utc': stamp, 'completion_sha256': sha(ROOT / 'completion.json'),
        'deadline_utc': completion['deadline_utc'],
        'main_primary_means': primary, 'main_primary_contrasts': contrasts,
        'residual_primary_means': [r for r in residual['means'] if r['step'] in [0, 10000]] if residual else [],
        'residual_primary_contrasts': [r for r in residual['contrasts'] if r['primary']] if residual else [],
        'extra_length_means': lengths['means'] if lengths else [],
        'source_readout_means': source_probe['means'] if source_probe else [],
        'target_readout_means': target_probe['means'] if target_probe else [],
        'independent_verification_available': checks,
        'all_requested_primary_neural_fits_complete': bool(
            main['all36_fits_complete_before_test'] and residual
            and 10000 in residual['complete_matched_endpoints']),
        'limitations': [
            'Known inverse relation was provided; spontaneous algebra discovery is not tested.',
            'Six fine-tuning repeats depend on three frozen pretrained source/target pairs and one parent data world.',
            'Main and residual assays share supports, U pools and original pretrained models.',
            'Extra lengths were absent only from the present intervention, not from original pretraining.',
            'Removing answer group means does not remove all output information.',
            'Different length-specific readout optimization is diagnostic, not a matched neural intervention.',
        ],
    }
    atomic_json(ROOT / 'final_results.json', summary)
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>取逆关系实验：夜间结果</title><style>body{font-family:system-ui;max-width:1200px;margin:30px auto;padding:20px;line-height:1.7}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{border:1px solid #ddd;padding:8px;text-align:left}img{max-width:100%}.note{background:#f5f5f5;padding:15px}</style><body><h1>取逆关系实验：夜间结果</h1>'
    html += '<p>截止时间：2026年10月7日上午10点（洛杉矶）；汇总UTC：' + escape(stamp) + '。</p>'
    html += '<p>核心任务关系：recoils(x)=descents(inverse(x))。冻结descents教师，比较正确输入对应与答案匹配错配是否带来可泛化的几何及目标准确率收益。全部主要终点预先固定，不根据测试选模型。</p>'
    html += '<h2>主实验：有能力的预训练起点</h2><p>36个模型、6组配对重复、每个固定40000更新；每组192训练标签＋64验证标签＋16384无标签输入。目标已有原任务预训练，标签预算指本轮微调。共同的新测试集有5120个输入。</p>'
    html += means_table(primary) + contrasts_table(contrasts)
    html += '<p>区间是共享固定测试集下的经验汇总，模型对仅有三个，不能按六个独立源模型解释。非众数由测试打开前冻结的教师训练期长度众数定义。</p><img src="trajectory.png" alt="所有预先固定更新数的测试准确率和CKA">'
    html += '<h2>补充干预：同答案内部的几何</h2>'
    if residual:
        html += '<p>24个模型，主要终点10000更新；共同独立测试有5120输入。移除教师预测答案分组的均值，比较同组内正确与错配。训练数据、模型对与主实验共享，批次及几何行数不同。</p>'
        html += means_table(summary['residual_primary_means'])
        html += contrasts_table(summary['residual_primary_contrasts'])
        html += '<p>完整匹配的已登记更新数：' + escape(str(residual['complete_matched_endpoints'])) + '。未完成全部条件的更高剂量不作匹配对比。</p>'
    else:
        html += '<p>补充干预未形成完整结果，不将其计作验证。</p>'
    html += '<h2>补充泛化：本轮微调未使用的长度</h2>'
    if lengths:
        html += '<p>新的4096个排列，长度12、18、24、28。这些长度曾用于原专门模型预训练，因此这里测试的是微调干预的长度泛化。</p>'
        html += means_table(lengths['means']) + contrasts_table(lengths['contrasts'])
    else:
        html += '<p>尚无完整结果。</p>'
    html += '<h2>信息来源：冻结前缀的线性答案读出</h2>'
    html += '<p>每个重复、每个长度各拟合一个固定惩罚读出器，共60个。仅使用无标签输入的教师预测，网络权重不更新。教师前缀与冻结目标前缀分别报告；这是信息诊断，优化及读出方式与主要干预不同。</p>'
    for probe in [source_probe, target_probe]:
        if probe:
            html += table(['读出方法', '整体准确率', '非众数准确率'], [
                [NAMES.get(row['condition'], row['condition']), percent(row['accuracy']), percent(row['nonmodal_accuracy'])]
                for row in probe['means']])
    html += '<h2>独立核验与资料</h2>'
    html += table(['核验', '完整记录'], [
        [key, '已有' if value else '缺失；相关补充结论需保留'] for key, value in checks.items()])
    html += '<p>旧模型和原数据的哈希、输入轨道排除、原前向计算、准确率分层和Gram几何均保留可复核记录。CKA变化与准确率收益分别报告；答案均值残差只控制线性分组均值，不能据此证明输出信息已被全部排除。</p>'
    links = [('final_results.json', '此次汇总数据'), ('completion.json', '截止审计清单'),
             ('report.html', '主实验逐次结果'), ('../overnight_inverse_residual_geometry/report.html', '补充干预逐次结果'),
             ('heldout_lengths/report.html', '其他长度'), ('readout_probe/report.html', '教师前缀读出'),
             ('frozen_target_probe/report.html', '冻结目标读出'), ('addendum_verification/verification.json', '补充独立核验'),
             ('confirmation/diagnostic/report.html', '旧模型新测试确认')]
    html += '<p>' + ' · '.join('<a href="' + href + '">' + label + '</a>' for href, label in links if (ROOT / href).exists()) + '</p></body></html>'
    (ROOT / 'final_report.html').write_text(html)
    atomic_json(ROOT / 'bundle_state.json', {'status': 'complete', 'updated_utc': stamp,
                'all_primary_fits_complete': summary['all_requested_primary_neural_fits_complete'],
                'independent_verification_available': checks})
    print(json.dumps({'report': str(ROOT / 'final_report.html'), 'verification': checks}), flush=True)


def run():
    plan = read(Path('configs/overnight_inverse_functional.json'))
    deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
    atomic_json(ROOT / 'bundle_protocol.json', {'registered_utc': datetime.now(timezone.utc).isoformat(),
                'code_sha256': sha(__file__), 'reporting_only': True,
                'primary_endpoints': {'main': 40000, 'residual': 10000},
                'scope': 'Assemble existing reports after the timed campaign completion; no numerical choices or training changes.'})
    while not (ROOT / 'completion.json').exists():
        campaign = read(ROOT / 'campaign_state.json') or {}
        if campaign.get('status') == 'failed' or time.time() > deadline + 600:
            atomic_json(ROOT / 'bundle_state.json', {'status': 'incomplete',
                        'updated_utc': datetime.now(timezone.utc).isoformat(),
                        'reason': 'Campaign failed or no completion within10 minutes of the deadline.',
                        'campaign_state': campaign})
            return
        time.sleep(10)
    assemble()


if __name__ == '__main__':
    run()
