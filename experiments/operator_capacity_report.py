"""Render the completed confirmation without changing its fitted protocol."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now

ROOT = Path('results/operator_capacity_confirmation')
LABELS = {'original': '原始冻结算子', 'linear_correct': '两层线性／正确',
          'linear_wrong': '两层线性／错配', 'nonlinear_correct': '两层 GELU／正确',
          'nonlinear_wrong': '两层 GELU／错配'}


def table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def interval(contrast):
    lo, hi = contrast['two_way_bootstrap_95_pp']
    return f'{contrast["mean_pp"]:+.2f} [{lo:+.2f}, {hi:+.2f}]'


def run():
    config = json.loads(Path('configs/operator_capacity_confirmation.json').read_text())
    result = json.loads((ROOT/'evaluation.json').read_text())
    verification = json.loads((ROOT/'independent_verification.json').read_text())
    assert verification['status'] == 'passed'
    tests = (ROOT/'pytest.log').read_text()
    assert 'failed' not in tests.lower() and 'passed' in tests.lower()
    means = {r['condition']: r for r in result['means'] if r['split']=='collisions'}
    contrasts = {r['contrast']: r for r in result['contrasts'] if r['endpoint']=='accuracy'}
    primary = contrasts['interaction']
    nl = contrasts['nonlinear']; linear = contrasts['linear']
    lo, hi = primary['two_way_bootstrap_95_pp']
    if lo > 0 and min(primary['source_effects_pp']) > 0:
        conclusion = '在这三个新源和新留出测试上，非线性扩大了正确配对相对答案匹配错配的复合准确率优势，符合事先登记的方向预测。'
    elif hi < 0:
        conclusion = '非线性没有释放更大的关系特异收益；交互作用方向与事先登记的预测相反。'
    else:
        conclusion = '交互作用区间跨零，当前确认实验没有提供非线性扩大关系特异收益的明确证据。'
    if not result['all_source_gates_pass']:
        conclusion += ' 存在源模型未通过单步门槛，所有种子仍完整报告，解释需保留这一限制。'

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    colors = ['#4682b4', '#a9bed1', '#c66328', '#e8b393']
    short = ['Linear\ncorrect', 'Linear\nmatched wrong', 'GELU\ncorrect', 'GELU\nmatched wrong']
    for j, condition in enumerate(config['conditions']):
        values = [100*r['accuracy'] for r in result['records'] if r['condition']==condition and r['split']=='collisions']
        axes[0].bar(j, np.mean(values), color=colors[j], width=.65)
        axes[0].scatter(j+np.array([-.13, 0, .13]), values, color='#222222', s=26, zorder=3)
        axes[0].text(j, max(values)+2, f'{np.mean(values):.2f}%', ha='center', va='bottom')
    axes[0].set_xticks(range(4), short)
    axes[0].set_ylim(0, min(105, max(60, 100*max(r['accuracy'] for r in means.values())+15)))
    axes[0].set_ylabel('Held-out collision AB accuracy (%)')
    axes[0].set_title('A. Same 65,920 parameters and fixed B / encoder / readout')
    names = ['linear', 'nonlinear', 'interaction']
    for j, name in enumerate(names):
        row = contrasts[name]; low, high = row['two_way_bootstrap_95_pp']
        axes[1].errorbar(j, row['mean_pp'], yerr=[[row['mean_pp']-low], [high-row['mean_pp']]],
                         fmt='o', color='#27577a', capsize=5, markersize=7)
        axes[1].scatter(j+np.array([-.13, 0, .13]), row['source_effects_pp'], marker='x', s=34, color='#b84b23', zorder=4)
    axes[1].axhline(0, color='gray', linewidth=1)
    axes[1].set_xticks(range(3), ['Correct − wrong\nlinear', 'Correct − wrong\nGELU', 'Interaction\nGELU − linear'])
    axes[1].set_ylabel('Paired accuracy contrast (percentage points)')
    axes[1].set_title('B. Three source effects and paired two-way 95% intervals')
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(ROOT/f'capacity_confirmation.{ext}', dpi=180)
    plt.close(fig)

    rows = [[LABELS[c], f'{100*means[c]["accuracy"]:.2f}%', f'{100*means[c]["pair_both_correct"]:.2f}%',
             f'{100*means[c]["first_accuracy"]:.2f}%', f'{means[c]["first_state_nmse"]:.5f}',
             f'{means[c]["compound_state_nmse"]:.5f}', f'{means[c]["cka"]:.4f}']
            for c in ['original']+config['conditions']]
    source_rows = []
    for i, seed in enumerate(config['source_seeds']):
        r = {x['condition']: x for x in result['records'] if x['source']==i and x['split']=='collisions'}
        source_rows.append([seed]+[f'{100*r[c]["accuracy"]:.2f}%' for c in config['conditions']]+[
            f'{linear["source_effects_pp"][i]:+.2f}', f'{nl["source_effects_pp"][i]:+.2f}', f'{primary["source_effects_pp"][i]:+.2f}'])
    contrast_rows = [[name, interval(contrasts[name])] for name in
                     ['linear', 'nonlinear', 'interaction', 'nonlinear_gain_correct', 'nonlinear_gain_wrong']]
    gates = [[config['source_seeds'][i], ', '.join(f'{100*v:.2f}%' for v in g['native']),
              ', '.join(f'{100*v:.2f}%' for v in g['generators'])] for i, g in enumerate(result['source_known_grades'])]
    text = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>矩阵算子表达能力确认实验</title><style>body{font:16px/1.75 system-ui;max-width:1180px;margin:36px auto;padding:0 20px;color:#18232d}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{border:1px solid #d7dee3;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%}h1{font-size:27px}.lead{padding:18px;background:#eef4f7;border-left:4px solid #4682b4}code{background:#f4f4f4;padding:2px}</style><h1>矩阵：第一算子表达能力 × 配对正确性</h1>'
    text += '<p class="lead">'+conclusion+'</p>'
    text += f'<p>已完成 12 个固定 2,000 步的第一算子拟合，3 个新源初始化（{", ".join(map(str, config["source_seeds"]))}）。主终点为碰撞样本 AB 准确率的交互作用：<strong>{interval(primary)} 个百分点</strong>。区间由三个源与 128 对共享测试样本双向重采样得到；三个源仍不足以估计广泛的源间变化。</p>'
    text += '<img src="capacity_confirmation.svg" alt="四组碰撞复合准确率和正确配对收益差">'
    text += '<h2>主结果与辅助指标</h2>'+table(['第一算子','碰撞 AB','配对双正确','单步 A','第一状态 NMSE','复合状态 NMSE','复合 CKA'], rows)
    text += '<h2>逐源复核</h2>'+table(['源种子','线性正确','线性错配','GELU 正确','GELU 错配','线性正确差 pp','GELU 正确差 pp','交互 pp'], source_rows)
    text += '<h2>配对效应</h2>'+table(['对比','均值 [双向 bootstrap 95%] pp'], contrast_rows)
    text += '<h2>冻结来源的可见任务门槛</h2>'+table(['源种子','原生 e / A / B','生成元 A / B'], gates)
    text += '<p>所有组共享每个源的编码器、读出器、原始第一算子跳连与第二算子。两层残差宽度均为 256，参数量严格相同；唯一结构差别是中间 Identity / GELU。末层零初始化使四组的起始映射相同。各组使用同一开始表征、同一正确单步输出蒸馏、同一初始化、更新序列与预算；几何目标才使用正确对应或三个可见真答案完全匹配的错配。输出蒸馏并未随错配交换。</p>'
    text += '<p>新普通来源先训练 500 轮，再做一份 900 轮正确单步关系准备；随后全部冻结。第一算子只学习 h(x)→h(Ax)，不使用真实 AB 标签或复合状态。复合预测由第一算子输出接冻结第二算子 B 得到，真实中间表征只用于诊断。测试整组轨道与新训练、验证轨道分离，并排除两轮历史测试的 1,024 个轨道。有限世界中的旧训练轨道可再次出现，不声称全历史未见。</p>'
    text += '<h2>结论边界</h2><p>这项干预检验在已接受正确关系训练的冻结编码器上，改变第一算子激活是否扩大关系特异的功能收益；它不能证明普通训练自行发现代数关系，也不能单独区分表达能力与优化差异。旧 50.65% 是更大参数量、仅几何损失、旧源旧测试的探索性诊断；本轮同时加入参数匹配、错配、正确输出监督和新源新划分，不能把两轮数值差全部归于一个因素。三真答案相同的输出编码具有 50% 碰撞准确率上限和零配对双正确上限，但完整输出分布与隐藏状态不受这个信息限制；超过 50% 本身不等于输出信息之外的机制。</p>'
    text += f'<p>独立检查通过：{verification["checks"]:,} 项数据、预算、冻结状态、参数及预测复算；NumPy 双精度 + 标量 erf 复算 GELU，与保存的第一状态最大差 {verification["max_numpy_double_first_state_replay_difference"]:.3g}。测试记录：{html.escape(tests.strip())}。</p>'
    text += '<p><a href="protocol.json">训练前方案</a> · <a href="evaluation.json">所有结果</a> · <a href="independent_verification.json">独立复算</a> · <a href="capacity_confirmation.pdf">导出图 PDF</a> · <a href="../../paper/operator_capacity_confirmation.tex">实验稿源文件</a></p></html>'
    (ROOT/'report.html').write_text(text)

    texrows = '\n'.join(' & '.join([c.replace('_', r'\_'), f'{100*means[c]["accuracy"]:.2f}',
                                   f'{100*means[c]["pair_both_correct"]:.2f}', f'{100*means[c]["first_accuracy"]:.2f}',
                                   f'{means[c]["first_state_nmse"]:.5f}'])+r' \\' for c in config['conditions'])
    tex = r'''\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,hyperref}
\title{A Parameter-Matched Test of First-Operator Nonlinearity and Relation Correctness}
\author{Experiment record}
\date{October 7, 2026}
\begin{document}
\maketitle
\paragraph{Question and registration.}
We tested whether changing the activation of a first latent operator increases the correct-versus-answer-matched-wrong advantage on an unsupervised two-step endpoint. The primary endpoint was fixed before new training and compound test inference:
\[
\delta=(a_{\mathrm{GELU,correct}}-a_{\mathrm{GELU,wrong}})
       -(a_{\mathrm{Identity,correct}}-a_{\mathrm{Identity,wrong}}).
\]
The finite world is invertible $2\times2$ matrices over $\mathbb F_{13}$ with left actions $A=\left(\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\right)$ and $B=\left(\begin{smallmatrix}0&-1\\1&-1\end{smallmatrix}\right)$. Here AB denotes first A, then B, thus endpoint $BAX$; the readout task is matrix trace modulo 13.
\paragraph{Controlled fitting.}
Three new ordinary source initializations (SEEDS) were trained for 500 epochs, then prepared for 900 epochs with the existing correct, forward-only single-generator protocol. Within each source the encoder, second operator, readout, and original first-map skip connection were frozen and identical in all four arms. Each first operator adds a residual Linear(128,256), Identity or GELU, Linear(256,128), with exactly 65,920 trainable parameters. Zero initialization of the final layer and matched random initialization make all initial maps identical. All arms receive the same correct first-state output-distillation targets. Only the activation and first-state geometric pairing vary. Wrong pairing is a full derangement preserving the three visible gold labels. The geometry coefficient is 0.25 and the KD coefficient is 1, with temperature 2 and a fixed correct-target variance normalization. Fitting uses 2,000 AdamW updates of 128 anchors, learning rate 0.001 and weight decay 0.0001. No compound target, loss, checkpoint selection or early stopping is used. The second operator is never refitted.
\paragraph{Data and opening policy.}
All twelve first-operator fits completed before compound test inference. Source, adaptation, validation and test group orbits are separated within this study. Both historical matrix test orbit sets (1,024 orbits) are excluded from the entire new experiment. Historical training orbits may recur with newly initialized models. Each source has 1,024 adaptation anchors. The common held-out test has 256 IID examples and 128 collision pairs, each pair agreeing on the three visible gold answers and disagreeing on AB.
\begin{table}[ht]
\centering
\begin{tabular}{lrrrr}
\toprule
First operator & Collision AB (\%) & Both correct (\%) & Single A (\%) & First NMSE\\
\midrule
TABLE
\bottomrule
\end{tabular}
\caption{Means over three newly initialized sources. Four arms share a fixed encoder, second operator and readout within each source.}
\end{table}
\paragraph{Primary contrast and uncertainty.}
The nonlinear correct-minus-wrong advantage is NL pp; the linear advantage is LINEAR pp. Their interaction is INTERACTION pp, with a paired two-way 95\% bootstrap interval of [LOW,HIGH] pp, resampling three source clusters and 128 common collision pairs (10,000 draws). Per-source interactions are EFFECTS pp. Only three source initializations were used, so these intervals have limited coverage of source variation and should be treated as descriptive.
\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{../results/operator_capacity_confirmation/capacity_confirmation.pdf}
\caption{Parameter-matched four-arm results and paired effects. Dots on accuracy bars and crosses on contrasts show individual source outcomes.}
\end{figure}
\paragraph{Interpretation.}
INTERPRETATION These results are conditional on a correctly relation-trained frozen encoder and do not demonstrate spontaneous relation discovery. Activation also changes optimization, which cannot be separated from expressivity by this intervention alone. The prior 50.65\% diagnostic used more nonlinear parameters, geometry-only fitting and old sources and tests; it is not a directly matched baseline for the present study. A 50\% information ceiling applies to deterministic encodings of the three gold visible answers, not to full output distributions or hidden states. Exceeding that ceiling does not by itself establish an output-independent mechanism.
\paragraph{Verification.}
CHECKS independent checks passed, including scalar finite-field action and label reconstruction, orbit exclusion, matched parameter count and exposure schedules, unchanged frozen buffers and shared backbone hashes, and NumPy double-precision replay using scalar erf for GELU. All existing project tests and the new manipulation tests passed. This document is supplied as LaTeX source; a TeX compiler is unavailable in the current environment.
\end{document}
'''
    interpretation = ('The registered directional interaction was supported on these three new sources and new test inputs.' if lo > 0 and min(primary['source_effects_pp']) > 0 else
                      'The registered directional interaction was not supported by this confirmation.')
    replacements = {'SEEDS': ', '.join(map(str, config['source_seeds'])), 'TABLE': texrows,
        'NL': f'{nl["mean_pp"]:+.2f}', 'LINEAR': f'{linear["mean_pp"]:+.2f}',
        'INTERACTION': f'{primary["mean_pp"]:+.2f}', 'LOW': f'{lo:+.2f}', 'HIGH': f'{hi:+.2f}',
        'EFFECTS': ', '.join(f'{v:+.2f}' for v in primary['source_effects_pp']),
        'INTERPRETATION': interpretation, 'CHECKS': str(verification['checks'])}
    for key, value in replacements.items():
        tex = tex.replace(key, value)
    paper = Path('paper/operator_capacity_confirmation.tex'); paper.write_text(tex)

    preserved = 0
    for manifest in ['results/algebra_relation_review/delivery.json', 'results/readout_null_confirmation/delivery.json']:
        for p, expected in json.loads(Path(manifest).read_text())['artifact_sha256'].items():
            assert sha(p) == expected, p
            preserved += 1
    paths = [p for p in ROOT.rglob('*') if p.is_file() and p.name not in ['delivery.json', 'state.json', 'training.log', 'report_build.log']]
    paths += [paper, Path(__file__), Path('experiments/operator_capacity_verify.py'),
              Path('experiments/operator_capacity_confirmation.py'), Path('configs/operator_capacity_confirmation.json'),
              Path('tests/test_operator_capacity.py')]
    atomic_json(ROOT/'delivery.json', {'status': 'complete', 'completed_utc': now(), 'formal_fits': 12,
        'new_independent_sources': 3, 'old_artifacts_preserved': preserved, 'independent_verification_checks': verification['checks'],
        'pytest': tests.strip(), 'primary': primary, 'conclusion': conclusion,
        'report': str(ROOT/'report.html'), 'experiment_latex_source': str(paper),
        'latex_compiled': False, 'artifact_sha256': {str(p): sha(p) for p in paths}})
    atomic_json(ROOT/'state.json', {'status': 'complete', 'updated_utc': now()})
    print(json.dumps({'status': 'complete', 'primary': primary, 'conclusion': conclusion,
                      'old_artifacts_preserved': preserved}), flush=True)


if __name__ == '__main__':
    run()
