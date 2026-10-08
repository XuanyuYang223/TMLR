"""Generate the reviewer-control report and paper supplement from saved data."""
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


def run():
    root = Path('results/null_space_review_controls')
    report = Path('results/null_space_review_reporting'); report.mkdir(exist_ok=True)
    data = json.loads((root/'results.json').read_text())
    verification = json.loads(Path('results/null_space_review_audit/verification.json').read_text())
    assert verification['status']=='passed'
    select = lambda c, d='fresh': next(r for r in data['means'] if
               (r['condition'], r['dataset'], r['split'])==(c, d, 'collisions'))
    labels = {
        'linear_correct': '线性接收方',
        'swap_linear_correct_null_nonlinear_correct': '正确 GELU 零空间供体',
        'swap_linear_correct_null_nonlinear_wrong': '错配 GELU 零空间供体',
        'wrong_covariance_transported': '验证集均值/协方差校正的错配供体',
        'wrong_norm_error_calibrated': '真实状态辅助范数/误差校准的错配供体',
        'random_norm_error_calibrated': '真实状态辅助范数/误差校准的随机供体',
        'random_validation_covariance': '验证集零空间协方差随机供体',
        'natural_true_intermediate': '真实中间状态（诊断上界）'}
    def table(headers, rows):
        return '<table><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
    contents = ['<h1>读出零空间：审稿控制与因果解释边界</h1>',
        '<p>固定五个已有源模型，不重新训练编码器或关系算子。旧测试追加诊断与新轨道复核分开报告。新集含64个IID输入和32对碰撞输入，128个完整轨道全部排除五个来源的训练、验证与历史测试轨道。</p>',
        '<p><b>主要判断：</b>正确供体的局部功能收益在新集复现，但不能认定这些分量就是数学关系编码。混合状态仍偏离自然状态；读出零空间仍有答案信息；同输入误差匹配的重叠样本很少。</p>',
        '<h2>新碰撞测试</h2>', table(['供体/状态', '复合准确率', '配对双正确率', '状态MSE', '自然训练状态最近邻RMS'],
            [[label, f"{100*select(c)['accuracy']:.2f}%", f"{100*select(c)['pair_both_correct']:.2f}%",
              f"{select(c)['true_state_mse']:.3f}", f"{select(c)['nearest_reference_rms']:.3f}"] for c, label in labels.items()]),
        '<p>正确与错配供体相差8.125个百分点，五个来源均为正；来源和碰撞对交叉bootstrap的描述性95%区间为[1.25,15.625]个百分点。这不是五个独立数据集的区间。</p>',
        '<h2>控制的解释范围</h2>',
        '<p>协方差变换只用已有验证集拟合，不用复合答案，主要校正零空间的边际分布。它没有匹配每个输入的状态误差、行空间—零空间依赖或完整自然状态分布。正确供体仍领先，因而单纯验证集边际均值/协方差变化不足以复现原收益；更广泛的状态分布解释仍未排除。</p>',
        '<p>逐输入范数/误差校准精确保持正确混合状态的零空间范数、到真实中间状态的距离，以及全部第一步logits，但使用真实中间状态的方向。随机校准供体达到70.47%，并不是一个可用的隐藏推断方法，也不能用来声称错误关系训练优于正确关系。它表明相同误差与范数可以对应很不同的下游行为，标量状态误差不足以识别机制。</p>',
        '<p>正确/错配原生预测的同输入误差和范数均在10%以内时，新碰撞集只有3个来源×样本观测；25%阈值下只有18个（320个来源×样本观测中）。25%子集的条件差为−5.56个百分点，不能支持稳定优势或稳定劣势。这里条件化的是训练后的潜在中介，属于描述性分析，不是总因果效应。</p>',
        '<h2>零空间答案信息与数值审计</h2>',
        '<p>独立线性岭探针只用支持集训练、已知验证集选正则。对真实中间状态的零空间，新碰撞集当前答案准确率54.06%，未来答案52.81%；当前答案训练众数基线8.13%。未来标签仅用于独立诊断probe，没有进入关系算子或其选择。成功说明零空间仍含可解码答案信息；不能据此认定信息独立于输出或等同数学关系。</p>',
        '<p>实际读出矩阵为13×128，五个来源均秩13。SVD与直接伪逆投影一致。所有供体控制的第一步logits最大变化4.11×10⁻¹³；逐输入范数和平方误差匹配误差均低于1.14×10⁻¹³。独立重放499,530项检查通过。代码采用行向量：z=hWᵀ+b，混合状态=h_LP+h_G(I−P)。</p>',
        '<p>旧五来源充分拟合仿射19.92%与GELU20.31%的结果保持原样。不能声称非线性必要，也不推广为所有任务都不需要非线性。</p>',
        '<p><a href="../null_space_review_controls/results.json">逐来源完整结果</a> · <a href="../null_space_review_audit/verification.json">独立重放与协方差匹配质量</a></p>']
    (report/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:16px/1.7 system-ui;max-width:1150px;margin:30px auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}</style>'+''.join(contents))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), gridspec_kw={'width_ratios': [1.3, 1]})
    groups = [(['linear_correct', 'swap_linear_correct_null_nonlinear_correct',
                'swap_linear_correct_null_nonlinear_wrong', 'wrong_covariance_transported'],
               ['Linear', 'Correct null', 'Wrong null', 'Cov. adjusted'], 'Predicted states; no true-state calibration'),
              (['swap_linear_correct_null_nonlinear_correct', 'wrong_norm_error_calibrated',
                'random_norm_error_calibrated'], ['Correct null', 'Wrong calibrated', 'Random calibrated'],
               'Oracle controls use true intermediate states')]
    for ax, (conditions, names, title) in zip(axes, groups):
        heights = [100*select(c)['accuracy'] for c in conditions]
        ax.bar(np.arange(len(conditions)), heights, color=['#3176a0' if 'correct' in c else '#9eabb3' for c in conditions])
        for j, c in enumerate(conditions):
            sources = [100*r['accuracy'] for r in data['rows'] if
                       (r['condition'], r['dataset'], r['split'])==(c, 'fresh', 'collisions')]
            ax.scatter(j+np.linspace(-.13, .13, len(sources)), sources, color='black', s=12, zorder=3)
        ax.set_xticks(np.arange(len(conditions)), names, rotation=25, ha='right')
        ax.set_title(title, fontsize=9); ax.set_ylim(0, 85); ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylabel('Fresh collision accuracy (%)'); ax.grid(axis='y', alpha=.15)
    fig.tight_layout()
    out = Path('paper/figures/reviewer_controls')
    for extension in ['pdf', 'png', 'svg']:
        fig.savefig(out.with_suffix('.'+extension), dpi=180, bbox_inches='tight')
    plt.close(fig)
    tex = r'''\section{Reviewer controls: state matching and interpretation}
These diagnostics were registered after the final confirmation, without changing any encoder, relation operator, or readout. We separately report previously inspected tests and a new set of 128 complete orbits (64 IID inputs and 32 collision pairs). The new orbits exclude training and validation of all five sources and every previously inspected test orbit. This is a small post-confirmation replication, not a new five-dataset study.

\paragraph{Predicted-state replication and distribution controls.}
On new collision pairs, the linear recipient reaches 10.00\%; correct predicted-null transfer reaches 15.31\%, versus 7.19\% for incorrect transfer. The difference is 8.125 points, with a descriptive crossed source/pair interval [1.25,15.625]; all five source differences are positive.
Validation-fitted mean/covariance transport of the incorrect donor reaches 7.50\%. This control addresses marginal null-coordinate distribution, not conditional state error, row/null dependence, or complete manifold membership.
Nearest-reference RMS distance is 0.565 for the correct hybrid, 0.616 for the incorrect hybrid, and 0.388 for true intermediate states. These distances concern natural training intermediates; they do not constitute a manifold-membership test. The distribution-shift explanation remains unresolved.

\paragraph{Error matching and its limitations.}
Within-input restriction to original correct/incorrect predictions whose null norms and errors both agree within 10\% retains only 3 source--sample observations; a 25\% restriction retains 18 of 320. The latter conditional difference is $-5.56$ points and is too sparse to support a stable mechanism conclusion. Conditioning on prediction error can remove a pathway through which correspondence acts; this is descriptive, not causal isolation.
An additional oracle diagnostic exactly matches each correct hybrid's null norm and error to the true intermediate, while preserving its row component. Reorienting a wrong donor yields 34.69\%, and randomized orientations yield 70.47\%. These controls use the true intermediate's direction. They are not learned predictions or evidence that wrong-pair training generalizes better. Their scope is narrower: equal scalar state error and norm do not imply equal downstream utility. Norm/error and marginal covariance controls are separate, rather than simultaneously matching all distributions.

\paragraph{Answers remain decodable from the null space.}
Independent ridge probes use support labels and known validation for regularization selection; they never update the composition pipeline. On true intermediate null coordinates, fresh-collision current-answer accuracy is 54.06\% (training-mode baseline 8.13\%) and future-answer accuracy is 52.81\%. Future labels are supplied only to these diagnostic probes. Thus the null component is invisible to one fixed linear readout, but is not independent of answer information.

\paragraph{Numerical audit.}
All five $13\times128$ readouts have numerical rank 13. SVD and pseudoinverse projectors agree; across every tested intervention and random draw, the maximum first-logit change is $4.11\times10^{-13}$. Norm/error matching discrepancies are below $1.14\times10^{-13}$. Independent reconstruction checks 499,530 sample/identity entries. These checks validate the defined interventions, not a natural relation-specific causal mechanism.

\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{figures/reviewer_controls.pdf}
\caption{Frozen-model reviewer controls on new collision pairs. Dots show five source initializations, sharing test inputs. Right-hand calibrated controls use true intermediate states and are diagnostics only; they are not available predictors.}
\end{figure}
'''
    (Path('paper/sections')/'reviewer_controls.tex').write_text(tex)
    atomic_json(report/'report_audit.json', {'completed_utc': now(), 'results_sha256': sha(root/'results.json'),
                'verification_sha256': sha('results/null_space_review_audit/verification.json'),
                'paper_section_sha256': sha('paper/sections/reviewer_controls.tex'),
                'oracle_controls_not_reported_as_learned_performance': True,
                'post_treatment_matching_not_reported_as_causal_isolation': True})


if __name__=='__main__':
    run()
