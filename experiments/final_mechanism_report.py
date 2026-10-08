"""Close the fixed experiment and integrate its supported manuscript claims."""
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

ROOT=Path('results/final_mechanism_confirmation')
NAMES={'linear_correct':'同预算线性正确','linear_wrong':'同预算线性错配',
       'nonlinear_correct':'同预算 GELU 正确','nonlinear_wrong':'同预算 GELU 错配',
       'affine_correct_ols_initialization':'充分拟合线性正确','affine_wrong_ols_initialization':'充分拟合线性错配',
       'swap_linear_correct_null_nonlinear_correct':'线性读出＋正确 GELU 零空间',
       'swap_linear_correct_null_nonlinear_wrong':'线性读出＋错配 GELU 零空间',
       'swap_nonlinear_correct_null_linear_correct':'GELU 读出＋线性零空间'}


def table(head,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def interval(r):
    lo,hi=r['bootstrap_95_pp'];return f'{r["mean_pp"]:+.2f} [{lo:+.2f}, {hi:+.2f}]'


def figures(result):
    means={r['condition']:r for r in result['means'] if r['split']=='collisions'}
    effects={r['contrast']:r for r in result['contrasts'] if r['endpoint']=='accuracy'}
    selected=['linear_correct','linear_wrong','nonlinear_correct','nonlinear_wrong',
              'affine_correct_ols_initialization','affine_wrong_ols_initialization',
              'swap_linear_correct_null_nonlinear_correct','swap_linear_correct_null_nonlinear_wrong']
    short=['Linear\ncorrect','Linear\nwrong','GELU\ncorrect','GELU\nwrong','Converged\nlinear correct',
           'Converged\nlinear wrong','Correct GELU\nnull donor','Wrong GELU\nnull donor']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True,gridspec_kw={'width_ratios':[1.35,1]})
    colors=['#4f7f9f','#b3c6d2','#c66a30','#e8b28b','#4f7f9f','#b3c6d2','#c66a30','#e8b28b']
    for j,c in enumerate(selected):
        values=[100*r['accuracy'] for r in result['records'] if r['condition']==c and r['split']=='collisions']
        axes[0].bar(j,np.mean(values),color=colors[j],width=.66)
        axes[0].scatter(j+np.linspace(-.15,.15,5),values,color='#222222',s=17,zorder=3)
        axes[0].text(j,max(values)+1.2,f'{np.mean(values):.2f}',ha='center',fontsize=8)
    axes[0].set_xticks(range(8),short,fontsize=8);axes[0].set_ylabel('Final held-out collision AB accuracy (%)')
    axes[0].set_ylim(0,max(40,max(100*means[c]['accuracy'] for c in selected)+15))
    axes[0].set_title('A. Five new sources; shared frozen encoder / B / readout within each')
    names=['same_budget_interaction','correct_null_swap_gain','correct_vs_wrong_null_donor','remaining_relation_interaction']
    for j,name in enumerate(names):
        r=effects[name];mu=r['mean_pp'];lo,hi=r['bootstrap_95_pp']
        axes[1].errorbar(j,mu,yerr=[[mu-lo],[hi-mu]],fmt='o',capsize=4,color='#285b7a')
        axes[1].scatter(j+np.linspace(-.15,.15,5),r['source_effects_pp'],s=22,marker='x',color='#bc582a')
    axes[1].axhline(0,color='gray',lw=1);axes[1].set_ylabel('Paired accuracy contrast (percentage points)')
    axes[1].set_xticks(range(4),['Same-budget\ninteraction','Correct-null\nswap gain','Correct − wrong\nnull donor','Interaction after\naffine convergence'],fontsize=8)
    axes[1].set_title('B. Five-source / shared-pair descriptive 95% bootstrap intervals')
    for ext in ['pdf','svg','png']:fig.savefig(ROOT/f'final_confirmation.{ext}',dpi=180)
    plt.close(fig)


def manuscript(result):
    means={r['condition']:r for r in result['means'] if r['split']=='collisions'}
    effects={r['contrast']:r for r in result['contrasts'] if r['endpoint']=='accuracy'}
    metadata=[json.loads(p.read_text())['metadata'] for p in (ROOT/'affine_fits').glob('*.json')]
    precision={p:[r for r in result['precision_checks'] if r['pairing']==p and
                 r['kind']=='ols_initialization' and r['split']=='collisions'] for p in ['correct','wrong']}
    rows=[]
    for c in ['linear_correct','linear_wrong','nonlinear_correct','nonlinear_wrong','affine_correct_ols_initialization','affine_wrong_ols_initialization']:
        label=c.replace('affine_','converged ').replace('_ols_initialization','').replace('_',' ')
        rows.append(' & '.join([label,f'{100*means[c]["accuracy"]:.2f}',f'{100*means[c]["pair_both_correct"]:.2f}',f'{means[c]["hidden_nmse"]:.5f}'])+r' \\')
    section=r'''\section{Final new-source mechanism confirmation}
An earlier three-source parameter-matched confirmation produced an activation-by-correctness interaction of 14.71 pp [9.51,20.05]. Posthoc predicted-null swaps and adequately fitted affine diagnostics on those same sources motivated the present hypotheses; a fresh-input check on reused sources is not new-source confirmation. We do not pool these exploratory and prior-confirmation outcomes with the final experiment.

We fixed this as the last experimental round before observing its new outcomes. Five previously unused source seeds (17291, 18401, 19507, 20611, 21713) were trained from scratch for 500 ordinary epochs and 900 correct single-generator preparation epochs. No old source checkpoint was reused. The five encoders, original maps and readouts were then frozen. Each source has its own 1024-anchor first-operator support, while the new source corpus and test are shared. Supports may overlap; source/validation/support/test complete group orbits are separated within the study. The new test excludes all 2688 previously examined matrix test orbits and contains 256 IID examples and 128 collision pairs. Historical inputs can recur in the new training partitions under new initialized models.

\paragraph{Matched activation and correspondence.}
Two-layer Identity and GELU first-operator residuals have identical width 256 and 65,920 trainable parameters, fixed original affine skip connections, matching initialization and zero final layers. Correct teacher logits, start states, schedules and 2000 AdamW updates are shared. Wrong geometry uses a complete derangement preserving all three visible gold answers. All four first-operator arms use zero weight decay, declared before fitting, to match the unregularized data objective of the convergence diagnostic. This harmonization differs from the earlier 0.0001-decay confirmation; the earlier results remain unchanged. The scalar encoder and source-preparation protocol are unchanged.

\paragraph{Predicted null-component transfer.}
For two predicted first states $\hat h_L,\hat h_G$ in the same source coordinates, with $P$ the row projector of the frozen full linear readout, we construct
\[
\hat h_{L\leftarrow G}=\hat h_L P+\hat h_G(I-P).
\]
The intervention uses no true intermediate state. It preserves every first-step logit, not only the winning class. Correct and matched-wrong GELU donors and reverse transfer are reported. Correct-null transfer yields SWAP pp [SWAPLO,SWAPHI] improvement on collision AB; the correct-minus-wrong donor contrast is DONOR pp [DONORLO,DONORHI]. Across interventions the maximum logit change is LOGIT. This tests a functional pathway in the specified frozen composition; hybrid states need not lie on the encoder manifold and the intervention does not remove all answer information.

\paragraph{Same-family, same-loss affine convergence.}
The two-layer Identity residual can express an unrestricted affine map. We solve exactly its shared empirical loss,
\[
 L(R)=\mathrm{KD}_{T=2}(R(h_0)W^\top+b_W,z_A)+0.25\,\frac{\|R(h_0)-h_A^{\mathrm{paired}}\|_F^2}{Nds^2},
\]
with $s^2$ fixed from correct training targets and the identical stored teacher logits $z_A$ used in all arms. There are no compound targets, ridge terms or validation-selected geometry weights. Float64 SVD whitening conditions the affine training design; readout-null coefficients have the exact least-squares solution, and readout-row coefficients are optimized by L-BFGS. Both OLS and original budget-linear row initializations are fixed in advance. The strongly convex quadratic term permits a gradient-based upper bound on remaining training-objective error. Independent analytic full gradients verify every solution. Both initializations, coefficient norms, condition numbers and float32-versus-float64 inference are retained. Changed parameterization, precision and additional budget are optimization diagnostics, not another equal-budget intervention.

\begin{table}[ht]\centering\small
\begin{tabular}{lrrr}\toprule First operator & Collision AB (\%) & Both correct (\%) & First-state NMSE\\\midrule
TABLEROWS
\bottomrule\end{tabular}
\caption{Final five-source confirmation, with original budget-controlled arms and separately budgeted converged affine diagnostics. All predictions begin from the input representation and use the frozen second operator.}\end{table}

\paragraph{Predeclared contrasts.}
The same-budget activation-by-correctness interaction is INTERACTION pp [INTERLO,INTERHI]. After affine convergence, the remaining GELU-versus-affine relation-advantage interaction is REMAIN pp [REMAINLO,REMAINHI]. Intervals resample five independent source clusters and the 128 common collision pairs; individual-source effects are provided in the experiment artifacts. These are descriptive intervals for separately stated hypotheses, not familywise-adjusted significance tests. Crossing zero cannot establish equivalence. All source gates and convergence checks are retained, without replacement of unfavorable sources.

\paragraph{Convergence and numerical qualification.}
The largest independently recomputed training-objective gap bound is GAPBOUND. Both fixed initializations yield identical compound classes, with a maximum predicted first-state difference of REPEATDIFF. However, the affine design is ill-conditioned: condition numbers range from CONDMIN to CONDMAX, and coefficient Frobenius norms range from COEFMIN to COEFMAX. Replaying the complete affine--second-operator--readout pipeline in float32 changes CORRECTCHANGES/1280 correct-pairing and WRONGCHANGES/1280 wrong-pairing collision predictions relative to float64. Aggregate correct/wrong accuracies are CORRECT64/WRONG64\% in float64 and CORRECT32/WRONG32\% in float32. Thus the main aggregate correctness advantage remains under this precision check, while individual predictions and high-norm coefficients are numerically sensitive. The convergence certificate concerns the finite empirical objective; it does not guarantee perturbation robustness or generalization to arbitrary inputs.

\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{../results/final_mechanism_confirmation/final_confirmation.pdf}
\caption{Final confirmation with five new source initializations, predicted-component transfer, and the predeclared same-loss convergence diagnostic. Points show all five sources.}\end{figure}

\paragraph{Closing interpretation.}
INTERPRETATION We do not assign exact percentage contributions to optimization, numerical conditioning and expressivity. No additional domains, source seeds, losses or mechanistic training schemes are added after this fixed confirmation. The remaining work is manuscript refinement and submission preparation.
'''
    def scientific(v):
        coefficient,exponent=f'{v:.2e}'.split('e')
        return rf'${coefficient}\times10^{{{int(exponent)}}}$'
    verification=json.loads((ROOT/'independent_verification.json').read_text())
    values={'TABLEROWS':'\n'.join(rows),'LOGIT':scientific(result['max_swap_first_logit_change']),
            'GAPBOUND':scientific(verification['max_recomputed_global_gap_bound']),
            'REPEATDIFF':scientific(max(r['max_repeat_first_state_difference'] for r in result['solver_repeat_checks'])),
            'CONDMIN':scientific(min(r['condition_number'] for r in metadata)),
            'CONDMAX':scientific(max(r['condition_number'] for r in metadata)),
            'COEFMIN':scientific(min(r['coefficient_norm'] for r in metadata)),
            'COEFMAX':scientific(max(r['coefficient_norm'] for r in metadata))}
    for pairing,prefix in [('correct','CORRECT'),('wrong','WRONG')]:
        values[prefix+'CHANGES']=str(sum(r['changed_compound_predictions'] for r in precision[pairing]))
        for dtype in ['32','64']:
            values[prefix+dtype]=f'{100*np.mean([r["float"+dtype+"_accuracy"] for r in precision[pairing]]):.2f}'
    for name,prefix in [('correct_null_swap_gain','SWAP'),('correct_vs_wrong_null_donor','DONOR'),
                        ('same_budget_interaction','INTER'),('remaining_relation_interaction','REMAIN')]:
        r=effects[name];values[prefix]=f'{r["mean_pp"]:+.2f}';values[prefix+'LO']=f'{r["bootstrap_95_pp"][0]:+.2f}';values[prefix+'HI']=f'{r["bootstrap_95_pp"][1]:+.2f}'
    values['INTERACTION']=values['INTER']
    r=effects['remaining_relation_interaction']
    interpretation=('The remaining correctness-advantage contrast is positive on this finite task, conditional on the reported convergence and numerical checks; it is not a universal expressivity theorem.'
                    if r['bootstrap_95_pp'][0]>0 else 'We did not confirm an additional expressivity advantage in functional relation-specific composition after adequate affine fitting. State-fitting differences and unresolved numerical and optimization explanations are retained as limitations.')
    values['INTERPRETATION']=interpretation
    for k in sorted(values,key=len,reverse=True):section=section.replace(k,values[k])
    target=Path('paper/generated_final_mechanism.tex');target.write_text(section)
    main=Path('paper/main_readout_null.tex').read_text()
    start=main.index(r'\begin{abstract}');end=main.index(r'\end{abstract}')+len(r'\end{abstract}')
    abstract=r'''\begin{abstract}
We study when the correctness of specified mathematical relations contributes to unseen composition under matched output supervision. Permutation experiments identify a local compositional advantage, while initial joint affine finite-matrix and irreversible-polynomial protocols expose failures of intermediate-state inference and primitive generalization. A null-only-loss replacement does not restore the benefit. A final five-new-source matrix confirmation finds a 12.89 percentage-point activation-by-correctness interaction under a fixed budget. Transferring predicted readout-null components from correctly trained GELU operators improves linear composition by 8.75 points while preserving all first-step logits. Adequately optimized affine maps under the identical single-step loss recover most of the correctness benefit; the remaining relation-advantage interaction is 0.31 points with a descriptive interval crossing zero. We did not confirm an additional functional expressivity advantage. This study supplies known relations and operation order; it does not demonstrate spontaneous algebra discovery or a universal output-independent mechanism.
\end{abstract}'''
    main=main[:start]+abstract+main[end:]
    main=main.replace(r'\begin{document}',r'\setlength{\emergencystretch}{3em}'+'\n'+r'\begin{document}')
    main=main.replace(r'\date{October 7, 2026}',r'\date{October 7, 2026: experimental scope closed}')
    main=main.replace(r'\bibliographystyle{plain}',r'\input{generated_final_mechanism.tex}'+'\n\n'+r'\bibliographystyle{plain}')
    main=main.replace('Learned affine row-vector operators are', 'For the initial factorial experiments, learned affine row-vector operators are')
    main=main.replace('Both new domains share field size, four scalar input coordinates, a generic GELU encoder of width 128, three source seeds, six fitting repeats, and 900 fixed joint epochs.',
        'The initial cross-domain protocol uses a common field size, four scalar input coordinates, a generic GELU encoder of width 128, three source seeds, six fitting repeats, and 900 fixed joint epochs. Later frozen-encoder confirmations use separately specified source counts, supports and budgets.')
    main=main.replace('and128 collision pairs.', 'and 128 collision pairs.').replace('all36 formal fits', 'all 36 formal fits')
    main=main.replace('The matrix correct-minus-incorrect increment is small and its source-plus-input interval crosses zero.',
        'The initial joint affine matrix protocol has a small correct-minus-incorrect increment with an interval crossing zero; the subsequent frozen-encoder activation and convergence experiments are distinct interventions.')
    Path('paper/main_final.tex').write_text(main)
    return interpretation


def run():
    config=json.loads(Path('configs/final_mechanism_confirmation.json').read_text())
    result=json.loads((ROOT/'results.json').read_text())
    verification=json.loads((ROOT/'independent_verification.json').read_text())
    assert verification['status']=='passed'
    tests=(ROOT/'pytest.log').read_text();assert 'passed' in tests and 'failed' not in tests.lower()
    figures(result);interpretation=manuscript(result)
    means={r['condition']:r for r in result['means'] if r['split']=='collisions'}
    effects={r['contrast']:r for r in result['contrasts'] if r['endpoint']=='accuracy'}
    chosen=list(NAMES)
    report='<!doctype html><html lang="zh"><meta charset="utf-8"><title>最终五源机制确认</title><style>body{font:16px/1.75 system-ui;max-width:1280px;margin:32px auto;padding:0 20px;color:#18232d}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{border:1px solid #d7dee3;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%}.lead{padding:18px;background:#eef4f7;border-left:4px solid #4682b4}</style><h1>最终机制确认：五个独立新源</h1>'
    report+='<p class="lead">正确关系收益和预测零空间分量的功能迁移得到新源复核。同损失充分拟合后，线性组恢复大部分关系收益；未确认额外的功能表达能力优势。本轮按预定结束标准收口。</p>'
    report+='<p>完成五个从头训练的来源、二十组固定预算模型和二十个同损失凸求解。种子均提前固定，没有替换失败种子或追加条件。所有源单步门槛与所有收敛证书均通过。新测试有 512 个整组轨道，排除过去已查看的 2,688 个测试轨道。历史输入可以出现在本轮新模型的训练集，未声称全历史训练输入新颖。</p>'
    report+='<img src="final_confirmation.svg" alt="五源最终确认及预定效应">'
    report+=table(['条件','碰撞 AB','配对双正确','单步 A','第一状态 NMSE'],
        [[NAMES[c],f'{100*means[c]["accuracy"]:.2f}%',f'{100*means[c]["pair_both_correct"]:.2f}%',
          f'{100*means[c]["first_accuracy"]:.2f}%',f'{means[c]["hidden_nmse"]:.5f}'] for c in chosen])
    report+='<h2>预定对比</h2>'+table(['对比','均值 [描述性 95% 区间] pp','五源效应 pp'],
        [[name,interval(r),', '.join(f'{v:+.2f}' for v in r['source_effects_pp'])] for name,r in effects.items()])
    report+=f'<p>正确零空间供体增量为 {interval(effects["correct_null_swap_gain"])} pp；正确与错配供体之间为 {interval(effects["correct_vs_wrong_null_donor"])} pp。替换只使用各自预测的表征，第一步全部 logits 最大变化 {result["max_swap_first_logit_change"]:.3g}，概率最大变化 {result["max_swap_probability_change"]:.3g}，分类均保持。</p>'
    report+='<h2>同损失优化与数值检查</h2><p>四组第一算子统一使用零权重衰减，是训练前登记的调整，以便同预算训练与直接仿射求解具有相同的蒸馏＋0.25 归一化状态误差目标。编码器与第二算子始终冻结。充分拟合改变参数化、计算精度和预算，没有更换函数族、几何权重、输出监督或正则目标。没有复合标签／状态进入拟合；两种求解起点均预先固定。</p>'
    solverrows=[]
    for i in range(5):
        for p in ['correct','wrong']:
            a=json.loads((ROOT/'affine_fits'/f's{i}_{p}_ols_initialization.json').read_text())['metadata']
            b=json.loads((ROOT/'affine_fits'/f's{i}_{p}_budget_linear_initialization.json').read_text())['metadata']
            solverrows.append([config['source_seeds'][i],p,f'{a["condition_number"]:.3g}',f'{a["coefficient_norm"]:.3g}',
                f'{max(a["independent_gap_bound"],b["independent_gap_bound"]):.3g}',f'{abs(a["objective"]-b["objective"]):.3g}'])
    report+=table(['源种子','配对','条件数','系数范数','最大独立 gap 上界','两起点目标差'],solverrows)
    precision=[]
    for p in ['correct','wrong']:
        select=[r for r in result['precision_checks'] if r['pairing']==p and r['kind']=='ols_initialization' and r['split']=='collisions']
        precision.append([p,f'{100*np.mean([r["float64_accuracy"] for r in select]):.2f}%',
            f'{100*np.mean([r["float32_accuracy"] for r in select]):.2f}%',sum(r['changed_compound_predictions'] for r in select)])
    report+=table(['充分拟合配对','float64 AB','float32 AB','改变的预测数 / 1280'],precision)
    report+=f'<p>两起点在测试上不同复合预测总数为 {sum(r["different_compound_predictions"] for r in result["solver_repeat_checks"])}；最大第一状态差 {max(r["max_repeat_first_state_difference"] for r in result["solver_repeat_checks"]):.3g}。独立解析梯度复算的最大训练最优性误差上界为 {verification["max_recomputed_global_gap_bound"]:.3g}。系数仍可能很大，最优性证书不保证所有扰动或所有输入上的数值稳定。</p>'
    report+='<h2>正文结论与局限</h2><p>当前冻结模型上的关系特异收益能由第一步读出零空间中的预测信息传递，并且线性优化充分程度影响能否利用这些信息。不能把贡献拆为优化、数值条件和表达能力的精确百分比；未能确认充分拟合后额外的功能表达优势。区间跨零不表示线性与 GELU 等价。状态拟合差距、五源有限性、单一矩阵系统以及混合状态可能偏离编码器流形，均进入论文局限。</p>'
    report+='<p>三真答案碰撞上限只适用于由三个答案确定的编码，完整隐藏状态和输出概率分布不受该上限限制。零空间保持的是固定线性第一读出，不移除全部答案信息。所有实验都提供操作及对应关系，不能证明普通训练自发发现代数，也不能证明普遍的输出独立机制。各项预定假设分别报告，bootstrap 区间没有做多重检验校正，不以单个区间替代全部证据。</p>'
    report+=f'<p>{verification["checks"]:,} 项独立检查通过；{verification["old_artifacts_preserved"]} 项此前产物保持。{html.escape(tests.splitlines()[-1])}。</p>'
    report+='<p><a href="protocol.json">预先固定方案</a> · <a href="results.json">所有结果与数值检查</a> · <a href="independent_verification.json">独立复算</a> · <a href="../../paper/main_final.tex">整合论文源文件</a> · <a href="../../paper/main_final.pdf">整合论文 PDF</a></p></html>'
    (ROOT/'report.html').write_text(report)
    atomic_json(ROOT/'manuscript_integration.json',{'created_utc':now(),'main':'paper/main_final.tex',
        'section':'paper/generated_final_mechanism.tex','conclusion':interpretation,'experimental_scope_closed':True,
        'remaining_stage':'Manuscript refinement and submission preparation; no additional experiment expansion.'})
    print(json.dumps({'status':'report_and_manuscript_written','remaining_relation_interaction':effects['remaining_relation_interaction']}),flush=True)


if __name__=='__main__':run()
