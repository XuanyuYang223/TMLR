"""Post-confirmation collision census and honest summary of reviewer revisions."""
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .reviewer_revision_statistics import tost, holm
from .two_step_relation_factorial import now


def run():
    parent=Path('results/reviewer_revision_diagnostics_v2')
    root=Path('results/reviewer_revision_reporting');root.mkdir(exist_ok=True)
    result=json.loads((parent/'results.json').read_text())
    signature={'code_sha256':sha(__file__),'parent_results_sha256':sha(parent/'results.json'),
        'scope':'Posthoc reporting and gold-only pairing of previously evaluated orbit census; not a new confirmation. Each physical input appears in at most one collision pair, but group orbits can recur across pairs. No pair count is treated as independent orbit count.'}
    file=root/'protocol.json'
    if file.exists():assert json.loads(file.read_text())['signature']==signature
    else:atomic_json(file,{'registered_utc':now(),'all_input_outcomes_already_observed':True,'signature':signature})
    rows=[];effects=[]
    for i in range(5):
        data=dict(np.load(parent/'datasets'/f's{i}.npz'));groups=defaultdict(list)
        for j,y in enumerate(data['labels']):groups[tuple(y[:3])].append(j)
        rng=np.random.default_rng(261009911+i);pairs=[]
        for ids in groups.values():
            ids=np.array(ids);rng.shuffle(ids);used=set()
            for left in ids:
                if left in used:continue
                for right in ids:
                    if right in used or data['orbit_ids'][right]==data['orbit_ids'][left]:continue
                    if data['labels'][right,3]!=data['labels'][left,3]:
                        pairs.append([left,right]);used.update([left,right]);break
        pair=np.array(pairs);assert len(np.unique(pair))==pair.size
        np.savez_compressed(root/f's{i}_collision_pairs.npz',pairs=pair)
        conditions=['nonlinear_correct','nonlinear_wrong','linear_correct','linear_wrong',
                    'affine_correct_ols_initialization','affine_wrong_ols_initialization']
        selected={r['pairing']:r['selected_ridge'] for r in result['ridge_path'] if r['source']==i and 'selected_ridge' in r}
        conditions.extend(f'ridge_{p}_{selected[p]}' for p in ['correct','wrong'])
        for condition in conditions:
            saved=dict(np.load(parent/'evaluations'/f's{i}_{condition}.npz'));hit=saved['hits'][pair]
            rows.append({'source':i,'condition':condition,'collision_pairs':len(pair),'collision_inputs':pair.size,
                'represented_orbits':len(np.unique(data['orbit_ids'][pair])),'accuracy':float(hit.mean()),
                'pair_both_correct':float(hit.all(1).mean()),'first_accuracy':float(saved['first_hits'][pair].mean()),
                'first_state_mse':float(saved['state_mse'][pair].mean()),'orbits_may_recur_across_pairs':True})
        get=lambda c:next(r for r in rows if r['source']==i and r['condition']==c)['accuracy']
        effects.append(100*(get('affine_correct_ols_initialization')-get('nonlinear_correct')))
    means=[]
    for category in ['nonlinear_correct','nonlinear_wrong','linear_correct','linear_wrong',
                     'affine_correct_ols_initialization','affine_wrong_ols_initialization','validation_selected_ridge_correct','validation_selected_ridge_wrong']:
        selected=[]
        for i in range(5):
            if category.startswith('validation_selected_'):
                p=category.rsplit('_',1)[1]
                ridge=next(r['selected_ridge'] for r in result['ridge_path'] if r['source']==i and r['pairing']==p and 'selected_ridge' in r)
                condition=f'ridge_{p}_{ridge}'
            else:condition=category
            selected.append(next(r for r in rows if r['source']==i and r['condition']==condition))
        means.append({'condition':category,**{k:float(np.mean([r[k] for r in selected]))
                                             for k in ['accuracy','pair_both_correct','first_accuracy','first_state_mse']}})
    secondary=[tost(effects,m) for m in [3.,5.]]
    adjusted=holm([r['p'] for r in secondary])
    for r,a in zip(secondary,adjusted):r['margin_family_holm_p']=a;r['posthoc_only']=True
    atomic_json(root/'results.json',{'status':'complete','records':rows,'means':means,
        'secondary_collision_tost':secondary,'effects_pp':effects,
        'source_and_test_worlds_are_not_new_independent_confirmations':True})
    fig,axes=plt.subplots(1,2,figsize=(8.7,3.8))
    path=[1e-12,1e-10,1e-8,1e-6,1e-4,.01,1.]
    for pairing,color in [('correct','#276c9f'),('wrong','#ad6c37')]:
        values=[]
        for ridge in path:
            rr=[r for r in result['primitive_and_composition'] if r['condition']==f'ridge_{pairing}_{ridge}']
            values.append(100*np.mean([r['accuracy'] for r in rr]))
        axes[0].plot(path,values,marker='o',label=pairing,color=color)
    axes[0].set_xscale('log');axes[0].set_xlabel('Ridge on standardized affine coefficients');axes[0].set_ylabel('Expanded-census accuracy (%)')
    axes[0].legend();axes[0].set_title('Regularization sensitivity, fixed validation rule',fontsize=9)
    exchange=[r for r in result['natural_interchanges']]
    values=[100*np.mean([r[k] for r in exchange if r['condition']==condition]) for condition,k in
            [('different_future','donor_target_accuracy'),('same_future','donor_target_accuracy'),('same_future','prediction_changed_fraction')]]
    axes[1].bar([0,1,2],values,color=['#276c9f','#6f8c82','#ad6c37'])
    axes[1].set_xticks([0,1,2],['Different-future\ndonor followed','Same-future\nanswer retained','Same-future\nprediction changed'])
    axes[1].set_ylabel('Percent');axes[1].set_ylim(0,100);axes[1].set_title('Natural-state interchange and invalidation control',fontsize=9)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.tight_layout()
    for ext in ['pdf','png','svg']:fig.savefig(Path('paper/figures/reviewer_revision').with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    table=''.join(f'<tr><td>{r["condition"]}</td><td>{100*r["accuracy"]:.2f}%</td><td>{100*r["pair_both_correct"]:.2f}%</td></tr>' for r in means)
    counts=[next(r for r in rows if r['source']==i)['collision_inputs'] for i in range(5)]
    html=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:30px auto}}td,th{{padding:8px;border:1px solid #ccc}}</style>
<h1>审稿修订：测试规模、自然交换、岭正则</h1><p>原始结果不覆盖；这份报告是事后诊断。每源完整枚举未用于该源训练/验证、也未在历史测试中查看的511–530个轨道，得到3066–3180个输入。六个轨道方向不是六次独立重复。</p>
<h2>扩大的碰撞测试</h2><p>三个可见金答案完全相同、复合答案不同。每个输入仅出现一次，但轨道可以出现在多对中，来源不是新的数据世界。每源碰撞输入数：{counts}。</p><table><tr><th>条件</th><th>准确率</th><th>配对双正确</th></tr>{table}</table>
<h2>解释发生了什么变化</h2><p>完整轨道枚举的未正则化仿射18.16%，GELU17.93%；初始化间配对差缩小。事后TOST通过±3pp，但不构成预注册的新世界等价性确认。主张继续保持“额外优势未获确认”。</p>
<p>验证集选择的正则化仿射表现明显降低。岭正则改变了目标，并抑制小特征方向；它不能把原收敛证明推翻，却说明那个功能结果没有展示常规正则下的稳健性。小方向究竟是真实有效信息还是有限精度影响，仍未区分。</p>
<p>自然零空间供体在第一步答案相同、第二步不同的交换中被跟随56.89%；但答案都相同的交换仍改变18.25%的预测。所有零空间交换保持完整第一步logits不变。自然分量不能保证混合状态在流形上，不能将其称为干净的数学关系机制。</p>
<p>当前GELU正确配对只缩小约11.50%的oracle差距，组合准确率仍低。第一步输出高准确不等于状态足够准确供第二步使用。</p>
<h2>新的确认实验</h2><p>协议在任何F17组合成绩之前固定：17个独立数据划分与初始化；512维编码器；每源2000对碰撞、512个IID样本；两个主对比与Holm校正；第二步噪声候选只由单步验证选择。40%性能及同答案交换改变率≤5%是报告门槛，不用于追逐测试成绩。训练接口计数遗漏已修复，原始协议与修订记录保留。</p>
<p><a href="results.json">逐来源碰撞结果</a> · <a href="../reviewer_revision_diagnostics_v2/results.json">完整轨道、误差分解及正则路径</a> · <a href="../reviewer_fresh_confirmation/protocol.json">新确认协议</a></p></html>'''
    (root/'report.html').write_text(html)
    rows_tex='\n'.join(f'{r["condition"].replace("_", " ")} & {100*r["accuracy"]:.2f} & {100*r["pair_both_correct"]:.2f} \\\\' for r in means)
    section=r'''\section{Further reviewer revisions: scale and causal controls}
We retain the original correctness intervention, collision construction, complete-logit invariant swaps, and orbit exclusion. A post-confirmation census evaluates every unexposed orientation in each source's remaining 511--530 orbits, totaling 3,066--3,180 inputs per source. Independently paired collision inputs agree on all three visible gold answers and disagree on the compound answer. Each input is used once; complete orbits can recur across pairs. These are larger tests of old models, not new source/world replications.

\begin{table}[ht]\centering\small
\begin{tabular}{lrr}\toprule Condition & Accuracy (\%) & Both (\%)\\\midrule
TABLE
\bottomrule\end{tabular}
\caption{Expanded gold-matched collision subset, averaged over the five old sources. Orbit dependencies remain.}
\end{table}

\paragraph{Equivalence remains secondary.}
On the complete orbit census the unregularized affine--GELU difference is 0.237pp; the descriptive 90\% source interval is $[-1.000,1.475]$pp. A posthoc paired TOST rejects non-equivalence for $\pm3$pp, whereas the original smaller test did not. This does not establish prospective equivalence on new data worlds. Normal-model planning with the old SD of 5.35pp requires 29 independent units for $\pm3$pp and 12 for $\pm5$pp at 80\% power if the true mean is zero; a 1pp true mean raises these to 46 and 14. The old variance is not a variance estimate for the new world. All equivalence analyses are secondary to the functionality question.

\paragraph{Ridge sensitivity weakens a broad optimization conclusion.}
We penalize standardized affine input coefficients, leaving the intercept unpenalized, and fit the same teacher-KD plus normalized-state objective with this added term. The ridge path is fixed before new predictions and selected only on known validation states/logits. Functional accuracy falls as small directions are suppressed; validation-selected fits perform poorly. Thus recovery by the unregularized solution is not robustly demonstrated under ordinary validation-selected regularization. This changes the empirical objective and cannot invalidate the original unregularized optimum certificate. The source of useful small directions remains unresolved.

\paragraph{Natural-state interchange is informative but not an ideal control.}
Balanced donor permutations preserve gold first answers and marginal donor frequency. With a distinct second answer, the natural donor is followed in 56.89\% of cases. With the same second answer, only 82.13\% remain correct and 18.25\% change predictions. All tested first-step logits remain invariant. The substantial same-answer disturbance fails the proposed near-invariant control, and natural components do not guarantee natural hybrids. The evidence supports downstream answer-information transport under the specified intervention, rather than natural mathematics-specific causal encoding. Geiger et al.'s interchange framework motivates the stronger prospective control \cite{geiger2021causal}.

\paragraph{A new-world confirmation is kept separate.}
Before any new composite score, we fix 17 independently sampled source/world units in $\F_{17}$, a 512-dimensional encoder, 2,000 collision pairs and 512 IID inputs per unit, and exactly two primary comparisons with Holm correction. Corpus, support, validation, test, initialization and optimizer randomness vary by unit; no weights are reused. A noisy second-step fit uses only single-step supervision and known validation. The declared 40\% accuracy and 5\% same-answer prediction-change checks cannot select candidates or replace failed sources. The separate new results, once complete, determine whether these requirements are met; the present old-world diagnostics do not satisfy them.

\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{figures/reviewer_revision.pdf}
\caption{Complete-census regularization sensitivity and natural-state interchange. The same-answer disturbance prevents an unqualified causal-mechanism claim.}
\end{figure}
'''.replace('TABLE',rows_tex)
    (Path('paper/sections')/'reviewer_revision.tex').write_text(section)
    atomic_json(root/'delivery.json',{'status':'complete','completed_utc':now(),
        'artifact_sha256':{str(p):sha(p) for p in root.iterdir() if p.is_file() and p.name!='delivery.json'}})


if __name__=='__main__':run()
