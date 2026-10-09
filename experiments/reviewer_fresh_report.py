"""Report the completed frozen F17 confirmation without selecting new fits."""
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
    parent = Path('results/reviewer_fresh_confirmation')
    result = json.loads((parent / 'results.json').read_text())
    audit = json.loads(Path('results/reviewer_fresh_audit/verification.json').read_text())
    assert result['status'] == 'complete' and audit['status'] == 'passed'
    root = Path('results/reviewer_fresh_reporting_v2')
    root.mkdir(exist_ok=True)
    signature = {'code_sha256': sha(__file__), 'parent_results_sha256': sha(parent / 'results.json'),
                 'independent_audit_sha256': sha('results/reviewer_fresh_audit/verification.json'),
                 'scope': 'Reporting of fixed primary contrasts and gates after all registered fits. No new model selection or inference.'}
    file = root / 'protocol.json'
    if file.exists():
        assert json.loads(file.read_text())['signature'] == signature
    else:
        atomic_json(file, {'registered_utc': now(), 'outcomes_already_observed': True, 'signature': signature})
    conditions = ['linear_correct', 'linear_wrong', 'nonlinear_correct', 'nonlinear_wrong']
    rows = []
    for second in ['original', 'known_selected']:
        for split in ['iid', 'collisions']:
            for condition in conditions:
                selected = [r for r in result['records']
                            if (r['second'], r['split'], r['condition']) == (second, split, condition)]
                assert len(selected) == 17
                keys = ['accuracy', 'first_accuracy', 'first_state_mse', 'oracle_accuracy']
                if split == 'collisions':
                    keys.append('pair_both_correct')
                rows.append({'second': second, 'split': split, 'condition': condition,
                             **{k: float(np.mean([r[k] for r in selected])) for k in keys}})
    get = lambda condition, second='known_selected', split='collisions': next(
        r for r in rows if (r['condition'], r['second'], r['split']) == (condition, second, split))
    correct, wrong = get('nonlinear_correct'), get('nonlinear_wrong')
    normalized = []
    for i in range(17):
        selected = [r for r in result['records'] if r['unit'] == i and r['second'] == 'known_selected' and r['split'] == 'collisions']
        c, w = [next(r for r in selected if r['condition'] == condition) for condition in ['nonlinear_correct', 'nonlinear_wrong']]
        denominator = c['oracle_accuracy'] - w['accuracy']
        normalized.append((c['accuracy'] - w['accuracy']) / denominator if denominator > 0 else None)
    interchanges = []
    for condition in ['null_different', 'row_different', 'null_same']:
        selected = [r for r in result['interchanges'] if r['condition'] == condition]
        interchanges.append({'condition': condition, **{k: float(np.mean([r[k] for r in selected]))
                            for k in ['donor_target_accuracy', 'prediction_changed_fraction']},
                            'maximum_first_logit_change': max(r['maximum_first_logit_change'] for r in selected)})
    normalized_mean = float(np.mean([v for v in normalized if v is not None]))
    summary = {'status': 'complete', 'means': rows, 'interchanges': interchanges,
               'normalized_oracle_gap_closed_by_unit': normalized,
               'normalized_oracle_gap_closed_mean': normalized_mean,
               'primary_contrasts': result['primary_contrasts'],
               'floor_check': result['floor_check'], 'same_answer_stability': result['same_answer_stability'],
               'source_gates': audit['source_primitive_gates'],
               'equivalence_primary': False,
               'equivalence_scope': 'No new converged affine fitting in F17. Old-world paired TOST is secondary and posthoc.',
               'independence_scope': 'Independent randomly sampled datasets and model initializations in one previously untrained field system. Cross-unit overlap is permitted. The legacy joint-preparation minibatch index RNG is shared; first/second fitting RNGs vary.',
               'all_parent_files_preserved': True}
    atomic_json(root / 'results.json', summary)
    table = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in [
        r['second'], r['condition'], f'{100*r["accuracy"]:.2f}', f'{100*r["pair_both_correct"]:.2f}',
        f'{100*r["first_accuracy"]:.2f}', f'{r["first_state_mse"]:.4f}', f'{100*r["oracle_accuracy"]:.2f}']) + '</tr>'
        for r in rows if r['split'] == 'collisions')
    contrast_table = ''.join(f'<tr><td>{html.escape(r["contrast"])}</td><td>{r["mean_pp"]:.3f}</td><td>{r["ci95_pp"]}</td><td>{r["holm_adjusted_p"]:.6g}</td></tr>' for r in result['primary_contrasts'])
    exchange_table = ''.join(f'<tr><td>{r["condition"]}</td><td>{100*r["donor_target_accuracy"]:.2f}</td><td>{100*r["prediction_changed_fraction"]:.2f}</td><td>{r["maximum_first_logit_change"]:.3g}</td></tr>' for r in interchanges)
    floor = result['floor_check']['passed']; stable = result['same_answer_stability']['passed']
    body = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ccc}}code{{overflow-wrap:anywhere}}</style>
<h1>F17 冻结协议确认：17 个独立数据与源模型单元</h1>
<p>所有 68 个第一算子拟合、51 个第二算子候选和 17 个新源模型完成后，才打开复合测试。每单元 2000 对碰撞（4000 个输入）及 512 个 IID 输入。模型宽度 512，是旧模型的四倍。数据世界与初始化一起改变；这是一个新有限域中的 17 个随机数据划分，不是 17 种数学系统。</p>
<h2>两个固定主对比</h2><p>以独立数据／源模型单元作为统计单位；双侧配对 t 检验，两项 Holm 校正。区间是逐项 95% 区间，未经同时校正。碰撞样本数不冒充源模型数。</p>
<table><tr><th>对比</th><th>平均百分点</th><th>95%区间</th><th>Holm p</th></tr>{contrast_table}</table>
<h2>复合与误差分解</h2><p>第二步噪声强度只按单步验证选择，共享给所有第一算子；没有使用复合标签或复合状态训练。oracle 使用真实中间状态，只是诊断。原第二算子也完整报告。</p>
<table><tr><th>第二算子</th><th>第一算子</th><th>复合准确率%</th><th>配对双正确%</th><th>第一步准确率%</th><th>第一步状态MSE</th><th>oracle%</th></tr>{table}</table>
<p>正确关系相对错配平均缩小 {100*normalized_mean:.2f}% 的 oracle 差距。状态 MSE 是本世界的原始坐标误差，不直接用于跨领域大小比较。</p>
<h2>自然状态交换</h2><table><tr><th>交换</th><th>供体答案命中%</th><th>预测改变%</th><th>最大第一步logit差</th></tr>{exchange_table}</table>
<p>Q 交换保持全部第一步 logits，P 交换是会改变 logits 的定位诊断；二者都匹配第一步金答案，跨轨道并保持供体使用频率。自然组成分量仍不保证混合状态在流形上。测试的是下游答案信息传递，不能单凭它识别数学关系编码。</p>
<h2>预定门槛与限制</h2><p>40% 复合准确率门槛：<b>{'通过' if floor else '未通过'}</b>；同答案交换改变率≤5%：<b>{'通过' if stable else '未通过'}</b>。所有失败单元保留，没有按测试结果补换来源或选择更多候选。</p>
<p>等价性仍作次要分析。新世界没有拟合收敛仿射，所以不能以同预算线性／GELU差异回答收敛后的等价性。旧世界的事后 TOST 结果不升级为新世界确认。</p>
<p>联合编码器／生成元预训练的 minibatch 索引随机流沿用共享的旧实现；各单元的数据、初始化和第一／第二算子拟合随机流不同。数据之间允许偶然重叠；宽度、输入覆盖与有限域同时改变，因此不能把变化归因于宽度一个因素。</p>
<p>独立审计通过 {audit['checks']:,} 项重算检查。<a href="results.json">汇总及逐来源效应</a> · <a href="../reviewer_fresh_confirmation/results.json">原始逐单元结果</a> · <a href="../reviewer_fresh_confirmation/protocol.json">冻结协议与接口修订</a> · <a href="../reviewer_fresh_audit/verification.json">独立审计</a></p></html>'''
    (root / 'report.html').write_text(body)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    labels = ['Linear\ncorrect', 'Linear\nwrong', 'GELU\ncorrect', 'GELU\nwrong']
    for j, condition in enumerate(conditions):
        values = [100*r['accuracy'] for r in result['records'] if r['second'] == 'known_selected' and r['split'] == 'collisions' and r['condition'] == condition]
        axes[0].bar(j, np.mean(values), color='#276c9f' if condition.endswith('correct') else '#ad6c37')
        axes[0].scatter(j + np.linspace(-.16, .16, len(values)), values, s=9, color='black', alpha=.6)
    axes[0].axhline(40, ls='--', color='gray', lw=1); axes[0].set_xticks(range(4), labels); axes[0].set_ylabel('Collision accuracy (%)')
    axes[0].set_title('17 independent source/data units', fontsize=10)
    axes[1].bar(range(3), [100*r['donor_target_accuracy'] for r in interchanges], color=['#276c9f', '#ad6c37', '#6f8c82'])
    axes[1].set_xticks(range(3), ['Q: different\nfuture', 'P: different\nfuture', 'Q: same\nfuture']); axes[1].set_ylabel('Donor answer followed (%)'); axes[1].set_title('Natural-state interchange', fontsize=10)
    for ax in axes: ax.set_ylim(0, 100); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    for ext in ['pdf', 'png', 'svg']: fig.savefig(Path('paper/figures/reviewer_fresh').with_suffix('.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    a, b = result['primary_contrasts']; null, row, same = interchanges
    def fmt_p(p):
        value = f'{p:.3g}'
        if 'e' not in value:
            return value
        mantissa, exponent = value.split('e')
        return mantissa + r'\times10^{' + str(int(exponent)) + '}'
    tex_rows = '\n'.join(f'{r["condition"].replace("_", " ")} & {100*r["accuracy"]:.2f} & {100*r["pair_both_correct"]:.2f} & {100*r["first_accuracy"]:.2f} & {r["first_state_mse"]:.4f} \\\\' for r in rows if r['second'] == 'known_selected' and r['split'] == 'collisions')
    tex = r'''\section{Prospective confirmation in a new field and larger model}\label{sec:fresh}
The frozen new-world protocol was completed without inspecting any compound score during fitting. Seventeen independently sampled corpora, supports, validation sets, test sets and source initializations in $\F_{17}$ replace the shared old-world data. Each test contains 2,000 collision pairs and 512 IID inputs. The hidden width is 512 instead of 128. The joint encoder/generator preparation minibatch index stream retains the shared legacy scheduling code; data, model initialization and first/second-operator fitting streams vary. Randomly sampled units can overlap across units, but no weights are reused. This is one new algebraic system with independent random datasets, rather than 17 distinct systems.

\paragraph{Fixed design and two primary tests.}
Correct versus answer-matched incorrect first-step correspondence is crossed with the same parameter-matched Identity/GELU predictor. Correct output distillation, inputs, steps and encoder are identical within a unit. All 68 first predictors and 51 second-step noise candidates were fit before the test was opened. Second-step noise is chosen only by known single-step validation and is shared across the first-predictor arms. No compound label or compound target state enters training. The two predeclared source-level contrasts use paired two-sided $t$ tests with Holm correction; reported 95\% intervals are marginal rather than simultaneous.

GELU correct minus incorrect compound accuracy is EFFECTA pp, with interval CIA pp and Holm $p=PA$. Natural null versus row interchange donor-target accuracy is EFFECTB pp, with interval CIB pp and Holm $p=PB$. Row interchange is a localization diagnostic that can change first logits; null interchange preserves them. All 17 units are retained.

\begin{table}[ht]\centering\small
\begin{tabular}{lrrrr}\toprule First predictor & Compound (\%) & Both (\%) & First (\%) & State MSE\\\midrule
TABLE
\bottomrule\end{tabular}
\caption{Mean collision performance with the validation-selected second operator in the new field. Datasets and initializations both vary across the 17 statistical units.}
\end{table}

\paragraph{Error decomposition and the performance gate.}
The selected second operator reaches ORACLE\% on true intermediate states, versus SCORE\% with correctly paired GELU predictions. The mean fraction of the oracle--incorrect gap closed is GAP\%. This contrasts first-state error with downstream failure and does not treat first-answer accuracy as state sufficiency. The original second operator, all IID results and all individual units are also archived. The preregistered 40\% performance gate FLOOR. No further candidate was selected by compound results.

\paragraph{Natural-state interchange and its invalidation control.}
Donors have the same first gold answer, lie in a different orbit and are used once per arm. Distinct-future null donors are followed in NULL\%, compared with ROW\% for distinct-future row donors. Same-future null donors retain the donor answer in SAME\%, but change CHANGE\% of predictions; the preregistered 5\% change gate STABLE. The maximum first-logit change for null interchange is LOGIT. Even natural components can yield off-manifold hybrids. These findings concern downstream answer-information transport, not identification of natural mathematics-specific relation coding.

\paragraph{Scope and equivalence.}
The larger model, increased input coverage, new field and noisy second-step fitting change together. This confirmation cannot isolate a width effect. It also does not fit the converged affine baseline in the new world; any finite-budget activation contrast is not a convergence or equivalence result. The old affine--GELU TOST analyses remain secondary and posthoc, with the primary wording ``additional advantage was not confirmed'' limited to the old unregularized diagnostic. Independent replay reconstructs CHECKS label, split, prediction and invariant entries; it validates the registered computation without strengthening its causal scope.

\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{figures/reviewer_fresh.pdf}
\caption{Frozen new-field confirmation. Points are independent source/data units; the dashed line is the predeclared 40\% accuracy gate. Natural-state swaps address downstream information transport.}
\end{figure}
'''
    replacements = {'EFFECTA': f'{a["mean_pp"]:.3f}', 'CIA': f'[{a["ci95_pp"][0]:.3f},{a["ci95_pp"][1]:.3f}]', 'PA': fmt_p(a['holm_adjusted_p']),
                    'EFFECTB': f'{b["mean_pp"]:.3f}', 'CIB': f'[{b["ci95_pp"][0]:.3f},{b["ci95_pp"][1]:.3f}]', 'PB': fmt_p(b['holm_adjusted_p']),
                    'TABLE': tex_rows, 'ORACLE': f'{100*correct["oracle_accuracy"]:.2f}', 'SCORE': f'{100*correct["accuracy"]:.2f}', 'GAP': f'{100*normalized_mean:.2f}',
                    'FLOOR': 'is passed' if floor else 'is not passed', 'NULL': f'{100*null["donor_target_accuracy"]:.2f}', 'ROW': f'{100*row["donor_target_accuracy"]:.2f}',
                    'SAME': f'{100*same["donor_target_accuracy"]:.2f}', 'CHANGE': f'{100*same["prediction_changed_fraction"]:.2f}', 'STABLE': 'is passed' if stable else 'is not passed',
                    'LOGIT': '$' + fmt_p(null['maximum_first_logit_change']) + '$', 'CHECKS': f'{audit["checks"]:,}'}
    for key in sorted(replacements, key=len, reverse=True): tex = tex.replace(key, replacements[key])
    (Path('paper/sections') / 'reviewer_fresh.tex').write_text(tex)
    old_section = Path('paper/sections/reviewer_revision.tex')
    old = old_section.read_text()
    start = old.index('\\paragraph{A new-world confirmation is kept separate.}')
    end = old.index('\\begin{figure}', start)
    old_section.write_text(old[:start] + r'''\paragraph{A new-world confirmation is kept separate.}
The following section reports the completed frozen $\F_{17}$ experiment. Those independent dataset/source units are distinguished from the posthoc old-world census above. The interface-only parameter-count amendment was made before first-operator fitting or new composite evaluation, with both protocol versions retained. Failed performance and interchange gates are reported rather than used to replace units.

''' + old[end:])
    framing = {
        'abstract.tex': rf'''\begin{{abstract}}
We test whether correct mathematical state correspondence supports unseen composition while output supervision is held fixed. A frozen confirmation in a previously untrained finite field uses 17 independently sampled datasets and source initializations, a fourfold-width encoder, and 2,000 collision pairs per unit. With single-step-selected second operators, correctly paired GELU predictors reach {100*correct['accuracy']:.2f}\% compound accuracy versus {100*wrong['accuracy']:.2f}\% for answer-matched incorrect pairing; the source-level difference is {a['mean_pp']:.2f} points with Holm $p={fmt_p(a['holm_adjusted_p'])}$. Natural readout-null interchange preserves all first-step logits and follows distinct-future donor answers in {100*null['donor_target_accuracy']:.2f}\%; same-future interchange changes {100*same['prediction_changed_fraction']:.2f}\% of predictions, restricting its causal interpretation. Earlier predicted-null transfer and adequately fit unregularized affine controls show a functional pathway but leave an additional expressivity advantage unconfirmed; validation-selected ridge regularization weakens that recovery. Equivalence remains secondary and posthoc. The results concern supplied correspondences in relation-prepared representations, without establishing spontaneous algebra discovery or universal relation-specific causal encoding.
\end{{abstract}}
''',
        'introduction.tex': r'''\section{Introduction}
Exact input transformations provide a controlled way to study what a neural representation preserves. A task answer may transform predictably when an input is inverted, complemented, translated, or differentiated. Matching transformed answers, aligning hidden vectors and predicting an unseen composition are different achievements. A representation can support an accurate first-step answer while failing to preserve information needed by the next operation.

We ask whether \emph{the correctness of a supplied state correspondence} improves held-out composition when output supervision is fixed. Exact finite mathematical worlds provide incorrect correspondences that preserve visible gold answers, endpoint marginals, input exposure and correct teacher-output supervision. The intervention changes which intermediate hidden state is treated as the geometric target. It identifies the functional contribution of this correspondence within the specified training procedure; it does not isolate algebra from every output-related statistical signal.

The evidence has three stages. Permutation and small matrix studies motivate correspondence-sensitive composition. A five-initialization matrix study, with shared training and test data, then crosses first-operator activation and correspondence correctness and tests predicted readout-null transfer plus same-family affine convergence. Reviewer diagnostics enlarge the old tests, examine natural-state counterfactual swaps and add a validation-selected ridge path. Finally, a prospectively frozen confirmation changes both datasets and initializations in a previously untrained field, uses a fourfold-width model and retains exactly two primary contrasts with source-level Holm correction.

The new-field confirmation contains 17 independent random dataset/source units, each with 2,000 collision pairs and 512 IID inputs. The first primary contrast asks whether correct versus answer-matched incorrect GELU correspondence improves composition. The second compares natural null and row donor following under same-first-answer, different-future-answer interchange. A same-future-answer disturbance control and a 40\% performance gate are recorded separately; neither can select models or replace sources. Natural-state interchange tests downstream answer-information transport and still requires scrutiny of mixed-state validity.

All matrix encoders receive ordinary task training and correct single-generator preparation before the frozen operator intervention. The evidence is conditional on these prepared representations. It does not test whether ordinary training discovers algebra spontaneously, and the program supplies operation identity and order. Width, input coverage and field change together in the new confirmation, so that study cannot estimate the effect of width alone.

The contribution is a controlled empirical account of when supplied geometric correspondence is functionally useful and how its interpretation can fail. Answer-matched correspondence, downstream prediction, full-logit invariant swaps, optimization diagnostics, natural-state invalidation controls and independent dataset/source confirmation separate answer preservation from compositional state sufficiency. Polynomial primitive-learning failures and null-only-loss failures delimit the result. No universal mathematical mechanism or new diagnostic in isolation is claimed.
''',
        'discussion.tex': rf'''\section{{Discussion and limitations}}
The prospective new-field comparison measures a {a['mean_pp']:.3f}pp correctness benefit for GELU compound prediction, with marginal 95\% interval $[{a['ci95_pp'][0]:.3f},{a['ci95_pp'][1]:.3f}]$ and Holm $p={fmt_p(a['holm_adjusted_p'])}$. Independent datasets and initializations replace the shared-data limitation of the earlier five-source comparison. The evidence remains conditional on correctly prepared encoders and this supplied correspondence intervention.

\paragraph{{What the interventions identify.}}
The readout-null projector concerns one fixed linear readout. It neither removes nonlinear answer information nor ensures independence from current or future answers; diagnostic probes recover both. Predicted-null transfer establishes a functional effect of the defined hybrid. Natural-state interchange supplies a more direct counterfactual information-transport check, but natural components can still produce off-manifold mixtures. In the new test same-future swaps change {100*same['prediction_changed_fraction']:.2f}\% of predictions, and the fixed 5\% stability gate {'passes' if stable else 'fails'}. This gate must accompany any donor-following result. The row-swap comparison is a localization diagnostic, not an intervention preserving all first-step logits. We do not identify the transferred content uniquely as mathematical relation information.

Correct versus incorrect hidden targets preserve gold answer triples and use the same soft output targets. They still change target-state consistency with those soft targets and the natural hidden-state distribution. Marginal covariance transport and sparse scalar-error matching do not eliminate conditional state-distribution explanations. Oracle error-calibrated random donors use true intermediate states and cannot be interpreted as available learned predictors.

\paragraph{{Performance and generalization.}}
New-field correct GELU accuracy is {100*correct['accuracy']:.2f}\%, compared with {100*correct['oracle_accuracy']:.2f}\% for true intermediates; the 40\% performance gate {'passes' if floor else 'fails'}. High first-step accuracy does not guarantee a state that the second operator can use. Every registered unit and every failed gate is retained. Noise-candidate selection uses single-step validation only, and the original second operator is reported alongside the selected one. These changes were fixed before composite scores, not selected to repair an observed compound result.

The 17 source/data units come from independent random partitions of one new finite-field system; they are not 17 distinct algebraic systems. Cross-unit data overlap is permitted. Ordinary source training, initializations and predictor fitting have varying random streams; the joint-preparation batch-index stream retains a shared legacy schedule. The primary statistical unit is the dataset/source pair, with Holm adjustment for two predeclared tests. Old-world bootstrap and TOST analyses remain descriptive or secondary. Confidence intervals in the new report are marginal, not simultaneous. Width, coverage, field and second-step robustness change together; none is isolated as the sole cause of a performance change.

\paragraph{{Optimization and numerical limits.}}
Adequate unregularized affine fitting largely recovers the old-world nonlinear functional advantage; an additional advantage was not confirmed there. A zero-crossing interval is not an equivalence test. Expanded old-world TOST analyses are explicitly secondary and posthoc, and no converged affine fitting was performed in the new field. Validation-selected ridge fits lose much of the old affine benefit, preventing a broad claim that optimization explains away nonlinear advantages. Ridge changes the objective, so this loss does not invalidate the original unregularized optimum diagnostic.

Direct affine fitting changes parameterization, precision and budget. Ill-conditioned designs, large coefficients and tiny directions associated with LayerNorm's exact affine constraint leave numerical robustness unresolved. Empirical convergence does not bound held-out state error. No precise allocation of the effects to optimization, conditioning and expressivity is supported. The irreversible polynomial setting has primitive-generalization failures; different domain architectures and supervision directions prevent a universal algebra-based explanation.

\section{{Conclusion}}
Correct supplied state correspondence and compositional state sufficiency are separate from first-step answer accuracy. The fixed new-world study tests correspondence functionality and natural-state information transport with independent datasets, larger tests and two corrected primary comparisons. Earlier predicted-null and affine diagnostics reveal functional pathways while the ridge and same-answer controls restrict mechanistic claims. The paper reports the benefits, failed gates and numerical boundaries of these interventions; it does not equate detectable geometry with spontaneous or universal algebraic reasoning.
'''}
    for name, content in framing.items():
        (Path('paper/sections') / name).write_text(content)
    manuscript = Path('paper/manuscript.tex')
    main = manuscript.read_text()
    if r'\input{sections/reviewer_fresh.tex}' not in main:
        main = main.replace(r'\input{sections/reviewer_revision.tex}',
                            '\\input{sections/reviewer_revision.tex}\n\\input{sections/reviewer_fresh.tex}')
        manuscript.write_text(main)
    atomic_json(root / 'delivery.json', {'status': 'complete', 'completed_utc': now(),
                'artifact_sha256': {str(p): sha(p) for p in root.iterdir() if p.is_file() and p.name != 'delivery.json'}})
    print(json.dumps({'report': str(root / 'report.html'), 'means': [correct, wrong], 'floor_passed': floor, 'same_answer_gate_passed': stable}), flush=True)


if __name__ == '__main__':
    run()
