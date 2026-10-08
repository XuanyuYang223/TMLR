"""Generate revised manuscript results directly from immutable study records."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now

ROOT = Path('results/final_mechanism_confirmation')
PAPER = Path('paper')
LABELS = {
    'linear_correct': 'Budget linear, correct',
    'linear_wrong': 'Budget linear, incorrect',
    'nonlinear_correct': 'Budget GELU, correct',
    'nonlinear_wrong': 'Budget GELU, incorrect',
    'affine_correct_ols_initialization': 'Converged affine, correct',
    'affine_wrong_ols_initialization': 'Converged affine, incorrect',
    'swap_linear_correct_null_nonlinear_correct': 'Linear + correct GELU null',
    'swap_linear_correct_null_nonlinear_wrong': 'Linear + incorrect GELU null',
    'swap_nonlinear_correct_null_linear_correct': 'GELU + linear null',
}

def sci(v):
    a, b = f'{v:.2e}'.split('e')
    return '$'+rf'{a}\times10^{{{int(b)}}}'+'$'

def effect(r):
    return f'{r["mean_pp"]:+.2f} pp [{r["bootstrap_95_pp"][0]:+.2f}, {r["bootstrap_95_pp"][1]:+.2f}]'

def write(name, content):
    (PAPER / 'sections' / name).write_text(content)

def run():
    records = json.loads((ROOT / 'results.json').read_text())
    means = {r['condition']: r for r in records['means'] if r['split'] == 'collisions'}
    effects = {r['contrast']: r for r in records['contrasts'] if r['endpoint'] == 'accuracy'}
    for d in ['sections', 'figures']:
        (PAPER / d).mkdir(exist_ok=True)
    write('abstract.tex', r'''\begin{abstract}
We study whether the correctness of supplied mathematical state correspondences supports unseen composition when correct output supervision is fixed. In a final finite-matrix confirmation with five new source initializations and relation-prepared frozen encoders, a parameter-matched activation-by-correctness intervention yields a 12.89 percentage-point interaction. Transferring predicted readout-null components from correctly trained GELU operators improves a linear recipient by 8.75 points while preserving all first-step logits. Adequate affine fitting under the identical single-step loss recovers most of the functional correspondence benefit; the remaining interaction is 0.31 points with a descriptive interval crossing zero. Ill-conditioning and precision sensitivity qualify that optimization result. Permutation experiments provide a second correspondence-sensitive setting, while initial matrix, polynomial, and null-only-loss studies establish boundaries. The evidence supports a conditional functional pathway and an optimization-sensitive interpretation, without establishing spontaneous algebra discovery, robust hidden-relation inference, or a universal output-independent mechanism.
\end{abstract}
''')
    table_rows = []
    for c, label in LABELS.items():
        r = means[c]
        table_rows.append(' & '.join([label, f'{100*r["accuracy"]:.2f}',
                                     f'{100*r["pair_both_correct"]:.2f}',
                                     f'{100*r["first_accuracy"]:.2f}',
                                     f'{r["hidden_nmse"]:.5f}']) + r' \\')
    section = r'''\section{Final confirmation: relation benefit, information transfer, and convergence}
\subsection{Budget-controlled correspondence benefit}
Table~\ref{tab:final} reports all original arms and the separate convergence and transfer diagnostics.
GELU's correct-minus-incorrect collision advantage is GELUBENEFIT.
The activation-by-correctness interaction is INTERACTION.
Its direction is positive in all five new sources. Budget-linear correct-minus-incorrect performance is LINEARBENEFIT.
All four budget arms have 100\% first-generator accuracy on the collision set.
Thus matching the winning first-step answer is insufficient to explain the compound-performance differences.

\begin{table}[ht]\centering\small
\begin{tabular}{lrrrr}\toprule
First-state predictor & AB (\%) & Both (\%) & A (\%) & State NMSE\\\midrule
TABLEROWS
\bottomrule\end{tabular}
\caption{Final collision test averaged over five independent source initializations. Both requires both members of a collision pair to be correct. Converged affine fitting receives additional optimization budget; hybrid states are predicted-component interventions. State NMSE uses the fixed training variance, not displacement energy.}\label{tab:final}
\end{table}

The correct GELU compound accuracy is GELUACCURACY\%, and pairwise double correctness is GELUPAIR\%.
These are local improvements; stable inference above the three-gold-answer-only 50\% ceiling is not established.
True-intermediate composition averages ORACLE\%, showing that a reliable second map exists when supplied the actual intermediate.
This oracle result is a diagnostic, not a predictor.

\subsection{A predicted component transmits functional benefit}
For the fixed full first-step linear readout, let $P$ project onto its row span and $Q=I-P$.
We construct
\[
 \hat h_{L\leftarrow G}=\hat h_LP+\hat h_GQ.
\]
The donor and recipient are their own predicted first states in the same frozen source coordinates. No true intermediate enters the swap.
Because $QW^\top=0$, all recipient first-step logits are preserved.
The maximum measured logit change across the interventions is LOGITCHANGE.

A correct GELU donor raises the correct linear recipient from LINEARACCURACY\% to SWAPACCURACY\%, an increment of SWAPGAIN.
The same swap with an incorrect GELU donor reaches WRONGDONORACCURACY\%.
The correct-minus-incorrect donor contrast is DONORCONTRAST, positive in every source.
Reverse replacement of the correct GELU null component by the linear one lowers GELU performance to REVERSEACCURACY\%.
This supports a correspondence-sensitive functional pathway through predicted information invisible to the fixed first readout.
It does not identify that subspace with all answer-independent information, or guarantee that hybrid states lie on the encoder manifold.

\subsection{Adequate affine fitting changes the interpretation}
Identity residuals express the unrestricted affine family. We therefore optimize that family directly under Equation~\ref{eq:loss}, retaining the same correct teacher logits, geometric targets, geometry coefficient, and zero regularization.
Float64 SVD whitening conditions the training design; readout-null coefficients take their least-squares optimum, while readout-row coefficients are optimized by L-BFGS.
Both OLS and budget-linear row initializations were fixed before final training.
No compound labels, states, or checkpoint selection enter fitting.
Appendix~\ref{app:convergence} details the objective and numerically evaluated gradient-based convergence bounds.

Converged correct affine prediction reaches AFFINEACCURACY\%, and its correct-minus-incorrect gain is AFFINEBENEFIT.
The remaining GELU-versus-converged-affine relation-advantage interaction is REMAINING.
Its interval crosses zero. We did not confirm an additional functional expressivity advantage; neither does this establish equivalence.
Correct affine improvement over the original budget-linear arm is OPTIMIZATIONGAIN.
This is a same-family optimization diagnostic with changed parameterization, precision and budget, rather than another budget-matched causal comparison.
GELU retains a smaller first-state NMSE (GELUNMSE versus AFFINENMSE); total state-fitting differences alone do not determine compound accuracy.

\begin{figure}[ht]\centering
\includegraphics[width=.95\linewidth]{figures/final_contrasts.pdf}
\caption{All five source effects and descriptive crossed source/pair 95\% intervals for predeclared contrasts. Convergence narrows the original activation-by-correctness interaction.}\label{fig:contrasts}
\end{figure}

\subsection{Convergence and precision qualifications}
All twenty affine fits pass the registered convergence threshold. The largest independently recomputed training-objective gap bound is GAPBOUND.
The two initializations give identical compound classes; their largest first-state difference is REPEATDIFF.
However, design condition numbers range from CONDMIN to CONDMAX, and coefficient norms from COEFMIN to COEFMAX.
Complete float32 replay of the affine, second-operator, and readout pipeline changes CORRECTCHANGES/1280 correct-pairing and WRONGCHANGES/1280 incorrect-pairing collision predictions relative to float64.
Aggregate correct/incorrect accuracy is CORRECT64/WRONG64\% in float64 and CORRECT32/WRONG32\% in float32.
The aggregate correspondence benefit survives this precision check, while individual predictions are numerically sensitive.
LayerNorm's exact affine constraint makes finite-precision rank effects a remaining candidate explanation.
Convergence of the empirical objective does not guarantee numerically robust state identification or out-of-distribution stability.
'''
    verification = json.loads((ROOT / 'independent_verification.json').read_text())
    metadata = [json.loads(p.read_text())['metadata'] for p in (ROOT / 'affine_fits').glob('*.json')]
    values = {
        'TABLEROWS': '\n'.join(table_rows),
        'GELUBENEFIT': effect(effects['gelu_relation_advantage']),
        'LINEARBENEFIT': effect(effects['linear_relation_advantage']),
        'INTERACTION': effect(effects['same_budget_interaction']),
        'SWAPGAIN': effect(effects['correct_null_swap_gain']),
        'DONORCONTRAST': effect(effects['correct_vs_wrong_null_donor']),
        'AFFINEBENEFIT': effect(effects['converged_linear_relation_advantage']),
        'REMAINING': effect(effects['remaining_relation_interaction']),
        'OPTIMIZATIONGAIN': effect(effects['linear_optimization_gain']),
        'LOGITCHANGE': sci(records['max_swap_first_logit_change']),
        'GAPBOUND': sci(verification['max_recomputed_global_gap_bound']),
        'REPEATDIFF': sci(max(r['max_repeat_first_state_difference'] for r in records['solver_repeat_checks'])),
        'CONDMIN': sci(min(r['condition_number'] for r in metadata)),
        'CONDMAX': sci(max(r['condition_number'] for r in metadata)),
        'COEFMIN': sci(min(r['coefficient_norm'] for r in metadata)),
        'COEFMAX': sci(max(r['coefficient_norm'] for r in metadata)),
        'ORACLE': f'{100*means["nonlinear_correct"]["oracle_accuracy"]:.2f}',
        'GELUPAIR': f'{100*means["nonlinear_correct"]["pair_both_correct"]:.2f}',
        'GELUNMSE': f'{means["nonlinear_correct"]["hidden_nmse"]:.5f}',
        'AFFINENMSE': f'{means["affine_correct_ols_initialization"]["hidden_nmse"]:.5f}',
    }
    for key, c in [('GELUACCURACY', 'nonlinear_correct'), ('LINEARACCURACY', 'linear_correct'),
                   ('SWAPACCURACY', 'swap_linear_correct_null_nonlinear_correct'),
                   ('WRONGDONORACCURACY', 'swap_linear_correct_null_nonlinear_wrong'),
                   ('REVERSEACCURACY', 'swap_nonlinear_correct_null_linear_correct'),
                   ('AFFINEACCURACY', 'affine_correct_ols_initialization')]:
        values[key] = f'{100*means[c]["accuracy"]:.2f}'
    for pairing, prefix in [('correct', 'CORRECT'), ('wrong', 'WRONG')]:
        rows = [r for r in records['precision_checks'] if r['pairing'] == pairing
                and r['kind'] == 'ols_initialization' and r['split'] == 'collisions']
        values[prefix+'CHANGES'] = str(sum(r['changed_compound_predictions'] for r in rows))
        for dtype in ['32', '64']:
            values[prefix+dtype] = f'{100*np.mean([r["float"+dtype+"_accuracy"] for r in rows]):.2f}'
    for k in sorted(values, key=len, reverse=True):
        section = section.replace(k, values[k])
    write('results.tex', section)
    perm = Path('paper/generated_permworld.tex').read_text().split(r'\end{table}')[0] + r'\end{table}' + '\n'
    perm = perm.replace('matched20epochs', 'matched 20 epochs').replace('held-out20-epoch', 'held-out 20-epoch')
    perm = perm.replace(r'\begin{table}[h]', r'\begin{table}[ht]')
    write('supporting_tables.tex', perm)
    rows = []
    keys = ['linear_correct', 'linear_wrong', 'nonlinear_correct', 'nonlinear_wrong',
            'affine_correct_ols_initialization', 'swap_linear_correct_null_nonlinear_correct']
    for i, seed in enumerate([17291, 18401, 19507, 20611, 21713]):
        values_i = [next(r['accuracy'] for r in records['records'] if r['source'] == i
                         and r['condition'] == c and r['split'] == 'collisions') for c in keys]
        rows.append(str(seed)+' & '+' & '.join(f'{100*x:.2f}' for x in values_i)+r' \\')
    contrasts = '\n'.join(
        k.replace('_', r'\_')+' & '+f'{r["mean_pp"]:+.2f}'+' & '+f'[{r["bootstrap_95_pp"][0]:+.2f},{r["bootstrap_95_pp"][1]:+.2f}]'+r' \\'
        for k, r in effects.items())
    cross = Path('paper/generated_cross_domain.tex').read_text()
    for a,b in [('by0.52','by 0.52'),('reaches99.22','reaches 99.22'),('reaches8.27','reaches 8.27'),('The128','The 128')]:
        cross = cross.replace(a,b)
    write('appendix_results.tex', r'''\section{Source-level results and the complete contrast inventory}
\begin{table}[ht]\centering\small
\begin{tabular}{lrrrrrr}\toprule
Source seed & L correct & L wrong & G correct & G wrong & Affine correct & Correct-null swap\\\midrule
'''+'\n'.join(rows)+r'''
\bottomrule\end{tabular}
\caption{Collision AB accuracy (\%) for every final source. All ten affine cases also include the second fixed solver initialization in the archived results.}
\end{table}
\begin{table}[ht]\centering\small
\begin{tabular}{lrr}\toprule
Accuracy contrast & Mean (pp) & Descriptive 95\% interval\\\midrule
'''+contrasts+r'''
\bottomrule\end{tabular}
\caption{All recorded final accuracy contrasts, including secondary explanatory contrasts. These are not eleven independent confirmatory tests.}
\end{table}
\section{Initial cross-domain outcome tables}
The following tables preserve the initial joint affine outcomes. Their six adaptation repeats reuse three sources. They are separate from the five-source frozen-encoder confirmation.
'''+cross)
    plt.rcParams.update({'font.size':10, 'pdf.fonttype':42, 'svg.fonttype':'none',
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout='constrained')
    names = ['same_budget_interaction','correct_null_swap_gain',
             'correct_vs_wrong_null_donor','remaining_relation_interaction']
    for j, name in enumerate(names):
        r = effects[name];lo, hi = r['bootstrap_95_pp'];mu = r['mean_pp']
        ax.errorbar(j, mu, yerr=[[mu-lo], [hi-mu]], fmt='o', color='#285b7a', capsize=4)
        ax.scatter(j+np.linspace(-.13,.13,5), r['source_effects_pp'], marker='x',
                   s=30, color='#bc582a', zorder=3)
    ax.axhline(0, color='gray', lw=1)
    ax.set_ylabel('Paired accuracy difference (percentage points)')
    ax.set_xticks(range(4), ['Budget activation\ninteraction','Correct-null\ntransfer gain',
                            'Correct − incorrect\nnull donor','Interaction after\naffine convergence'], fontsize=9)
    ax.set_ylim(-7,19)
    for ext in ['pdf','svg','png']:
        fig.savefig(PAPER/'figures'/f'final_contrasts.{ext}', dpi=200)
    plt.close(fig)
    audit = {'created_utc':now(), 'experiments_reopened':False,
             'result_source':str(ROOT/'results.json'), 'result_sha256':sha(ROOT/'results.json'),
             'main':'paper/manuscript.tex', 'sections':[str(p) for p in sorted((PAPER/'sections').glob('*.tex'))]}
    atomic_json(PAPER/'revision_sources.json', audit)
    print(json.dumps({'status':'revised_manuscript_sources_written','source_results_sha256':audit['result_sha256']}), flush=True)

if __name__ == '__main__':
    run()
