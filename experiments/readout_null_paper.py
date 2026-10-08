"""Append follow-up evidence to a separate manuscript revision."""
import json
import re
from pathlib import Path
import numpy as np
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now
ROOT=Path('results/readout_null_confirmation')


def run():
    studies={d:json.loads((ROOT/d/'independent_results.json').read_text()) for d in ['matrix','permworld']}
    probe=json.loads(Path('results/first_state_predictor_diagnostic_v2/results.json').read_text())['records']
    lines=[r'\section{Margin-aware diagnostics and a directional-loss confirmation}',
      'This follow-up was specified after the preceding experiments. Old-model diagnostics motivate a new-source, new-test intervention; they are not independent confirmation.',
      r'With row-vector conventions and first-step error $e$, the output change induced by its readout-nullspace part is $d_{\mathrm{null}}=eQ M_B W^\top$, where $Q$ projects onto $\ker W$. This preserves all first-step numeric logits, not every possible nonlinear answer code. We compare harmful rival-score shifts with the corresponding true-intermediate classification margins. Oracle-correct inputs are analyzed separately.',
      r'At equal perturbation norm (10\% of centered intermediate hidden RMS), old matrix-model collision accuracy falls from 99.2\% to 53.6\% in the actual null-error direction; the corresponding permutation figures are 93.3\% and 92.3\%. Random null directions are also tested at identical norm. The analysis is conditional on old trained models and test distributions and does not establish a universal domain ordering.',
      r'\begin{figure}[h]\centering\includegraphics[width=\linewidth]{../results/downstream_harm_diagnostic_v2/perturbations.pdf}',
      r'\caption{Equal-norm actual-error and random-null perturbations of true intermediate states. Real intermediates are diagnostic information, not inputs available to the deployed predictor.}\end{figure}',
      r'\paragraph{Predictor capacity diagnostic.} We fit an affine residual and a one-hidden-layer GELU residual on only base-to-first-generator training states, using 2000 fixed updates and a frozen second operator and readout. Neither compound targets nor compound labels enter fitting. The nonlinear residual has approximately four times as many trainable parameters as the affine residual; this is not a capacity-matched comparison.',
      r'\begin{center}\begin{tabular}{llrr}\toprule Domain & First predictor & Compound (\%) & Both correct (\%)\\\midrule']
    for d in ['matrix','permworld']:
        for kind,label in [('original','Original affine'),('affine_residual','Affine residual'),('nonlinear_residual','Nonlinear residual')]:
            rs=[r for r in probe if (r['domain'],r['kind'],r['split'])==(d,kind,'test')]
            lines.append(f'{d} & {label} & {100*np.mean([r["collision_compound_accuracy"] for r in rs]):.2f} & {100*np.mean([r["collision_pair_both_correct"] for r in rs]):.2f}'+r'\\')
    lines.extend([r'\bottomrule\end{tabular}\end{center}',
      'The capacity diagnostic uses three old source models per domain. Without matched-wrong nonlinear targets it cannot identify the causal effect of mathematical correctness. Identical-budget replay saves all predictors and forecasts and independently reproduces the initial diagnostic without tuning.',
      r'\paragraph{Prospective loss-direction intervention.} Each domain uses three new ordinary-source initializations and one fresh support pool per source. Full versus null-only geometric objectives are crossed with correct versus answer-matched wrong pairings. Each objective is normalized by the fixed ordinary teacher variance in its own space, using training states only. Native labels, teacher outputs, schedules, frozen readout, and all training budgets remain identical across the four arms. Permutation models train 20 epochs from initialization (distinct from the earlier dependent 10-to-20 continuation), and matrix models train 900 epochs. All twelve fits complete before the domain compound test opens.',
      'Permutation sources reuse the original known-state corpus, so the new-source claim concerns initialization rather than new corpora. Their support and test orbits are excluded from all local archived input datasets. Matrix partitions are newly sampled, exclude previously examined matrix test orbits, and separate entire source/validation/support/test group orbits. Historical training orbits may recur under entirely new models.',
      r'\begin{center}\begin{tabular}{llrr}\toprule Domain & Direction/pairing & Compound (\%) & Both correct (\%)\\\midrule'])
    for d,r in studies.items():
        for row in r['means']:
            if row['split']=='collisions':
                label=row['condition'].replace('_',' ')
                lines.append(f'{d} & {label} & {100*row["accuracy"]:.2f} & {100*row["pair_both_correct"]:.2f}'+r'\\')
    lines.extend([r'\bottomrule\end{tabular}\end{center}',
      r'The predeclared primary statistic is the matrix accuracy interaction: $(\mathrm{null,correct}-\mathrm{null,wrong})-(\mathrm{full,correct}-\mathrm{full,wrong})$. Confidence intervals resample three source clusters and complete collision pairs. They do not treat the twelve models as twelve independent sources.',
      r'\begin{center}\begin{tabular}{lrr}\toprule Domain & Interaction (pp) & Source+pair 95\% interval\\\midrule'])
    for d,r in studies.items():
        c=r['contrasts']['accuracy']['interaction_null_minus_full'];ci=c['source_and_whole_pair_interval_pp']
        lines.append(f'{d} & {c["mean_pp"]:+.2f} & [{ci[0]:+.2f}, {ci[1]:+.2f}]'+r'\\')
    lines.extend([r'\bottomrule\end{tabular}\end{center}',
      'Null-only loss removes the full-space row penalty as well as emphasizing null directions. In the new matrix study it lowers null error but permits larger readout-row state error; high primitive classification scores do not bound that error. Thus this experiment tests the specified replacement objective, not a scheme that retains the full-space term and adds null weighting.',
      'No compound-outcome tuning or checkpoint selection was performed. A missing shared KD import was caught by a toy loss test and corrected before any permutation intervention fit; the initial script and pre-fit implementation amendment are retained. An initial diagnostic self-class division produced nonfinite margin statistics; the corrected implementation and independent rival-loop verification are preserved. The perturbation curves were unaffected.'])
    path=Path('paper/generated_readout_null.tex');body='\n\n'.join(lines)+'\n'
    body=re.sub(r'(?s)(\\begin\{tabular\}.*?\\end\{tabular\})',lambda m:m.group(0).replace('\n\n','\n'),body)
    path.write_text(body)
    main=Path('paper/main.tex').read_text();main=main.replace(r'\end{abstract}', 'A prospective null-only-loss replacement does not improve correctness-specific composition and degrades permutation performance despite accurate primitives. A larger nonlinear single-step predictor improves old matrix-model composition in a posthoc diagnostic, without matched-wrong or capacity-matched controls.\n'+r'\end{abstract}')
    revision=main.replace(r'\bibliographystyle{plain}',r'\input{generated_readout_null.tex}'+'\n\n'+r'\bibliographystyle{plain}')
    Path('paper/main_readout_null.tex').write_text(revision)
    atomic_json(ROOT/'paper_revision.json',{'created_utc':now(),'original_paper_unchanged':True,'revision':'paper/main_readout_null.tex',
        'generated_section_sha256':sha(path),'revision_sha256':sha('paper/main_readout_null.tex'),
        'tex_compilation':'Source only; no local TeX engine available.'})

if __name__=='__main__':run()
