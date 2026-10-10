"""Freeze analysis rules and create mapping-fit inputs; no source training."""
import json
from pathlib import Path
import numpy as np
from inventory import ROOT,WS,NR,sha,js,save,csv_read
from relations import relations
from metrics import verify
from experiments.permutation_audit import transform,TRANSFORMS
from neurips_permutations.math_ops import PROPERTY32_TASK_NAMES,PROPERTY_FUNCTIONS
from neurips_permutations.passage import TOKEN_TO_ID,one_line_tokens

ACTIONS=('identity','complement','inverse','reverse')


def run():
    if (ROOT/'metric_protocol.json').exists():
        assert not (ROOT/'metric_results.csv').exists(),'Never overwrite protocol after results.'
        return
    parent=WS/'results/specialist_cka_controls/dataset.npz'
    protocol={
        'status':'frozen_before_new_neural_metric_results','exploratory_not_confirmation':True,
        'new_source_training':0,'relations':relations(),'evaluation_pairs_sha256':sha(ROOT/'evaluation_pairs.json'),
        'primary_landmark':'final_norm at ONE_END before task token; exact logical input end in each model',
        'auxiliary_landmark':'single-task query_final_norm, separate branch; never called source_query_concat',
        'perm_lengths':list(range(10,31)),'length_weights':'1/21, require all21 defined; retain partial per-length values without silently reweighting',
        'test_inputs':{'existing_source':str(parent),'sha256':sha(parent),'split':2,'per_length':128},
        'fit_inputs':{'per_length':128,'existing_split0_per_length':32,'additional_per_length':96,
            'rng_seed':2026100941,'orbit_split':'Reject any8-element orbit overlapping any original specialist fit/validation/test orbit or prior new fit orbit.',
            'scope':'Only analysis map-fitting holdout. No source parameters trained; fit inputs may have appeared in historical source training.'},
        'actions':list(ACTIONS),
        'controls':{'within_length':'Center each length separately.',
            'answer_strata':'For each relation and each split, joint key=(n,left identity answer,right identity/correct/wrong answers); retain groups>=3 and subtract own split group means. Same label-selected anchors across all conditions/branches/metrics.',
            'test_centering_scope':'Test-stratum mean subtraction is an oracle measurement control, not a deployable prediction operation; PCA/scales/orthogonal map are fit-only.'},
        'metrics':{
            'linear_cka':'Centered linear Gram alignment; full feature dimensions; denominator<=1e-20 undefined.',
            'debiased_cka':'U-centered off-diagonal Gram alignment; n<4 undefined; sensitivity only.',
            'knn_overlap':{'k':10,'distance':'Euclidean on split-centered features; global positive scaling leaves ranks unchanged; no per-row norm.',
                'overlap':'Mean intersection size/k of paired sample-ID neighbor sets','ties':'Ascending sample ID then stable distance sort',
                'exclusions':'Self and duplicate underlying objects cannot be neighbors. Less than10 eligible neighbors for any row or constant centered representation => undefined.'},
            'procrustes':{'meaning':'Held-out orthogonal Procrustes in a fixed low-dimensional fitted PCA subspace; does not claim full-hidden-state predictability.',
                'rank':'r=min(8,left feature dimension,right feature dimension), fixed independently of neural outcomes; no adaptive rank reduction.',
                'fit':'Each side PCA on split-centered independent fit orbits; retain r PCs; scale by fit RMS of retained energy; SVD gives r×r orthogonal map.',
                'test':'Apply fit-only bases/scales/map on separately centered test points. NRMSE=sqrt(total squared residual/total squared right projected test state).',
                'direction':'score=-NRMSE (higher better); keep raw error too.',
                'undefined':'fit samples<=r, numerical fit rank<r using relative1e-6 singular-value cutoff, or zero fit/test target energy.'}},
        'branches':['prefix_final_norm','query_final_norm','output_prob','answer_onehot','answer_scalar'],
        'gold_baseline':'Static true-answer geometry only; no invented initial label representation or label learning Delta. Constant strata remain undefined.',
        'decomposition':'B0=correct_initial-wrong_initial;Bt=correct_trained-wrong_trained;dc=correct_trained-correct_initial;dw=wrong_trained-wrong_initial;Delta=Bt-B0=dc-dw. Error metrics use negative raw errors.',
        'initialization':'A actual original weights;B original training step0 activation on documented inputs/landmark;C constructor/RNG rebuild unverified original identity;D missing. A cache reconstructed post hoc remains C, never B.',
        'cohorts':'Original48, native_confirmation single20 and available completed six_hour singles separately; same-seed within-cohort comparisons only; no across-cohort matched-effect claims.',
        'dependency':'Report all sources/relations. Summaries and leave-one-family descriptives do not create independent mathematical relations. Fixed-domain seeds share data worlds.',
        'F5':{'cohort':'field_symmetry rank4,45 existing source weights,3 physical-coordinate bases×3 initializations×5groups',
            'relations':'P vs Mk,k1..4: correct latent transform Gk^-1; wrong Gnext^-1 where next=1+k%4; identity also retained.',
            'fit_test':'All625 source-seen inputs. Analysis fit z1 in{0,1} (250);test z1 in{2,3,4} (375). Every action fixes z1, so fit/test transformation orbits do not overlap.',
            'init':'Actual saved encoder step0 snapshot B for hidden geometry on these625 inputs; any reconstructed initial readout remains C.',
            'answer_control':'Full rank4 source answer tuple identifies input. Common-answer groups are singleton, so exact answer-strata controls are undefined; do not weaken control to manufacture a positive result.',
            'scope':'Second instance; simultaneously changes architecture, input representation and mathematical domain. No unseen-source-input claim, no domain-only attribution.'},
        'k_auxiliary':{'models':50,'scope':'Reuse all fixed original positions;R3/R4 task-subset sensitivities at seed17, not extra seeds.',
            'lengths':list(range(2,31)),'weights':'1/29 with complete-length rule',
            'fit_test':'Partition cached probe anchors by complete mathematical orbit key using fixed SHA parity; never fit/test the same orbit. Degenerate short lengths remain undefined.',
            'comparisons':'Between pools and each trained pool vs reconstructed own initialization; no source decomposition claim from similarity/retention alone.'},
        'stop':'Complete inventory/exploratory metrics/minimal-gap report, then stop; no source training, no paused continuation, no release.'}
    save('metric_protocol.json',protocol);save('metric_validation.json',verify())
    old=dict(np.load(parent));ns=old['lengths'];seen=set()
    for row,n in zip(old['permutations'][:,0],ns):
        p=tuple(map(int,row[:n]));seen.add(min(tuple(transform(p,t)) for t in TRANSFORMS))
    rng=np.random.default_rng(2026100941);rows=[];lengths=[];splits=[];oldindices=[]
    for n in range(10,31):
        fitids=np.flatnonzero((ns==n)&(old['split']==0));testids=np.flatnonzero((ns==n)&(old['split']==2))
        assert len(fitids)==32 and len(testids)==128
        selected=[(tuple(map(int,old['permutations'][i,0,:n])),0,int(i)) for i in fitids]
        while len(selected)<128:
            p=tuple(map(int,rng.permutation(n)+1));orbit=[transform(p,t) for t in TRANSFORMS]
            if len(set(orbit))!=8:continue
            key=min(orbit)
            if key in seen:continue
            seen.add(key);selected.append((p,0,-1))
        selected += [(tuple(map(int,old['permutations'][i,0,:n])),1,int(i)) for i in testids]
        for p,split,index in selected:rows.append(p);lengths.append(n);splits.append(split);oldindices.append(index)
    inputs=np.full((len(rows),4,64),TOKEN_TO_ID['<PAD>'],np.int64)
    perms=np.zeros((len(rows),4,30),np.int64);labels=np.zeros((len(rows),4,32),np.int64)
    for i,p in enumerate(rows):
        for a,action in enumerate(ACTIONS):
            q=transform(p,action);perms[i,a,:len(q)]=q
            tokens=[TOKEN_TO_ID['<BOS>'],TOKEN_TO_ID['<SIZE>'],len(q)]+[TOKEN_TO_ID[t] for t in one_line_tokens(q)]
            inputs[i,a,:len(tokens)]=tokens;labels[i,a]=[PROPERTY_FUNCTIONS[t](q) for t in PROPERTY32_TASK_NAMES]
    path=ROOT/'evaluation_inputs.npz'
    np.savez_compressed(path,input=inputs,permutations=perms,labels=labels,lengths=lengths,split=splits,
        old_indices=oldindices,tasks=np.array(PROPERTY32_TASK_NAMES),actions=np.array(ACTIONS))
    fitset={min(transform(p,t) for t in TRANSFORMS) for p,s in zip(rows,splits) if s==0}
    testset={min(transform(p,t) for t in TRANSFORMS) for p,s in zip(rows,splits) if s==1}
    assert fitset.isdisjoint(testset)
    save('analysis_input_audit.json',{'sha256':sha(path),'anchors':len(rows),'fit_orbits':len(fitset),
        'test_orbits':len(testset),'fit_test_orbit_overlap':0,'old_test_reused_exactly':True,
        'source_training':0,'parent_sha256':sha(parent)})
    save('protocol_freeze.json',{'metric_protocol_sha256':sha(ROOT/'metric_protocol.json'),
        'metrics_code_sha256':sha(ROOT/'metrics.py'),'relation_code_sha256':sha(ROOT/'relations.py'),
        'evaluation_inputs_sha256':sha(path),'model_task_definitions_sha256':sha(NR/'src/neurips_permutations/math_ops.py')})
    print({'protocol':'frozen','analysis_inputs':len(rows),'new_source_training':0},flush=True)


if __name__=='__main__':run()
