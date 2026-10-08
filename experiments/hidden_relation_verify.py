"""Independent replay of source sampling and compound prediction equations."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .algebra_structure_verify import check_direct
from .answer_prompt_baseline import answer_features,remove_linear_answer_component
from .hidden_relation_data_verify import run as verify_data
from .hidden_relation_train import load_plan
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_confirmation import setup
from .permworld_combinations import sha
from .representation_algebra import permutation_action_table,word_action


def run():
    plan,config,root=load_plan();verify_data();_,functions,_,_=setup(config)
    protocol=json.loads((root/'protocol.json').read_text())['signature']
    assert sha('configs/algebra_hidden_relations.json')==protocol['plan_sha256']
    for p,digest in protocol['core_sha256'].items():assert sha(p)==digest
    effective=json.loads((root/'effective_config.json').read_text());assert sha(plan['base_config'])==effective['base_config_sha256'] and config==effective['effective_base_config']
    gate=json.loads((root/'claim_gate.json').read_text());assert sha('experiments/hidden_relation_analysis.py')==gate['signature']['analysis_code_sha256']
    ev=json.loads((root/'evaluation_protocol.json').read_text())['signature'];assert sha('experiments/hidden_relation_evaluate.py')==ev['code_sha256']
    for p,digest in ev['core_sha256'].items():assert sha(p)==digest
    source=dict(np.load(root/'source_data.npz'));data=dict(np.load(root/'probe_dataset.npz'))
    buckets={n:np.flatnonzero(source['train_lengths']==n) for n in range(10,31)};sources=0;matrix_values=0;archive_count=0;checkpoints={}
    for seed in plan['source_seeds']:
        model=make_model(config,plan['architecture'],seed,'cpu')
        initial=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest();del model
        signatures=[]
        for condition in plan['conditions']:
            name=f'{condition}_s{seed}';r=json.loads((root/'source'/f'{name}.json').read_text())
            assert r['steps']==20000 and r['source_label_exposures']==1920000 and r['source_input_actions']==[0,1,4]
            assert r['initial_parameter_sha256']==initial
            cp=root/'checkpoints'/f'{name}.pt';assert sha(cp)==r['checkpoint_sha256'];checkpoints[name]=sha(cp)
            rng=np.random.default_rng(seed+2026100662);wrong=np.random.default_rng(seed+2026100663);digest=hashlib.sha256();seen=set()
            for _ in range(20000):
                n=int(rng.integers(10,31));ids=rng.choice(buckets[n],32,replace=False);wrong.integers(1,32)
                digest.update(np.array([n],dtype=np.int64).tobytes()+ids.tobytes());seen.update(map(int,ids))
            assert digest.hexdigest()==r['sample_sha256'] and rng.bit_generator.state==r['core_rng'] and wrong.bit_generator.state==r['wrong_rng']
            assert len(seen)==r['unique_source_anchors_seen'];signatures.append((r['initial_parameter_sha256'],r['sample_sha256']));sources+=1
        assert len(set(signatures))==1
    table=permutation_action_table();letters={'c':1,'i':4}
    for path in sorted((root/'evaluations').glob('*.json')):
        record=json.loads(path.read_text());fp=root/'features'/f'{path.stem}.npz';assert sha(fp)==record['feature_sha256']
        features=dict(np.load(fp));state=torch.load(root/'checkpoints'/f'{path.stem}.pt',weights_only=True,map_location='cpu')
        weights=state['model']['lm_head.weight'][:31].numpy().astype(np.float64)
        for method in record['methods']:
            h=features['source_query_concat' if method['view']=='query' else 'ONE_END'][:,:,-1].astype(np.float64)
            ap=root/'arrays'/f"{path.stem}_{method['view']}_{method['method']}.npz";assert sha(ap)==record['array_sha256'][ap.name];archive_count+=1
            a=dict(np.load(ap));np.testing.assert_array_equal(a['readout_weight'],weights)
            means={int(n):m for n,m in zip(a['mean_lengths'],a['mean_vectors'])}
            width=h.shape[-1];aug={}
            for g in ['c','i']:
                mat=np.eye(width+1);mat[:-1,:-1]=a['rho_'+g];mat[-1,:-1]=a['bias_'+g];aug[g]=mat
                if method['method']=='native_operators':
                    j=['c','i'].index(g)
                    np.testing.assert_array_equal(a['rho_'+g],np.eye(width)+state['operators']['offset'][j].numpy().astype(np.float64))
                    np.testing.assert_array_equal(a['bias_'+g],state['operators']['bias'][j].numpy().astype(np.float64))
            for row in method['results']:
                k=2 if row['split']=='iid' else 3;use=data['split']==k
                center=np.array([means.get(int(n),np.zeros(width)) for n in data['lengths'][use]])
                base=h[use,0];product=np.eye(width+1)
                for g in row['word']:product=product @ aug[g]
                # A single homogeneous product reproduces sequential affine
                # application without accessing any compound hidden inputs.
                predicted=np.column_stack([base-center,np.ones(len(base))]) @ product[:,:-1]+center
                key=row['split']+'_'+row['word']
                np.testing.assert_allclose(predicted,a[key+'_hidden'],rtol=3e-5,atol=3e-5)
                action=word_action(row['word'],table,letters);assert action==row['action']
                target=h[use,action];denom=np.square(target-base).sum()
                if row['displacement_nmse'] is not None:
                    np.testing.assert_allclose(np.square(predicted-target).sum()/denom,row['displacement_nmse'],rtol=3e-5,atol=3e-5);matrix_values+=1
                if method['view']=='query':
                    pred=(predicted @ a['readout_weight'].T+a['readout_bias']).argmax(-1);truth=data['labels'][use,action]
                    np.testing.assert_array_equal(pred,a[key+'_answers'])
                    np.testing.assert_allclose(np.mean(pred==truth),row['answer_accuracy'],rtol=0,atol=1e-12);matrix_values+=1
                    if k==3:
                        pairs=data['pair_ids'][use];both=[np.all(pred[pairs==pair]==truth[pairs==pair]) for pair in np.unique(pairs)]
                        np.testing.assert_allclose(np.mean(both),row['pair_both_correct'],rtol=0,atol=1e-12);matrix_values+=1
    # Recheck oracle controls and linear residuals against their frozen source.
    answer_root=root/'answer_controls';ap=json.loads((answer_root/'protocol.json').read_text())['signature']
    assert sha('experiments/answer_prompt_baseline.py')==ap['code_sha256']
    previous=Path(plan['previous_replication']);old=json.loads(Path('configs/algebra_structure_replication.json').read_text());od=dict(np.load(previous/'probe_dataset.npz'))
    answer_values=0;residuals=0
    for group in old['groups']:
        labels=np.array([[[functions[t](tuple(map(int,p[:n]))) for t in group['tasks']] for p in orbit] for orbit,n in zip(od['permutations'],od['lengths'])])
        for seed in plan['answer_baseline_seeds']:
            name=f"{group['id']}_answer_prompt_s{seed}";record=json.loads((answer_root/'probes'/f'{name}.json').read_text());a=answer_root/'arrays'/f'{name}.npz';assert sha(a)==record['array_sha256']
            h=answer_features(labels,group['tasks'],seed)
            answer_values+=check_direct(h,od['split'],od['lengths'],table,{'c':1,'r':2,'i':4},record['result'],a)
        for seed in old['source_seeds']:
            name=f"{group['id']}_answer_residual_s{seed}";record=json.loads((answer_root/'probes'/f'{name}.json').read_text());fp=previous/'features'/f"{group['id']}_s{seed}_trained.npz";assert sha(fp)==record['source_feature_sha256']
            h=np.load(fp)['source_query_concat'][:,:,-1];residual,audit,w=remove_linear_answer_component(h,labels,od,old['ridge_grid'])
            np.testing.assert_allclose(residual,np.load(answer_root/'arrays'/f'{name}_residual.npy'),rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(audit['heldout_h_reconstruction_nmse'],record['answer_audit']['heldout_h_reconstruction_nmse'],rtol=1e-10,atol=1e-10);residuals+=1
            answer_values+=check_direct(residual,od['split'],od['lengths'],table,{'c':1,'r':2,'i':4},record['result'],answer_root/'arrays'/f'{name}.npz')
    summary=json.loads((root/'summary.json').read_text());allrows=summary['all_query_summaries']
    get=lambda c,m,w:next(r for r in allrows if r['condition']==c and r['method']==m and r['split']=='answer_collisions' and r['word']==w)
    avg=lambda c,m:float(np.mean([get(c,m,w)['answer_accuracy'] for w in ['ci','ici']]));correct=avg('correct_relations','native_operators')
    expected={'both_native_hidden_words_above_answer_ceiling':all(get('correct_relations','native_operators',w)['answer_accuracy']>.5 for w in ['ci','ici']),
        'native_CI_positive_pair_both_correct':get('correct_relations','native_operators','ci')['pair_both_correct']>0,
        'native_correct_beats_shuffled_and_ordinary_identity':correct>max(avg('ordinary','native_operators'),avg('shuffled_relations','native_operators')),
        'stronger_gate_correct_beats_ordinary_posthoc':correct>avg('ordinary','posthoc_correct_generators') and get('correct_relations','native_operators','ci')['source_accuracy']>=.9}
    assert expected==summary['fixed_predictions']
    paused=json.loads(Path('results/native_ablation/state.json').read_text());assert paused['status']=='paused_for_research_focus_change'
    result={'status':'passed','verified_utc':datetime.now(timezone.utc).isoformat(),'new_sources_and_sampler_traces_verified':sources,
        'within_seed_initialization_and_source_exposure_matched':True,'compound_maps_and_answer_archives_hash_checked':archive_count,
        'compound_numeric_values_recomputed':matrix_values,'answer_control_numeric_values_recomputed':answer_values,
        'linear_answer_residuals_independently_reconstructed':residuals,'compound_predictor_uses_base_state_and_generator_maps_only':True,
        'fixed_predictions_recomputed':expected,'source_checkpoint_sha256':checkpoints,'old_LIS_branch_still_paused':True}
    atomic_json(root/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':run()
