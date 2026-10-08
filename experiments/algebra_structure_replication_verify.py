"""Independent checks of sampling, held-out matrices, and readout separation."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .algebra_noninvertible import coordinate_zero_projection,polynomial_law_audit
from .algebra_structure_replication import collect_prior_inputs,load_plan
from .algebra_structure_verify import check_direct
from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_ablation import sample
from .native_confirmation import setup,permutation_row
from .native_source_subspace import contrast_basis
from .permworld_combinations import sha
from .permutation_audit import transform
from .representation_algebra import ACTION_NAMES,permutation_action_table


def run():
    plan,config,root=load_plan();names,functions,tokens,one_line=setup(config)
    protocol=json.loads((root/'protocol.json').read_text())['signature']
    assert sha('configs/algebra_structure_replication.json')==protocol['config_sha256']
    assert sha('experiments/algebra_structure_replication.py')==protocol['replication_code_sha256']
    for p,digest in protocol['core_sha256'].items():assert sha(p)==digest,p
    assert sha(plan['source_data'])==protocol['source_data_sha256']
    seen=collect_prior_inputs(plan['excluded_datasets']);data=dict(np.load(root/'probe_dataset.npz'))
    assert sha(root/'probe_dataset.npz')==json.loads((root/'dataset_audit.json').read_text())['probe_dataset_sha256']
    probe=set();split_sets=[set(),set(),set()]
    for orbit,raw,n,k in zip(data['input'],data['permutations'],data['lengths'],data['split']):
        base=tuple(map(int,raw[0,:n]));assert len({tuple(p[:n]) for p in raw})==8
        for j,action in enumerate(ACTION_NAMES):
            p=transform(base,action)
            assert p==tuple(map(int,raw[j,:n]))==permutation_row(orbit[j],int(n))
            assert p not in seen and p not in probe;probe.add(p);split_sets[int(k)].add(p)
            prefix=[tokens['<BOS>'],tokens['<SIZE>'],int(n)]+[tokens[t] for t in one_line(p)]
            prefix+=[tokens['<PAD>']]*(orbit.shape[-1]-len(prefix));np.testing.assert_array_equal(prefix,orbit[j])
    with np.load(plan['source_data']) as a:source={k:a[k] for k in a.files}
    buckets={n:np.flatnonzero(source['train_lengths']==n) for n in range(10,31)}
    sources_verified=0;model_hashes={}
    for seed in plan['source_seeds']:
        initial=make_model(config,plan['architecture'],seed,'cpu')
        expected_initial_hash=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in initial.parameters())).hexdigest()
        del initial
        pairs=[]
        for g in plan['groups']:
            name=f"{g['id']}_s{seed}";r=json.loads((root/'source'/f'{name}.json').read_text())
            assert r['status']=='complete' and r['steps']==20000 and r['total_labels']==2560000
            assert r['per_task_exposures']==640000 and r['labels_per_update']==128
            path=root/'source/checkpoints'/f'{name}.pt';assert sha(path)==r['checkpoint_sha256'];model_hashes[name]=sha(path)
            core=np.random.default_rng(seed+20261005);extra=np.random.default_rng(seed+202610061);digest=hashlib.sha256();coverage=np.zeros(len(source['train_lengths']),bool)
            for _ in range(20000):
                n,ids=sample(core,extra,buckets,32,10,30);digest.update(np.array([n],dtype=np.int64).tobytes()+ids.tobytes());coverage[ids]=True
            assert digest.hexdigest()==r['sample_sha256'] and core.bit_generator.state==r['core_state'] and extra.bit_generator.state==r['extra_state']
            assert int(coverage.sum())==r['unique_source_inputs_seen']
            assert r['initial_parameter_sha256']==expected_initial_hash
            pairs.append((r['sample_sha256'],r['initial_parameter_sha256']));sources_verified+=1
        assert len(set(pairs))==1,'Groups do not share initialization and sampled inputs within a seed'
    table=permutation_action_table();checked=0;array_hashes=0;readout_checks=0
    tasks=['left_to_right_maxima','right_to_left_maxima','left_to_right_minima','right_to_left_minima']
    labels=np.array([[[functions[t](tuple(map(int,p[:n]))) for t in tasks] for p in orbit] for orbit,n in zip(data['permutations'],data['lengths'])])
    for action,ids in [(1,[2,3,0,1]),(2,[1,0,3,2]),(4,[3,1,2,0])]:
        np.testing.assert_array_equal(labels[:,table[:,action]],labels[:,:,ids])
    for p in sorted((root/'probes').glob('*.json')):
        if p.name.startswith('teacher_'):continue
        r=json.loads(p.read_text());fp=root/'features'/f'{p.stem}.npz';assert sha(fp)==r['feature_sha256']
        if r['model_status']=='trained':
            src=Path(plan['source_reference']) if r['seed']==plan['pilot_seed'] else root/'source'
            state=torch.load(src/'checkpoints'/f"{r['group']}_s{r['seed']}.pt",weights_only=True,map_location='cpu',mmap=True)['model']
            weights=state['lm_head.weight'][:31].numpy().astype(np.float64)
        else:
            model=make_model(config,plan['architecture'],r['seed'],'cpu');weights=model.lm_head.weight[:31].detach().numpy().astype(np.float64)
        numeric_basis,contrasts=contrast_basis(weights);assert numeric_basis.shape[1]==30
        features=dict(np.load(fp));full=features['source_query_concat'][:,:,-1].astype(np.float64)
        blocks=full.reshape(*full.shape[:-1],4,256);numeric=((blocks @ numeric_basis) @ numeric_basis.T).reshape(full.shape)
        null=full-numeric;null_blocks=null.reshape(blocks.shape)
        np.testing.assert_allclose(null_blocks @ contrasts.T,0,atol=1e-9,rtol=0);readout_checks+=1
        views={'ONE_END':features['ONE_END'][:,:,-1],'source_query_concat':full,
            'source_query_numeric_contrast':numeric,'source_query_numeric_null':null,
            'source_query_numeric_logits':(blocks @ contrasts.T).reshape(*full.shape[:-1],-1)}
        for view,h in views.items():
            for suffix in ['']+(['_full_width'] if view in plan['full_width_sensitivity'] else []):
                name=view+suffix;path=root/'arrays'/f'{p.stem}_{name}.npz'
                assert sha(path)==r['array_sha256'][path.name];array_hashes+=1
                checked+=check_direct(h,data['split'],data['lengths'],table,{'c':1,'r':2,'i':4},r['results'][name],path)
                h=h.astype(np.float64)
                centered=h.copy()
                for n in np.unique(data['lengths']):
                    mean=h[(data['lengths']==n)&(data['split']==0)].reshape(-1,h.shape[-1]).mean(0)
                    centered[data['lengths']==n]-=mean
                with np.load(path) as a:
                    x=centered[data['split']==2].reshape(-1,h.shape[-1]);q=a['basis'].astype(np.float64);z=x @ q;inv=a['map_i'].astype(np.float64)
                    error=float(np.square((z @ inv @ inv-z) @ q.T).sum()/np.square(x).sum())
                    np.testing.assert_allclose(error,r['results'][name]['explicit_checks']['two_inverse_full_space_reconstruction_nmse'],rtol=3e-5,atol=3e-5)
                assert r['results'][name]['explicit_checks']['identity_displacement_nmse']==1
    summary=json.loads((root/'summary.json').read_text())
    rows={r['group']:r for r in summary['primary_new_seed_summaries']};record=rows['novel_records'];control=[rows[g['id']] for g in plan['groups'] if g['id']!='novel_records']
    expected={
        'lower_mean_generator_than_both_controls':all(record['generator_nmse']<r['generator_nmse'] for r in control),
        'lower_mean_composite_than_both_controls':all(record['composite_nmse']<r['composite_nmse'] for r in control),
        'lower_generator_and_composite_than_identity_shuffled_random':record['generator_nmse']<min(1,record['generator_shuffled_nmse'],record['random_generator_nmse']) and record['composite_nmse']<min(1,record['composite_shuffled_nmse'],record['random_composite_nmse']),
        'positive_wrong_order_gap':record['wrong_order_gap']>0}
    assert expected==summary['fixed_predictions']
    projection=coordinate_zero_projection(4,2);np.testing.assert_array_equal(projection @ projection,projection)
    # Reading existing archived hashes also verifies that the paused transfer
    # branch was not resumed by this independent structural training branch.
    paused=json.loads(Path('results/native_ablation/state.json').read_text());assert paused['status']=='paused_for_research_focus_change'
    atomic_json(root/'verification.json',{'status':'passed','verified_utc':datetime.now(timezone.utc).isoformat(),
        'probe_states_exactly_checked':len(probe),'excluded_distinct_inputs':len(seen),
        'split_states':[len(s) for s in split_sets],'new_source_models_verified':sources_verified,
        'source_rng_sequences_replayed':sources_verified,'within_seed_initialization_and_exposure_matched':True,
        'matrix_error_and_law_values_recomputed':checked,'probe_matrix_archives_hash_checked':array_hashes,
        'numeric_null_readout_checks':readout_checks,'identity_baseline':1,
        'exact_record_task_closure_on_new_probe_states':len(probe),
        'fixed_predictions_recomputed':expected,'polynomial_laws':polynomial_law_audit(),
        'old_lis_branch_still_paused':True,'new_source_checkpoint_sha256':model_hashes})
    print(json.dumps({'verification':'passed','source_models':sources_verified,'matrix_values':checked,'predictions':expected},indent=2))


if __name__=='__main__':run()
