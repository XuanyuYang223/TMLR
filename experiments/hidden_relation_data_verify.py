"""Independently verify source endpoint exclusion and answer collisions."""
import json
from pathlib import Path

import numpy as np

from .algebra_structure_replication import collect_prior_inputs
from .hidden_relation_train import load_plan
from .longrun_engine import atomic_json
from .native_confirmation import setup,permutation_row
from .permworld_combinations import sha
from .permutation_audit import transform
from .representation_algebra import ACTION_NAMES


def run():
    plan,config,root=load_plan();_,functions,tokens,one_line=setup(config);f=functions[plan['source_task']]
    audit=json.loads((root/'dataset_audit.json').read_text());previous=collect_prior_inputs(audit['excluded_datasets'])
    for path,digest in audit['excluded_datasets'].items():assert sha(path)==digest
    for path,digest in json.loads((root/'data_hashes.json').read_text()).items():assert sha(root/path)==digest
    source=dict(np.load(root/'source_data.npz'));training=dict(np.load(root/'training_orbit_audit.npz'));probe=dict(np.load(root/'probe_dataset.npz'))
    visible=set();withheld=set();all_states=set();probe_states=[set() for _ in range(4)]
    for j,(raw,n) in enumerate(zip(training['permutations'],training['lengths'])):
        base=tuple(map(int,raw[0,:n]));assert len({tuple(p[:n]) for p in raw})==8
        for k,action in enumerate(ACTION_NAMES):
            p=transform(base,action);assert p==tuple(map(int,raw[k,:n])) and p not in previous and p not in all_states
            all_states.add(p)
            if k in plan['observed_actions']:
                visible.add(p);idx=plan['observed_actions'].index(k)
                assert permutation_row(source['train_input'][j,idx],int(n))==p and source['train_labels'][j,idx]==f(p)
            else:withheld.add(p)
    assert not visible&withheld
    for orbit,raw,labels,n,k in zip(probe['input'],probe['permutations'],probe['labels'],probe['lengths'],probe['split']):
        base=tuple(map(int,raw[0,:n]));assert len({tuple(p[:n]) for p in raw})==8
        for j,action in enumerate(ACTION_NAMES):
            p=transform(base,action)
            assert p==tuple(map(int,raw[j,:n]))==permutation_row(orbit[j],int(n))
            assert p not in previous and p not in all_states;all_states.add(p);probe_states[int(k)].add(p)
            assert f(p)==labels[j]
            row=[tokens['<BOS>'],tokens['<SIZE>'],int(n)]+[tokens[t] for t in one_line(p)]
            row+=[tokens['<PAD>']]*(orbit.shape[-1]-len(row));np.testing.assert_array_equal(row,orbit[j])
    val=probe['split']==1
    np.testing.assert_array_equal(source['validation_input'],probe['input'][val][:,plan['observed_actions']])
    np.testing.assert_array_equal(source['validation_labels'],probe['labels'][val][:,plan['observed_actions']])
    collision=probe['split']==3;ids=probe['pair_ids'][collision];labels=probe['labels'][collision];lengths=probe['lengths'][collision]
    pairs=0
    for pair in np.unique(ids):
        rows=np.flatnonzero(ids==pair);assert len(rows)==2 and lengths[rows[0]]==lengths[rows[1]]
        np.testing.assert_array_equal(labels[rows[0],[0,1,4]],labels[rows[1],[0,1,4]])
        assert labels[rows[0],5]!=labels[rows[1],5]
        np.testing.assert_array_equal(labels[rows,5],labels[rows,2]);pairs+=1
    result={'status':'passed','excluded_prior_inputs':len(previous),'source_visible_states':len(visible),
        'source_withheld_states':len(withheld),'probe_states_per_split':[len(s) for s in probe_states],
        'collision_pairs_verified':pairs,'any_answer_only_deterministic_pair_both_correct_ceiling':0.,
        'any_answer_only_deterministic_accuracy_ceiling':.5,'hidden_source_endpoint_input_overlap':0,
        'source_archive_contains_only_observed_state_labels':True}
    atomic_json(root/'data_verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':run()
