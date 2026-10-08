"""Registered wrong-order readout and commutator diagnostic; no fitting."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .longrun_engine import atomic_json
from .overnight_inverse_statistics_verify import bootstrap_interval
from .permworld_combinations import sha
from .two_step_relation_factorial import now


def register(config):
    plan=json.loads(config.read_text());root=Path(plan['output'])
    signature={'code_sha256':sha(__file__),'config_sha256':sha(config),
               'comparison':'Evaluate correct CI and reversed IC predictions against the same heldout CI answers. Also compare the learned hidden commutator CI-IC with true hidden difference h(T_CI x)-h(T_IC x). Report all four conditions, splits and source repeats.',
               'scope':'Secondary diagnostic only; no training, tuning or alteration of fixed primary success criteria. IC tested against its own answers is a separate existing diagnostic; this analysis uses CI truth for both orders.'}
    file=root/'order_control_protocol.json'
    if file.exists():assert json.loads(file.read_text())['signature']==signature
    else:
        atomic_json(file,{'registered_utc':now(),'compound_test_opened_at_registration':(root/'test_opened.json').exists(),'signature':signature})
        (root/'order_control_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return plan,root


def run(config):
    plan,root=register(config)
    while not (root/'completion.json').exists():
        state=json.loads((root/'state.json').read_text()) if (root/'state.json').exists() else {}
        if state.get('status') in ['failed','not_started_visible_feasibility_failed']:raise RuntimeError('Study incomplete')
        time.sleep(10)
    data=np.load(root/'dataset/test/dataset.npz');rows=[]
    for i in range(6):
        for condition in plan['conditions']:
            name=f'n{i}_hidden_{condition}';a=np.load(root/'evaluations'/f'{name}.npz')
            h=np.load(root/'features'/f'{name}_test.npz')['hidden'].astype(float)
            for split,label in [(0,'iid'),(1,'collisions')]:
                use=data['split']==split;truth=data['labels'][use,5]
                ci=a['ci_answers'][use]==truth;ic=a['ic_answers'][use]==truth
                predicted=a['ci_hidden'][use].astype(float)-a['ic_hidden'][use].astype(float)
                true=h[use,5]-h[use,6];den=np.square(true).sum()
                row={'replicate':f'n{i}','source_seed':plan['source_seeds'][i%3],'condition':condition,'split':label,
                     'correct_order_ci_accuracy':float(ci.mean()),'wrong_order_ic_against_ci_accuracy':float(ic.mean()),
                     'correct_minus_wrong_order_accuracy_pp':100*float(ci.mean()-ic.mean()),
                     'hidden_commutator_match_nmse':float(np.square(predicted-true).sum()/den) if den>1e-20 else None,
                     'prediction_orders_disagree_fraction':float((a['ci_answers'][use]!=a['ic_answers'][use]).mean()),
                     'correct_order_pair_both_correct':None,'wrong_order_pair_both_correct':None}
                if split==1:
                    pairs=data['pair_ids'][use]
                    row.update(correct_order_pair_both_correct=float(np.mean([ci[pairs==p].all() for p in np.unique(pairs)])),
                               wrong_order_pair_both_correct=float(np.mean([ic[pairs==p].all() for p in np.unique(pairs)])))
                rows.append(row)
    means=[]
    for condition in plan['conditions']:
        for split in ['iid','collisions']:
            selected=[r for r in rows if (r['condition'],r['split'])==(condition,split)]
            values=[next(r['correct_minus_wrong_order_accuracy_pp'] for r in selected if r['replicate']==f'n{i}') for i in range(6)]
            sources=[float(np.mean([values[j],values[j+3]])) for j in range(3)]
            means.append({'condition':condition,'split':split,
                          'correct_order_ci_accuracy':float(np.mean([r['correct_order_ci_accuracy'] for r in selected])),
                          'wrong_order_ic_against_ci_accuracy':float(np.mean([r['wrong_order_ic_against_ci_accuracy'] for r in selected])),
                          'correct_minus_wrong_order_accuracy_pp':float(np.mean(values)),
                          'three_source_order_effects_pp':sources,'three_source_bootstrap_95_pp':bootstrap_interval(sources),
                          'hidden_commutator_match_nmse':float(np.mean([r['hidden_commutator_match_nmse'] for r in selected]))})
    atomic_json(root/'order_control.json',{'status':'complete','completed_utc':now(),'secondary':True,
                'same_ci_truth_for_both_orders':True,'primary_predictions_unchanged':True,'means':means,'records':rows,
                'limitation':'Order sensitivity alone is not evidence of spontaneous relation discovery or an output-independent mechanism. Three source initializations are reused.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config',type=Path);parser.add_argument('phase',choices=['register','run']);args=parser.parse_args()
    register(args.config) if args.phase=='register' else run(args.config)
