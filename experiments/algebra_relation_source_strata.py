"""Known-generator strata and simple source-label priors; no model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now


def run(domain):
    root=Path('results/algebra_relation_v3')/domain
    summary=json.loads((root/'summary.json').read_text())
    sources=[]
    for source in range(3):
        grades=[json.loads((root/'fits'/f'n{i}_both_correct.json').read_text())['curve'][-1] for i in [source,source+3]]
        native=np.mean([g['native'] for g in grades],axis=0)
        generators=np.mean([g['generators'] for g in grades],axis=0)
        passed=bool(min(native)>=.95 and min(generators)>=.9)
        sources.append({'source_index':source,'known_native':native.tolist(),'known_generators':generators.tolist(),
                        'passed_known_gate':passed,'collision_means':{metric:float(np.mean([r[metric] for r in summary['records']
                           if r['replicate']%3==source and r['condition']=='both_correct' and r['split']=='collisions' and r['word']=='ab']))
                           for metric in ['accuracy','pair_both_correct','prediction_nmse','true_intermediate_accuracy']}})
    passed=[r['source_index'] for r in sources if r['passed_known_gate']]
    selected=[r for r in summary['records'] if r['replicate']%3 in passed and r['split']=='collisions' and r['word']=='ab']
    means=[]
    for c in ['both_correct','a_correct_b_wrong','a_wrong_b_correct','both_wrong','no_geometry','output_space']:
        rows=[r for r in selected if r['condition']==c]
        if rows:means.append({'condition':c,**{metric:float(np.mean([r[metric] for r in rows])) for metric in
                       ['accuracy','pair_both_correct','prediction_nmse','true_intermediate_accuracy']}})
    source_data=dict(np.load(root/'dataset/source.npz'));test=dict(np.load(root/'dataset/test.npz'))
    # Native source labels only; do not compute unseen compound answers on source inputs.
    counts=np.bincount(source_data['labels'][:,0],minlength=13);prior=int(counts.argmax())
    baselines=[{'split':name,'source_native_label_mode':prior,'source_mode_accuracy':float((test['labels'][test['split']==sid,3]==prior).mean()),
                'source_mode_pair_both_correct':0. if sid==1 else None,'uniform_guess_expected_accuracy':1/13,
                'independent_uniform_guess_expected_pair_double_correct':1/169 if sid==1 else None} for sid,name in [(0,'iid'),(1,'collisions')]]
    atomic_json(root.parent/f'{domain}_verification/source_strata.json',{'status':'complete','completed_utc':now(),
        'sources':sources,'passed_source_indices':passed,'known_gate_conditioned_means':means,'source_label_prior_baselines':baselines,
        'code_sha256':sha(__file__),
        'scope':'Supplement defined after matrix outcomes and before polynomial compound outcomes; thresholds95% native/90% generators were already fixed. Source-level generator means over two fitting pools determine strata. No source removed from primary means, no checkpoint or hyperparameter selection. Source-mode baseline uses only native source labels; no compound source labels.'})
    print(json.dumps({'status':'complete','domain':domain,'passed_sources':passed,'source_mode':prior}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('domain',choices=['matrix','polynomial']);args=parser.parse_args()
    run(args.domain)
