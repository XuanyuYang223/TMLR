"""Exploratory exact intermediate error diagnosis in completed additional worlds."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now


def run(domain):
    torch.set_num_threads(1)
    root=Path('results/algebra_relation_v3')/domain
    assert json.loads((root/'state.json').read_text())['status']=='complete'
    d=dict(np.load(root/'dataset/test.npz'));rows=[];max_logits=0.;checks=0
    for i in range(6):
        for c in ['both_correct','a_correct_b_wrong','a_wrong_b_correct','both_wrong','no_geometry']:
            name=f'n{i}_{c}';cp=torch.load(root/'models'/f'{name}.pt',map_location='cpu',weights_only=True)
            h=np.load(root/'evaluations'/f'{name}.npz')['native'].astype(float)
            a,b=[cp['operators'][f'maps.{j}.weight'].numpy().T.astype(float) for j in range(2)]
            ba,bb=[cp['operators'][f'maps.{j}.bias'].numpy().astype(float) for j in range(2)]
            w=cp['model']['readout.weight'].numpy().astype(float);bias=cp['model']['readout.bias'].numpy().astype(float)
            first=h[:,0]@a+ba;delta=first-h[:,1];propagated=delta@b
            true_mid=h[:,1]@b+bb;second=true_mid-h[:,3];composed=first@b+bb;total=composed-h[:,3]
            np.testing.assert_allclose(total,propagated+second,atol=1e-9,rtol=1e-9)
            visible=np.linalg.lstsq(w.T,delta.T,rcond=1e-10)[0].T@w
            null=delta-visible
            repair_null=first-null;repair_visible=first-visible
            np.testing.assert_allclose(repair_null@w.T,first@w.T,atol=1e-9,rtol=1e-9)
            max_logits=max(max_logits,float(np.abs(repair_null@w.T-first@w.T).max()));checks+=2
            candidates={'composed':composed,'true_intermediate':true_mid,'oracle_null_repair':repair_null@b+bb,
                        'oracle_row_repair':repair_visible@b+bb}
            hits={k:(v@w.T+bias).argmax(-1)==d['labels'][:,3] for k,v in candidates.items()}
            saved=np.load(root/'evaluations'/f'{name}.npz')['ab_answers']
            assert np.array_equal(hits['composed'],saved==d['labels'][:,3])
            for sid,split in [(0,'iid'),(1,'collisions')]:
                use=d['split']==sid;denom=float(np.square(h[use,3]-h[use,0]).sum())
                rows.append({'replicate':i,'source_index':i%3,'condition':c,'split':split,
                    **{k+'_accuracy':float(v[use].mean()) for k,v in hits.items()},
                    'first_error_normalized':float(np.square(delta[use]).sum()/denom),
                    'propagated_first_error_normalized':float(np.square(propagated[use]).sum()/denom),
                    'second_at_true_intermediate_error_normalized':float(np.square(second[use]).sum()/denom),
                    'total_error_normalized':float(np.square(total[use]).sum()/denom),
                    'cross_term_normalized':float(2*(propagated[use]*second[use]).sum()/denom),
                    'empirical_first_error_amplification':float(np.square(propagated[use]).sum()/max(np.square(delta[use]).sum(),1e-30)),
                    'first_answer_preserved_by_null_repair':bool(np.array_equal((repair_null[use]@w.T+bias).argmax(-1),(first[use]@w.T+bias).argmax(-1)))})
    means=[]
    numeric=[k for k in rows[0] if k not in ['replicate','source_index','condition','split','first_answer_preserved_by_null_repair']]
    for c in ['both_correct','a_correct_b_wrong','a_wrong_b_correct','both_wrong','no_geometry']:
        for split in ['iid','collisions']:
            selected=[r for r in rows if (r['condition'],r['split'])==(c,split)]
            means.append({'condition':c,'split':split,**{k:float(np.mean([r[k] for r in selected])) for k in numeric}})
    output=root.parent/f'{domain}_verification/error_localization.json'
    atomic_json(output,{'status':'complete','completed_utc':now(),'records':rows,'means':means,
        'exact_hidden_error_and_full_logit_invariance_checks':checks,'maximum_first_step_logit_change':max_logits,
        'all_original_composed_answers_replayed':True,'code_sha256':sha(__file__),
        'scope':'Exploratory extension after matrix aggregate outcomes; identical diagnostic specified before polynomial compound outcomes. Repairs use true intermediate hidden states and are not deployable predictions. Linear readout nullspace is not all-answer removal. Output-space condition excluded from hidden-state diagnosis.'})
    print(json.dumps({'status':'complete','domain':domain,'checks':checks,'correct_collision':next(r for r in means if r['condition']=='both_correct' and r['split']=='collisions')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('domain',choices=['matrix','polynomial']);args=parser.parse_args()
    run(args.domain)
