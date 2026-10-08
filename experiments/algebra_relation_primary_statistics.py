"""All registered endpoints, factorial effects, and dependence-aware intervals."""
import argparse
import json
from pathlib import Path
import numpy as np
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now
from .overnight_inverse_statistics_verify import bootstrap_interval


def run(domain):
    root=Path('results/algebra_relation_v3')/domain
    summary=json.loads((root/'summary.json').read_text())
    verification=json.loads((root.parent/f'{domain}_verification/verification.json').read_text())
    records=[r for r in summary['records'] if r['word']=='ab']
    all_conditions=['both_correct','a_correct_b_wrong','a_wrong_b_correct','both_wrong','no_geometry','output_space']
    contrasts=[]
    for split in ['iid','collisions']:
        for metric in ['accuracy','pair_both_correct','prediction_nmse']:
            if metric=='pair_both_correct' and split=='iid':continue
            values={c:np.array([next(r[metric] for r in records if
                (r['condition'],r['replicate'],r['split'])==(c,i,split)) for i in range(6)]) for c in all_conditions}
            definitions={'both_correct-'+c:values['both_correct']-values[c] for c in all_conditions[1:]
                if metric!='prediction_nmse' or c!='output_space'}
            definitions['A_main_effect']=(values['both_correct']-values['a_wrong_b_correct']+
                                          values['a_correct_b_wrong']-values['both_wrong'])/2
            definitions['B_main_effect']=(values['both_correct']-values['a_correct_b_wrong']+
                                          values['a_wrong_b_correct']-values['both_wrong'])/2
            definitions['interaction']=values['both_correct']-values['a_correct_b_wrong']-values['a_wrong_b_correct']+values['both_wrong']
            scale=1 if metric=='prediction_nmse' else 100
            for name,diff in definitions.items():
                diff=diff*scale
                sources=[float(np.mean(diff[[i,i+3]])) for i in range(3)]
                contrasts.append({'split':split,'metric':metric,'contrast':name,'mean':float(diff.mean()),
                    'units':'NMSE difference (negative is lower error)' if metric=='prediction_nmse' else 'percentage points',
                    'per_fit':diff.tolist(),'source_means':sources,'three_source_bootstrap_95':bootstrap_interval(sources)})
    laws=[]
    relation_records=verification['auxiliary_learned_relation_checks']
    metrics=[k for k in relation_records[0] if k not in ['replicate','condition','split','normalization']]
    for c in all_conditions:
        for split in ['iid','collisions']:
            rows=[r for r in relation_records if (r['condition'],r['split'])==(c,split)]
            laws.append({'condition':c,'split':split,**{k:float(np.mean([r[k] for r in rows])) for k in metrics}})
    atomic_json(root.parent/f'{domain}_verification/primary_statistics.json',{
        'status':'complete','completed_utc':now(),'all_registered_primary_endpoints_reported':True,
        'contrasts':contrasts,'auxiliary_law_means':laws,
        'collision_block_dependence':verification['collision_block_dependence'],
        'uncertainty':'Three-source bootstrap only, conditional on fixed test inputs; not six independent sources or independent polynomial pairs. Intervals are descriptive with only three source clusters.',
        'output_nmse_not_compared_to_hidden_nmse':True,'primary_summary_sha256':sha(root/'summary.json'),
        'verification_sha256':sha(root.parent/f'{domain}_verification/verification.json'),'code_sha256':sha(__file__)})
    print(json.dumps({'status':'complete','domain':domain,'primary_contrasts':len(contrasts)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('domain',choices=['matrix','polynomial']);args=parser.parse_args()
    run(args.domain)
