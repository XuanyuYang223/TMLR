"""Prospective secondary main effects, carrying forward exact action theory."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .longrun_engine import atomic_json
from .overnight_inverse_statistics_verify import bootstrap_interval
from .permworld_combinations import sha
from .two_step_relation_factorial import now

ROOT = Path('results/two_step_relation_coverage_confirmation')
THEORY = Path('results/two_step_relation_joint/structural_predictions.json')


def register():
    signature = {'code_sha256':sha(__file__), 'prior_structural_prediction_sha256':sha(THEORY),
                 'metrics':['accuracy','pair_both_correct','cka'], 'words':['ci','ic'],
                 'main_effect_C':'0.5*(both_correct-c_wrong_i_correct+c_correct_i_wrong-both_wrong)',
                 'main_effect_I':'0.5*(both_correct-c_correct_i_wrong+c_wrong_i_correct-both_wrong)',
                 'secondary_prediction':'The exact four-record-count theory registered before the earlier joint test implies an asymmetric role. Complement carries the missing fourth statistic into the visible three-statistic span; inverse preserves that span. Under common correct generator-output supervision, complement geometry may matter more for CI readout than inverse geometry. Both-correct need not outperform complement-only-correct. This secondary analysis does not replace the original fixed primary predictions or interpretation gate.',
                 'method':'Calculate effects in each of six paired fits, then average pairs sharing each of three source initializations. Use the already verified exact multinomial three-source bootstrap for secondary intervals. No source, endpoint or metric is selected by compound outcomes.'}
    file=ROOT/'factorial_effects_protocol.json'
    if file.exists():
        assert json.loads(file.read_text())['signature']==signature
    else:
        atomic_json(file,{'registered_utc':now(),'new_compound_test_opened':(ROOT/'test_opened.json').exists(),'signature':signature})
        (ROOT/'factorial_effects_source_snapshot.py').write_bytes(Path(__file__).read_bytes())


def run():
    register()
    while not (ROOT/'completion.json').exists():
        state=json.loads((ROOT/'state.json').read_text()) if (ROOT/'state.json').exists() else {}
        if state.get('status') in ['failed','not_started_visible_feasibility_failed']:
            raise RuntimeError('Confirmation did not complete')
        time.sleep(10)
    rows=json.loads((ROOT/'evaluation_records.json').read_text())['records'];results=[]
    weights={'C':[.5,.5,-.5,-.5], 'I':[.5,-.5,.5,-.5], 'interaction':[1,-1,-1,1]}
    conditions=['both_correct','c_correct_i_wrong','c_wrong_i_correct','both_wrong']
    for split in ['iid','collisions']:
        for word in ['ci','ic']:
            for metric in ['accuracy','pair_both_correct','cka']:
                if metric=='pair_both_correct' and split=='iid':continue
                unit=100 if metric!='cka' else 1
                for effect,w in weights.items():
                    differences=[]
                    for i in range(6):
                        values={r['condition']:r[metric] for r in rows if
                                (r['replicate'],r['split'],r['word'],r['view'])==(f'n{i}',split,word,'hidden')}
                        differences.append(unit*sum(weight*values[c] for weight,c in zip(w,conditions)))
                    source_means=[float(np.mean([differences[j],differences[j+3]])) for j in range(3)]
                    results.append({'split':split,'word':word,'metric':metric,'unit':'percentage points' if unit==100 else 'CKA units',
                                    'effect':effect,'mean':float(np.mean(differences)),'paired_fit_effects':differences,
                                    'three_source_effects':source_means,'three_source_bootstrap_95':bootstrap_interval(source_means)})
    atomic_json(ROOT/'factorial_effects.json',{'status':'complete','completed_utc':now(),'secondary':True,
                'primary_predictions_unchanged':True,'prior_theory_sha256':sha(THEORY),'results':results,
                'limitation':'Three reused source initializations; these secondary effects do not establish a hidden-computation mechanism or remove all output information.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','run'])
    register() if parser.parse_args().phase=='register' else run()
