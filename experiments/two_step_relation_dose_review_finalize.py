"""Freeze a separate combined report after the dependent dose checks finish."""
import json
from pathlib import Path
import time

from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now
from . import two_step_relation_final_report as report

ROOT=Path('results/two_step_relation_dose_review')
DOSE=Path('results/two_step_relation_dose_confirmation')
PRIMARY=Path('results/two_step_relation_coverage_confirmation')
BASELINE=Path('results/two_step_relation_output_only')


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    required=['completion.json','uncertainty.json','factorial_effects.json','paired_dose.json','order_control.json']
    while not (all((DOSE/f).exists() for f in required) and
               (PRIMARY/'order_control.json').exists() and (BASELINE/'results.json').exists() and
               (BASELINE/'statistics_verification.json').exists()):
        state=json.loads((DOSE/'state.json').read_text()) if (DOSE/'state.json').exists() else {}
        if state.get('status') in ['failed','not_started_visible_feasibility_failed']:
            atomic_json(ROOT/'state.json',{'status':'dose_not_complete','dose_state':state,'updated_utc':now()});return
        atomic_json(ROOT/'state.json',{'status':'waiting_matched_dose_and_audits','updated_utc':now(),
                    'completed_dose_fits':len(list((DOSE/'fits').glob('*.json'))),'planned_dose_fits':24})
        time.sleep(10)
    for f in required:assert json.loads((DOSE/f).read_text())['status']=='complete'
    assert json.loads((BASELINE/'results.json').read_text())['status']=='complete'
    assert json.loads((BASELINE/'statistics_verification.json').read_text())['status']=='complete'
    assert json.loads((DOSE/'prefix_budget_verification.json').read_text())['all24_prefixes_preserved']
    assert json.loads((DOSE/'verification.json').read_text())['all24_readouts_preserved']
    report.ROOT=ROOT;report.run()
    results=json.loads((ROOT/'results.json').read_text());assert len(results['completed_studies'])==4
    files=[ROOT/'report.html',ROOT/'results.json',Path('results/two_step_relation_review/completion.json'),
           Path('results/two_step_relation_review/project_test_verification.json'),
           Path('results/two_step_relation_final_project_tests.log')]
    for f in required+['verification.json','statistics_verification.json','prefix_budget_verification.json',
                       'prefix_records.json','dose_confirmation_protocol.json','dose_implementation_amendments.json',
                       'data_verification.json','uncertainty_protocol.json','factorial_effects_protocol.json',
                       'paired_dose_protocol.json','uncertainty_draws.npz']:
        files.append(DOSE/f)
    files.extend(Path(s['root'])/'summary.json' for s in results['completed_studies'])
    files.extend(Path(f) for f in ['results/two_step_relation_dose_pilot/visible_dose_results.json',
                                   'experiments/two_step_relation_dose_review_finalize.py',
                                   'experiments/two_step_relation_final_report.py',
                                   'experiments/two_step_relation_paired_dose.py'])
    files.extend(BASELINE/f for f in ['results.json','statistics_verification.json','output_only_protocol.json'])
    files.extend([PRIMARY/'order_control.json', PRIMARY/'order_control_protocol.json',
                  DOSE/'order_control_protocol.json',Path('experiments/two_step_relation_output_only_verify.py')])
    atomic_json(ROOT/'completion.json',{'status':'complete','completed_utc':now(),'factorial_studies':4,
                'factorial_fits':sum(s['operator_fits'] for s in results['completed_studies']),
                'visible_only_feasibility_fits':4,'dose_is_not_independent_replication':True,
                'output_only_baseline_fits':6,'output_only_baseline_matches_primary10_only':True,
                'primary10_epoch_result_preserved':True,'same_input_paired_dose_comparison_complete':True,
                'artifact_sha256':{str(p):sha(p) for p in files}})
    atomic_json(ROOT/'state.json',{'status':'complete','updated_utc':now()})
    print(json.dumps({'status':'complete','report':str(ROOT/'report.html')}),flush=True)


if __name__=='__main__':run()
