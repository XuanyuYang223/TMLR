"""Finish queued assays, verify them, and enforce the user's local10AM limit."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .longrun_engine import atomic_json
from .permworld_combinations import sha

ROOT=Path('results/overnight_inverse_functional')
SUPP=Path('results/overnight_inverse_residual_geometry')


def stage(label,module,phase=None):
    atomic_json(ROOT/'campaign_state.json',{'status':'running','stage':label,'updated_utc':datetime.now(timezone.utc).isoformat()})
    command=[sys.executable,'-u','-m',module]+([phase]if phase else[])
    with open(ROOT/f'campaign_{label}.log','a')as handle:result=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{label} failed with exit{result.returncode}; inspect campaign_{label}.log')


def final_manifest():
    base_protocol=json.loads((ROOT/'protocol.json').read_text())['signature'];previous=base_protocol['prior_artifacts_sha256']
    for p,digest in previous.items():assert sha(p)==digest,p
    for item in base_protocol['sources']+base_protocol['targets']:assert sha(item['checkpoint'])==item['checkpoint_sha256']
    artifacts={}
    dynamic={'campaign_state.json','live_state.json','progress.html'}
    for folder in [ROOT,SUPP]:
        for p in sorted(folder.rglob('*')):
            if p.is_file()and p.name not in dynamic|{'completion.json'} and not p.name.endswith('.tmp'):artifacts[str(p)]=sha(p)
    paths=['experiments/overnight_inverse_functional.py','experiments/overnight_inverse_diagnostic.py','experiments/overnight_inverse_analysis.py',
           'experiments/overnight_inverse_verify.py','experiments/overnight_inverse_residual.py','experiments/overnight_inverse_residual_analysis.py',
           'experiments/overnight_inverse_budget_check.py','experiments/overnight_inverse_campaign.py','experiments/overnight_inverse_status.py',
           'configs/overnight_inverse_functional.json','configs/overnight_inverse_residual.json','tests/test_overnight_inverse_functional.py','tests/test_overnight_inverse_residual.py',
           'results/overnight_inverse_functional_project_tests.log','results/overnight_inverse_functional_residual_tests.log']
    for p in paths:artifacts[p]=sha(p)
    main=json.loads((ROOT/'summary.json').read_text());supp=json.loads((SUPP/'summary.json').read_text())if(SUPP/'summary.json').exists()else None
    atomic_json(ROOT/'completion.json',{'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),
        'requested_local_deadline':'2026-10-07 10:00 America/Los_Angeles','deadline_utc':base_protocol['plan']['deadline_utc'],
        'primary_models':36,'main_updates':40000,'supplement_models':24,'supplement_primary_update':10000,
        'supplement_complete_matched_endpoints':supp['complete_matched_endpoints']if supp else[],
        'previous_artifacts_preserved':len(previous),'frozen_checkpoints_preserved':6,'new_oracle_U_answers_used':False,
        'main_verification':json.loads((ROOT/'verification.json').read_text()),
        'supplement_verification':json.loads((SUPP/'verification.json').read_text())if(SUPP/'verification.json').exists()else None,
        'budget_verification':json.loads((ROOT/'budget_verification.json').read_text()),
        'main_primary_means':[r for r in main['means']if r['step']in[0,40000]],
        'main_primary_contrasts':[r for r in main['contrasts']if r['primary']],
        'supplement_primary_means':[r for r in supp['means']if r['step']in[0,10000]]if supp else[],
        'supplement_primary_contrasts':[r for r in supp['contrasts']if r['primary']]if supp else[],
        'spontaneous_discovery_confirmed':False,'output_independent_mechanism_confirmed':False,'artifact_sha256':artifacts})


def run():
    plan=json.loads(Path('configs/overnight_inverse_functional.json').read_text());deadline=datetime.fromisoformat(plan['deadline_utc']).timestamp()
    atomic_json(ROOT/'campaign_controller.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'code_sha256':sha(__file__),'deadline_utc':plan['deadline_utc'],
        'queue':['wait_base','verify_main','train_residual','evaluate_residual','analyze_residual','verify_residual','verify_budgets','deadline_manifest'],
        'queue_rule':'All supplementary choices registered before main outcomes; no numerical-result-dependent retuning.'})
    try:
        while time.time()<deadline:
            state=json.loads((ROOT/'controller_state.json').read_text())if(ROOT/'controller_state.json').exists()else{}
            if state.get('status')=='failed':raise RuntimeError('Base controller failed: '+str(state))
            if state.get('status')=='base_assay_complete_timed_goal_still_active':break
            time.sleep(10)
        assert(ROOT/'summary.json').exists(),'Primary study did not complete before deadline'
        stage('verify_main','experiments.overnight_inverse_verify','main')
        stage('train_residual','experiments.overnight_inverse_residual','train')
        state=json.loads((SUPP/'training/state.json').read_text())
        if state['primary_complete']:
            stage('evaluate_residual','experiments.overnight_inverse_residual','evaluate')
            stage('analyze_residual','experiments.overnight_inverse_residual_analysis','analyze')
            stage('verify_residual','experiments.overnight_inverse_residual_analysis','verify')
        stage('verify_budgets','experiments.overnight_inverse_budget_check')
        atomic_json(ROOT/'campaign_state.json',{'status':'fits_and_checks_complete_timed_goal_active','updated_utc':datetime.now(timezone.utc).isoformat(),'deadline_utc':plan['deadline_utc']})
        while time.time()<deadline:time.sleep(min(10,max(.1,deadline-time.time())))
        final_manifest();atomic_json(ROOT/'campaign_state.json',{'status':'complete','updated_utc':datetime.now(timezone.utc).isoformat()})
    except BaseException as error:
        atomic_json(ROOT/'campaign_state.json',{'status':'failed','error':repr(error),'updated_utc':datetime.now(timezone.utc).isoformat()});raise


if __name__=='__main__':run()
