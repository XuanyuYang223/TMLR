"""Finish verification, figures, revision, and immutable delivery manifest."""
import json
from pathlib import Path
import time
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now
from .readout_null_finalize import run as finalize,ROOT
from .readout_null_figures import run as figures
from .readout_null_paper import run as paper
from .readout_null_loss_audit import run as loss_audit


def run():
    started=time.monotonic()
    while not (ROOT/'permworld/completion.json').exists():
        log=Path('results/readout_null_permworld_parallel_v2.log').read_text()
        if 'Traceback' in log:raise RuntimeError('PermWorld intervention process failed; tests remain closed until all12 fits')
        if time.monotonic()-started>14400:raise TimeoutError('PermWorld intervention not completed within four hours')
        time.sleep(20)
    finalize('permworld');loss_audit();figures();paper()
    old=json.loads(Path('results/algebra_relation_review/delivery.json').read_text())['artifact_sha256']
    for p,h in old.items():assert sha(p)==h,p
    pp=ROOT/'pytest_tests.log';assert '164 passed' in pp.read_text()
    diagnostics={d:json.loads(Path(d).read_text()) for d in ['results/downstream_harm_diagnostic_v2/independent_verification.json','results/first_state_predictor_diagnostic_v2/independent_verification.json']}
    assert all(x['status']=='complete' for x in diagnostics.values())
    # Registered input archives and test-opening fit records are immutable.
    archives={}
    matrix=json.loads((ROOT/'matrix/data_audit.json').read_text())['dataset_sha256']
    for p,h in matrix.items():assert sha(p)==h,p;archives[p]=h
    perm=json.loads((ROOT/'permworld/data_manifest.json').read_text())['cohorts']
    for c in perm:assert sha(c['path'])==c['sha256'];archives[c['path']]=c['sha256']
    for d in ['matrix','permworld']:
        for p,h in json.loads((ROOT/d/'test_opened.json').read_text())['fit_record_sha256'].items():assert sha(p)==h,p
    snapshots=['configs/readout_null_confirmation.json','configs/readout_null_permworld.json',
        'experiments/readout_null_sources.py','experiments/readout_null_matrix.py','experiments/readout_null_permworld_v2.py',
        'experiments/readout_null_permworld_parallel_v2.py','experiments/readout_null_finalize.py','experiments/readout_null_paper.py',
        'experiments/readout_null_figures.py','experiments/readout_null_deliver.py','experiments/readout_null_loss_audit.py',
        'paper/main_readout_null.tex','paper/generated_readout_null.tex']
    manifest={str(p):sha(p) for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink() and p.name!='delivery.json' and not p.name.endswith('.tmp')}
    manifest.update({p:sha(p) for p in snapshots})
    atomic_json(ROOT/'delivery.json',{'status':'complete','completed_utc':now(),'formal_fits':24,'new_ordinary_sources':6,
        'source_clusters_per_domain':3,'old_delivery_files_preserved':len(old),'pytest':164,
        'diagnostic_replay_checks':sum(x['checks'] for x in diagnostics.values()),'registered_input_sha256':archives,
        'artifact_sha256':manifest,'report':str(ROOT/'report.html'),'manuscript_revision':'paper/main_readout_null.tex',
        'scope':'All12 fits per domain precede test opening. Three new source seeds per domain; PermWorld ordinary corpus reused. Null-only replacement, not full-plus-null augmentation. Nonlinear probe posthoc and not capacity-matched.'})
    print(json.dumps({'status':'complete','formal_fits':24,'new_ordinary_sources':6,'delivery':str(ROOT/'delivery.json')}),flush=True)

if __name__=='__main__':run()
