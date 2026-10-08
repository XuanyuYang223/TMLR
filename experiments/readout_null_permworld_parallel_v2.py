"""Two independent fit processes; unchanged preregistered losses and budgets."""
import json
import multiprocessing as mp
from pathlib import Path
import time
import torch
from . import readout_null_permworld_v2 as study
from .native_confirmation import setup
from .permworld_combinations import sha


def worker(job):
    plan,parent,config,root,sig=study.initialize();_,_,tokens,_=setup(config);torch.set_num_threads(2)
    i,c=job;study.train_one(plan,parent,config,root,sig,i,c,tokens,'cuda')
    return {'replicate':i,'condition':c,'status':'complete'}


def run():
    plan=json.loads(study.CONFIG.read_text());root=Path(plan['output']);root.mkdir(exist_ok=True,parents=True)
    pp=root/'parallel_execution_v2.json';signature={'code_sha256':sha(__file__),'fit_code_sha256':sha(study.__file__),
        'config_sha256':sha(study.CONFIG),'independent_processes':2,'implementation_correction':'Before formal fitting, toy loss test caught missing import of unchanged shared KD helper; v1 retained, v2 explicitly imports it. No model fit/outcome from v1.',
        'scope':'Execution-only amendment before any formal fits: all loss/initialization/batch order/budget/inputs unchanged. Per-fit updates serialized; independent fits may run concurrently.'}
    if pp.exists():assert json.loads(pp.read_text())['signature']==signature
    else:study.atomic_json(pp,{'registered_utc':study.core.now(),'new_compound_outcomes_observed':False,'signature':signature})
    done=Path(plan['parent'])/'completion.json';started=time.monotonic()
    while not done.exists():
        if 'Traceback' in Path('results/readout_null_sources.log').read_text():raise RuntimeError('Ordinary source process failed')
        if time.monotonic()-started>7200:raise TimeoutError('Sources incomplete')
        time.sleep(10)
    plan,parent,config,root,sig=study.initialize();_,_,tokens,_=setup(config)
    study.core.prepare_data(plan,parent,config,root,sig);study.core.cache_supports(plan,parent,config,root,sig)
    jobs=[(i,c) for i in range(3) for c in study.FLAGS]
    with mp.get_context('spawn').Pool(2) as pool:
        for result in pool.imap_unordered(worker,jobs):print(json.dumps(result),flush=True)
    study.evaluate(plan,parent,config,root,sig,tokens)

if __name__=='__main__':run()
