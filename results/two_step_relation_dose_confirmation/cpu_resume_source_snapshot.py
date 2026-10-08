"""Keep optimizer scalar counters onCPU while resuming the fixed dose plan."""
import json
from pathlib import Path

import torch

from . import two_step_relation_dose_confirmation as dose
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now

ROOT=Path('results/two_step_relation_dose_confirmation')


def run():
    verification=json.loads((ROOT/'resume_compatibility_verification.json').read_text())
    assert verification['parameter_and_moment_checks_bitwise_equal']==160
    original=torch.load
    signature={'code_sha256':sha(__file__),'dose_runner_sha256':sha(dose.__file__),
               'compatibility_verification_sha256':sha(ROOT/'resume_compatibility_verification.json'),
               'change':'For dose maps/*_resume.pt only, load the checkpoint onCPU. Model.load_state_dict copies parameters toCUDA; optimizer.load_state_dict moves moments to their parameter devices while scalar step counters stay onCPU. All other torch.load calls retain their original options.',
               'scope':'Performance correction only. Optimizer settings, tensors, schedules, exact exposures, source data and model endpoints are unchanged. The independent optimizer compatibility case gives bitwise-identical parameters/moments for10 updates.'}
    file=ROOT/'resume_load_protocol.json'
    if file.exists():assert json.loads(file.read_text())['signature']==signature
    else:
        atomic_json(file,{'registered_utc':now(),'compound_test_opened_at_registration':(ROOT/'test_opened.json').exists(),'signature':signature})
        (ROOT/'cpu_resume_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    def load(file,*args,**kwargs):
        if isinstance(file,(str,Path)):
            p=Path(file)
            if p.name.endswith('_resume.pt') and p.parent.resolve()==(ROOT/'maps').resolve():
                kwargs={**kwargs,'map_location':'cpu'}
        return original(file,*args,**kwargs)
    torch.load=load
    dose.run('run')


if __name__=='__main__':run()
