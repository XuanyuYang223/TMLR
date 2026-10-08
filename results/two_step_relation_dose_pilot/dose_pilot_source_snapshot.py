"""Visible-only extension of the weaker source; never uses compound grades."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from . import two_step_relation_joint as joint
from .longrun_engine import atomic_json
from .native_confirmation import setup
from .permworld_combinations import sha

CONFIG = Path('configs/two_step_relation_dose_pilot.json')
BASE = Path('results/two_step_relation_coverage_confirmation').resolve()


def run():
    joint.CONFIG=CONFIG
    plan,parent,config,root,sig=joint.initialize()
    name='n2_hidden_both_correct';old_record=BASE/'fits'/f'{name}.json';old_resume=BASE/'maps'/f'{name}_resume.pt'
    signature={'code_sha256':sha(__file__),'joint_code_sha256':sha(joint.__file__),
               'old_fit_record_sha256':sha(old_record),'old_resume_sha256':sha(old_resume),'plan':plan,
               'criterion':'Native e/C/I validation all>=0.95 and generator C/I both>=0.90 at fixed cumulative20 epochs.',
               'schedule':'Continue the registered10-epoch fit, retaining optimizer and parameters. Additional epochs11-20 use the second half of the20-epoch cosine schedule in the unchanged joint training implementation. First10 epochs retain their original10-epoch cosine; there is no refitting of the prefix.',
               'selection':'Only existing n2 visible validation. No compound prediction or grade is read. A pass permits registering a matched20-epoch dose extension of all24 conditions; neither dose replaces the fixed10-epoch primary.'}
    file=root/'dose_pilot_protocol.json'
    if file.exists():assert json.loads(file.read_text())['signature']==signature
    else:
        atomic_json(file,{'registered_utc':joint.core.now(),'base_compound_test_opened_at_registration':(BASE/'test_opened.json').exists(),'signature':signature})
        (root/'dose_pilot_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for name2 in ['dataset/n2/support','features/n2_support.npz','features/n2_support.json']:
        dst=root/name2;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():dst.symlink_to(BASE/name2,target_is_directory=name2.endswith('support'))
    dst=root/'maps'/f'{name}_resume.pt'
    if not dst.exists():shutil.copyfile(old_resume,dst)
    atomic_json(root/'state.json',{'status':'running','stage':'visible_only_extension','updated_utc':joint.core.now()})
    _,_,tokens,_=setup(config);device='cuda' if torch.cuda.is_available() else 'cpu'
    joint.train_one(plan,parent,config,root,sig,2,'both_correct',tokens,device)
    record=json.loads((root/'fits'/f'{name}.json').read_text());checkpoint=root/'maps'/f'{name}_e20.pt'
    model=joint.core.source_model(sig['sources'][2],parent,config,device,accelerated=False)
    head=model.lm_head.weight.detach().cpu().clone()
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(state['model'])
    torch.testing.assert_close(head,model.lm_head.weight.detach().cpu(),atol=0,rtol=0)
    ops=joint.AffineOperators(parent['architecture']['d_model']).to(device);ops.load_state_dict(state['operators'])
    a=dict(np.load(root/'dataset/n2/support/dataset.npz'))
    grade=joint.validate(model,ops,{k:a[k][a['split']==1] for k in ['input','lengths','labels']},parent,tokens)
    for key in grade:np.testing.assert_allclose(grade[key],record['curve'][-1][key],atol=1e-8,rtol=1e-8)
    passed=min(grade['native_e_C_I_accuracy'])>=.95 and min(grade['generator_C_I_accuracy'])>=.90
    old=json.loads(old_record.read_text())
    atomic_json(root/'visible_dose_results.json',{'completed_utc':joint.core.now(),'feasible':passed,'source_seed':7103,
                'visible_validation':grade,'cumulative_epochs':20,'additional_epochs':10,
                'prefix_updates':old['updates'],'additional_updates':record['updates']-old['updates'],
                'original_model_validation_replayed':True,'numeric_readout_preserved':True,
                'compound_test_used_for_selection':False,'compound_predictions_generated':False,
                'fit_record_sha256':sha(root/'fits'/f'{name}.json'),'limitation':'One visible-only continuation, not an independent source replication.'})
    atomic_json(root/'state.json',{'status':'complete','updated_utc':joint.core.now()})
    print(json.dumps({'feasible':passed,**grade}),flush=True)


if __name__=='__main__':run()
