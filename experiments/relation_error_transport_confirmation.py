"""Extend the fixed error-transport diagnosis to additional source repetitions."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .relation_composition_diagnostics import diagnose


ROOT=Path('results/relation_error_transport_confirmation')
DEADLINE=datetime(2026,10,6,17,tzinfo=timezone.utc).timestamp()


def now():return datetime.now(timezone.utc).isoformat()


def initialize():
    ROOT.mkdir(exist_ok=True)
    (ROOT/'evaluations').mkdir(exist_ok=True)
    (ROOT/'arrays').mkdir(exist_ok=True)
    signature={'code_sha256':sha(__file__),'diagnostic_helper_sha256':sha('experiments/relation_composition_diagnostics.py'),
        'original_diagnostic_protocol_sha256':sha('results/frozen_relation_followup/diagnostics/protocol.json'),
        'word_endpoints':['ci','ici'],'maps':'native operators unchanged',
        'scope':'Exploratory mechanism replication after the original transported-error diagnosis and six additional native hidden-word outcomes were inspected. No new fitting, supervision or selection. True intermediates are used only for diagnostic local errors and teacher-forcing, never as extra inputs to the base prediction. All source conditions and repetitions are retained. Component energies include cross terms and are not causal fractions.'}
    path=ROOT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['signature']==signature
    else:atomic_json(path,{'registered_utc':now(),'signature':signature,'new_transport_diagnostics':0})


def evaluate(path,scope):
    rec=json.loads(path.read_text());folder=path.parent.parent
    name=rec.get('source',f"{rec['condition']}_s{rec['seed']}")
    dest=ROOT/'evaluations'/(name+'.json')
    if dest.exists():return
    old=folder.name=='algebra_hidden_relations'
    mp=folder/'arrays'/(name+('_query_native_operators.npz' if old else '_native_operators.npz'))
    fp=folder/'features'/(name+'.npz')
    data_folder=folder if folder.name.startswith('world') else Path('results/algebra_hidden_relations')
    data=dict(np.load(data_folder/'probe_dataset.npz'))
    hidden=np.load(fp)['source_query_concat'][:,:,-1].astype(np.float64)
    archive=dict(np.load(mp))
    maps={g:(archive['rho_'+g].astype(np.float64),archive['bias_'+g].astype(np.float64)) for g in ['c','i']}
    means={int(n):v for n,v in zip(archive['mean_lengths'],archive['mean_vectors'])}
    rows,saved=diagnose(hidden,data,maps,means,archive['readout_weight'].astype(np.float64),archive['readout_bias'].astype(np.float64))
    use=data['split']==3;h=hidden[use]
    pred_c=h[:,0]@maps['c'][0]+maps['c'][1]
    error=pred_c-h[:,1]
    transported=error@maps['i'][0]
    energy=float(np.square(error).sum())
    gain=float(np.square(transported).sum()/energy) if energy>1e-20 else None
    _,singular,_=np.linalg.svd(maps['i'][0])
    ap=ROOT/'arrays'/(name+'.npz');np.savez_compressed(ap,**saved)
    atomic_json(dest,{'source':name,'source_condition':rec['condition'],'seed':rec['seed'],'cohort':scope,
        'map_path':str(mp),'feature_path':str(fp),'data_path':str(data_folder/'probe_dataset.npz'),
        'map_sha256':sha(mp),'feature_sha256':sha(fp),'prediction_archive_sha256':sha(ap),
        'C_error_effective_I_gain_squared':gain,'I_spectral_norm_squared':float(singular[0]**2),
        'I_isotropic_direction_mean_gain_squared':float(np.square(singular).mean()),'rows':rows,'completed_utc':now()})
    print(json.dumps({'transport_diagnosed':name,'effective_C_error_gain_squared':gain,
        'collision_composed':[r for r in rows if r['kind']=='composed_from_base' and r['split']=='answer_collisions']}),flush=True)


if __name__=='__main__':
    initialize()
    while time.time()<DEADLINE:
        locations=[(Path('results/algebra_hidden_relations'),'original_exploratory'),(Path('results/relation_seed_extension'),'additional_six_seeds')]
        locations.extend((p,p.name) for p in Path('results/relation_world_confirmation').glob('world*'))
        for folder,scope in locations:
            for path in (folder/'evaluations').glob('*.json'):
                if time.time()<DEADLINE:evaluate(path,scope)
        count=len(list((ROOT/'evaluations').glob('*.json')))
        atomic_json(ROOT/'state.json',{'status':'complete' if count==36 else 'waiting_or_evaluating','sources':count,'updated_utc':now()})
        if count==36:break
        time.sleep(min(45,max(0,DEADLINE-time.time())))
