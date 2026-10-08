"""Freeze new targets with physical support >=3 before their CPU readouts."""
from datetime import datetime,timezone
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from . import field_symmetry_transfer
from .algebra import composition_size, projective, rank_mod
from .field_forecast import VARIANTS, augment, design, predict, source_statistics
from .field_symmetry import group_sources
from .longrun_engine import atomic_json
from .matched_field import controlled_world
from .permworld_combinations import sha
from .six_hour_report import write_rows


def select_targets(source,old):
    p=source['p'];basis={w:controlled_world(p,4,w)[3] for w in source['world_seeds']}
    excluded={projective(row,p) for g in source['groups'] for row in group_sources(g,p)}|{projective(t,p) for t in old['targets']}
    seen=set();candidates=[]
    for values in product(range(p),repeat=4):
        if not any(values):continue
        key=projective(values,p)
        if key in seen or key in excluded:continue
        seen.add(key);v=np.array(key)
        orders={g:composition_size(group_sources(g,p),v,p) for g in source['groups']}
        physical={str(w):int(np.count_nonzero(v@b%p)) for w,b in basis.items()}
        if min(physical.values())>=3 and min(orders.values())<max(orders.values()):
            candidates.append({'target':v.tolist(),'source_orders':orders,'physical_support':physical})
    rng=np.random.default_rng(202610054)
    selected=[]
    for minimum in (2,3):
        eligible=[r for r in candidates if min(r['source_orders'].values())==minimum]
        selected.extend(eligible[i] for i in rng.choice(len(eligible),4,replace=False))
    assert rank_mod(np.array([r['target'] for r in selected]),p)==4
    return candidates,selected


def future_design(rows,variant,old_config,old_source):
    # New identities have no fitted target-specific effect. Use the mean old
    # target effect, independently of their new display IDs or outcome labels.
    matrix=design([{**r,'target_id':100+r['target_id']} for r in rows],variant,old_config,old_source)
    count=len(old_config['targets']);matrix[:,4:4+count-1]=1/count
    return matrix


def prepare():
    root=Path('results/controlled_target_followup');root.mkdir(exist_ok=True)
    source=json.loads(Path('configs/field_matched_support.json').read_text())
    old=json.loads(Path('configs/field_matched_support_transfer.json').read_text())
    original_path=Path('results/field_forecast/forecasts.json')
    original=json.loads(original_path.read_text())
    candidates,selected=select_targets(source,old)
    config={**old,'output':'results/controlled_target_transfer','targets':[r['target'] for r in selected],
            'analysis_reserve_seconds':600,'maximum_runtime_seconds':1800,
            'runtime_device':'CPU readouts of the same frozen GPU-trained encoders',
            'target_selection':'new scalar directions; physical support >=3 in every basis; four minimum-order-2 and four minimum-order-3 targets; only mathematical selection',
            'followup_wrapper_sha256':sha(__file__)}
    protocol={'registered_utc':datetime.now(timezone.utc).isoformat(),
              'status':'exploratory follow-up after mixed-target negative gains; same encoders, new targets; frozen predictions before these readouts',
              'selection_seed':202610054,'all_candidates':candidates,'selected':selected,
              'source_models_retrained':False,'original_target_outcomes_retained':True,
              'original_predictor_sha256':sha(original_path),'code_sha256':sha(__file__),
              'source_config':source,'transfer_config':config,
              'new_target_fixed_effect':'mean of all eight original fitted target effects',
              'new_basis_effect':'mean of all three original fitted basis effects',
              'limitations':['Target restriction is motivated by observed earlier failure and is not an unconditional replication.',
                             'Physical support is three or four, rather than identical across all targets.',
                             'All source models and formula families are already known; only target identities are new.',
                             'Readout arithmetic is FP32 CPU; both trained and random conditions use the same backend.',
                             'Mathematical task definitions are privileged analytical inputs to prediction; not given to target neural learners.']}
    frozen_path=root/'forecasts.json'
    if frozen_path.exists():
        saved=json.loads(frozen_path.read_text())
        assert saved['protocol']['code_sha256']==sha(__file__)
        assert saved['protocol']['original_predictor_sha256']==sha(original_path)
        assert saved['protocol']['transfer_config']==config
        return config
    assert not Path(config['output']).joinpath('metadata.json').exists()
    atomic_json(root/'protocol.json',protocol)
    atomic_json('configs/controlled_target_transfer.json',config)
    old_config=json.loads(Path('configs/field_symmetry_transfer.json').read_text())
    old_source=json.loads(Path(old_config['source_config']).read_text())
    rows=[]
    for g,w,m in product(source['groups'],source['world_seeds'],source['model_seeds']):
        statistics=source_statistics(source,g,w,m,controlled_world)
        for t,b,mode in product(range(8),config['budgets'],('linear','mlp','categorical_subset')):
            rows.append(augment({'group':g,'world_seed':w,'model_seed':m,'target_id':t,'budget':b,'mode':mode},config,source,statistics))
    forecasts=[]
    for mode,budget,variant in product(('linear','mlp','categorical_subset'),config['budgets'],VARIANTS):
        subset=[r for r in rows if (r['mode'],r['budget'])==(mode,budget)]
        values=predict(original['fitted_models'][f'{mode}_{budget}_{variant}'],future_design(subset,variant,old_config,old_source))
        forecasts.extend({**r,'variant':variant,'predicted_outcome':float(v)} for r,v in zip(subset,values))
    atomic_json(frozen_path,{'protocol':protocol,'frozen_utc':datetime.now(timezone.utc).isoformat(),
                           'independent_behavior_not_started':True,'forecasts':forecasts,
                           'fitted_models':original['fitted_models']})
    write_rows(root/'forecasts.csv',forecasts)
    print(json.dumps({'new_target_forecasts_saved':len(forecasts),'before_new_target_readouts':True}),flush=True)
    return config


def run():
    assert not torch.cuda.is_available(),'Run with CUDA_VISIBLE_DEVICES="" to keep the native GPU queue available'
    prepare()
    field_symmetry_transfer.world=controlled_world
    field_symmetry_transfer.run('configs/controlled_target_transfer.json')


if __name__=='__main__':run()
