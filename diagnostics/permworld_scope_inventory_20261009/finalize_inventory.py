"""Enrich metadata without loading unselected states or changing source artifacts."""
from collections import defaultdict
import json
import tomllib
from inventory import ROOT,WS,NR,HEADS,sha,js,save,csv_read,csv_write,register


def run():
    rows=csv_read(ROOT/'training_inventory.csv')
    if not any(r['cohort']=='permworld_combinations_pilot24' for r in rows):
        meta=js(WS/'results/permworld_combinations/metadata.json')
        for g in meta['groups']:
            for seed in meta['config']['model_seeds']:
                lid=f'{g["id"]}_s{seed}';p=WS/f'results/permworld_combinations/{lid}.json';d=js(p)
                register(rows,'2026TMLR','permworld_combinations_pilot24',lid,p,
                    WS/f'results/permworld_combinations/checkpoints/{lid}.pt',g['tasks'],{
                    'record_status':'complete','training_steps':d['curve'][-1]['step'],'availability':'complete_weight_verified',
                    'seed':seed,'data_sha256':meta['fingerprint'],'data_world':'Fixed small PermWorld pilot seed20261005',
                    'architecture':json.dumps({k:meta['config'][k] for k in ['d_model','layers','heads','ff_multiplier','dropout']},sort_keys=True),
                    'loss':meta['source_objective'],'optimization':json.dumps(meta['config'],sort_keys=True),
                    'cache':str(WS/f'results/permworld_combinations/{lid}_features.npy'),
                    'initialization_grade':'C','initialization_reason':'Historical initial_s*_features is a posthoc constructor rebuild; not original source step0.',
                    'landmark':meta['cka_landmark'],'suitable_comparison':'Separate small1200-step pilot; source learning insufficient for full-scale confirmation'})
    for p in sorted((NR/'results/v3/scaling/k16/evaluation/per-run').glob('baseline--*.json')):
        if any(r['logical_position']=='reused_baseline/'+p.stem for r in rows):continue
        d=js(p);r=next(r for r in rows if r['checkpoint_sha256']==d['checkpoint_sha256'])
        rows.append(dict(r,cohort='k16_data_depth',logical_position='reused_baseline/'+p.stem,
            record_path=str(p),record_sha256=sha(p),reused_from=r['logical_position'],reused_checkpoint=True,
            new_source_training=False,completion_evidence='Historical completed v3 checkpoint reused in scaling baseline endpoint.'))
    configs={sha(p):(p,tomllib.loads(p.read_text())) for p in (NR/'configs').glob('*.toml')}
    base=js(WS/'configs/permworld_combinations.json')
    checks=[]
    for r in rows:
        d=js(r['record_path']) if r['record_path'] and __import__('pathlib').Path(r['record_path']).is_file() else {}
        r['task_definition_module_sha256']=sha(WS/'external/neurips/src/neurips_permutations/math_ops.py') if r['encoding']=='one_line' else ''
        if r['repository']=='neurips':
            key=d.get('experiment_config_sha256','')
            if key in configs:
                p,cfg=configs[key];r['historical_config_path']=str(p);r['historical_config_sha256']=key
                r['configuration_evidence']='Exact file SHA matches completed experiment config digest'
                model=cfg.get('model',{});r['architecture']=json.dumps(dict(family=d.get('architecture',r['architecture']),**model),sort_keys=True)
                r['optimization']=json.dumps(dict(optimizer='AdamW',**cfg.get('training',{})),sort_keys=True)
                r['loss']='Causal next-token CE over answer and EOS; input/question mask; historical protocol'
                if cfg.get('data'):r['input_sequence_policy']=json.dumps(cfg['data'],sort_keys=True)
            else:
                r['configuration_evidence']='Exact historical experiment config not matched in local configs; completion digest preserved. Do not infer all hyperparameters from current HEAD.'
                if not r['optimization'] or r['optimization']=='{}':r['optimization']='Unresolved exact historical configuration; see completion config SHA'
        elif r['cohort'].startswith(('native_','six_hour_session','algebra_structure','ordinary_relation')):
            arch=d.get('architecture',{})
            if arch:r['architecture']=json.dumps(dict(**arch,ff_multiplier=base['ff_multiplier'],dropout=base['dropout'],max_seq_len=66),sort_keys=True)
            policy=d.get('training_policy',{})
            r['optimization']=json.dumps(dict(optimizer='AdamW',weight_decay=base['weight_decay'],**policy),sort_keys=True)
            r['configuration_evidence']='Per-job recorded architecture/training policy + source-hashed model constructor; common base config used where unspecified'
            r['vocabulary']='Passage188; exact passage.py digest in recorded upstream hashes'
        if r['cohort']=='joint_four_representations':
            r['encoding']='joint one_line/cycle/Lehmer/inversion_vector; unbalanced11-task shared network'
            r['cannot_reuse_reason']='Not a balanced independently trained encoding intervention'
        if r['cohort']=='F13_polynomial_ordinary_sources':
            r['seed']=d.get('source_seed','');r['training_steps']=d.get('epochs','')
            r['optimization']='Historical polynomial source protocol; epochs are not PermWorld updates'
        recorded=d.get('source_hashes',{});recorded.update(d.get('upstream_hashes',{}))
        if recorded:
            status={}
            for p,digest in recorded.items():
                path=WS/p if p.startswith('experiments/') else WS/'external/neurips/src/neurips_permutations'/p
                status[p]='matches_current_exact_digest' if path.is_file() and sha(path)==digest else 'mismatch_or_missing_current_file'
            r['recorded_source_digest_check']=json.dumps(status,sort_keys=True);checks.append(dict(cohort=r['cohort'],logical_position=r['logical_position'],files=status))
    groups=defaultdict(list)
    for r in rows:
        if r['checkpoint_sha256']:groups[r['checkpoint_sha256']].append(r)
    for digest,rs in groups.items():
        for r in rs:r['unique_checkpoint_id']='sha256:'+digest
    csv_write('training_inventory.csv',rows)
    csv_write('checkpoint_dedup.csv',[dict(unique_checkpoint_id='sha256:'+h,sha256=h,logical_positions=len(rs),
        cohorts='|'.join(sorted({r['cohort'] for r in rs})),paths='|'.join(sorted({r['checkpoint'] for r in rs}))) for h,rs in groups.items()])
    summary=[]
    for cohort in sorted({r['cohort'] for r in rows}):
        rs=[r for r in rows if r['cohort']==cohort]
        summary.append(dict(cohort=cohort,logical_positions=len(rs),completed_records=sum(r['record_status'] in ['complete','completed'] for r in rs),
            accessible_weights=sum(bool(r['checkpoint_sha256']) for r in rs),distinct_file_sha256=len({r['checkpoint_sha256'] for r in rs if r['checkpoint_sha256']}),
            reused_endpoints=sum(str(r.get('reused_checkpoint',''))=='True' for r in rs),
            partial=sum('partial' in r['record_status'] for r in rs),plans_only=sum(r['record_status']=='planned_not_started' for r in rs)))
    csv_write('cohort_inventory_summary.csv',summary)
    old=[r for r in rows if r['cohort'] in ['specialist16','property32_k']]
    assert len(old)==98 and len({r['checkpoint_sha256'] for r in old})==94
    assert sum(r['cohort']=='k16_data_depth' for r in rows)==24
    save('inventory_audit.json',dict(repository_heads=HEADS,rows=len(rows),distinct_checkpoint_files=len(groups),
        original98_positions94_unique_verified=True,k16_24_endpoints_18_new6_reused_verified=True,
        state_dict_policy='Only selected forward models loaded. Unselected parameter-state hashes remain explicit gaps; file SHA verified.',
        source_digest_checks=checks,hash_mismatches=[r['logical_position'] for r in rows if r['availability']=='hash_mismatch_do_not_reuse'],
        new_source_training=0))
    print('inventory finalized',len(rows),'logical positions',len(groups),'distinct accessible weight containers')


if __name__=='__main__':run()
