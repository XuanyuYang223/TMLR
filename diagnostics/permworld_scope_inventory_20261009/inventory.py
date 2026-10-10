"""Read-only source inventory. Never imports or invokes a training entry point."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parent
WS=ROOT.parents[1]
NR=Path('/home/yangx/neurips')
sys.path[:0]=[str(WS),str(WS/'external/neurips/src')]
from neurips_permutations.math_ops import PROPERTY_FUNCTIONS
from experiments.algebra import scenarios


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def js(path):return json.loads(Path(path).read_text())


def save(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def csv_write(name,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def csv_read(path):return list(csv.DictReader(open(path)))


HEADS={r:subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip()
       for r,p in [('neurips',NR),('2026TMLR',WS)]}
FILE_HASH={}


def file_hash(path):
    path=Path(path).resolve()
    if not path.is_file():return ''
    if path not in FILE_HASH:FILE_HASH[path]=sha(path)
    return FILE_HASH[path]


def register(rows,repository,cohort,logical,record,checkpoint,tasks=None,extra=None):
    d=js(record) if record and Path(record).exists() else {}
    job=d.get('job',d.get('signature',{}).get('job',{}))
    tasks=tasks if tasks is not None else job.get('tasks',d.get('tasks',[]))
    cp=Path(checkpoint) if checkpoint else None
    expected=d.get('checkpoint_sha256','')
    actual=file_hash(cp) if cp else ''
    state=d.get('status','planned_not_started' if not d else 'record_only')
    complete=state in ['complete','completed']
    step=d.get('step',d.get('global_step',d.get('steps','')))
    if not step and d.get('curve'):step=d['curve'][-1].get('step','')
    if not actual:available='missing_checkpoint' if complete else 'partial_or_plan_without_checkpoint'
    elif expected and actual!=expected:available='hash_mismatch_do_not_reuse'
    elif complete:available='complete_weight_verified' if expected else 'complete_weight_sha_computed_no_original_digest'
    else:available='partial_weight_do_not_resume' if 'partial' in state else 'weight_present_status_needs_context'
    row=dict(repository=repository,repository_commit=HEADS[repository],cohort=cohort,logical_position=logical,
        record_path=str(Path(record).resolve()) if record else '',record_sha256=file_hash(record) if record else '',
        tasks='|'.join(map(str,tasks)),task_definitions='|'.join(f'{t}:PROPERTY_FUNCTIONS[{t}]' if t in PROPERTY_FUNCTIONS else str(t) for t in tasks),
        single_multi='single' if len(tasks)==1 else 'multi',seed=job.get('seed',d.get('seed','')),
        data_world='',data_sha256=d.get('data_sha256',d.get('signature',{}).get('data_sha256',d.get('training_manifest_sha256',''))),
        encoding='one_line',vocabulary='Passage188; source module hash retained where supplied',
        architecture=json.dumps(d.get('architecture',{}),sort_keys=True),loss='',optimization=json.dumps(d.get('training_policy',{}),sort_keys=True),
        exposure=json.dumps(d.get('exposures_per_task',d.get('per_task_exposures',d.get('task_accounting',{}))),sort_keys=True),
        record_status=state,training_steps=step,checkpoint=str(cp.resolve()) if cp else '',checkpoint_exists=bool(actual),
        checkpoint_sha256=actual,expected_checkpoint_sha256=expected,checkpoint_hash_matches=(actual==expected) if expected and actual else '',
        parameter_sha256='',parameter_hash_status='Not loaded solely for inventory; populated for selected forward models.',
        initial_parameter_hash_recorded=d.get('initial_parameter_sha256',''),initialization_grade='D',
        initialization_reason='Original step0 not yet located or reconstructed; no identity inference from seed or shape.',
        original_initial_weight_identity_verified=False,cache='',landmark='',availability=available,
        suitable_comparison='cohort-only exploratory' if complete and actual else 'not a completed source comparison',
        cannot_reuse_reason='' if complete and actual else available,
        training_source_hashes=json.dumps(d.get('source_hashes',d.get('upstream_hashes',{})),sort_keys=True),
        source_code_version_status='Recorded hash where available; repository HEAD alone does not identify historical training version.')
    if extra:row.update(extra)
    rows.append(row)


def build():
    rows=[]
    old=csv_read(WS/'diagnostics/permworld_cka_20261009/model_inventory.csv')
    for r in old:
        d=js(r['marker']);cohort='specialist16' if r['family']=='specialist' else 'property32_k'
        register(rows,'neurips',cohort,r['model_id'],r['marker'],r['checkpoint'],d['tasks'],{
            'initialization_grade':'C','initialization_reason':'Prior audited constructor/RNG reconstruction; original initial weight identity explicitly unverified.',
            'cache':r['activation_cache'] if cohort=='property32_k' else str(WS/f'results/specialist_cka_controls/features/trained_{r["task"]}_s{r["seed"]}.npz'),
            'landmark':'final-layer input end before task','initial_rebuilt_parameter_hash':r['initial_weight_hash_rebuilt'],
            'k':r.get('k',''),'pool':r.get('pool',''),'replicate':r.get('replicate',''),
            'loss':'answer+EOS causal CE','data_world':'original 16m Property32 corpus; source seeds share data',
            'architecture':'Transformer256/4layers/8heads; prior checkpoint/config audit',
            'parameter_hash_status':'Prior checkpoint-only inventory did not save trained-state hash; selected forwards populate it.'})
    for runroot,cohort in [('property-task-geometry/bundles','property_bundles4'),('henry-permutation','v2_nested'),
            ('henry-permutation-v3','v3_nested_category'),('permutation-representation-transfer-v1','joint_four_representations'),
            ('permutation-scaling-v3','k16_data_depth'),('property32-relation-controlled','relation_controlled_paused_pilot')]:
        for p in sorted((NR/'runs'/runroot).rglob('completed.json')):
            d=js(p);cp=NR/d.get('checkpoint',str(p.parent/'checkpoint.pt'))
            register(rows,'neurips',cohort,str(p.parent.relative_to(NR/'runs')),p,cp,extra={
                'data_world':d.get('training_manifest_sha256','not in completion record'),
                'architecture':d.get('architecture',''),
                'suitable_comparison':'historical protocol, separate exploratory cohort; not homogeneous confirmation'})
    # Six scaling baselines are logical endpoints reusing v3 weights, not new training.
    for p in sorted((NR/'results/v3/scaling/k16/evaluation/per-run').glob('baseline--*.json')):
        d=js(p);matches=[r for r in rows if r['checkpoint_sha256']==d['checkpoint_sha256']]
        assert matches, ('missing local scaling baseline',str(p))
        source=matches[0]
        alias=dict(source,cohort='k16_data_depth',logical_position='reused_baseline/'+p.stem,
            record_path=str(p),record_sha256=file_hash(p),reused_from=source['logical_position'],
            reused_checkpoint=True,new_source_training=False,
            completion_evidence='Completed historical source checkpoint plus scaling evaluation manifest; endpoint reuses weights.')
        rows.append(alias)
    # Planned relation-controlled positions remain plans, not completed models.
    for f in (NR/'configs').glob('*relation*controlled*.toml'):
        save('relation_controlled_config_snapshot.json',{'path':str(f),'sha256':sha(f),
            'note':'72 logical /60 distinct are design counts, not completion counts. Actual local completed records listed separately.'})
    meta=js(WS/'results/permworld_combinations/metadata.json')
    for g in meta['groups']:
        for seed in meta['config']['model_seeds']:
            lid=f'{g["id"]}_s{seed}';p=WS/f'results/permworld_combinations/{lid}.json'
            d=js(p)
            register(rows,'2026TMLR','permworld_combinations_pilot24',lid,p,
                WS/f'results/permworld_combinations/checkpoints/{lid}.pt',g['tasks'],{
                'record_status':'complete','training_steps':d['curve'][-1]['step'],
                'availability':'complete_weight_verified','seed':seed,'data_sha256':meta['fingerprint'],
                'data_world':'Fixed small PermWorld pilot data seed20261005',
                'architecture':json.dumps({k:meta['config'][k] for k in ['d_model','layers','heads','ff_multiplier','dropout']},sort_keys=True),
                'loss':meta['source_objective'],'optimization':json.dumps(meta['config'],sort_keys=True),
                'cache':str(WS/f'results/permworld_combinations/{lid}_features.npy'),
                'initialization_grade':'C','initialization_reason':'Historical posthoc initial_s*_features is reconstructed, not original source step0.',
                'landmark':meta['cka_landmark'],'suitable_comparison':'Small1200-step pilot, separate; source learning insufficient for full-scale confirmation',
                'completion_evidence':'Saved source curve reaches configured1200 steps and final source result persisted.'})
    for cohort,folder in [('six_hour_session','results/six_hour_session'),('native_confirmation','results/native_confirmation')]:
        for phase in ['single','multi','calibration']:
            for p in sorted((WS/folder/phase).glob('*.json')):
                d=js(p)
                if 'job' not in d:continue
                cp=p.parent/'checkpoints'/f'{p.stem}.pt'
                register(rows,'2026TMLR',cohort+'_'+phase,p.stem,p,cp,extra={
                    'data_world':'one fixed native corpus '+str(d.get('data_sha256','')),
                    'loss':'answer-only CE; single mean, multi summed per-task means',
                    'architecture':json.dumps(d.get('architecture',{}),sort_keys=True),
                    'initialization_grade':'D','initialization_reason':'Reconstruction candidate from documented historical constructor/RNG path, not executed in inventory. Selected forwards will upgrade to C; original step0 not saved.',
                    'landmark':'ONE_END primary; query separate',
                    'cache':str(WS/folder/'initial') if cohort=='six_hour_session' else ''})
    for folder,cohort in [('results/native_ablation','native_ablation_paused'),
                          ('results/algebra_structure_replication','algebra_structure_replication9'),
                          ('results/ordinary_relation_seed_confirmation','ordinary_relation_seed_confirmation')]:
        for p in sorted((WS/folder/'source').glob('*.json')):
            d=js(p)
            if 'job' not in d:continue
            register(rows,'2026TMLR',cohort,p.stem,p,p.parent/'checkpoints'/f'{p.stem}.pt',extra={
                'loss':'ordinary answer-only multi/single source; no geometric supervision',
                'data_world':d.get('signature',{}).get('data_sha256',d.get('data_sha256','')),
                'cache':str(WS/folder/'features'),'landmark':'ONE_END and query assays must be separated'})
        jobs=WS/folder/'jobs.json'
        if jobs.exists():
            actual={r['logical_position'] for r in rows if r['cohort']==cohort}
            for j in js(jobs):
                lid=f'{j.get("id","")}_b{j.get("batch",96)}_s{j.get("seed","")}'
                if not any(str(j.get('id')) in x and f's{j.get("seed")}' in x for x in actual):
                    register(rows,'2026TMLR',cohort,lid,None,None,j.get('tasks',[]),{'seed':j.get('seed',''),'record_status':'planned_not_started'})
    pause=WS/'results/native_ablation/pause_record.json'
    if pause.exists():
        rec=js(pause);job=rec['last_job'];lid=f'{job["id"]}_s{job["seed"]}'
        rows=[r for r in rows if not (r['cohort']=='native_ablation_paused' and r['record_status']=='planned_not_started'
            and r['seed']==job['seed'] and r['tasks']=='|'.join(job['tasks']))]
        register(rows,'2026TMLR','native_ablation_paused',lid,pause,
            WS/f'results/native_ablation/source/checkpoints/{lid}.pt',job['tasks'],{
                'seed':job['seed'],'training_steps':rec['saved_last_job_step'],'record_status':'partial_paused',
                'availability':'partial_weight_do_not_resume','suitable_comparison':'partial, exclude from complete comparison',
                'cannot_reuse_reason':'3000/20000 steps; explicitly paused, no automatic continuation'})
    for rootname,cohort in [('results/pilot','F5_rank3_pilot24'),('results/replication','F5_rank3_replication48'),
            ('results/field_symmetry','F5_rank4_symmetry45'),('results/field_matched_support','F5_matched_support27')]:
        root=WS/rootname;meta=js(root/'metadata.json');cfg=meta['config']
        groupdefs={g['id']:g['sources'].tolist() for g in scenarios()}
        if 'audit' in meta:groupdefs={g['group']:g['sources'] for g in meta['audit']}
        for p in sorted(root.glob('*.json')):
            d=js(p)
            rank3='source_curve' in d and 'run_id' in d
            if ('checkpoint_sha256' not in d and not rank3) or p.stem=='metadata':continue
            parts=p.stem.split('_');group=parts[0]
            ws=int(next(x[1:] for x in parts if x.startswith('w')))
            ms=int(next(x[1:] for x in parts if x.startswith('m')))
            initial=root/f'{p.stem}_step0_features.npy'
            vectors=groupdefs.get(group,[])
            extra={}
            if rank3:extra=dict(record_status='complete',training_steps=d['source_curve'][-1]['step'],
                availability='complete_weight_sha_computed_no_original_digest',completion_evidence='Source curve reaches configured1200 and final run result persisted; no explicit status/digest in historical JSON.',
                suitable_comparison='Separate rank3 cohort, weights accessible; not pooled with rank4')
            register(rows,'2026TMLR',cohort,p.stem,p,root/'checkpoints'/f'{p.stem}.pt',
                [f'linear_F5:{v}' for v in vectors],{
                'seed':ms,'data_world':f'F5^4 full625; input basis seed{ws}; not unseen-input test',
                'data_sha256':meta['fingerprint'],'encoding':'categorical one-hot four physical coordinates',
                'vocabulary':'F5 each coordinate 5 categories;20 inputs',
                'architecture':f'MLP20→{cfg["hidden"]}→{cfg["features"]},GELU,LayerNorm,4x5 linear readout',
                'loss':'four task CE; all625 inputs every update','optimization':json.dumps(cfg,sort_keys=True),
                'exposure':f'all625/step × {cfg.get("steps",cfg.get("pretrain_steps"))}',
                'initialization_grade':'B' if initial.exists() else 'D',
                'initialization_reason':'Actual pre-update encoder activation snapshot in source training loop, restricted to all625 saved inputs/encoder landmark; full step0 weights not saved.' if initial.exists() else 'No original step0 snapshot identified.',
                'cache':str(initial) if initial.exists() else '', 'landmark':'encoder normalized hidden state; all source inputs exposed',
                'world_seed':ws,'group':group,'cache_sha256':file_hash(initial) if initial.exists() else '',
                'source_code_version_status':'metadata carries source/model/algebra exact hashes; original provenance in metadata.json',**extra})
    # Keep the F13 source fits and F17 functional work as separate, non-natural cohorts.
    for p in sorted((WS/'results/algebra_relation_v3/polynomial/sources').glob('*.json')):
        register(rows,'2026TMLR','F13_polynomial_ordinary_sources',p.stem,p,p.with_suffix('.pt'),
                 ['f(0)','f(1)','derivative_f(0)'],{'encoding':'F13 coefficients','suitable_comparison':'Different mathematical source/functional protocol; not pooled into natural geometry'})
    for folder,cohort in [('results/final_mechanism_confirmation','F17_functional_original5'),
                          ('results/reviewer_fresh_confirmation','F17_functional_fresh17_local')]:
        root=WS/folder
        files=sorted((root/'models').glob('*.pt')) if 'original' in cohort else sorted(root.glob('u*/models/*.pt'))
        for p in files:
            register(rows,'2026TMLR',cohort,str(p.relative_to(root)),None,p,['F17 fixed functional source'],{
                'record_status':'existing_functional_bundle','single_multi':'functional_source_plus_predictors',
                'encoding':'F17','availability':'weight_present_functional_not_natural_geometry',
                'suitable_comparison':'F17 inventory only; excluded from new geometry claims and never modified',
                'cannot_reuse_reason':'Functional relation supervision/architecture differs from natural cross-task source comparison'})
    save('preparation_only_audit.json',{
        'encoding_plan':{'units':24,'models':768,'actually_trained':0,'status':'unexecuted_plan'},
        'superseded_engineering_preflight':{'development_data_generated':True,'overfit_models':0,'full_source_models':0,'status':'stopped before source training'},
        'new_source_training_this_task':0,'paused_pilots_resumed':False})
    groups=defaultdict(list)
    for r in rows:
        if r['checkpoint_sha256']:groups[r['checkpoint_sha256']].append(r)
    for digest,rs in groups.items():
        for r in rs:r['unique_checkpoint_id']='sha256:'+digest
    dedup=[dict(unique_checkpoint_id='sha256:'+h,sha256=h,logical_positions=len(rs),
        cohorts='|'.join(sorted({r['cohort'] for r in rs})),paths='|'.join(sorted({r['checkpoint'] for r in rs}))) for h,rs in groups.items()]
    csv_write('training_inventory.csv',rows);csv_write('checkpoint_dedup.csv',dedup)
    summary=[]
    for cohort in sorted({r['cohort'] for r in rows}):
        rs=[r for r in rows if r['cohort']==cohort]
        summary.append({'cohort':cohort,'logical_positions':len(rs),'completed_records':sum(r['record_status'] in ['complete','completed'] for r in rs),
            'accessible_weights':sum(bool(r['checkpoint_sha256']) for r in rs),'distinct_file_sha256':len({r['checkpoint_sha256'] for r in rs if r['checkpoint_sha256']}),
            'partial':sum('partial' in r['record_status'] for r in rs),'plans_only':sum(r['record_status']=='planned_not_started' for r in rs)})
    csv_write('cohort_inventory_summary.csv',summary)
    save('inventory_audit.json',{'repository_heads':HEADS,'rows':len(rows),'distinct_checkpoint_files':len(groups),
        'selected_state_dicts_not_loaded_yet':True,'hash_mismatches':[r['logical_position'] for r in rows if r['availability']=='hash_mismatch_do_not_reuse'],
        'new_source_training':0,'file_hashes_computed':len(FILE_HASH)})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':build()
