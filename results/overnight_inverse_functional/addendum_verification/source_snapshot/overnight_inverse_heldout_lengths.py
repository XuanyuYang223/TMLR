"""Supplemental length-generalization measurement, with no student fitting."""
from datetime import datetime
import gzip
import json
from pathlib import Path
import time

import numpy as np

from .inverse_functional_alignment import configure, extract, now
from .inverse_functional_auxiliary import scores
from .inverse_functional_report import table
from .longrun_engine import atomic_json
from .overnight_inverse_analysis import paired
from .overnight_inverse_diagnostic import stratified
from .overnight_inverse_functional import initialize, load_model
from .overnight_inverse_verify import data_check
from .permworld_combinations import sha
from .permutation_audit import TRANSFORMS, transform
from .specialist_cka_controls import api, input_key, old_inputs, original_input_key

LENGTHS=[12,18,24,28]


def register(plan,root,sig):
    folder=root/'heldout_lengths'
    for sub in ['', 'dataset', 'teacher', 'evaluations']:(folder/sub).mkdir(parents=True,exist_ok=True)
    file=folder/'protocol.json'
    if file.exists():
        registered=json.loads(file.read_text())['signature'];assert registered['code_sha256']==sha(__file__)
        return folder,registered
    assert not(root/'test_opened.json').exists()
    names={'data.npz','probe_dataset.npz','source_data.npz','training_orbit_audit.npz','dataset.npz'};entries=[];digests=set()
    for p in sorted(Path('results').rglob('*.npz')):
        if p.name not in names or folder in p.parents:continue
        digest=sha(p)
        if digest not in digests:entries.append({'path':str(p),'sha256':digest});digests.add(digest)
    signature={'code_path':__file__,'code_sha256':sha(__file__),'base_protocol_sha256':sha(root/'protocol.json'),'excluded_archives':entries,
        'lengths':LENGTHS,'calibration_seed':261072017,'test_seed':261072019,'calibration_per_length':512,'test_per_length':1024,
        'measurement':'All36 fixed40000-update models and native starts; accuracy and128-row per-length hidden/output CKA, answer-matched shuffle and linear answer residual. Same shared test and mask within each source pair. No model/checkpoint/hyperparameter selection.',
        'modes':'Source-specific modes fitted only from512 fresh teacher-argmax calibration examples per length; no true calibration answers computed. Freeze before opening this test.',
        'scope':'Supplementary assessment chosen after inspecting early validation curves, before main test results. These lengths are unseen in the current fine-tuning intervention, but were present in original source/target pretraining. Not zero-shot unseen-length learning or independent source-world replication.'}
    atomic_json(file,{'registered_utc':now(),'main_test_opened':False,'new_length_results_observed':False,'signature':signature});return folder,signature


def create_data(plan,folder,sig):
    if(folder/'dataset/audit.json').exists():return
    functions,tokens,one_line,_,_=api(plan);seen=old_inputs(sig['excluded_archives']);used=set();lookup={};proposals={}
    for world,seed in enumerate([sig['calibration_seed'],sig['test_seed']]):
        rng=np.random.default_rng(seed);need=sig['calibration_per_length']if world==0 else sig['test_per_length']
        for n in LENGTHS:
            rows=[]
            while len(rows)<2*need:
                p=tuple(map(int,rng.permutation(n)+1));keys=[input_key(transform(p,t))for t in TRANSFORMS]
                if len(set(keys))!=8 or any(k in seen or k in used for k in keys):continue
                anchor=(world,n,len(rows));rows.append(p)
                for key in keys:used.add(key);lookup[key]=anchor
            proposals[world,n]=rows
    manifest=Path(plan['repository'])/'data/permutation-properties-16m-v1/manifest.json';parent=json.loads(manifest.read_text());rejected=set();scanned=0
    for shard in parent['shards']:
        file=manifest.parent/shard['filename'];assert sha(file)==shard['sha256']
        with gzip.open(file,'rb')as handle:
            for line in handle:
                anchor=lookup.get(original_input_key(line))
                if anchor is not None:rejected.add(anchor)
                scanned+=1
    records=[]
    for world,name in enumerate(['calibration','test']):
        inputs=[];perms=[];lengths=[];labels=[];need=sig['calibration_per_length']if world==0 else sig['test_per_length']
        for n in LENGTHS:
            rows=[p for j,p in enumerate(proposals[world,n])if(world,n,j)not in rejected];assert len(rows)>=need
            for p in rows[:need]:
                pair=[]
                for q in [p,transform(p,'inverse')]:
                    row=[tokens['<BOS>'],tokens['<SIZE>'],tokens[f'{n:02d}']]+[tokens[v]for v in one_line(q)];pair.append(row+[tokens['<PAD>']]*(64-len(row)))
                inputs.append(pair);perms.append([list(transform(p,t))+[0]*(30-n)for t in TRANSFORMS]);lengths.append(n)
                if world==1:labels.append(functions[plan['target_task']](p))
        arrays={'input':np.asarray(inputs),'permutations':np.asarray(perms),'lengths':np.asarray(lengths)}
        if world==1:arrays['labels']=np.asarray(labels)
        dest=folder/'dataset'/name;dest.mkdir(exist_ok=True);file=dest/'dataset.npz';np.savez_compressed(file,**arrays)
        records.append({'cohort':name,'path':str(file),'sha256':sha(file),'examples':len(lengths),'oracle_labels_present':world==1})
    atomic_json(folder/'dataset/audit.json',{'created_utc':now(),'cohorts':records,'original_inputs_scanned':scanned,'original_manifest_sha256':sha(manifest),'all_eight_orbit_states_excluded':True,'calibration_oracle_answers_computed':False})


def run():
    plan,root,base=initialize();folder,sig=register(plan,root,base);create_data(plan,folder,sig)
    if not(folder/'data_verification.json').exists():atomic_json(folder/'data_verification.json',{'completed_utc':now(),**data_check(plan,folder,sig)})
    deadline=datetime.fromisoformat(plan['deadline_utc']).timestamp()
    while not(root/'evaluation_state.json').exists()and time.time()<deadline:time.sleep(10)
    if not(root/'evaluation_state.json').exists():atomic_json(folder/'state.json',{'status':'not_evaluated_main_incomplete','updated_utc':now()});return
    device=configure();_,tokens,_,_,_=api(plan);cal=dict(np.load(folder/'dataset/calibration/dataset.npz'));assert 'labels'not in cal;mappings={}
    for source in base['sources']:
        model=load_model(source,plan,device);out=extract(model,cal,plan['source_task'],tokens,1);np.savez_compressed(folder/'teacher'/f"cal_s{source['seed']}.npz",**out);prediction=out['logits'].argmax(-1);mapping={}
        for n in LENGTHS:
            values,counts=np.unique(prediction[cal['lengths']==n],return_counts=True);mapping[str(n)]=int(values[counts.argmax()])
        mappings[str(source['seed'])]=mapping;del model
    atomic_json(folder/'length_modes.json',{'frozen_utc':now(),'mappings':mappings,'calibration_oracle_answers_computed':False})
    atomic_json(folder/'test_opened.json',{'opened_utc':now(),'all36_base_fits_complete':True,'fixed_base_update':40000,'modes_sha256':sha(folder/'length_modes.json')})
    data=dict(np.load(folder/'dataset/test/dataset.npz'));n=data['lengths'];y=data['labels'];records=[];sources=[]
    for source in base['sources']:
        model=load_model(source,plan,device);out=extract(model,data,plan['source_task'],tokens,1);np.savez_compressed(folder/'teacher'/f"test_s{source['seed']}.npz",**out);del model
        mode=np.array([mappings[str(source['seed'])][str(int(nn))]for nn in n]);sources.append({'source_seed':source['seed'],**stratified(out['logits'].argmax(-1)==y,mode==y,n)})
    for rep in plan['replicates']:
        target=next(s for s in base['targets']if s['seed']==rep['target_pretrain_seed']);teacher=np.load(folder/'teacher'/f"test_s{rep['source_seed']}.npz")
        mode=np.array([mappings[str(rep['source_seed'])][str(int(nn))]for nn in n]);modal=mode==y
        for condition in ['initial']+plan['conditions']:
            model=load_model(target,plan,device)
            if condition!='initial':
                import torch
                cp=torch.load(root/'checkpoints'/f"{rep['id']}_{condition}_u40000.pt",map_location='cpu',weights_only=True);model.load_state_dict(cp['model']);del cp
            out=extract(model,data,plan['target_task'],tokens,0);del model;file=folder/'evaluations'/f"{rep['id']}_{condition}.npz";np.savez_compressed(file,**out)
            hidden=scores(out['hidden'],teacher['hidden'],n,y,LENGTHS);outputs=scores(out['logits'],teacher['logits'],n,y,LENGTHS)
            geom={k:float(np.mean([r[k]for r in hidden]))for k in ['correct_cka','answer_residual_correct_cka','answer_residual_wrong_cka']};geom['answer_residual_contrast']=geom['answer_residual_correct_cka']-geom['answer_residual_wrong_cka']
            records.append({'replicate':rep['id'],'source_seed':rep['source_seed'],'condition':condition,**stratified(out['logits'].argmax(-1)==y,modal,n),'geometry':geom,'per_length_geometry':hidden,'per_length_output_geometry':outputs,'archive_sha256':sha(file)})
        print({'heldout_lengths_evaluated_repeat':rep['id']},flush=True)
    means=[{'condition':c,'accuracy':float(np.mean([r['accuracy']for r in records if r['condition']==c])),
        'nonmodal_accuracy':float(np.mean([r['nonmodal']['accuracy']for r in records if r['condition']==c])),
        'cka':float(np.mean([r['geometry']['correct_cka']for r in records if r['condition']==c]))}for c in ['initial']+plan['conditions']]
    contrasts=[]
    for contrast in plan['primary_contrasts']:
        a,b=contrast.split('-');differences=[]
        for rep in plan['replicates']:
            left=next(r for r in records if r['replicate']==rep['id']and r['condition']==a);right=next(r for r in records if r['replicate']==rep['id']and r['condition']==b);differences.append(100*(left['accuracy']-right['accuracy']))
        contrasts.append({'contrast':contrast,'accuracy_pp':paired(differences,plan['replicates'])})
    atomic_json(folder/'summary.json',{'completed_utc':now(),'supplementary':True,'lengths':LENGTHS,'unseen_during_current_intervention_only':True,'means':means,'records':records,'contrasts':contrasts,'source_accuracy':sources,'test_examples':len(y)})
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>干预外长度的泛化</title><style>body{font-family:system-ui;max-width:1100px;margin:30px auto;padding:20px;line-height:1.7}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}</style><body><h1>当前微调没有使用的长度</h1><p>长度12、18、24、28未用于本轮监督或对齐微调，但原预训练见过这些长度。属于补充泛化诊断，在主测试揭晓前登记。所有主实验模型固定40000更新，使用新的4096输入；512×4教师预测仅用于提前拟合长度众数，不更新学生，不计算校准集真实答案。</p>'
    html+=table(['方法','整体准确率','非众数准确率','逐长度CKA'],[[r['condition'],f"{100*r['accuracy']:.2f}%",f"{100*r['nonmodal_accuracy']:.2f}%",f"{r['cka']:.4f}"]for r in means])
    html+='<p><a href="summary.json">所有重复和对比</a> · <a href="protocol.json">登记</a> · <a href="data_verification.json">独立数据核验</a></p></body></html>'
    (folder/'report.html').write_text(html)


if __name__=='__main__':run()
