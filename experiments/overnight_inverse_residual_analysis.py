"""All matched residual-intervention doses; fresh-test verification included."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.nn import functional as F

from .inverse_functional_alignment import configure, now
from .inverse_functional_auxiliary import scores
from .inverse_functional_report import table
from .inverse_functional_verify import direct
from .longrun_engine import atomic_json
from .overnight_inverse_analysis import paired
from .overnight_inverse_diagnostic import stratified
from .overnight_inverse_residual import initialize
from .overnight_inverse_verify import data_check, preserve, summaries
from .permworld_combinations import sha
from .specialist_cka_controls import api


def register(plan,root,bp,base,bs,sig):
    file=root/'analysis_protocol.json';signature={'code_sha256':sha(__file__),'training_protocol_sha256':sha(root/'protocol.json'),
        'helpers_sha256':{p:sha(p)for p in ['experiments/inverse_functional_auxiliary.py','experiments/overnight_inverse_diagnostic.py','experiments/overnight_inverse_analysis.py','experiments/overnight_inverse_verify.py']},
        'rules':'Fixed10000 primary; report all complete matched doses, omit unmatched20000 contrasts. Same first128 test rows per length; correct-minus-answer-matched shuffle and answer-mean residual CKA, output-logit CKA. Frozen base-U modes define common nonmodal masks. Paired differences and weighted decomposition; exact6^6/3^3 descriptive bootstrap. Independently check native forward, Gram scores and data exclusion.'}
    if file.exists():assert json.loads(file.read_text())['signature']==signature;return
    assert not(root/'test_opened.json').exists();assert(base/'length_modes.json').exists()
    shutil.copy2(base/'length_modes.json',root/'length_modes.json');assert sha(base/'length_modes.json')==sha(root/'length_modes.json')
    for rep in bp['replicates']:
        directory=root/'dataset'/rep['id']
        if not directory.exists():directory.symlink_to((base/'dataset'/rep['id']).resolve(),target_is_directory=True)
        for part in ['support','unlabeled']:
            for suffix in ['', '_mismatch']:
                source=base/'teacher'/f"{rep['id']}_{part}{suffix}.npz";dest=root/'teacher'/source.name
                if not dest.exists():dest.symlink_to(source.resolve())
    atomic_json(file,{'registered_utc':now(),'main_test_already_opened':(base/'test_opened.json').exists(),'supplement_test_opened':False,'signature':signature,
        'base_length_modes_sha256':sha(base/'length_modes.json'),'input_symlinks_are_read_only_references':True})


def joined(plan,bp,steps):
    return {**bp,**plan,'replicates':bp['replicates'],'lengths':bp['lengths'],'checkpoint_updates':steps,'updates':plan['primary_update']}


def analyze(plan,root,bp,base,bs,sig):
    register(plan,root,bp,base,bs,sig);state=json.loads((root/'evaluation_state.json').read_text());steps=state['complete_matched_endpoints'];assert plan['primary_update']in steps
    data=np.load(root/'dataset/test/dataset.npz');y=data['labels'];n=data['lengths'];modes=json.loads((root/'length_modes.json').read_text())['records'];records=[];source_grades=[]
    for rep in bp['replicates']:
        teacher=np.load(root/'teacher'/f"test_s{rep['source_seed']}.npz");mapping=next(r['modes']for r in modes if r['replicate']==rep['id']and r['prior']=='teacher')
        modal=np.array([mapping[str(int(a))]==int(b)for a,b in zip(n,y)]);source_grades.append({'replicate':rep['id'],**stratified(teacher['logits'].argmax(-1)==y,modal,n)})
        for condition in ['initial']+plan['conditions']:
            for step in ([0]if condition=='initial'else steps):
                file=root/'evaluations'/(f"{rep['id']}_initial.npz"if condition=='initial'else f"{rep['id']}_{condition}_u{step}.npz");out=np.load(file)
                hidden=scores(out['hidden'],teacher['hidden'],n,y,bp['lengths']);outputs=scores(out['logits'],teacher['logits'],n,y,bp['lengths'])
                mean={k:float(np.mean([r[k]for r in hidden]))for k in ['correct_cka','matched_correct_cka','matched_wrong_cka','answer_residual_correct_cka','answer_residual_wrong_cka']}
                mean['matched_contrast']=mean['matched_correct_cka']-mean['matched_wrong_cka'];mean['answer_residual_contrast']=mean['answer_residual_correct_cka']-mean['answer_residual_wrong_cka']
                records.append({'replicate':rep['id'],'source_seed':rep['source_seed'],'condition':condition,'step':step,**stratified(out['logits'].argmax(-1)==y,modal,n),
                    'geometry':mean,'per_length_geometry':hidden,'per_length_output_geometry':outputs,'archive_sha256':sha(file)})
    means=[];contrasts=[]
    for condition in ['initial']+plan['conditions']:
        for step in ([0]if condition=='initial'else steps):
            rows=[r for r in records if r['condition']==condition and r['step']==step]
            means.append({'condition':condition,'step':step,'accuracy':float(np.mean([r['accuracy']for r in rows])),
                'modal_accuracy':float(np.mean([r['modal']['accuracy']for r in rows])),'nonmodal_accuracy':float(np.mean([r['nonmodal']['accuracy']for r in rows])),
                'geometry':{k:float(np.mean([r['geometry'][k]for r in rows]))for k in rows[0]['geometry']}})
    for step in steps:
        for contrast in plan['primary_contrasts']:
            left,right=contrast.split('-');delta={k:[]for k in ['accuracy_pp','modal_accuracy_pp','nonmodal_accuracy_pp','modal_contribution_pp','nonmodal_contribution_pp','cka','answer_residual_contrast']}
            for rep in bp['replicates']:
                a=next(r for r in records if r['replicate']==rep['id']and r['condition']==left and r['step']==step)
                b=next(r for r in records if r['replicate']==rep['id']and r['condition']==right and r['step']==step)
                delta['accuracy_pp'].append(100*(a['accuracy']-b['accuracy']))
                for group in ['modal','nonmodal']:
                    difference=100*(a[group]['accuracy']-b[group]['accuracy']);delta[group+'_accuracy_pp'].append(difference);delta[group+'_contribution_pp'].append(difference*a[group]['count']/len(y))
                assert abs(delta['accuracy_pp'][-1]-delta['modal_contribution_pp'][-1]-delta['nonmodal_contribution_pp'][-1])<1e-10
                delta['cka'].append(a['geometry']['correct_cka']-b['geometry']['correct_cka']);delta['answer_residual_contrast'].append(a['geometry']['answer_residual_contrast']-b['geometry']['answer_residual_contrast'])
            contrasts.append({'contrast':contrast,'step':step,'primary':step==plan['primary_update'],**{k:paired(v,bp['replicates'])for k,v in delta.items()}})
    summary={'created_utc':now(),'primary_update':plan['primary_update'],'complete_matched_endpoints':steps,'target_models':24,'means':means,'records':records,'contrasts':contrasts,'source_accuracy':source_grades,
        'shared_six_support_and_unlabeled_pools_with_base':True,'test_examples':len(y),'unmatched_high_doses_excluded':True,'spontaneous_discovery_tested':False,'output_independent_mechanism_established':False}
    atomic_json(root/'summary.json',summary)
    primary=[r for r in means if r['step']in[0,plan['primary_update']]]
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>同答案内部的几何干预</title><style>body{font-family:system-ui;max-width:1100px;margin:30px auto;padding:20px;line-height:1.7}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}</style><body><h1>同答案内部的几何干预</h1><p>在主实验结果揭晓前登记。所有条件接受相同的正确教师输出蒸馏；额外几何只作用于128个无标签样本。残差几何按教师预测的答案减去组均值，保留组大小至少3的相同行。正确和错配只改变具体输入对应。此操作移除线性答案均值，不能移除全部答案信息。</p>'
    html+=table(['方法','固定更新','整体准确率','非众数准确率','隐藏CKA','答案残差正确−错配'],[[r['condition'],r['step'],f"{100*r['accuracy']:.2f}%",f"{100*r['nonmodal_accuracy']:.2f}%",f"{r['geometry']['correct_cka']:.4f}",f"{r['geometry']['answer_residual_contrast']:.4f}"]for r in primary])
    html+=table(['预先固定对比','准确率差(pp)','三个模型对bootstrap95','非众数差(pp)'],[[r['contrast'],f"{r['accuracy_pp']['mean']:+.3f}",r['accuracy_pp']['three_pair_bootstrap_95'],f"{r['nonmodal_accuracy_pp']['mean']:+.3f}"]for r in contrasts if r['primary']])
    html+='<p>复用主实验六份训练数据和三个预训练模型对，使用全新的5120输入测试集。所有24条件完成同一训练档位后才比较；高档位未配齐则不报告该档对比。相对主实验还改变了批量与学习率轨迹，跨实验比较只能作为描述。</p><p><a href="summary.json">逐次、逐长度及全部配齐档位</a> · <a href="protocol.json">实验登记</a> · <a href="analysis_protocol.json">分析登记</a> · <a href="../overnight_inverse_functional/report.html">主实验</a></p></body></html>'
    (root/'report.html').write_text(html);print(json.dumps({'primary_means':primary,'primary_accuracy_contrasts':[{k:r[k]for k in['contrast','accuracy_pp','nonmodal_accuracy_pp']}for r in contrasts if r['primary']]}),flush=True)


def verify(plan,root,bp,base,bs,sig):
    summary=json.loads((root/'summary.json').read_text());steps=summary['complete_matched_endpoints'];checks=summaries(joined(plan,bp,steps),root,bs)
    device=configure();_,tokens,_,TrainConfig,factory=api(bp);data=dict(np.load(root/'dataset/test/dataset.npz'));ix=np.concatenate([np.flatnonzero(data['lengths']==n)[::128]for n in bp['lengths']]);errors=[];replayed=0
    for rep in bp['replicates']:
        source=next(s for s in bs['targets']if s['seed']==rep['target_pretrain_seed']);support=dict(np.load(base/'dataset'/rep['id']/'support/dataset.npz'));val=np.flatnonzero(support['split']==1)
        for condition in plan['conditions']:
            for step in steps:
                name=rep['id']+'_'+condition;cp=torch.load(source['checkpoint'],map_location='cpu',weights_only=True);model=factory(TrainConfig.from_value(cp['config']));del cp
                cp=torch.load(root/'checkpoints'/f'{name}_u{step}.pt',map_location='cpu',weights_only=True);model.load_state_dict(cp['model']);grade=cp['grade'];del cp;model.to(device)
                saved=np.load(root/'evaluations'/f'{name}_u{step}.npz');out=direct(model,data,bp['target_task'],tokens,0,ix)
                for branch in ['hidden','logits']:
                    np.testing.assert_allclose(out[branch],saved[branch][ix],atol=3e-4,rtol=3e-4);errors.append(float(np.abs(out[branch]-saved[branch][ix]).max()))
                out=direct(model,support,bp['target_task'],tokens,0,val);assert float((out['logits'].argmax(-1)==support['labels'][val]).mean())==grade['accuracy']
                ce=float(F.cross_entropy(torch.tensor(out['logits']),torch.tensor(support['labels'][val])));np.testing.assert_allclose(ce,grade['cross_entropy'],atol=3e-4,rtol=3e-4);replayed+=1;del model
    assert sha(base/'length_modes.json')==sha(root/'length_modes.json')
    atomic_json(root/'verification.json',{'completed_utc':now(),**checks,**preserve(bp,base,bs),'native_forward_and_validation_endpoints_replayed':replayed,'maximum_cached_feature_difference':max(errors)})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','data_verify','analyze','verify']);args=parser.parse_args();values=initialize();plan,root,bp,base,bs,sig=values
    if args.phase=='register':register(*values)
    elif args.phase=='data_verify':register(*values);atomic_json(root/'data_verification.json',{'completed_utc':now(),**data_check(bp,root,sig)})
    elif args.phase=='analyze':analyze(*values)
    else:verify(*values)
