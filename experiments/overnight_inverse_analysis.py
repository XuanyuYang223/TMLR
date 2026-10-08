"""Frozen endpoints, length-prior decomposition and controlled CKA summaries."""
import argparse
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .inverse_functional_alignment import now
from .inverse_functional_auxiliary import scores
from .inverse_functional_report import paired_interval, table
from .longrun_engine import atomic_json
from .overnight_inverse_diagnostic import stratified
from .overnight_inverse_functional import initialize
from .permworld_combinations import sha


def register(plan,root):
    file=root/'analysis_protocol.json'
    signature={'code_sha256':sha(__file__),'protocol_sha256':sha(root/'protocol.json'),
        'geometry_helper_sha256':sha('experiments/inverse_functional_auxiliary.py'),
        'stratum_helper_sha256':sha('experiments/overnight_inverse_diagnostic.py'),
        'rules':{'fixed_primary_update':plan['updates'],'all_checkpoints_reported':plan['checkpoint_updates'],
            'geometry':'First128 test rows per length, equal-weight five lengths; answer strata>=3, cyclic one-row matched shuffle; subtract stratum means as linear answer control. Output-logit CKA also reported; no output-independent conclusion from CKA alone.',
            'modes':'Fit primary length modes only from each16384-U frozen teacher argmax; secondary modes from192 true training answers. Lowest answer breaks ties. Save mappings before final test is opened.',
            'subgroups':'Shared modal/nonmodal masks per repeat from frozen primary modes; report exact overall weighted decomposition for each primary contrast. No test-refitted length prior.',
            'uncertainty':'Exhaustive6^6 paired empirical and3^3 source/target-pair cluster bootstrap; shared test/data world and three related pretrained pairs; descriptive, not population significance.',
            'primary_outcomes':'New-test target full-vocabulary accuracy first; raw CKA and matched/residual specificity secondary. Four predeclared contrasts and all nulls reported. No test selection.',
            'initial':'Replay frozen target pretrained starts and inverse teachers on identical test rows; source accuracy per length and subgroup is reported.'}}
    if file.exists():assert json.loads(file.read_text())['signature']==signature;return
    assert not (root/'test_opened.json').exists()
    atomic_json(file,{'registered_utc':now(),'new_test_outcomes_observed':False,'signature':signature})


def freeze_modes(plan,root):
    file=root/'length_modes.json'
    if file.exists():return
    assert not (root/'test_opened.json').exists();records=[]
    for rep in plan['replicates']:
        support=np.load(root/'dataset'/rep['id']/'support/dataset.npz');pool=np.load(root/'dataset'/rep['id']/'unlabeled/dataset.npz')
        teacher=np.load(root/'teacher'/f"{rep['id']}_unlabeled.npz");assert 'labels'not in pool
        for prior,lengths,answers in [('teacher',pool['lengths'],teacher['predicted_answer']),('labeled',support['lengths'][support['split']==0],support['labels'][support['split']==0])]:
            mapping={}
            for n in plan['lengths']:
                values,counts=np.unique(answers[lengths==n],return_counts=True);mapping[str(n)]=int(values[counts.argmax()])
            records.append({'replicate':rep['id'],'prior':prior,'modes':mapping,'fit_examples':len(answers)})
    atomic_json(file,{'frozen_utc':now(),'new_test_inspected':False,'records':records})


def paired(values,reps):
    values=np.asarray(values,dtype=float);cluster=[float(np.mean([values[i] for i,r in enumerate(reps) if r['source_seed']==s])) for s in [17,42,101]]
    return {'mean':float(values.mean()),'paired_values':values.tolist(),'positive_repeats':int((values>0).sum()),
            'paired_bootstrap_95':paired_interval(values),'three_pair_means':cluster,'three_pair_bootstrap_95':paired_interval(cluster)}


def analyze(plan,root):
    register(plan,root);assert json.loads((root/'evaluation_state.json').read_text())['status']=='complete'
    data=dict(np.load(root/'dataset/test/dataset.npz'));n=data['lengths'];y=data['labels'];modes=json.loads((root/'length_modes.json').read_text())['records'];records=[];sources=[]
    for rep in plan['replicates']:
        teacher=dict(np.load(root/'teacher'/f"test_s{rep['source_seed']}.npz"))
        mapping=next(r['modes'] for r in modes if r['replicate']==rep['id'] and r['prior']=='teacher');modal=np.array([mapping[str(int(nn))]==int(yy) for nn,yy in zip(n,y)])
        sources.append({'replicate':rep['id'],'source_seed':rep['source_seed'],**stratified(teacher['logits'].argmax(-1)==y,modal,n)})
        for condition in ['initial']+plan['conditions']:
            for step in ([0] if condition=='initial' else plan['checkpoint_updates']):
                file=root/'evaluations'/(f"{rep['id']}_initial.npz" if condition=='initial' else f"{rep['id']}_{condition}_u{step}.npz")
                out=dict(np.load(file));grade=stratified(out['logits'].argmax(-1)==y,modal,n)
                hidden_geometry=scores(out['hidden'],teacher['hidden'],n,y,plan['lengths']);output_geometry=scores(out['logits'],teacher['logits'],n,y,plan['lengths'])
                mean={k:float(np.mean([r[k] for r in hidden_geometry])) for k in ['correct_cka','matched_correct_cka','matched_wrong_cka','answer_residual_correct_cka','answer_residual_wrong_cka']}
                mean['matched_contrast']=mean['matched_correct_cka']-mean['matched_wrong_cka'];mean['answer_residual_contrast']=mean['answer_residual_correct_cka']-mean['answer_residual_wrong_cka']
                records.append({'replicate':rep['id'],'source_seed':rep['source_seed'],'condition':condition,'step':step,**grade,
                    'geometry':mean,'per_length_geometry':hidden_geometry,'per_length_output_geometry':output_geometry,'archive_sha256':sha(file)})
        print({'analyzed_repeat':rep['id']},flush=True)
    means=[];contrasts=[]
    for condition in ['initial']+plan['conditions']:
        for step in ([0] if condition=='initial' else plan['checkpoint_updates']):
            rows=[r for r in records if r['condition']==condition and r['step']==step]
            means.append({'condition':condition,'step':step,'accuracy':float(np.mean([r['accuracy'] for r in rows])),
                'modal_accuracy':float(np.mean([r['modal']['accuracy'] for r in rows])), 'nonmodal_accuracy':float(np.mean([r['nonmodal']['accuracy'] for r in rows])),
                'geometry':{k:float(np.mean([r['geometry'][k] for r in rows])) for k in rows[0]['geometry']}})
    for step in plan['checkpoint_updates']:
        for contrast in plan['primary_contrasts']:
            left,right=contrast.split('-');deltas={k:[] for k in ['accuracy_pp','modal_accuracy_pp','nonmodal_accuracy_pp','modal_contribution_pp','nonmodal_contribution_pp','cka','matched_contrast','answer_residual_contrast']}
            for rep in plan['replicates']:
                a=next(r for r in records if r['replicate']==rep['id'] and r['condition']==left and r['step']==step)
                b=next(r for r in records if r['replicate']==rep['id'] and r['condition']==right and r['step']==step)
                deltas['accuracy_pp'].append(100*(a['accuracy']-b['accuracy']))
                for group in ['modal','nonmodal']:
                    delta=100*(a[group]['accuracy']-b[group]['accuracy']);deltas[group+'_accuracy_pp'].append(delta);deltas[group+'_contribution_pp'].append(delta*a[group]['count']/len(y))
                assert abs(deltas['modal_contribution_pp'][-1]+deltas['nonmodal_contribution_pp'][-1]-deltas['accuracy_pp'][-1])<1e-10
                for k,key in [('cka','correct_cka'),('matched_contrast','matched_contrast'),('answer_residual_contrast','answer_residual_contrast')]:deltas[k].append(a['geometry'][key]-b['geometry'][key])
            contrasts.append({'step':step,'contrast':contrast,'primary':step==plan['updates'],**{k:paired(v,plan['replicates']) for k,v in deltas.items()}})
    priors=[]
    for prior in ['teacher','labeled']:
        for rep in plan['replicates']:
            mapping=next(r['modes'] for r in modes if r['replicate']==rep['id'] and r['prior']==prior)
            prediction=np.array([mapping[str(int(nn))] for nn in n]);priors.append({'replicate':rep['id'],'prior':prior,'accuracy':float((prediction==y).mean()),'modes':mapping})
    summary={'created_utc':now(),'all36_fits_complete_before_test':True,'primary_update':plan['updates'],'means':means,'contrasts':contrasts,'records':records,
        'source_accuracy':sources,'length_priors':priors,'test_examples':len(y),'target_models':36,'sources':3,'target_pretrained_starts':3,
        'fresh_support_U_training_repeats':6,'spontaneous_discovery_tested':False,'output_independent_mechanism_established':False}
    atomic_json(root/'summary.json',summary)
    report(plan,root,summary)


def report(plan,root,summary):
    names={'initial':'预训练起点','ordinary':'普通微调','correct_geometry':'正确几何','matched_mismatch':'答案匹配错配','distillation':'输出蒸馏','distillation_geometry':'蒸馏＋正确几何','distillation_mismatch':'蒸馏＋错配'}
    rows=[r for r in summary['means'] if r['step'] in [0,plan['updates']]]
    display=[[names[r['condition']],f"{100*r['accuracy']:.2f}%",f"{100*r['modal_accuracy']:.2f}%",f"{100*r['nonmodal_accuracy']:.2f}%",f"{r['geometry']['correct_cka']:.4f}",f"{r['geometry']['answer_residual_contrast']:.4f}"] for r in rows]
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>取逆关系：有能力起点的功能实验</title><style>body{font-family:system-ui;max-width:1200px;margin:30px auto;padding:20px;line-height:1.7}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}img{max-width:100%}</style><body><h1>取逆关系：从可预测任务的模型出发</h1><p>同一冻结descents教师在inverse(x)上提供监督，目标从已有recoils专门模型开始。六份新训练数据各有192训练标签、64验证标签、16384无标签输入；各条件同输入、同初始化、同40000更新。主要终点固定，不根据测试选择。几何位置是任务提示之前的ONE_END。</p>'
    html+=table(['方法','整体准确率','众数样本','非众数样本','留出CKA','答案残差正确−错配'],display)
    selected=[r for r in summary['contrasts'] if r['primary']]
    html+=table(['预先固定对比','准确率差(pp)','三模型对bootstrap95','CKA差','非众数差(pp)'],[[r['contrast'],f"{r['accuracy_pp']['mean']:+.3f}",str(r['accuracy_pp']['three_pair_bootstrap_95']),f"{r['cka']['mean']:+.5f}",f"{r['nonmodal_accuracy_pp']['mean']:+.3f}"] for r in selected])
    html+='<p>非众数由训练期教师的长度众数映射确定。长度基线在这组样本上按定义为零；重点是条件之间的配对差。六次训练重复依赖三个预训练模型对和同一个原始数据世界，区间是描述性汇总。已知数学关系由实验提供，结果不能证明普通训练自发发现代数关系。</p><img src="trajectory.png" alt="训练更新数与准确率、CKA"><p><a href="summary.json">全部逐次逐长度结果</a> · <a href="protocol.json">主实验登记</a> · <a href="analysis_protocol.json">分析登记</a> · <a href="confirmation/diagnostic/report.html">旧模型新留出集确认</a> · <a href="existing_diagnostic/report.html">既有预测诊断</a></p></body></html>'
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for condition in plan['conditions']:
        trajectory=[r for r in summary['means'] if r['condition']==condition];x=[r['step'] for r in trajectory]
        axes[0].plot(x,[100*r['accuracy'] for r in trajectory],marker='o',label=condition)
        axes[1].plot(x,[r['geometry']['correct_cka'] for r in trajectory],marker='o',label=condition)
    axes[0].set_ylabel('Fresh-test accuracy (%)');axes[1].set_ylabel('Per-length mean hidden CKA')
    for ax in axes:ax.set_xlabel('Training updates');ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(root/'trajectory.png',dpi=160);fig.savefig(root/'trajectory.svg');plt.close(fig)
    (root/'report.html').write_text(html)
    print(json.dumps({'fixed_primary_means':rows,'primary_accuracy_contrasts':[{k:r[k] for k in ['contrast','accuracy_pp','nonmodal_accuracy_pp','cka']} for r in selected]}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','modes','analyze']);args=parser.parse_args();plan,root,_=initialize()
    if args.phase=='register':register(plan,root)
    elif args.phase=='modes':register(plan,root);freeze_modes(plan,root)
    else:analyze(plan,root)
