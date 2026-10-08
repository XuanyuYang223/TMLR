"""Exploratory fixed-encoder target readout, registered before the main test."""
import argparse
import json
import time

import numpy as np

from .inverse_functional_alignment import configure, extract, now
from .inverse_functional_report import table
from .longrun_engine import atomic_json
from .overnight_inverse_diagnostic import stratified
from .overnight_inverse_functional import initialize, load_model
from .permworld_combinations import sha
from .specialist_cka_controls import api


def register(plan,root,sig):
    folder=root/'frozen_target_probe';folder.mkdir(exist_ok=True);file=folder/'protocol.json'
    signature={'code_path':__file__,'code_sha256':sha(__file__),'base_protocol_sha256':sha(root/'protocol.json'),
        'scope':'Exploratory extension after viewing teacher-prefix validation monitoring, before main test outcomes. No change to main or residual-intervention settings.30 fixed-penalty heads fitted on frozen native target ONE_END features and inverse-teacher pseudo-answers from the same U pools. No true U answers or neural-weight updates.',
        'fit':'Same per-length standardized ridge as teacher probe:188 one-hot pseudo-answer classes, U-only mean/std floor1e-6, centered outputs, fixed penalty.01. No validation tuning.',
        'comparators':'Frozen native query predictions; inverse teacher query; original teacher-U length mode. Reuse main test after all main target fitting. All heads/lengths/repeats reported.',
        'limitation':'Different decoder and length-specific routing from the main autoregressive models, not a matched optimization baseline. Diagnoses linear information available in frozen features, not an independent source-world or hidden algebra discovery.'}
    if file.exists():assert json.loads(file.read_text())['signature']==signature;return folder
    assert not(root/'test_opened.json').exists();atomic_json(file,{'registered_utc':now(),'new_main_test_opened':False,'signature':signature});return folder


def fit(plan,root,sig):
    folder=register(plan,root,sig);device=configure();_,tokens,_,_,_=api(plan);records=[];validation_rows=[]
    for target in sig['targets']:
        model=load_model(target,plan,device)
        for rep in plan['replicates']:
            if rep['target_pretrain_seed']!=target['seed']:continue
            pool=dict(np.load(root/'dataset'/rep['id']/'unlabeled/dataset.npz'));assert 'labels'not in pool
            teacher=np.load(root/'teacher'/f"{rep['id']}_unlabeled.npz");out=extract(model,pool,plan['target_task'],tokens,0)
            cache=folder/f"{rep['id']}_native_U.npz";np.savez_compressed(cache,**out)
            h=out['hidden'].astype(np.float64);labels=teacher['predicted_answer'];arrays={}
            for n in plan['lengths']:
                use=pool['lengths']==n;x=h[use];mean=x.mean(0);std=np.maximum(x.std(0),1e-6);z=(x-mean)/std;y=np.eye(188)[labels[use]];intercept=y.mean(0)
                weights=np.linalg.solve(z.T@z/len(z)+.01*np.eye(z.shape[1]),z.T@(y-intercept)/len(z))
                residual=z.T@(z@weights-(y-intercept))/len(z)+.01*weights;assert np.abs(residual).max()<1e-9
                for k,v in [('mean',mean),('std',std),('weights',weights),('intercept',intercept)]:arrays[f'n{n}_{k}']=v
                records.append({'replicate':rep['id'],'length':n,'fit_examples':int(use.sum()),'pseudo_answer_fit_accuracy':float(((z@weights+intercept).argmax(-1)==labels[use]).mean()),'normal_equation_max_residual':float(np.abs(residual).max())})
            file=folder/f"{rep['id']}_heads.npz";np.savez_compressed(file,**arrays)
            support=dict(np.load(root/'dataset'/rep['id']/'support/dataset.npz'));validation=extract(model,support,plan['target_task'],tokens,0);ix=np.flatnonzero(support['split']==1);pred=[]
            for i in ix:
                n=int(support['lengths'][i]);z=(validation['hidden'][i]-arrays[f'n{n}_mean'])/arrays[f'n{n}_std'];pred.append(int((z@arrays[f'n{n}_weights']+arrays[f'n{n}_intercept']).argmax()))
            validation_rows.append({'replicate':rep['id'],'validation_examples':len(ix),'ridge_accuracy':float((np.array(pred)==support['labels'][ix]).mean()),
                'native_query_accuracy':float((validation['logits'][ix].argmax(-1)==support['labels'][ix]).mean()),'no_validation_selection':True})
            print({'fitted_frozen_target_readout':rep['id']},flush=True)
        del model
    atomic_json(folder/'fit.json',{'completed_utc':now(),'linear_heads':30,'neural_models_updated':0,'true_U_answers_computed':False,'records':records,
        'validation_monitoring':validation_rows,'head_sha256':{p.name:sha(p)for p in folder.glob('*_heads.npz')}})


def evaluate(plan,root,sig):
    folder=register(plan,root,sig);assert json.loads((root/'evaluation_state.json').read_text())['status']=='complete'
    data=np.load(root/'dataset/test/dataset.npz');n=data['lengths'];y=data['labels'];modes=json.loads((root/'length_modes.json').read_text())['records'];info=json.loads((folder/'fit.json').read_text());records=[]
    assert info['completed_utc']<json.loads((root/'test_opened.json').read_text())['opened_utc']
    for rep in plan['replicates']:
        file=folder/f"{rep['id']}_heads.npz";assert sha(file)==info['head_sha256'][file.name];head=np.load(file)
        native=np.load(root/'evaluations'/f"{rep['id']}_initial.npz");teacher=np.load(root/'teacher'/f"test_s{rep['source_seed']}.npz");pred=np.empty(len(y),dtype=int)
        for length in plan['lengths']:
            ix=np.flatnonzero(n==length);z=(native['hidden'][ix]-head[f'n{length}_mean'])/head[f'n{length}_std'];pred[ix]=(z@head[f'n{length}_weights']+head[f'n{length}_intercept']).argmax(-1)
        mapping=next(r['modes']for r in modes if r['replicate']==rep['id']and r['prior']=='teacher');prior=np.array([mapping[str(int(v))]for v in n]);modal=prior==y
        for condition,answers in [('native_ONE_END_ridge',pred),('native_query_output',native['logits'].argmax(-1)),('inverse_teacher_query',teacher['logits'].argmax(-1)),('teacher_length_mode',prior)]:
            grade=stratified(answers==y,modal,n);assert grade['accuracy']==sum(int(a)==int(b)for a,b in zip(answers,y))/len(y)
            records.append({'replicate':rep['id'],'source_seed':rep['source_seed'],'condition':condition,**grade})
    means=[{'condition':condition,'accuracy':float(np.mean([r['accuracy']for r in records if r['condition']==condition])),
            'nonmodal_accuracy':float(np.mean([r['nonmodal']['accuracy']for r in records if r['condition']==condition]))}
           for condition in ['native_ONE_END_ridge','native_query_output','inverse_teacher_query','teacher_length_mode']]
    atomic_json(folder/'summary.json',{'completed_utc':now(),'exploratory_extension':True,'means':means,'records':records,'linear_heads':30,'neural_models_updated':0,'heads_fit_before_test_opened':True})
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>冻结目标表征的读出诊断</title><style>body{font-family:system-ui;max-width:1100px;margin:30px auto;padding:20px;line-height:1.7}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}</style><body><h1>冻结目标表征的线性答案信息</h1><p>在查看教师前缀的验证读出后提出的探索性扩展，主测试揭晓前登记并完成拟合。编码器来自原recoils预训练模型，保持冻结；只在原16384无标签池的教师预测上拟合固定线性读出，不计算无标签真实答案。</p>'
    html+=table(['方法','整体准确率','非众数准确率'],[[r['condition'],f"{100*r['accuracy']:.2f}%",f"{100*r['nonmodal_accuracy']:.2f}%"]for r in means])
    html+='<p>这是按长度路由的新读出器，优化预算与主实验不同；不能把其表现直接当成公平的优化对照或几何的因果效应。它检验冻结特征中线性可读出的信息。测试和源模型与主实验共享，不是独立复现。</p><p><a href="summary.json">全部结果</a> · <a href="protocol.json">登记</a> · <a href="fit.json">训练和验证监测</a></p></body></html>'
    (folder/'report.html').write_text(html);print(json.dumps({'means':means}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['fit','evaluate','watch']);args=parser.parse_args();plan,root,sig=initialize()
    if args.phase=='fit':fit(plan,root,sig)
    elif args.phase=='evaluate':evaluate(plan,root,sig)
    else:
        if not(root/'frozen_target_probe/fit.json').exists():fit(plan,root,sig)
        while not(root/'evaluation_state.json').exists():time.sleep(10)
        evaluate(plan,root,sig)
