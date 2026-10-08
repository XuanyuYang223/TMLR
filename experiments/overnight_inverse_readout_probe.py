"""Answer decodability at the exact source feature landmark being aligned."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from .inverse_functional_alignment import now
from .inverse_functional_report import table
from .longrun_engine import atomic_json
from .overnight_inverse_diagnostic import stratified
from .overnight_inverse_functional import initialize
from .permworld_combinations import sha


def register(plan,root,sig):
    folder=root/'readout_probe';folder.mkdir(exist_ok=True);file=folder/'protocol.json'
    signature={'code_path':__file__,'code_sha256':sha(__file__),'base_protocol_sha256':sha(root/'protocol.json'),
        'landmark':'Frozen inverse-input descents teacher final_norm at ONE_END, before task prompt.',
        'fitting':'For each of6 U pools and each length separately, fit a linear188-output ridge head to frozen teacher full-vocabulary argmax. Standardize features using U-only mean/std, std floor1e-6. Center one-hot outputs and add output-mean intercept. Solve (Z^T Z/N+.01I)W=Z^T Y_center/N. Fixed penalty, no validation tuning. No true U answers computed.',
        'comparators':'Teacher actual query/equals output; frozen teacher-U length mode; same masks and true labels on main new test after every main target fit. Report modal and nonmodal accuracy, per length, and training pseudo-answer fit.',
        'interpretation':'This diagnoses linearly decodable answer content at the landmark, not absence of nonlinear answer information. It does not intervene on students or prove that useful alignment is impossible.',
        'scope':'30 auxiliary linear heads, conditional on3 frozen teachers and6 U pools. Not independent source replications, no neural source retraining.'}
    if file.exists():assert json.loads(file.read_text())['signature']==signature;return folder
    assert not(root/'test_opened.json').exists()
    atomic_json(file,{'registered_utc':now(),'new_main_test_opened':False,'signature':signature});return folder


def fit(plan,root,sig):
    folder=register(plan,root,sig);records=[]
    for rep in plan['replicates']:
        pool=np.load(root/'dataset'/rep['id']/'unlabeled/dataset.npz');teacher=np.load(root/'teacher'/f"{rep['id']}_unlabeled.npz");assert 'labels'not in pool
        h=teacher['hidden'].astype(np.float64);labels=teacher['predicted_answer'];arrays={}
        for n in plan['lengths']:
            use=pool['lengths']==n;x=h[use];y=np.eye(188)[labels[use]];mean=x.mean(0);std=np.maximum(x.std(0),1e-6);z=(x-mean)/std;intercept=y.mean(0)
            weights=np.linalg.solve(z.T@z/len(z)+.01*np.eye(z.shape[1]),z.T@(y-intercept)/len(z))
            prediction=(z@weights+intercept).argmax(-1)
            for k,v in [('mean',mean),('std',std),('weights',weights),('intercept',intercept)]:arrays[f'n{n}_{k}']=v
            # Verify the fitted normal equations independently.
            residual=(z.T@(z@weights-(y-intercept)))/len(z)+.01*weights
            assert np.max(np.abs(residual))<1e-9
            records.append({'replicate':rep['id'],'length':n,'examples':int(use.sum()),'pseudo_answer_training_accuracy':float((prediction==labels[use]).mean()),'maximum_normal_equation_residual':float(np.abs(residual).max())})
        file=folder/f"{rep['id']}_heads.npz";np.savez_compressed(file,**arrays)
    atomic_json(folder/'fit.json',{'completed_utc':now(),'linear_heads':30,'true_U_answers_computed':False,'records':records,
                                'head_sha256':{p.name:sha(p)for p in folder.glob('*_heads.npz')}})


def evaluate(plan,root,sig):
    folder=register(plan,root,sig);assert json.loads((root/'evaluation_state.json').read_text())['status']=='complete'
    data=np.load(root/'dataset/test/dataset.npz');n=data['lengths'];y=data['labels'];modes=json.loads((root/'length_modes.json').read_text())['records'];records=[];checks=0
    fitinfo=json.loads((folder/'fit.json').read_text());assert fitinfo['completed_utc']<json.loads((root/'test_opened.json').read_text())['opened_utc']
    for rep in plan['replicates']:
        teacher=np.load(root/'teacher'/f"test_s{rep['source_seed']}.npz");file=folder/f"{rep['id']}_heads.npz";assert sha(file)==fitinfo['head_sha256'][file.name];head=np.load(file)
        pred=np.empty(len(y),dtype=int)
        for length in plan['lengths']:
            ix=np.flatnonzero(n==length);z=(teacher['hidden'][ix]-head[f'n{length}_mean'])/head[f'n{length}_std']
            pred[ix]=(z@head[f'n{length}_weights']+head[f'n{length}_intercept']).argmax(-1)
        mapping=next(r['modes']for r in modes if r['replicate']==rep['id']and r['prior']=='teacher');prior=np.array([mapping[str(int(v))]for v in n]);modal=prior==y
        for condition,answers in [('ONE_END_ridge',pred),('teacher_query_output',teacher['logits'].argmax(-1)),('teacher_length_mode',prior)]:
            grade=stratified(answers==y,modal,n)
            assert grade['accuracy']==sum(int(a)==int(b)for a,b in zip(answers,y))/len(y);checks+=1
            records.append({'replicate':rep['id'],'source_seed':rep['source_seed'],'condition':condition,**grade})
    means=[{'condition':condition,'accuracy':float(np.mean([r['accuracy']for r in records if r['condition']==condition])),
            'modal_accuracy':float(np.mean([r['modal']['accuracy']for r in records if r['condition']==condition])),
            'nonmodal_accuracy':float(np.mean([r['nonmodal']['accuracy']for r in records if r['condition']==condition]))}
           for condition in ['ONE_END_ridge','teacher_query_output','teacher_length_mode']]
    atomic_json(folder/'summary.json',{'completed_utc':now(),'means':means,'records':records,'independent_accuracy_checks':checks,'linear_heads':30,
        'heads_fit_before_new_test_opened':True,'neural_models_updated':0,'true_U_answers_computed':False})
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>教师对齐位置的线性答案信息</title><style>body{font-family:system-ui;max-width:1100px;margin:30px auto;padding:20px;line-height:1.7}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}</style><body><h1>教师的对齐位置能否读出答案？</h1><p>主测试打开前，在每个无标签池分别拟合固定惩罚的逐长度线性读出。监督只使用教师预测，未计算无标签真实答案。教师冻结，学生未改变。测试输入与功能主实验相同，因此不是一个独立数据重复。</p>'
    html+=table(['方法','整体准确率','众数样本','非众数样本'],[[r['condition'],f"{100*r['accuracy']:.2f}%",f"{100*r['modal_accuracy']:.2f}%",f"{100*r['nonmodal_accuracy']:.2f}%"]for r in means])
    html+='<p>低线性读出成绩只能限制线性可解码信息的判断，不能证明不存在非线性答案信息。高读出成绩也不能保证CKA约束会让学生获得可泛化的预测能力。</p><p><a href="summary.json">全部重复及逐长度结果</a> · <a href="protocol.json">登记规则</a> · <a href="fit.json">训练拟合及正规方程核验</a></p></body></html>'
    (folder/'report.html').write_text(html);print(json.dumps({'means':means,'independent_accuracy_checks':checks}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['fit','evaluate','watch']);args=parser.parse_args();plan,root,sig=initialize()
    if args.phase=='fit':fit(plan,root,sig)
    elif args.phase=='evaluate':evaluate(plan,root,sig)
    else:
        if not(root/'readout_probe/fit.json').exists():fit(plan,root,sig)
        while not(root/'evaluation_state.json').exists():time.sleep(10)
        evaluate(plan,root,sig)
