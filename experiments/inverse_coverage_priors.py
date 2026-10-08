"""Explicitly exploratory length-only baselines for output-information gains."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .inverse_alignment_coverage import initialize
from .inverse_functional_alignment import now
from .inverse_functional_report import table
from .longrun_engine import atomic_json
from .permworld_combinations import sha


def run():
    plan,root,sig=initialize();folder=root/'length_priors';folder.mkdir(exist_ok=True);old=Path(plan['previous_study']);test=np.load(root/'dataset/test/dataset.npz');records=[];checks=0
    protocol=folder/'protocol.json'
    if not protocol.exists():atomic_json(protocol,{'created_utc':now(),'code_sha256':sha(__file__),'exploratory_post_test':True,
        'reason':'After observing teacher-output gains over the neural ordinary baseline, check whether those gains establish input-dependent prediction rather than a better length-conditioned answer prior.',
        'policy':'For each length choose the most frequent answer in192 true training labels, or the most frequent teacher argmax in4096 unlabeled inputs. Ties choose the smallest answer. No validation or test labels used to fit either lookup. Predictions depend only on length. This diagnostic is not a predeclared primary comparison.'})
    else:assert json.loads(protocol.read_text())['code_sha256']==sha(__file__)
    for rep in sig['replicates']:
        support=np.load(old/'dataset'/rep['id']/'dataset.npz');pool=np.load(root/'dataset'/rep['id']/'dataset.npz');teacher=np.load(root/'teacher'/f"{rep['id']}.npz");assert 'labels'not in pool
        sources=[('labeled_length_mode',support['lengths'][support['split']==0],support['labels'][support['split']==0]),
                 ('teacher_length_mode',pool['lengths'],teacher['predicted_answer'])]
        for name,lengths,answers in sources:
            mapping={}
            for n in plan['lengths']:
                values,counts=np.unique(answers[lengths==n],return_counts=True);mapping[str(n)]=int(values[counts.argmax()])
                # Independent Counter implementation verifies fitting and tie policy.
                counted=Counter(map(int,answers[lengths==n]));expected=max(counted,key=lambda value:(counted[value],-value));assert expected==mapping[str(n)];checks+=1
            prediction=np.array([mapping[str(int(n))]for n in test['lengths']]);hit=prediction==test['labels'];accuracy=float(hit.mean())
            manual=sum(mapping[str(int(n))]==int(y)for n,y in zip(test['lengths'],test['labels']))/len(hit);assert accuracy==manual;checks+=1
            file=folder/f"{rep['id']}_{name}.npz";np.savez_compressed(file,predictions=prediction)
            records.append({'replicate':rep['id'],'condition':name,'modes':mapping,'accuracy':accuracy,'fit_examples':len(answers),'prediction_archive_sha256':sha(file),
                            'per_length':{str(n):float(hit[test['lengths']==n].mean())for n in plan['lengths']}})
    means={name:float(np.mean([r['accuracy']for r in records if r['condition']==name]))for name in ['labeled_length_mode','teacher_length_mode']}
    atomic_json(folder/'summary.json',{'created_utc':now(),'exploratory_post_test':True,'means':means,'records':records,'new_oracle_unlabeled_answers_used':False,
        'independent_count_and_accuracy_checks':checks,'neural_models_trained':0})
    tab=[[r['replicate'],r['condition'],r['fit_examples'],f"{100*r['accuracy']:.2f}%",json.dumps(r['modes'])]for r in records]
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>长度先验事后诊断</title><style>body{font-family:system-ui;max-width:1200px;margin:30px auto;padding:15px;line-height:1.7}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:7px}</style><body><h1>仅长度的答案先验：事后诊断</h1><p>在主结果揭晓后追加。本检查不训练神经模型，也不读取无标签真实答案。按长度选取训练标签或教师预测中频率最高的答案，平局选最小值；拟合时不使用验证或测试标签。基线预测只依赖长度，不依赖排列内容。</p><p>192标签的长度众数平均准确率：'+f"{100*means['labeled_length_mode']:.2f}%"+'；4096教师预测的长度众数：'+f"{100*means['teacher_length_mode']:.2f}%"+'。</p>'
    html+=table(['重复','基线','拟合样本','准确率','各长度众数'],tab)
    html+='<p>这不能证明神经模型完全没有使用排列信息；它限制了将模型相对较弱普通训练的增益解释为输入相关规律学习的结论。两种众数拟合与12个准确率均用独立计数方式核验。</p><p><a href="../report.html">主报告</a> · <a href="protocol.json">事后诊断范围</a> · <a href="summary.json">完整结果</a></p></body></html>'
    (folder/'report.html').write_text(html);print(json.dumps({'means':means,'independent_checks':checks}))


if __name__=='__main__':run()
