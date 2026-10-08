"""Post hoc source-output identities/equivariance on unseen input rows.

Source predictions are evaluated with FP32 CPU inference so the authorized
GPU source/target queue is not interrupted. No target task labels are read.
"""
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .longrun_engine import atomic_json
from .longrun_transfer import make_model
from .native_transform_audit import certificates
from .permworld_combinations import answer_logits, select_groups, sha
from .six_hour_report import write_rows


@torch.no_grad()
def predict_sources(model, prefixes, lengths, tasks, token_ids):
    model.eval(); result=[]
    for start in range(0,len(prefixes),32):
        x=torch.tensor(prefixes[start:start+32]);n=torch.tensor(lengths[start:start+32])
        predicted=answer_logits(model,x,n,tasks,token_ids).argmax(-1).reshape(len(tasks),-1).T
        result.append(predicted.numpy())
    return np.concatenate(result)


def score_identity(prediction, length, terms, n_coefficient, constant, positions):
    chosen=np.array([positions[t] for t in terms]);weight=np.array(list(terms.values()))
    valid=((prediction[:,chosen]>=0)&(prediction[:,chosen]<=length[:,None])).all(1)
    residual=prediction[:,chosen]@weight-n_coefficient*length-constant
    return valid&(residual==0)


def run():
    torch.set_num_threads(4)
    plan=json.loads(Path('configs/six_hour_session.json').read_text())
    config=json.loads(Path(plan['base_config']).read_text());root=Path(plan['output'])
    output=root/'source_consistency';output.mkdir(exist_ok=True)
    sys.path.insert(0,str(Path(config['repository'])/'src'))
    from neurips_permutations.passage import TOKEN_TO_ID
    groups=select_groups(config)
    arch=json.loads((root/'architecture_selection.json').read_text())['selected']
    names=json.loads((root/'dataset/metadata.json').read_text())['names']
    identities=json.loads(Path('results/permutation_audit/summary.json').read_text())['identities']
    signature={'code_sha256':sha(__file__),'source_model_sha256':sha('experiments/longrun_transfer.py'),
               'source_data_sha256':sha(root/'dataset/data.npz'),
               'transformed_data_sha256':sha(root/'native_transform_audit/transformed_representation.npz'),
               'architecture':arch,'inference':'FP32 CPU; no source or target retraining',
               'scope':'post hoc after observing geometry; all known same-input identities and proved source-vector actions are retained',
               'target_labels_read':False}
    if (output/'metadata.json').exists():assert json.loads((output/'metadata.json').read_text())==signature
    atomic_json(output/'metadata.json',signature)
    with np.load(root/'dataset/data.npz') as data:
        train_y=data['train_labels'];train_n=data['train_lengths']
        prefixes=data['representation_input'];lengths=data['representation_lengths'];labels=data['representation_labels']
    with np.load(root/'native_transform_audit/transformed_representation.npz') as transformed:
        transformed_arrays={k:transformed[k] for k in transformed.files}
    identity_rows,action_rows,source_rows=[],[],[]
    for group,seed in product(groups,plan['model_seeds']):
        tasks=group['tasks'];positions={t:i for i,t in enumerate(tasks)};task_ids=[names.index(t) for t in tasks]
        run_id=f"{arch['id']}_{group['id']}_s{seed}"
        source_record=json.loads((root/'multi'/f'{run_id}.json').read_text())
        checkpoint=root/'multi'/'checkpoints'/f'{run_id}.pt'
        assert source_record['status']=='complete' and sha(checkpoint)==source_record['checkpoint_sha256']
        model=make_model(config,arch,seed,'cpu')
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
        archive=output/f'{run_id}_predictions.npz'
        actions=certificates(group)
        if archive.exists():
            with np.load(archive) as cache:predicted={k:cache[k] for k in cache.files}
        else:
            predicted={'original':predict_sources(model,prefixes,lengths,tasks,TOKEN_TO_ID)}
            for op in actions:predicted[op]=predict_sources(model,transformed_arrays[op+'_input'],lengths,tasks,TOKEN_TO_ID)
            np.savez_compressed(archive,**predicted)
        truth=labels[:,task_ids];original=predicted['original']
        majority=np.zeros_like(truth)
        for n in np.unique(lengths):
            for k,t in enumerate(task_ids):majority[lengths==n,k]=np.bincount(train_y[train_n==n,t]).argmax()
        source_rows.append({'group':group['id'],'seed':seed,'source_accuracy':float((original==truth).mean()),
                            'source_joint_accuracy':float((original==truth).all(1).mean()),
                            'majority_accuracy':float((majority==truth).mean()),'prediction_sha256':sha(archive)})
        for identity in identities:
            if not set(identity['terms'])<=set(tasks):continue
            correct=score_identity(truth,lengths,identity['terms'],identity['n'],identity['constant'],positions)
            assert correct.all()
            satisfied=score_identity(original,lengths,identity['terms'],identity['n'],identity['constant'],positions)
            base=score_identity(majority,lengths,identity['terms'],identity['n'],identity['constant'],positions)
            chosen=[positions[t] for t in identity['terms']]
            all_correct=(original[:,chosen]==truth[:,chosen]).all(1)
            identity_rows.append({'group':group['id'],'seed':seed,'identity':identity['id'],
                                  'identity_satisfaction':float(satisfied.mean()),'majority_satisfaction':float(base.mean()),
                                  'excess_over_majority':float(satisfied.mean()-base.mean()),
                                  'all_identity_answers_correct':float(all_correct.mean()),
                                  'consistent_but_some_identity_answers_wrong':float((satisfied&~all_correct).mean())})
        for op,(matrix,_) in actions.items():
            changed=predicted[op];new_truth=transformed_arrays[op+'_labels'][:,task_ids]
            np.testing.assert_array_equal(truth@matrix.T,new_truth)
            valid=(original<=lengths[:,None]).all(1)&(changed<=lengths[:,None]).all(1)
            consistent=valid&(original@matrix.T==changed).all(1)
            base_consistent=(majority@matrix.T==majority).all(1)
            both_correct=(original==truth).all(1)&(changed==new_truth).all(1)
            action_rows.append({'group':group['id'],'seed':seed,'operator':op,
                                'source_accuracy_original':float((original==truth).mean()),
                                'source_accuracy_transformed':float((changed==new_truth).mean()),
                                'majority_accuracy_original':float((majority==truth).mean()),
                                'majority_accuracy_transformed':float((majority==new_truth).mean()),
                                'task_vector_equivariance':float(consistent.mean()),
                                'majority_task_vector_equivariance':float(base_consistent.mean()),
                                'equivariance_excess_over_majority':float(consistent.mean()-base_consistent.mean()),
                                'all_eight_answers_correct':float(both_correct.mean()),
                                'equivariant_but_some_answers_wrong':float((consistent&~both_correct).mean())})
        print(json.dumps({'source_prediction_complete':run_id}),flush=True)
    for filename,rows in (('source_accuracy.csv',source_rows),('identity_consistency.csv',identity_rows),('action_consistency.csv',action_rows)):
        write_rows(output/filename,rows)
    summary={'reported_utc':datetime.now(timezone.utc).isoformat(),'source_models':len(source_rows),
             'same_input_identity_endpoints':len(identity_rows),'proved_action_endpoints':len(action_rows),
             'mean_identity_satisfaction':float(np.mean([r['identity_satisfaction'] for r in identity_rows])),
             'mean_identity_joint_correctness':float(np.mean([r['all_identity_answers_correct'] for r in identity_rows])),
             'mean_action_equivariance':float(np.mean([r['task_vector_equivariance'] for r in action_rows])),
             'mean_action_eight_answer_correctness':float(np.mean([r['all_eight_answers_correct'] for r in action_rows])),
             'all_truth_identities_and_proved_actions_verified':True}
    atomic_json(output/'summary.json',summary)
    identity_table=''.join(f"<tr><td>{r['group']}</td><td>{r['seed']}</td><td>{100*r['identity_satisfaction']:.1f}%</td><td>{100*r['majority_satisfaction']:.1f}%</td><td>{100*r['all_identity_answers_correct']:.1f}%</td><td>{100*r['consistent_but_some_identity_answers_wrong']:.1f}%</td></tr>" for r in identity_rows)
    action_table=''.join(f"<tr><td>{r['group']}</td><td>{r['seed']}</td><td>{r['operator']}</td><td>{100*r['task_vector_equivariance']:.1f}%</td><td>{100*r['majority_task_vector_equivariance']:.1f}%</td><td>{100*r['all_eight_answers_correct']:.1f}%</td></tr>" for r in action_rows)
    (output/'report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>源任务答案：关系一致与真正正确</title><style>body{{max-width:1200px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}aside{{background:#f2f4f7;padding:16px}}</style>
<h1>源任务答案是否满足关系，与答案是否正确分开检查</h1><p>对 24 个已完成源模型，在 840 个未训练输入上做 FP32 CPU 推理。13 个已证明任务向量变换分别跨三个种子，共 39 个端点；变换输入同样未用于源训练。没有使用任何目标任务答案或改变模型。</p>
<h2>同输入恒等式</h2><table><tr><th>组</th><th>种子</th><th>关系成立</th><th>长度多数类成立</th><th>关系中答案全正确</th><th>关系成立但答案错误</th></tr>{identity_table}</table>
<h2>已证明的输入变换关系</h2><table><tr><th>组</th><th>种子</th><th>算子</th><th>输出等变一致</th><th>长度多数类一致</th><th>变换前后八答案全正确</th></tr>{action_table}</table>
<aside>长度多数类预测仅由源训练标签得到，常数或集中答案也可能满足恒等式及变换关系。一致性高不能代替源任务准确率，更不能单独证明算法学习或迁移。本检查是在隐藏几何结果之后追加的事后诊断；CPU 与 GPU 的浮点推理不保证边界样本逐位相同。19 个任务向量非闭合情况没有可适用的等变公式，不混入这一统计。</aside>
<p><a href="metadata.json">范围与指纹</a> · <a href="source_accuracy.csv">源准确率</a> · <a href="identity_consistency.csv">恒等式结果</a> · <a href="action_consistency.csv">所有变换结果</a> · <a href="summary.json">汇总</a></p></html>''')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
