"""Prospective same-task-set identity references, supplementary to fixed primaries."""
from datetime import datetime,timezone
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

from .native_confirmation import paths
from .native_confirmation_analysis import complete_endpoints,prediction_score,paired_summary
from .native_target_repeat import fit_weights,apply_weights
from .longrun_engine import atomic_json
from .permworld_combinations import sha,select_groups
from .six_hour_report import write_rows


def identity_features(row,groups):
    return row['baseline'][:2]+row['baseline'][7:10]+[float(row['group']==g) for g in groups[1:]]


def prepare():
    plan,config,root=paths();groups=sorted(g['id'] for g in select_groups(config))
    protocol_path=root/'identity_control_protocol.json'
    signature={'code_sha256':sha(__file__),'primary_protocol_sha256':sha(root/'protocol.json'),
               'scope':'supplementary same-seen-task-group reference; registered after some new source learning, before any new target adaptation or outcomes',
               'references':['old group-target mean gain','source learning plus group indicators'],
               'fitting':'two old pools averaged within old source group/seed/target; alpha 1; no new behavior used',
               'evaluation':'first three specified new seeds per group, all four targets, two budgets, shared and condition adaptation policies',
               'groups':groups}
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text())['signature']==signature;return
    assert not list((root/'transfer').glob('*.json'))
    state=json.loads((root/'state.json').read_text())
    assert state['status'] in ('prepared','source_training','all_sources_complete','forecasts_saved_before_new_adaptation')
    count=sum(json.loads(p.read_text())['status']=='complete' for phase in ('multi','single') for p in (root/phase).glob('*.json'))
    atomic_json(protocol_path,{'registered_utc':datetime.now(timezone.utc).isoformat(),'signature':signature,
                              'new_source_models_known_at_registration':count,'new_target_records_at_registration':0})
    old=Path('results/six_hour_session')
    features=json.loads((old/'transfer_prediction/features.json').read_text());pools=[]
    for pool in (old,Path('results/native_target_repeat')):
        rows=[{**r,'group':record['group'],'seed':record['seed']} for path in (pool/'transfer').glob('*.json')
              for record in [json.loads(path.read_text())] for r in record['rows']]
        random={(r['seed'],r['target'],r['budget'],r['mode']):r['test_accuracy'] for r in rows if r['group']=='random'}
        pools.append({(r['group'],r['seed'],r['target'],r['budget'],r['mode']):r['test_accuracy']-random[r['seed'],r['target'],r['budget'],r['mode']]
                      for r in rows if r['group']!='random'})
    fitted,means=[],[]
    for mode,budget in product(('shared','condition'),config['target_budgets']):
        old_mode='finetune_shared' if mode=='shared' else 'finetune'
        y=np.array([np.mean([p[r['group'],r['seed'],r['target'],budget,old_mode] for p in pools]) for r in features])
        weights=fit_weights(np.array([identity_features(r,groups) for r in features]),y)
        fitted.append({'mode':mode,'budget':budget,'weights':weights})
        for group,target in product(groups,config['target_tasks']):
            mask=np.array([(r['group'],r['target'])==(group,target) for r in features]);assert mask.sum()==3
            means.append({'group':group,'target':target,'mode':mode,'budget':budget,'mean_gain':float(y[mask].mean())})
    atomic_json(root/'identity_control_weights.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),
                'protocol_sha256':sha(protocol_path),'weights':fitted,'old_group_target_means':means})


def freeze():
    plan,config,root=paths()
    state=json.loads((root/'state.json').read_text())
    assert state['status'] in ('source_training','all_sources_complete','forecasts_saved_before_new_adaptation')
    assert not list((root/'transfer').glob('*.json'))
    weights=json.loads((root/'identity_control_weights.json').read_text())
    groups=json.loads((root/'identity_control_protocol.json').read_text())['signature']['groups']
    means={(r['group'],r['target'],r['budget'],r['mode']):r['mean_gain'] for r in weights['old_group_target_means']}
    rows=[]
    for group in groups:
        seeds=plan['model_seeds'] if group==plan['candidate_group'] else plan['prediction_seeds']
        for seed in seeds:
            rid=f"{plan['architecture']['id']}_{group}_s{seed}"
            record=json.loads((root/'multi'/f'{rid}.json').read_text());assert record['status']=='complete'
            learning=[float(np.mean([r['accuracy'] for r in record['source_audit']])),float(np.mean([r['accuracy'] for r in record['curve']]))]
            for target in config['target_tasks']:
                baseline=learning+[0.]*5+[float(target==t) for t in config['target_tasks'][1:]]
                rows.append({'group':group,'seed':seed,'target':target,'baseline':baseline})
    forecasts=[]
    for fitted in weights['weights']:
        prediction=apply_weights(np.array([identity_features(r,groups) for r in rows]),fitted['weights'])
        for row,p in zip(rows,prediction):
            base={k:row[k] for k in ('group','seed','target')};base.update({k:fitted[k] for k in ('mode','budget')})
            forecasts.append({**base,'reference':'learning_plus_group_identity','predicted_gain':float(p)})
            forecasts.append({**base,'reference':'old_group_target_mean','predicted_gain':means[row['group'],row['target'],fitted['budget'],fitted['mode']]})
    # Check again after feature construction to fail explicitly on a timing race.
    state=json.loads((root/'state.json').read_text())
    assert state['status'] in ('source_training','all_sources_complete','forecasts_saved_before_new_adaptation')
    assert not list((root/'transfer').glob('*.json'))
    atomic_json(root/'identity_control_forecasts.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),
                'frozen_before_new_adaptation':True,'new_worker_state_at_freeze':state['status'],
                'weights_sha256':sha(root/'identity_control_weights.json'),'rows':forecasts})


def evaluate():
    plan,config,root=paths();records,endpoints=complete_endpoints(root)
    forecast=json.loads((root/'identity_control_forecasts.json').read_text())
    assert forecast['weights_sha256']==sha(root/'identity_control_weights.json')
    assert all(datetime.fromisoformat(forecast['frozen_utc'])<datetime.fromisoformat(r['evaluated_utc']) for r in records)
    primary=json.loads((root/'summary.json').read_text())
    random={(r['seed'],r['target'],r['budget'],r['mode']):r['test_accuracy'] for r in endpoints if r['group']=='random'}
    lookup={(r['group'],r['seed'],r['target'],r['budget'],r['mode'],r['reference']):r['predicted_gain'] for r in forecast['rows']}
    primary_forecasts=json.loads((root/'forecasts.json').read_text())
    relations={(r['group'],r['seed'],r['target'],r['budget'],r['mode']):r['predicted_paired_gain'] for r in primary_forecasts['rows'] if r['variant']=='source_learning_plus_relations'}
    scores=[];comparisons=[]
    for mode,budget in product(('shared','condition'),config['target_budgets']):
        rows=[r for r in endpoints if r['phase']=='multi' and r['seed'] in plan['prediction_seeds'] and (r['mode'],r['budget'])==(mode,budget)]
        assert len(rows)==96
        y=np.array([r['test_accuracy']-random[r['seed'],r['target'],budget,mode] for r in rows])
        math=np.array([relations[r['group'],r['seed'],r['target'],budget,mode] for r in rows])
        for reference in ('old_group_target_mean','learning_plus_group_identity'):
            predicted=np.array([lookup[r['group'],r['seed'],r['target'],budget,mode,reference] for r in rows])
            score=prediction_score(y,predicted);scores.append({'mode':mode,'budget':budget,'reference':reference,**score})
            per_group=[float(np.mean(((y-predicted)**2-(y-math)**2)[[r['group']==g for r in rows]])) for g in sorted({r['group'] for r in rows})]
            summary=paired_summary(per_group,plan['inference_seed']+2,plan['bootstrap_repetitions'])
            comparisons.append({'mode':mode,'budget':budget,'reference':reference,
                                'math_mse_improvement':summary['mean_gain'],'descriptive_group_95_low':summary['bootstrap_95_low'],
                                'descriptive_group_95_high':summary['bootstrap_95_high'],'groups_math_beats_reference':summary['positive_seeds']})
    atomic_json(root/'identity_control_summary.json',{'status':'complete','scores':scores,'comparisons':comparisons,
                'registered_after_partial_source_learning':True,'new_target_outcomes_at_registration':0,
                'forecasts_saved_before_new_adaptation':True,'scope':'same eight seen task groups, supplementary references, no new group generalization claim'})
    write_rows(root/'identity_control_scores.csv',scores);write_rows(root/'identity_control_comparisons.csv',comparisons)
    table=''.join(f"<tr><td>{r['mode']}</td><td>{r['budget']}</td><td>{r['reference']}</td><td>{r['r2']:.3f}</td><td>{100*r['mae']:.2f} pp</td></tr>" for r in scores)
    (root/'identity_control.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>已见组身份的预测参照</title><style>body{{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}aside{{background:#f2f4f7;padding:16px}}</style><h1>数学关系的预测是否超出已见任务组身份</h1><p>补充控制在部分新源学习已知、任何新迁移优化或结果之前固定。原来四个版本的主要预测规则保持不变。本参照只使用旧数据：一项为旧两池、三个初始化的同组同目标平均收益；另一项为源学习成绩、目标指示变量与任务组指示变量的固定岭回归。两者均在新迁移优化前保存具体预测。</p><aside>新世界仍沿用相同八个任务组。如果数学特征比只用源成绩好、但不如记住组身份的参照，证据主要支持已见组的收益稳定性，而不足以说明数学描述提供了可推广到新组的解释。组均值参照知道完整源组身份与目标，不能直接作为未见组预测器。</aside><table><tr><th>策略</th><th>标签</th><th>参照</th><th>新世界 R²</th><th>平均绝对误差</th></tr>{table}</table><p><a href="identity_control_protocol.json">补充规则与时点</a> · <a href="identity_control_weights.json">旧数据权重</a> · <a href="identity_control_forecasts.json">新优化前的预测</a> · <a href="identity_control_comparisons.csv">数学特征对参照的组级误差差值</a> · <a href="report.html">原固定主分析</a></p></html>''')
    report=root/'report.html'
    html=report.read_text()
    if 'identity_control.html' not in html:
        report.write_text(html.replace('</html>','<p><a href="identity_control.html">补充：已见任务组身份的预测参照</a>在新迁移开始前追加固定，与原四个主要版本分开报告。</p></html>'))
    print(json.dumps({'status':'identity_controls_complete','scores':scores}),flush=True)


def run():
    plan,config,root=paths();prepare()
    if not (root/'identity_control_forecasts.json').exists():
        while True:
            count=sum(json.loads(p.read_text())['status']=='complete' for phase in ('multi','single') for p in (root/phase).glob('*.json'))
            if count==46:break
            time.sleep(1)
        freeze()
    while json.loads((root/'state.json').read_text())['status']!='complete':time.sleep(2)
    evaluate()


if __name__=='__main__':run()
