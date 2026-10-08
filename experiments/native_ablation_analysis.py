"""All registered budget/ablation contrasts and four held-out task sets."""
from datetime import datetime,timezone
from html import escape
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from .native_ablation import paths,load,jobs,references,catalog,rid,sample
from .native_confirmation import feature_vector,permutation_row
from .native_confirmation_analysis import paired_summary,prediction_score
from .native_target_repeat import apply_weights
from .permworld_combinations import sha
from .longrun_engine import atomic_json
from .six_hour_report import write_rows


def endpoints(r):
    records=[json.loads(p.read_text()) for p in (r/'transfer').glob('*.json')]
    assert len(records)==60 and all(x['status']=='complete' for x in records)
    rows=[{**x,'group':d['job']['id'],'seed':d['job']['seed'],'kind':d['job']['kind']} for d in records for x in d['rows']]
    assert len(rows)==240
    assert len({(x['group'],x['seed'],x['budget'],x['mode']) for x in rows})==240
    return records,rows


def analyze():
    p,c,r=paths();records,rows=endpoints(r)
    lookup={(x['group'],x['seed'],x['budget'],x['mode']):x['accuracy'] for x in rows}
    length={x['budget']:x['accuracy'] for x in json.loads((r/'length.json').read_text())}
    pairs=[('equal_total_drop','full_b96','drop_'+t+'_b96') for t in p['candidate_tasks']]
    pairs += [('equal_per_task_drop','full_b128','drop_'+t+'_b96') for t in p['candidate_tasks']]
    pairs += [('equal_total_single_96','full_b96','single_'+t+'_b96') for t in p['candidate_tasks']]
    pairs += [('equal_total_single_32','full_b32','single_'+t+'_b32') for t in p['candidate_tasks']]
    pairs += [('anchor','full_b128',ref) for ref in ('random','length')]
    comparisons=[];seed_rows=[]
    for budget,mode in product(p['budgets'],p['adaptation']):
        for question,full,ref in pairs:
            gains=[]
            for seed in p['seeds']:
                a=lookup[full,seed,budget,mode];b=length[budget] if ref=='length' else lookup[ref,seed,budget,mode]
                gains.append(a-b);seed_rows.append({'question':question,'full':full,'reference':ref,'seed':seed,'budget':budget,'mode':mode,'full_accuracy':a,'reference_accuracy':b,'gain':a-b})
            effect=paired_summary(gains,p['inference_seed'],p['bootstrap_repetitions'])
            effect['scope']='three paired source seeds conditional on a reused source world and one new common target-data pool'
            comparisons.append({'question':question,'full':full,'reference':ref,'budget':budget,'mode':mode,**effect})
    forecasts=json.loads((r/'forecasts.json').read_text());prior=json.loads((r/'prior_forecasts.json').read_text())
    descriptors=json.loads((r/'descriptors.json').read_text());scores=[];prediction_rows=[];increments=[]
    variants=['source_learning','source_learning_plus_relations','source_learning_and_label_stats','source_learning_label_stats_plus_relations']
    novel={g['id'] for g in p['novel_groups']}
    for budget in p['budgets']:
        selected=[x for x in rows if x['group'] in novel and x['budget']==budget and x['mode']=='shared'];assert len(selected)==12
        truth=np.array([x['accuracy']-lookup['random',x['seed'],budget,'shared'] for x in selected]);predictions={}
        for variant in variants:
            predicted=np.array([next(q['predicted_gain'] for q in forecasts['rows'] if (q['group'],q['seed'],q['budget'],q['variant'])==(x['group'],x['seed'],budget,variant)) for x in selected]);predictions[variant]=predicted
            scores.append({'budget':budget,'variant':variant,**prediction_score(truth,predicted),'task_sets':4,'rows':12})
            prediction_rows.extend({'group':x['group'],'seed':x['seed'],'budget':budget,'variant':variant,'actual_gain':float(y),'predicted_gain':float(z)} for x,y,z in zip(selected,truth,predicted))
        for base,aug in ((variants[0],variants[1]),(variants[2],variants[3])):
            delta=(truth-predictions[base])**2-(truth-predictions[aug])**2
            values=[float(delta[[x['group']==g for x in selected]].mean()) for g in sorted(novel)]
            effect=paired_summary(values,p['inference_seed']+1,p['bootstrap_repetitions'])
            increments.append({'budget':budget,'baseline':base,'augmented':aug,'mse_improvement':effect['mean_gain'],
                'descriptive_group_low':effect['bootstrap_95_low'],'descriptive_group_high':effect['bootstrap_95_high'],'groups_improved':effect['positive_seeds'],
                'scope':'four deliberately selected held-out task sets; descriptive resampling, not a population-level confidence interval'})
    qualitative=[]
    for budget in p['budgets']:
        per_seed=[]
        for seed in p['seeds']:
            closed=np.mean([lookup[g,seed,budget,'shared'] for g in ('novel_records','novel_minima_pair')])
            mixed=np.mean([lookup[g,seed,budget,'shared'] for g in ('novel_peak_mixed','novel_valley_mixed')]);per_seed.append(closed-mixed)
        qualitative.append({'budget':budget,**paired_summary(per_seed,p['inference_seed'],p['bootstrap_repetitions']),
                            'scope':'prespecified two-versus-two task-set contrast; shared random reference cancels, three source seeds'})
    novel_summary=[]
    for g,budget in product(p['novel_groups'],p['budgets']):
        actual=np.mean([lookup[g['id'],seed,budget,'shared']-lookup['random',seed,budget,'shared'] for seed in p['seeds']])
        predicted=np.mean([x['predicted_gain'] for x in forecasts['rows'] if (x['group'],x['budget'],x['variant'])==(g['id'],budget,variants[1])])
        early=next(x['predicted_gain'] for x in prior['rows'] if (x['group'],x['budget'],x['variant'])==(g['id'],budget,variants[1]))
        novel_summary.append({'group':g['id'],'tasks':' | '.join(g['tasks']),'budget':budget,'operators':' | '.join(descriptors[g['id']]['certified_operators']),
                              'early_prior_gain':early,'presaved_gain':float(predicted),'actual_gain':float(actual)})
    source_rows=[]
    for job in catalog(p):
        if job['kind']=='random':continue
        record=json.loads(Path(job['record']).read_text()) if job['kind']=='reused' else json.loads((r/'source'/f'{rid(job)}.json').read_text())
        source_rows.append({'group':job['id'],'seed':job['seed'],'kind':job['kind'],'tasks':' | '.join(job['tasks']),
            'labels_per_update':job['batch']*len(job['tasks']),'total_labels':20000*job['batch']*len(job['tasks']),
            'per_task_exposures':20000*job['batch'],'source_audit_mean':float(np.mean([x['accuracy'] for x in record['source_audit']])),
            'unique_inputs_seen':record.get('unique_source_inputs_seen','not logged for reused models')})
    for name,data in (('endpoints',rows),('comparisons',comparisons),('seed_gains',seed_rows),('prediction_scores',scores),('prediction_rows',prediction_rows),
                       ('relation_increments',increments),('novel_summary',novel_summary),('source_profiles',source_rows)):write_rows(r/f'{name}.csv',data)
    summary={'status':'complete','comparisons':comparisons,'prediction_scores':scores,'relation_increments':increments,'qualitative_contrast':qualitative,
             'novel_summary':novel_summary,'new_source_models':42,'reused_sources':15,'random_models':3,'target_endpoints':240,
             'source_data_worlds':1,'new_source_data_worlds':0,'source_seeds':p['seeds'],'fresh_target_inputs':6300,'reported_utc':datetime.now(timezone.utc).isoformat()}
    atomic_json(r/'summary.json',summary);report(p,r,summary,source_rows)
    print(json.dumps({'primary_drop':[x for x in comparisons if x['question']=='equal_total_drop' and x['budget']==256 and x['mode']=='candidate_shared'],
                      'prediction_scores':scores,'qualitative_contrast':qualitative},indent=2),flush=True)


def report(p,r,s,source_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary=[x for x in s['comparisons'] if x['question']=='equal_total_drop' and x['budget']==256 and x['mode']=='candidate_shared']
    fig,ax=plt.subplots(figsize=(8,4))
    for j,x in enumerate(primary):
        ax.scatter(j+np.arange(3)*.06-.06,np.array(x['seed_gains'])*100)
        ax.errorbar(j,x['mean_gain']*100,yerr=[[100*(x['mean_gain']-x['bootstrap_95_low'])],[100*(x['bootstrap_95_high']-x['mean_gain'])]],fmt='o',color='black',capsize=4)
    ax.axhline(0,color='gray');ax.set_xticks(range(4),['peaks','valleys','LR maxima','RL maxima']);ax.set_xlabel('Removed task')
    ax.set_ylabel('Full minus leave-one-out LIS accuracy (pp)');ax.set_title('Equal total supervision: 1.92M labels; three paired seeds')
    fig.tight_layout();fig.savefig(r/'ablation.png',dpi=170);fig.savefig(r/'ablation.pdf');plt.close(fig)
    def effect_rows(question):
        return ''.join(f"<tr><td>{escape(x['reference'])}</td><td>{100*x['mean_gain']:+.2f}</td><td>[{100*x['bootstrap_95_low']:+.2f}, {100*x['bootstrap_95_high']:+.2f}]</td><td>{x['positive_seeds']}/3</td></tr>" for x in s['comparisons'] if x['question']==question and x['budget']==256 and x['mode']=='candidate_shared')
    pred=''.join(f"<tr><td>{x['budget']}</td><td>{escape(x['variant'])}</td><td>{format(x['r2'],'.3f') if x['r2'] is not None else 'undefined'}</td><td>{100*x['mae']:.2f}</td></tr>" for x in s['prediction_scores'])
    novel=''.join(f"<tr><td>{escape(x['group'])}<br>{escape(x['tasks'])}</td><td>{escape(x['operators']) or '已覆盖规则下无完整作用'}</td><td>{100*x['early_prior_gain']:+.2f}</td><td>{100*x['presaved_gain']:+.2f}</td><td>{100*x['actual_gain']:+.2f}</td></tr>" for x in s['novel_summary'] if x['budget']==256)
    tables=''.join(f'<h2>{title}</h2><table><tr><th>参照</th><th>平均差值 pp</th><th>三种子 bootstrap 区间 pp</th><th>正种子</th></tr>{effect_rows(question)}</table>' for question,title in (
        ('equal_total_drop','总标签量匹配：四任务减去各三任务组合'),('equal_per_task_drop','每任务曝光匹配：原四任务减去三任务组合'),
        ('equal_total_single_96','192 万总标签：四任务减去各单任务'),('equal_total_single_32','64 万总标签：四任务减去各单任务'),('anchor','原组合在新目标池上的参照')))
    (r/'report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LIS 消融与未见组合检验</title><style>body{{max-width:1200px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}aside{{padding:16px;background:#f2f4f7}}img{{max-width:100%}}</style><h1>LIS：任务消融、监督量匹配与未见组合</h1><p>固定三个配对种子 {p['seeds']}，同一源训练世界，另生成全新的共同支持池 2,100 输入、测试池 4,200 输入。42 个新源模型、15 个既有源检查点、3 个随机模型，共 240 目标端点。主要消融与监督量比较采用旧候选选定的 0.0003/100 微调；未见组合预测采用已有预测器对应的共同 0.0001/100 微调。两预算、两策略均保留。</p><aside>这是根据先前结果选择的后续试验，规则在新训练和目标数据生成前固定。所有源模型仍为 20,000 更新。192 万总标签组：四任务每任务每步 24、三任务 32、单任务 96；64 万组：四任务 8、单任务 32。因此匹配了总标签量与更新数，但每步排列输入数与每任务曝光不同。源池、初始化、长度序列以及基础 32 输入采样相同，较小批取前缀，较大批追加独立抽样。区间仅反映条件于固定数据的三个源种子，不能作跨世界结论。</aside><img src="ablation.png">{tables}<h2>四个未见组合：预测在测试前保存</h2><p>组合此前未进行联合源训练；任务本身和数学对象仍属于 PermWorld。完整变换作用由通用计数与记录规则给出，已复现旧八组的数学特征；有限核验不替代对应计数证明。早期预测使用旧源学习成绩的统一均值；后期预测使用新源审计和学习曲线，但仍在任何新迁移优化之前保存。已有预测权重完全沿用，不根据本轮结果重新拟合。</p><table><tr><th>组合与任务</th><th>已覆盖的完整作用</th><th>训练前预测 pp</th><th>测试前预测 pp</th><th>实际随机配对收益 pp</th></tr>{novel}</table><p>以上列出学习成绩加数学关系的一个预先固定版本；全部四版本得分如下。误差增量的四组描述性重采样区间见 CSV，不能把 12 个源种子端点视作 12 个独立任务组。</p><table><tr><th>标签</th><th>版本</th><th>R²</th><th>MAE pp</th></tr>{pred}</table><p><a href="protocol.json">固定规则</a> · <a href="prior_forecasts.json">训练前预测</a> · <a href="forecasts.json">迁移前预测</a> · <a href="comparisons.csv">全部消融与监督量对照</a> · <a href="seed_gains.csv">逐种子差值</a> · <a href="source_profiles.csv">训练监督量与源学习成绩</a> · <a href="relation_increments.csv">未见组合预测增量</a> · <a href="summary.json">含预先固定结构对照的汇总</a> · <a href="verification.json">核验</a></p></html>''')


def verify():
    p,c,r,raw,data,names,functions,tokens,device=load();records,rows=endpoints(r)
    with np.load(Path(p['source_root'])/'dataset/data.npz') as old:
        for split in ('train','validation','representation','source_audit'):
            for suffix in ('input','lengths','labels'):np.testing.assert_array_equal(raw[split+'_'+suffix],old[split+'_'+suffix])
    metadata=json.loads((r/'dataset/metadata.json').read_text());seen=set()
    for path,h in metadata['excluded_datasets_sha256'].items():
        assert sha(path)==h
        with np.load(path) as a:
            for k in a.files:
                if k.endswith('_input'):seen.update(permutation_row(x,int(n)) for x,n in zip(a[k],a[k[:-6]+'_lengths']))
    checked_labels=0
    for split in ('support_pool','target_test'):
        for x,n,y in zip(raw[split+'_input'],raw[split+'_lengths'],raw[split+'_labels']):
            value=permutation_row(x,int(n));assert value not in seen;seen.add(value)
            np.testing.assert_array_equal(y,[functions[t](value) for t in names]);checked_labels+=len(names)
    support=json.loads((r/'dataset/support_indices.json').read_text());order=np.random.default_rng(p['target_data_seed']+50000).permutation(2100)
    for b in p['budgets']:assert support[str(b)]==order[:b].tolist()
    buckets={n:np.flatnonzero(raw['train_lengths']==n) for n in range(10,31)};initial={}
    for job in jobs(p):
        d=json.loads((r/'source'/f'{rid(job)}.json').read_text());assert d['status']=='complete' and d['steps']==20000
        assert d['total_labels']==20000*job['batch']*len(job['tasks'])
        path=r/'source/checkpoints'/f'{rid(job)}.pt';assert sha(path)==d['checkpoint_sha256']
        assert initial.setdefault(job['seed'],d['initial_parameter_sha256'])==d['initial_parameter_sha256']
        core=np.random.default_rng(job['seed']+20261005);extra=np.random.default_rng(job['seed']+202610061)
        import hashlib
        digest=hashlib.sha256();consumed=np.zeros(42000,dtype=bool)
        for _ in range(20000):
            n,ids=sample(core,extra,buckets,job['batch'],10,30);consumed[ids]=True;digest.update(np.array([n],dtype=np.int64).tobytes()+ids.tobytes())
        assert digest.hexdigest()==d['sample_sha256'] and int(consumed.sum())==d['unique_source_inputs_seen']
        assert core.bit_generator.state==d['core_state'] and extra.bit_generator.state==d['extra_state']
        state=torch.load(path,weights_only=True,map_location='cpu');assert state['sample_sha256']==d['sample_sha256'] and state['step']==20000
    forecast=json.loads((r/'forecasts.json').read_text());weights=json.loads((r/'weights.json').read_text());features=json.loads((r/'forecast_features.json').read_text())
    assert forecast['weights_sha256']==sha(r/'weights.json') and forecast['features_sha256']==sha(r/'forecast_features.json')
    checked_forecasts=0
    for w in weights['weights']:
        if w['mode']!='shared':continue
        predicted=apply_weights(np.array([feature_vector(x,w['variant']) for x in features]),w['weights'])
        for x,y in zip(features,predicted):
            q=next(z for z in forecast['rows'] if (z['group'],z['seed'],z['budget'],z['variant'])==(x['group'],x['seed'],w['budget'],w['variant']))
            assert abs(float(y)-q['predicted_gain'])<1e-12;checked_forecasts+=1
    assert checked_forecasts==96
    truth=raw['target_test_labels'][:,names.index('lis_length')];accuracy_checks=0
    for record in records:
        assert datetime.fromisoformat(forecast['frozen_utc'])<datetime.fromisoformat(record['evaluated_utc'])
        path=r/'transfer'/f"{rid(record['job'])}_predictions.npz";assert sha(path)==record['predictions_sha256']
        with np.load(path) as a:
            for x in record['rows']:
                assert len(a[f"{x['budget']}_{x['mode']}"])==4200
                assert float(np.mean(a[f"{x['budget']}_{x['mode']}"]==truth))==x['accuracy'];accuracy_checks+=1
    assert accuracy_checks==240
    prior=json.loads((r/'prior_forecasts.json').read_text());assert datetime.fromisoformat(prior['frozen_utc']).timestamp()<(r/'dataset/data.npz').stat().st_mtime
    prior_checks=0
    for w in weights['weights']:
        if w['mode']!='shared':continue
        predicted=apply_weights(np.array([feature_vector(x,w['variant']) for x in prior['features']]),w['weights'])
        for x,y in zip(prior['features'],predicted):
            q=next(z for z in prior['rows'] if (z['group'],z['budget'],z['variant'])==(x['group'],w['budget'],w['variant']))
            assert abs(float(y)-q['predicted_gain'])<1e-12;prior_checks+=1
    with np.load(r/'length_predictions.npz') as a:
        for x in json.loads((r/'length.json').read_text()):assert float(np.mean(a[str(x['budget'])]==truth))==x['accuracy']
    atomic_json(r/'verification.json',{'status':'passed','checked_utc':datetime.now(timezone.utc).isoformat(),'fresh_target_inputs':6300,
        'recomputed_fresh_labels':checked_labels,'source_blocks_exactly_equal_previous_world':True,'new_source_sampling_replayed':42,
        'source_initializations_equal_within_seed':True,'weights_unchanged':True,'recomputed_forecasts':96,'recomputed_early_forecasts':prior_checks,'recomputed_accuracies':accuracy_checks,
        'limitation':'same source world; three paired seeds and four chosen unseen task sets'})
    atomic_json(r/'state.json',{'status':'complete','updated_utc':datetime.now(timezone.utc).isoformat(),'new_sources':42,'target_endpoints':240,'verification':'passed'})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('analyze','verify'),default='analyze');args=parser.parse_args();globals()[args.phase]()
