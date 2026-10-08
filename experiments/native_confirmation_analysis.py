"""Seed-level candidate effects and frozen cross-world prediction checks."""
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import numpy as np

from .native_confirmation import paths, load, jobs, feature_vector, permutation_row, VARIANTS, MODES
from .native_target_repeat import apply_weights
from .longrun_engine import atomic_json
from .permworld_combinations import sha, select_groups
from .six_hour_report import write_rows


def paired_summary(values, seed, repetitions=20000):
    values=np.asarray(values,dtype=float)
    assert values.ndim==1 and len(values)>1 and np.isfinite(values).all()
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),(repetitions,len(values)))].mean(1)
    interval=np.quantile(means,[.025,.975])
    return {'mean_gain':float(values.mean()),'seed_sd':float(values.std(ddof=1)),
            'bootstrap_95_low':float(interval[0]),'bootstrap_95_high':float(interval[1]),
            'positive_seeds':int(np.sum(values>0)),'seeds':len(values),'seed_gains':values.tolist(),
            'scope':'source initialization/sampling seeds conditional on one fixed new data world; five seeds do not establish cross-world uncertainty'}


def prediction_score(actual,predicted):
    actual,predicted=np.asarray(actual),np.asarray(predicted)
    denominator=float(np.sum((actual-actual.mean())**2))
    return {'r2':float(1-np.sum((actual-predicted)**2)/denominator) if denominator else None,
            'mse':float(np.mean((actual-predicted)**2)),'mae':float(np.mean(abs(actual-predicted)))}


def complete_endpoints(root):
    records=[json.loads(p.read_text()) for p in (root/'transfer').glob('*.json')]
    assert len(records)==51 and all(r['status']=='complete' for r in records)
    rows=[{**r,'group':record['group'],'seed':record['seed'],'phase':record['phase']}
          for record in records for r in record['rows']]
    assert len(rows)==1224
    assert len({(r['group'],r['seed'],r['target'],r['budget'],r['mode']) for r in rows})==len(rows)
    return records,rows


def analyze():
    plan,config,root=paths()
    records,endpoints=complete_endpoints(root)
    lookup={(r['group'],r['seed'],r['target'],r['budget'],r['mode']):r['test_accuracy'] for r in endpoints}
    length=json.loads((root/'length_baselines.json').read_text())
    length_lookup={(r['target'],r['budget']):r['test_accuracy'] for r in length['rows'] if r['mode']=='smooth_length'}
    candidate=plan['candidate_group']
    comparisons=[];seed_rows=[]
    for target,budget,mode in product(config['target_tasks'],config['target_budgets'],MODES):
        seeds=plan['model_seeds']
        for reference in ['random','smooth_length']+plan['single_tasks']:
            gains=[]
            for seed in seeds:
                accuracy=lookup[candidate,seed,target,budget,mode]
                baseline=length_lookup[target,budget] if reference=='smooth_length' else lookup[reference,seed,target,budget,mode]
                gains.append(accuracy-baseline)
                seed_rows.append({'target':target,'budget':budget,'mode':mode,'seed':seed,
                                  'candidate_accuracy':accuracy,'reference':reference,'reference_accuracy':baseline,'gain':accuracy-baseline})
            comparisons.append({'target':target,'budget':budget,'mode':mode,'reference':reference,
                                **paired_summary(gains,plan['inference_seed'],plan['bootstrap_repetitions'])})
    gains=[]
    for r in endpoints:
        if r['group']=='random':continue
        gains.append({**r,'paired_random_gain':r['test_accuracy']-lookup['random',r['seed'],r['target'],r['budget'],r['mode']],
                      'gain_over_length':r['test_accuracy']-length_lookup[r['target'],r['budget']]})
    summaries=[]
    for group,target,budget,mode in sorted({(r['group'],r['target'],r['budget'],r['mode']) for r in gains}):
        selected=[r for r in gains if (r['group'],r['target'],r['budget'],r['mode'])==(group,target,budget,mode)]
        summaries.append({'group':group,'target':target,'budget':budget,'mode':mode,'seeds':len(selected),
                          'test_accuracy':float(np.mean([r['test_accuracy'] for r in selected])),
                          'random_gain':float(np.mean([r['paired_random_gain'] for r in selected])),
                          'length_gain':float(np.mean([r['gain_over_length'] for r in selected]))})
    # Balanced new-seed evaluation: three specified seeds for every group.
    forecasts=json.loads((root/'forecasts.json').read_text())
    forecast_lookup={(r['group'],r['seed'],r['target'],r['budget'],r['mode'],r['variant']):r['predicted_paired_gain'] for r in forecasts['rows']}
    groups=select_groups(config);scores=[];prediction_rows=[];increments=[];group_losses=[]
    for mode,budget in product(('shared','condition'),config['target_budgets']):
        rows=[r for r in gains if r['phase']=='multi' and r['seed'] in plan['prediction_seeds'] and (r['mode'],r['budget'])==(mode,budget)]
        assert len(rows)==96
        actual=np.array([r['paired_random_gain'] for r in rows])
        predictions={}
        for variant in VARIANTS:
            predicted=np.array([forecast_lookup[r['group'],r['seed'],r['target'],budget,mode,variant] for r in rows])
            predictions[variant]=predicted
            scores.append({'mode':mode,'budget':budget,'variant':variant,**prediction_score(actual,predicted),'rows':96,'groups':8,'new_seeds_per_group':3})
            prediction_rows.extend({**{k:r[k] for k in ('group','seed','target','budget','mode')},'variant':variant,
                                    'actual_gain':float(y),'predicted_gain':float(p)} for r,y,p in zip(rows,actual,predicted))
        for base,augmented in ((VARIANTS[0],VARIANTS[1]),(VARIANTS[2],VARIANTS[3])):
            loss_delta=(actual-predictions[base])**2-(actual-predictions[augmented])**2
            per_group=[]
            for group in groups:
                mask=np.array([r['group']==group['id'] for r in rows])
                assert mask.sum()==12
                improvement=float(loss_delta[mask].mean());per_group.append(improvement)
                group_losses.append({'mode':mode,'budget':budget,'baseline':base,'augmented':augmented,
                                     'group':group['id'],'mse_improvement':improvement,'rows':12})
            base_score=next(r for r in scores if (r['mode'],r['budget'],r['variant'])==(mode,budget,base))
            augmented_score=next(r for r in scores if (r['mode'],r['budget'],r['variant'])==(mode,budget,augmented))
            cluster=paired_summary(per_group,plan['inference_seed']+1,plan['bootstrap_repetitions'])
            increments.append({'mode':mode,'budget':budget,'baseline':base,'augmented':augmented,
                               'baseline_r2':base_score['r2'],'augmented_r2':augmented_score['r2'],
                               'delta_r2':augmented_score['r2']-base_score['r2'],
                               'mse_improvement':cluster['mean_gain'],'descriptive_group_bootstrap_95_low':cluster['bootstrap_95_low'],
                               'descriptive_group_bootstrap_95_high':cluster['bootstrap_95_high'],
                               'groups_with_lower_mse':cluster['positive_seeds'],'groups':8,
                               'scope':'eight fixed task sets; group bootstrap is a descriptive sensitivity analysis, not independent world replication'})
    source_rows=[]
    for phase in ('multi','single'):
        for path in (root/phase).glob('*.json'):
            record=json.loads(path.read_text());assert record['status']=='complete'
            source_rows.extend({'phase':phase,'group':record['job']['id'],'seed':record['job']['seed'],**r}
                               for r in record['source_audit'])
    for name,rows in (('endpoints',endpoints),('paired_gains',gains),('candidate_seed_gains',seed_rows),
                      ('candidate_comparisons',comparisons),('group_target_summary',summaries),('frozen_prediction_scores',scores),
                      ('frozen_prediction_rows',prediction_rows),('relation_increments',increments),('prediction_group_losses',group_losses),('source_audit',source_rows)):
        write_rows(root/f'{name}.csv',rows)
    primary=next(r for r in comparisons if (r['target'],r['budget'],r['mode'],r['reference'])==('lis_length',256,'candidate_shared','random'))
    summary={'status':'complete','reported_utc':datetime.now(timezone.utc).isoformat(),'source_multi_models':26,'source_single_models':20,
             'random_models':5,'transfer_models':51,'transfer_endpoints':1224,'primary':primary,
             'candidate_comparisons':comparisons,'prediction_scores':scores,'relation_increments':increments,
             'new_data_worlds':1,'initialization_seeds':plan['model_seeds'],'target_test_inputs':4200,
             'all_frozen_variants_retained':True,'source_combinations_not_new':True,
             'analysis_code_sha256':sha(__file__),'protocol_sha256':sha(root/'protocol.json')}
    atomic_json(root/'summary.json',summary)
    report(root,summary,seed_rows,source_rows)
    print(json.dumps({'primary':primary,'relation_increments':increments},indent=2),flush=True)


def report(root,summary,seed_rows,source_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from html import escape
    primary=[r for r in summary['candidate_comparisons'] if (r['target'],r['budget'],r['mode'])==('lis_length',256,'candidate_shared')]
    fig,ax=plt.subplots(figsize=(10,5))
    for i,r in enumerate(primary):
        values=np.array(r['seed_gains'])*100
        ax.scatter(np.arange(len(values))*.06+i-.12,values,s=25,alpha=.8)
        ax.errorbar(i,r['mean_gain']*100,yerr=[[100*(r['mean_gain']-r['bootstrap_95_low'])],[100*(r['bootstrap_95_high']-r['mean_gain'])]],fmt='o',color='black',capsize=5)
    ax.axhline(0,color='gray');ax.set_xticks(range(len(primary)),[r['reference'] for r in primary],rotation=20)
    ax.set_ylabel('Candidate accuracy minus reference (pp)');ax.set_title('LIS, 256 labels; identical candidate-selected adaptation policy; five new source seeds')
    fig.tight_layout();fig.savefig(root/'candidate_effects.png',dpi=170);fig.savefig(root/'candidate_effects.pdf');plt.close(fig)
    effects=''.join(f"<tr><td>{escape(r['reference'])}</td><td>{100*r['mean_gain']:+.2f} pp</td><td>[{100*r['bootstrap_95_low']:+.2f}, {100*r['bootstrap_95_high']:+.2f}] pp</td><td>{r['positive_seeds']}/{r['seeds']}</td></tr>" for r in primary)
    predictions=''.join(f"<tr><td>{r['mode']}</td><td>{r['budget']}</td><td>{escape(r['baseline'])}</td><td>{r['baseline_r2']:.3f}</td><td>{r['augmented_r2']:.3f}</td><td>{r['delta_r2']:+.3f}</td><td>{r['groups_with_lower_mse']}/8</td></tr>" for r in summary['relation_increments'])
    perseed=[]
    for seed in summary['initialization_seeds']:
        rows=[r for r in seed_rows if (r['seed'],r['target'],r['budget'],r['mode'])==(seed,'lis_length',256,'candidate_shared')]
        refs={r['reference']:r['reference_accuracy'] for r in rows}
        perseed.append(f"<tr><td>{seed}</td><td>{100*rows[0]['candidate_accuracy']:.2f}%</td>"+''.join(f"<td>{100*refs[name]:.2f}%</td>" for name in ['random','peaks','valleys','left_to_right_maxima','right_to_left_maxima','smooth_length'])+'</tr>')
    source=''.join(f"<tr><td>{task}</td><td>{100*np.mean([r['accuracy'] for r in source_rows if r['phase']=='multi' and r['group']=='interior_none' and r['task']==task]):.2f}%</td><td>{100*np.mean([r['accuracy'] for r in source_rows if r['phase']=='single' and r['task']==task]):.2f}%</td></tr>" for task in ('peaks','valleys','left_to_right_maxima','right_to_left_maxima'))
    (root/'report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>新源模型与单任务对照复核</title><style>body{{max-width:1200px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui;color:#18202b}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}aside{{background:#f2f4f7;padding:16px;margin:18px 0}}</style><h1>固定 interior_none → LIS：新源训练与单任务复核</h1><p>候选来自上一轮结果，本轮方案在新数据生成与源训练前保存。新生成一个源训练数据世界，排除两套旧目标数据与先导排列数据；五个新种子为 {summary['initialization_seeds']}。候选四任务组合与其四个单任务各训练五次，其余七组各三次，共 26 多任务模型、20 单任务模型。每模型固定 20,000 更新、每任务 640,000 曝光，采用同种子同输入采样与相同初始参数。</p><aside>本轮有一个新的训练数据世界，五个种子是条件于这个世界的初始化/采样重复，不能算五个独立世界。单任务与多任务的梯度目标不同，不能将差异直接归因于代数关系。所有原任务组均保留，新任务组泛化尚未检验。</aside><h2>预先固定的主要比较</h2><p>LIS、256 标签、所有候选与对照共同采用学习率 0.0003、100 次微调。这个参数来自旧候选验证选择，没有在新结果上调参。点为五个同种子准确率差值，区间为按源种子重采样的 95% 描述性 bootstrap 区间；测试输入共 4,200 个，各种子共用同一支持集及测试集。少量种子和固定数据下的区间不能替代跨数据世界复制。长度基线沿用旧带宽 2，仅用支持标签。</p><img src="candidate_effects.png"><table><tr><th>候选减参照</th><th>平均收益</th><th>种子 bootstrap 区间</th><th>正种子</th></tr>{effects}</table><h2>各个源种子的实际成绩</h2><table><tr><th>种子</th><th>四任务候选</th><th>随机</th><th>单 peaks</th><th>单 valleys</th><th>单 LR maxima</th><th>单 RL maxima</th><th>长度</th></tr>{''.join(perseed)}</table><p>全部四个单任务对照都列出；只有候选持续优于这些对照，才能进一步支持需要组合的说法。这里没有按测试挑选一个单任务或一个种子。</p><h2>源任务学习程度</h2><table><tr><th>性质</th><th>候选组合中的源审计</th><th>单任务源审计</th></tr>{source}</table><h2>数学关系是否提供额外预测信息</h2><p>旧两套目标池在同组同种子同目标内先取收益平均；每个预测器只看到 96 行，不能把两池当成独立源训练复制。岭惩罚固定为 1，四个版本的权重在新数据生成前保存，数值预测在新源学习后、新迁移测试前保存。学习基线为源审计准确率、源验证曲线均值与目标指示变量；稳健性基线再加入源标签熵、联合熵、相关性及源—目标条件互信息。数学特征为恒等式阶数、正确的无约束指示变量、精确类别答案的变换 CKA、可证明向量作用及单因子置换比例。标签相似性使用源训练输入上的精确目标标签，属于分析参照，并未用于源优化。</p><p>新得分只使用每组预先指定的前三个新种子，共八组、四目标、96 个配对收益。候选额外两个种子不增加其预测指标权重。新预测检验的是新数据世界与新模型的同名任务组，不是未见组合。负 R² 表示预测仍比测试均值参照差；ΔR² 为正本身不足以说明可靠。</p><table><tr><th>策略</th><th>标签</th><th>基线</th><th>基线 R²</th><th>加关系 R²</th><th>ΔR²</th><th>误差降低组数</th></tr>{predictions}</table><p>误差差值的八组描述性重采样区间见 <a href="relation_increments.csv">全部增量与区间</a>，每组差值见 <a href="prediction_group_losses.csv">组级误差</a>。固定八组合之间的组重采样不构成独立数据世界重复。</p><p><a href="protocol.json">测试前固定方案</a> · <a href="weights.json">数据生成前的权重</a> · <a href="forecasts.json">迁移测试前的预测</a> · <a href="old_group_holdout.json">旧数据整组留出</a> · <a href="candidate_comparisons.csv">所有目标、预算与策略的对照</a> · <a href="candidate_seed_gains.csv">逐种子差值</a> · <a href="group_target_summary.csv">全部组合结果</a> · <a href="frozen_prediction_scores.csv">全部新预测得分</a> · <a href="dataset/metadata.json">输入来源</a> · <a href="verification.json">核验</a></p></html>''')


def verify():
    plan,config,root,raw,data,names,functions,tokens,device=load()
    metadata=json.loads((root/'dataset/metadata.json').read_text())
    old=set()
    for path in plan['excluded_datasets']:
        assert sha(path)==metadata['excluded_sha256'][path]
        with np.load(path) as archive:
            for key in archive.files:
                if key.endswith('_input'):
                    old.update(permutation_row(row,int(n)) for row,n in zip(archive[key],archive[key[:-6]+'_lengths']))
    new=set();checked_labels=0
    for split,count in plan['examples_per_length'].items():
        assert len(raw[split+'_input'])==count*21
        for row,n,labels in zip(raw[split+'_input'],raw[split+'_lengths'],raw[split+'_labels']):
            p=permutation_row(row,int(n));assert p not in new and p not in old
            assert sorted(p)==list(range(1,int(n)+1))
            new.add(p)
            np.testing.assert_array_equal(labels,[functions[t](p) for t in names]);checked_labels+=len(names)
    support=json.loads((root/'dataset/support_indices.json').read_text())
    order=np.random.default_rng(plan['data_seed']+50000).permutation(len(raw['support_pool_input']))
    for budget in config['target_budgets']:assert support[str(budget)]==order[:budget].tolist()
    source_counts={}
    for phase in ('multi','single'):
        records=[json.loads(p.read_text()) for p in (root/phase).glob('*.json')]
        source_counts[phase]=len(records)
        for record in records:
            assert record['status']=='complete' and record['step']==20000 and record['exposures_per_task']==640000
            signature={key:record[key] for key in ('job','architecture','training_policy','data_sha256','source_hashes','upstream_hashes')}
            assert record['fingerprint']==__import__('hashlib').sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
            for p,expected in record['source_hashes'].items():assert sha(p)==expected
            assert sha(root/phase/'checkpoints'/f"{record['job_id']}.pt")==record['checkpoint_sha256']
            assert not set(record['job']['tasks'])&set(config['target_tasks'])
    assert source_counts=={'multi':26,'single':20}
    forecast=json.loads((root/'forecasts.json').read_text());weights=json.loads((root/'weights.json').read_text())
    assert forecast['weights_sha256']==sha(root/'weights.json') and forecast['features_sha256']==sha(root/'forecast_features.json')
    assert datetime.fromisoformat(weights['frozen_utc']).timestamp()<Path(root/'dataset/data.npz').stat().st_mtime
    features=json.loads((root/'forecast_features.json').read_text())
    prediction_lookup={(r['group'],r['seed'],r['target'],r['budget'],r['mode'],r['variant']):r['predicted_paired_gain'] for r in forecast['rows']}
    for fitted in weights['weights']:
        predicted=apply_weights(np.array([feature_vector(r,fitted['variant']) for r in features]),fitted['weights'])
        for row,p in zip(features,predicted):
            assert np.isclose(p,prediction_lookup[row['group'],row['seed'],row['target'],fitted['budget'],fitted['mode'],fitted['variant']],rtol=0,atol=1e-12)
    records,endpoints=complete_endpoints(root)
    accuracy_checks=0
    for record in records:
        assert datetime.fromisoformat(forecast['frozen_utc'])<datetime.fromisoformat(record['evaluated_utc'])
        rid=f"{plan['architecture']['id']}_{record['group']}_s{record['seed']}"
        path=root/'transfer'/f'{rid}_predictions.npz';assert sha(path)==record['predictions_sha256']
        with np.load(path) as archived:
            assert len(archived.files)==24
            for r in record['rows']:
                p=archived[f"{r['target']}_{r['budget']}_{r['mode']}"]
                truth=raw['target_test_labels'][:,names.index(r['target'])]
                assert len(p)==4200 and float(np.mean(p==truth))==r['test_accuracy']
                accuracy_checks+=1
    assert accuracy_checks==1224
    result={'status':'passed','checked_utc':datetime.now(timezone.utc).isoformat(),'globally_disjoint_fresh_inputs':len(new),
            'excluded_distinct_prior_inputs':len(old),'recomputed_property_labels':checked_labels,'source_models':source_counts,
            'new_source_seeds':plan['model_seeds'],'source_code_and_checkpoint_hashes':True,'forecasts_recomputed':len(forecast['rows']),
            'weights_saved_before_new_dataset':True,'forecasts_saved_before_new_adaptation':True,
            'recomputed_test_accuracies_from_saved_predictions':accuracy_checks,'nested_shared_supports':True,
            'verifier_sha256':sha(__file__),'limitations':plan['limitations']}
    atomic_json(root/'verification.json',result)
    atomic_json(root/'state.json',{'status':'complete','updated_utc':datetime.now(timezone.utc).isoformat(),
                                  'source_models':46,'target_endpoints':1224,'verification':'passed'})
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('analyze','verify'),default='analyze')
    args=parser.parse_args();globals()[args.phase]()
