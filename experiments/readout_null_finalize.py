"""Independent budget/prediction checks and paired interaction statistics."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import torch
from .permworld_combinations import sha
from .two_step_relation_factorial import epoch_schedule,now
from .downstream_harm_diagnostic_v2 import metrics
from .relation_error_localization import row_projection
from .longrun_engine import atomic_json
ROOT=Path('results/readout_null_confirmation')
CONDITIONS=['full_correct','full_wrong','null_correct','null_wrong']


def matrix_action(x,letter):
    a,b,c,d=map(int,x)
    return ((c,d,a,b) if letter=='a' else (-c,-d,a-c,b-d))


def matrix_word(x,word):
    for letter in word:x=tuple(v%13 for v in matrix_action(x,letter))
    return tuple(x)


def matrix_key(x):return min(matrix_word(x,w) for w in ['','a','b','bb','ab','abb'])


def audit_matrix_data(root):
    words=['','a','b','ab','ba','aa','bb','bbb','aba','bbbb'];checks=0;sets={}
    for file in (root/'dataset').glob('*.npz'):
        d=dict(np.load(file));ws=words if file.name=='test.npz' else words[:3]
        for row,labels in zip(d['x'],d['labels']):
            for x,y,w in zip(row,labels,ws):
                truth=matrix_word(tuple(row[0]),w);assert tuple(x)==truth and (truth[0]+truth[3])%13==y;checks+=1
        sets[file.stem]={matrix_key(x) for x in d['x'][:,0]}
        if 'pc' in d:
            ids=np.arange(len(d['labels']))
            for k in ['pc','pi']:
                partner=d[k];assert np.array_equal(np.sort(partner),ids) and np.all(partner!=ids)
                assert np.array_equal(d['labels'][partner],d['labels'])
            assert np.all(d['pc'][d['pi']]!=ids) and np.all(d['pi'][d['pc']]!=ids)
    train=set.union(sets['source'],*[sets[f'support{i}'] for i in range(3)])
    val=sets['validation']|sets['pilot_validation'];test=sets['test']
    assert not train&val and not train&test and not val&test
    old={matrix_key(x) for x in np.load('results/algebra_relation_v3/matrix/dataset/test.npz')['x'][:,0]}
    assert not set.union(*sets.values())&old
    return checks


def audit_fits(domain,root):
    checks=0;per_source=[]
    plan=json.loads(Path('configs/readout_null_permworld.json' if domain=='permworld' else 'configs/algebra_relation_common_v3.json').read_text())
    for i in range(3):
        if domain=='matrix':
            source=torch.load(root/'sources'/f's{i}.pt',map_location='cpu',weights_only=True)['model']
            d=dict(np.load(root/'dataset'/f'support{i}.npz'));rng=np.random.default_rng(261078101+i*997)
            digest=sha256()
            for epoch in range(1,901):
                order=rng.permutation(1024)
                for start in range(0,1024,64):digest.update(np.asarray([epoch],np.int64).tobytes()+order[start:start+64].tobytes())
            schedulesha=digest.hexdigest();exposures=900*1024;updates=900*16
        else:
            seed=[8123,9133,10151][i]
            source=torch.load(ROOT/'permworld_sources/checkpoints'/f'ordinary_s{seed}.pt',map_location='cpu',weights_only=True)['model']
            raw=dict(np.load(root/'dataset'/f'n{i}/support/dataset.npz'));fit=raw['split']==0
            pairing=dict(np.load(root/'dataset'/f'n{i}/support/pairings.npz'));digest=sha256();updates=0;exposures=0
            for epoch,ids in epoch_schedule(raw['lengths'][fit],pairing['eligible'],20,64,plan['support_seeds'][i]+8001):
                digest.update(np.asarray([epoch],np.int64).tobytes()+ids.tobytes());updates+=1;exposures+=len(ids)
            schedulesha=digest.hexdigest()
        grades=[]
        for c in CONDITIONS:
            name=f'n{i}_{c}' if domain=='matrix' else f'n{i}_hidden_{c}'
            record=json.loads((root/'fits'/f'{name}.json').read_text());assert record['epochs']==(900 if domain=='matrix' else 20)
            assert record['updates']==updates and record['anchor_exposures']==exposures and record['schedule_sha256']==schedulesha
            cp=root/'models'/f'{name}.pt' if domain=='matrix' else root/'maps'/f'{name}_e20.pt'
            assert sha(cp)==record['checkpoint_sha256']
            state=torch.load(cp,map_location='cpu',weights_only=True)
            if domain=='matrix':
                for k in ['readout.weight','readout.bias']:assert torch.equal(source[k],state['model'][k])
                assert record['native_forward_rows']==5*exposures
                ops=state['operators'];a=ops['maps.0.weight'].numpy().T.astype(float);b=ops['maps.1.weight'].numpy().T.astype(float)
                ba=ops['maps.0.bias'].numpy().astype(float);bb=ops['maps.1.bias'].numpy().astype(float)
            else:
                assert torch.equal(source['lm_head.weight'],state['model']['lm_head.weight'])
                assert torch.equal(source['token_embedding.weight'],state['model']['token_embedding.weight'])
                ops=state['operators'];a=np.eye(256)+ops['offset'][0].numpy().astype(float);b=np.eye(256)+ops['offset'][1].numpy().astype(float)
                ba=ops['bias'][0].numpy().astype(float);bb=ops['bias'][1].numpy().astype(float)
            cache=dict(np.load(root/'evaluations'/f'n{i}_{c}.npz'))
            for key,val in [('rho_a',a),('rho_b',b),('bias_a',ba),('bias_b',bb)]:np.testing.assert_array_equal(cache[key],val)
            replay=(cache['native'][:,0].astype(float)@a+ba)@b+bb
            np.testing.assert_allclose(replay,cache['ab_pred'],atol=1e-10,rtol=1e-10)
            answers=(replay@cache['readout_weight'].T+cache['readout_bias']).argmax(1)
            np.testing.assert_array_equal(answers,cache['ab_answers']);checks+=8
            grade=record['curve'][-1]
            grades.append({'source':i,'condition':c,'native':grade['native'] if domain=='matrix' else grade['native_e_C_I_accuracy'],
                           'generators':grade['generators'] if domain=='matrix' else grade['generator_C_I_accuracy'],
                           'geometry_scale':record['normalization_scale']})
        per_source.extend(grades)
    return checks,per_source


def bootstrap(per_source_pair,seed=261082001):
    rng=np.random.default_rng(seed);n=per_source_pair.shape[1];vals=np.empty(10000);sourceonly=np.empty(10000)
    sm=per_source_pair.mean(1)
    for j in range(10000):
        si=rng.integers(0,3,3);pi=rng.integers(0,n,n)
        sourceonly[j]=sm[si].mean();vals[j]=per_source_pair[np.ix_(si,pi)].mean()
    return {'mean_pp':float(100*per_source_pair.mean()),'source_differences_pp':(100*sm).tolist(),
            'source_only_interval_pp':(100*np.quantile(sourceonly,[.025,.975])).tolist(),
            'source_and_whole_pair_interval_pp':(100*np.quantile(vals,[.025,.975])).tolist(),
            'independent_source_clusters':3,'collision_pair_blocks':n}


def analyze(domain):
    root=ROOT/domain
    assert (root/'test_opened.json').exists()
    opened=json.loads((root/'test_opened.json').read_text())
    assert len(opened['fit_record_sha256'])==12
    for p,h in opened['fit_record_sha256'].items():assert sha(p)==h
    data=dict(np.load(root/('dataset/test.npz' if domain=='matrix' else 'dataset/test/dataset.npz')))
    k=3 if domain=='matrix' else 5;truth=data['labels'][:,k];use=data['split']==1;pairids=data['pair_ids'][use];unique=np.unique(pairids)
    matchecks=audit_matrix_data(root) if domain=='matrix' else json.loads((root/'data_verification.json').read_text())['full_orbit_states_checked']
    fitchecks,grades=audit_fits(domain,root)
    rows=json.loads((root/'evaluation_records.json').read_text())['records'];diag=[];pair_accuracy={};pair_both={};maxnull=0.
    for c in CONDITIONS:
        acc=[];both=[]
        for i in range(3):
            d=dict(np.load(root/'evaluations'/f'n{i}_{c}.npz'));h=d['native'].astype(float);w=d['readout_weight'].astype(float);bw=d['readout_bias'];b=d['rho_b'];bb=d['bias_b']
            pred1=h[:,0]@d['rho_a']+d['bias_a'];err=pred1-h[:,1];q=np.eye(w.shape[1])-row_projection(w);en=err@q
            z=(h[:,1]@b+bb)@w.T+bw;delta=en@b@w.T;mm=metrics(z,delta,truth)
            maxnull=max(maxnull,float(abs(en@w.T).max()))
            hit=d['ab_answers']==truth;acc.append(np.asarray([hit[use][pairids==p].mean() for p in unique]));both.append(np.asarray([hit[use][pairids==p].all() for p in unique],float))
            pos=use & mm['forced_correct']
            diag.append({'source':i,'condition':c,'split':'collisions','null_margin_crossing':float((mm['worst_margin_ratio'][pos]>=1).mean()),
                         'null_only_accuracy':float(mm['perturbed_correct'][use].mean()),'first_hidden_nmse':float(np.square(err[use]).sum()/np.square(h[use,1]-h[use,1].mean(0)).sum()),
                         'null_hidden_nmse':float(np.square(en[use]).sum()/np.square(h[use,1]-h[use,1].mean(0)).sum())})
            prior=next(r for r in rows if r['replicate']==i and r['condition']==c and r['split']=='collisions')
            assert prior['accuracy']==hit[use].mean() and prior['pair_both_correct']==both[-1].mean()
        pair_accuracy[c]=np.asarray(acc);pair_both[c]=np.asarray(both)
    assert maxnull<1e-8
    contrasts={}
    for name,arrays in [('accuracy',pair_accuracy),('pair_both',pair_both)]:
        contrasts[name]={
            'full_correct_minus_wrong':bootstrap(arrays['full_correct']-arrays['full_wrong']),
            'null_correct_minus_wrong':bootstrap(arrays['null_correct']-arrays['null_wrong']),
            'interaction_null_minus_full':bootstrap(arrays['null_correct']-arrays['null_wrong']-arrays['full_correct']+arrays['full_wrong']),
            'null_minus_full_correct':bootstrap(arrays['null_correct']-arrays['full_correct']),
            'null_minus_full_wrong':bootstrap(arrays['null_wrong']-arrays['full_wrong'])}
    means=[]
    for c in CONDITIONS:
        for split in ['iid','collisions']:
            rr=[r for r in rows if r['condition']==c and r['split']==split]
            means.append({'condition':c,'split':split,**{k:float(np.mean([r[k] for r in rr])) for k in ['accuracy','prediction_nmse','cka','true_intermediate_accuracy','direct_native_accuracy']},
                          'pair_both_correct':float(np.mean([r['pair_both_correct'] for r in rr])) if split=='collisions' else None})
    result={'status':'complete','domain':domain,'completed_utc':now(),'records':rows,'means':means,'contrasts':contrasts,'grades':grades,'diagnostic':diag,
            'verification':{'math_orbit_checks':matchecks,'fit_and_affine_checks':fitchecks,'maximum_null_logit_change':maxnull,'all12_budgets_and_schedules_match':True},
            'gates':{'native95_and_generators90':all(min(g['native'])>=.95 and min(g['generators'])>=.9 for g in grades)}}
    atomic_json(root/'independent_results.json',result)
    print(json.dumps({'domain':domain,'means':means,'primary_interaction':contrasts['accuracy']['interaction_null_minus_full'],'grades':grades}),flush=True)
    return result


def report(results):
    from .two_step_relation_final_report import table
    from html import escape
    pieces=['<h1>代数关系：下游有害误差与零空间约束</h1>']
    diag=json.loads(Path('results/downstream_harm_diagnostic_v2/results.json').read_text())
    pieces.append('<p>先诊断旧模型，再固定新实验。零空间保留第一步全部线性数字分数；真实中间状态仅用于诊断，不能计为推断成功。三次新源初始化并非三个新来源语料。</p>')
    pieces.append('<img style="width:100%" src="../downstream_harm_diagnostic_v2/perturbations.svg">')
    ds=[r for r in diag['means'] if r['split']=='collisions' and r['condition'] in ['both_correct','both_wrong']]
    pieces.append(table(['旧领域','配对','自然零空间误差跨界','复合准确率'],[[r['domain'],r['condition'],f'{100*r["null_margin_crossing_fraction"]:.2f}%',f'{100*r["composed_accuracy"]:.2f}%'] for r in ds]))
    if Path('results/first_state_predictor_diagnostic_v2/results.json').exists():
        probe=json.loads(Path('results/first_state_predictor_diagnostic_v2/results.json').read_text())['records']
        ps=[]
        for domain in ['matrix','permworld']:
            for kind in ['original','affine_residual','nonlinear_residual']:
                rr=[r for r in probe if (r['domain'],r['kind'],r['split'])==(domain,kind,'test')]
                ps.append([domain,kind,f'{np.mean([r["hidden_nmse"] for r in rr]):.4f}',f'{100*np.mean([r["collision_compound_accuracy"] for r in rr]):.2f}%',f'{100*np.mean([r["collision_pair_both_correct"] for r in rr]):.2f}%'])
        pieces.append('<h2>旧模型容量诊断</h2>'+table(['领域','单步预测器','状态误差','复合准确率','配对双正确'],ps)+'<p>2000次更新只使用单步表征，第二步与读出器固定。非线性预测器参数更多；没有错配组，不能识别关系正确性的因果作用。补保存权重与预测的同预算重放在第一次诊断结果后登记，没有调参。</p>')
    for r in results:
        selected=[s for s in r['means'] if s['split']=='collisions']
        pieces.append('<h2>'+r['domain']+'：新来源和新测试</h2>'+table(['几何损失','复合准确率','配对双正确','状态误差','CKA'],[[s['condition'],f'{100*s["accuracy"]:.2f}%',f'{100*s["pair_both_correct"]:.2f}%',f'{s["prediction_nmse"]:.4f}',f'{s["cka"]:.4f}'] for s in selected]))
        cc=r['contrasts']['accuracy'];pieces.append(table(['配对差值','百分点','来源＋完整碰撞对95%区间'],[[name,f'{v["mean_pp"]:+.3f}',str([round(x,3) for x in v['source_and_whole_pair_interval_pp']])] for name,v in cc.items()]))
        component_rows=[]
        for c in CONDITIONS:
            dd=[x for x in r['diagnostic'] if x['condition']==c]
            mean=lambda k:float(np.mean([x[k] for x in dd]))
            component_rows.append([c,f'{mean("first_hidden_nmse"):.4f}',f'{mean("null_hidden_nmse"):.4f}',f'{mean("first_hidden_nmse")-mean("null_hidden_nmse"):.4f}',f'{100*mean("null_margin_crossing"):.2f}%'])
        pieces.append(table(['组','第一步总误差','零空间误差','读出行空间误差','自然零空间误差跨界'],component_rows))
        pieces.append('<p>本轮零空间损失替代全空间项，释放了读出方向的几何约束。输出分类准确率高仍允许较大的读出方向状态误差；结果不能直接推广到保留全空间项再增加零空间权重的方案。</p>')
        pieces.append('<p>主要交互：［零空间正确−错配］−［全空间正确−错配］。'+escape(str(r['gates']))+'</p>')
        pieces.append(table(['源','组','可见准确率','两生成元准确率','损失归一化尺度'],[[g['source'],g['condition'],str([round(x,4) for x in g['native']]),str([round(x,4) for x in g['generators']]),round(g['geometry_scale'],5)] for g in r['grades']]))
    pieces.append('<p>预登记：每领域三个新源种子、四组，共12次固定预算拟合；PermWorld每组20轮，矩阵900轮。只改变配对和约束空间，各组输出监督、教师、曝光、训练顺序及读出一致。PermWorld沿用旧源语料并增加全新适应/测试轨道；矩阵排除旧测试轨道，新的训练/验证/测试轨道互不重叠。不同领域仍保留原架构与双向/单向监督差异。</p><p>旧诊断第一版的自身类别产生非有限分类间隔统计，已在v2修正并保留原始记录；扰动曲线未受该问题影响。区间仅有三个来源簇，须连同逐来源效应判断。</p><p><a href="../downstream_harm_diagnostic_v2/results.json">旧误差诊断</a> · <a href="../first_state_predictor_diagnostic_v2/results.json">状态预测器诊断</a> · <a href="matrix/independent_results.json">矩阵确认</a> · <a href="permworld/independent_results.json">PermWorld确认</a></p>')
    (ROOT/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:35px auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:7px}</style>'+''.join(pieces))


def run(domain):
    torch.set_num_threads(1)
    domains=['matrix','permworld'] if domain=='both' else [domain]
    for d in domains:analyze(d)
    results=[json.loads((ROOT/d/'independent_results.json').read_text()) for d in ['matrix','permworld'] if (ROOT/d/'independent_results.json').exists()]
    report(results)
    if len(results)==2:atomic_json(ROOT/'completion.json',{'status':'complete','completed_utc':now(),'formal_fits':24,'new_source_models':6,'independent_verification_complete':True,'results':{r['domain']:r['contrasts']['accuracy']['interaction_null_minus_full'] for r in results}})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('domain',choices=['matrix','permworld','both']);run(p.parse_args().domain)
