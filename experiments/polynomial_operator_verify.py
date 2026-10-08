"""Reconstruct the polynomial supplemental assay from saved generator maps."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from .algebra_structure_replication_analysis import table, write_csv
from .field_symmetry import group_sources
from .longrun_engine import atomic_json
from .matched_field import controlled_world
from .permworld_combinations import sha


def run():
    root=Path('results/algebra_structure_replication/polynomial')
    protocol=json.loads((root/'protocol.json').read_text())['signature']
    assert sha('experiments/polynomial_operator_assay.py')==protocol['code_sha256']
    assert sha('experiments/algebra_noninvertible.py')==protocol['operator_code_sha256']
    for path,digest in protocol['dependency_sha256'].items():assert sha(path)==digest
    config=json.loads(Path('configs/field_matched_support.json').read_text());source=Path(config['output'])
    assert sha(source/'metadata.json')==protocol['source_protocol_sha256']
    ops={k:np.asarray(v) for k,v in protocol['operators'].items()}
    np.testing.assert_array_equal(ops['d'] @ ops['p1'],ops['p0'] @ ops['d'])
    assert not np.array_equal(ops['d'] @ ops['p1'],ops['p1'] @ ops['d'])
    for p in ['p1','p0']:np.testing.assert_array_equal(ops[p] @ ops[p],ops[p])
    rows=[];counts={'conditions':0,'matrix_archives':0,'scalar_values':0,'numeric_null_checks':0};hashes={}
    for path in sorted((root/'probes').glob('*.json')):
        record=json.loads(path.read_text());g=record['group'];w=record['world_seed'];seed=record['seed'];status=record['model_status']
        inputs,_,_,basis=controlled_world(5,4,w);latent=inputs @ basis.T % 5;lookup={tuple(z):i for i,z in enumerate(latent)}
        ctx=np.random.default_rng(protocol['context_split_seed']).permutation(5)
        split=np.array([0 if z in ctx[:3] else 1 if z==ctx[3] else 2 for z in latent[:,3]])
        fit=split==0;test=split==2
        assert [int(np.sum(split==k)) for k in range(3)]==[375,125,125]
        assert record['heldout_context_values']==ctx[3:].tolist()
        def images(word):
            transformed=latent.copy()
            for op in word:transformed=transformed @ ops[op].T % 5
            ids=np.array([lookup[tuple(z)] for z in transformed]);np.testing.assert_array_equal(split[ids],split)
            return ids
        np.testing.assert_array_equal(images(['p1','d']),images(['d','p0']))
        assert np.any(images(['p1','d'])!=images(['d','p1']))
        step=0 if status=='random' else config['steps'];fp=source/f'{g}_w{w}_m{seed}_step{step}_features.npy'
        assert sha(fp)==record['feature_sha256'];h=np.load(fp).astype(np.float64);views={'hidden':h,'hidden_full_width':h}
        if status=='trained':
            cp=source/'checkpoints'/f'{g}_w{w}_m{seed}.pt';hashes[cp.name]=sha(cp)
            state=torch.load(cp,weights_only=True,map_location='cpu')
            weights=state['heads.weight'].numpy().astype(np.float64).reshape(4,5,64)
            contrasts=(weights-weights.mean(1,keepdims=True)).reshape(20,64)
            _,s,v=np.linalg.svd(contrasts,full_matrices=False);rank=int(np.sum(s>s[0]*1e-10));assert rank==16
            numeric=((h @ v[:rank].T) @ v[:rank]);null=h-numeric
            np.testing.assert_allclose(null @ contrasts.T,0,atol=1e-9,rtol=0);counts['numeric_null_checks']+=1
            labels=latent @ group_sources(g,5).T % 5
            views.update(readout_contrast=numeric,readout_null=null,exact_onehot=np.eye(5)[labels].reshape(625,-1))
        for view,result in record['results'].items():
            archive=root/'arrays'/f'{path.stem}_{view}.npz';assert sha(archive)==record['array_sha256'][archive.name]
            counts['matrix_archives']+=1
            features=views[view];mean=features[fit].mean(0)
            with np.load(archive) as a:
                np.testing.assert_allclose(mean,a['mean'],rtol=1e-10,atol=1e-10)
                q=a['basis'].astype(np.float64);z=(features-mean) @ q
                maps={op:(a['rho_'+op],a['bias_'+op]) for op in ops}
                shuffled={op:(a['shuffled_rho_'+op],a['shuffled_bias_'+op]) for op in ops}
            assert set(maps)=={'d','p1','p0'}
            def apply(word,chosen=maps):
                predicted=z[test]
                for op in word:predicted=predicted @ chosen[op][0]+chosen[op][1]
                return predicted
            def prediction(word,chosen=maps):
                return features[test]+(apply(word,chosen)-z[test]) @ q.T
            def error(word,chosen=maps):
                delta=features[images(word)[test]]-features[test]
                return float(np.square(prediction(word,chosen)-features[images(word)[test]]).sum()/np.square(delta).sum())
            def close(a,b):
                np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-6);counts['scalar_values']+=1
            for action in result['generators']+result['composites']:
                word=action['word'].split('_');close(error(word),action['displacement_nmse'])
                close(error(word,shuffled),action['shuffled_fit_displacement_nmse'])
                delta=features[images(word)[test]]-features[test]
                close(float(np.square(delta).sum()/np.square(features[test]-mean).sum()),action['displacement_energy'])
                assert action['identity_displacement_nmse']==1
                if g=='P' and view=='exact_onehot':assert action['displacement_nmse']<1e-8
            for law in result['laws']:
                left=law['left'].split('_');right=law['right'].split('_')
                np.testing.assert_array_equal(images(left),images(right))
                denom=np.square(features[images(left)[test]]-features[test]).sum()
                close(float(np.square((apply(left)-apply(right)) @ q.T).sum()/denom),law['full_space_consistency_nmse'])
                close(error(left),law['left_prediction_nmse']);close(error(right),law['right_prediction_nmse'])
            correct=features[images(['p1','d'])[test]];wrong=features[images(['d','p1'])[test]]
            predicted=prediction(['p1','d']);denom=np.square(correct-features[test]).sum()
            close(float((np.square(predicted-wrong).sum()-np.square(predicted-correct).sum())/denom),result['wrong_order_gap'])
            rows.append({'group':g,'world_seed':w,'model_seed':seed,'status':status,'view':view,
                'generator_nmse':float(np.mean([r['displacement_nmse'] for r in result['generators']])),
                'generator_shuffled_nmse':float(np.mean([r['shuffled_fit_displacement_nmse'] for r in result['generators']])),
                'cross_nmse':result['laws'][0]['left_prediction_nmse'],
                'other_cross_nmse':result['laws'][0]['right_prediction_nmse'],
                'cross_shuffled_nmse':result['composites'][0]['shuffled_fit_displacement_nmse'],
                'cross_consistency_nmse':result['laws'][0]['full_space_consistency_nmse'],
                'wrong_order_gap':result['wrong_order_gap']})
        counts['conditions']+=1
    assert counts['conditions']==54 and counts['matrix_archives']==189
    write_csv(root/'endpoints.csv',rows);summary=[]
    keys=['generator_nmse','generator_shuffled_nmse','cross_nmse','other_cross_nmse','cross_shuffled_nmse','cross_consistency_nmse','wrong_order_gap']
    for g in config['groups']:
        for status in ['random','trained']:
            for view in ['hidden','hidden_full_width','readout_contrast','readout_null','exact_onehot']:
                selected=[r for r in rows if r['group']==g and r['status']==status and r['view']==view]
                if not selected:continue
                summary.append({'group':g,'status':status,'view':view,'crossed_world_seed_cells':len(selected),
                    **{k:float(np.mean([r[k] for r in selected])) for k in keys},
                    'positive_wrong_order_cells':sum(r['wrong_order_gap']>0 for r in selected)})
    write_csv(root/'all_views.csv',summary)
    primary=[r for r in summary if r['status']=='trained' and r['view']=='hidden']
    p=next(r for r in primary if r['group']=='P');controls=[r for r in primary if r['group']!='P']
    predictions={'P_lower_generator_than_both_mixed_groups':all(p['generator_nmse']<r['generator_nmse'] for r in controls),
        'P_lower_cross_prediction_than_both_mixed_groups':all(p['cross_nmse']<r['cross_nmse'] for r in controls),
        'P_correct_cross_beats_wrong_order_on_average':p['wrong_order_gap']>0}
    atomic_json(root/'verified_summary.json',{'reported_utc':datetime.now(timezone.utc).isoformat(),'primary':primary,
        'all_views':summary,'fixed_predictions':predictions,'scope':protocol['scope'],
        'source_worlds':config['world_seeds'],'source_model_seeds':config['model_seeds'],
        'independent_source_seeds':3,'crossed_world_seed_cells':9,
        'source_inputs_all_previously_seen':625,'probe_split_rows':[375,125,125],
        'all_125_polynomials_seen_in_other_contexts':True,'test_context':int(ctx[4]),'new_source_training':False})
    atomic_json(root/'verification.json',{'status':'passed','verified_utc':datetime.now(timezone.utc).isoformat(),
        **counts,'all_action_images_stay_in_same_probe_context_split':True,
        'only_three_generators_fitted':True,'source_checkpoint_sha256':hashes,
        'exact_P_task_code_composite_errors_below':1e-8,'fixed_predictions':predictions})
    body='<h1>补充：求导与截断的交叉关系</h1><p>复用有限域模型，把四个潜在坐标解释为二次多项式的三个系数与保持不变的上下文。D P₁=P₀ D；错误顺序 D 后 P₁ 一般不等于 P₁ 后 D。只拟合 D、P₁、P₀ 的三个仿射探针，复合词由生成元复合得到。</p>'
    body+='<p>范围：没有新增神经网络训练；各模型此前已见全部 625 个源输入。探针按上下文划分 375/125/125，算子保持上下文，拟合、验证和测试的输入及其算子像互不重叠。但全部 125 种多项式系数在其它上下文中已见，这不是未见多项式或独立领域泛化。3 个模型种子与 3 个源世界交叉，共 9 个条件，不能当作 9 个独立源种子。</p>'
    body+=table(primary,[('group','源组'),('generator_nmse','生成元误差'),('cross_nmse','先截断后求导误差'),('other_cross_nmse','先求导后截断误差'),('cross_consistency_nmse','等价词一致性误差'),('wrong_order_gap','错误顺序减正确顺序'),('positive_wrong_order_cells','正顺序差条件数/9')])
    body+='<p>测量前保存的预测：</p>'+table([{'prediction':k,'supported':v} for k,v in predictions.items()],[('prediction','预测'),('supported','支持')])
    body+='<p>预定的 PCA 32 维主分析没有确认 P 组的整体优势：P 没有优于全部混合任务对照，平均生成元误差也没有优于匹配随机模型。完整维度敏感性另列于表中，部分误差下降，不能用它替换预定主分析。等价词的一致性低误差须和真实像预测同时看；恒等变换与收缩映射也可有低一致性误差。精确任务 one-hot 编码构成可恢复的正对照，但不是神经网络发现。</p>'
    body+=table(summary,[('group','源组'),('status','模型'),('view','空间'),('generator_nmse','生成元误差'),('generator_shuffled_nmse','错误配对误差'),('cross_nmse','未拟合交叉误差'),('cross_shuffled_nmse','错误配对交叉误差'),('cross_consistency_nmse','交叉一致性误差'),('wrong_order_gap','顺序判别差')])
    body+='<p><a href="verified_summary.json">核验后汇总</a> · <a href="verification.json">矩阵重算核验</a> · <a href="protocol.json">事前协议</a> · <a href="endpoints.csv">逐条件结果</a> · <a href="../report.html">PermWorld 复核</a></p>'
    css='body{font:16px system-ui;max-width:1300px;margin:32px auto;padding:0 18px}p{line-height:1.6}table{border-collapse:collapse;font-size:13px;margin:20px 0}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}'
    (root/'report.html').write_text(f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>Polynomial operator assay</title><style>{css}</style>{body}</html>')
    print(json.dumps({'verification':counts,'primary':primary,'predictions':predictions},indent=2))


if __name__=='__main__':run()
