"""Fixed exploratory interaction kernels of learned categorical source heads.

All degrees and direct-input/random-source-head controls are reported. This
supplies a generic factor-interaction hypothesis class; it is not ordinary
hidden-feature neural transfer and uses no target coefficient in fitting.
"""
from datetime import datetime, timezone
from itertools import product
import json
from math import comb
from pathlib import Path

import numpy as np
import torch

from .algebra import composition_size, world
from .field_symmetry import group_sources
from .field_symmetry_transfer import splits
from .longrun_engine import atomic_json
from .matched_field import controlled_world
from .models import SourceModel
from .permworld_combinations import sha
from .six_hour_report import write_rows


def interaction_kernel(left, right, degree):
    """Mean of categorical probability feature kernels for degrees 0..r."""
    similarities=np.einsum('nfp,mfp->fnm',left,right)
    coefficients=[np.ones((len(left),len(right)))]+[np.zeros((len(left),len(right))) for _ in range(degree)]
    for similarity in similarities:
        for k in range(degree,0,-1):coefficients[k]+=coefficients[k-1]*similarity
    return sum(coefficients[k]/comb(left.shape[1],k) for k in range(degree+1))/(degree+1)


def ridge_readout(factors, support_labels, support_ids, test_ids, degree, p, alpha):
    train=factors[support_ids];test=factors[test_ids]
    kernel=interaction_kernel(train,train,degree)
    weights=np.linalg.solve(kernel+alpha*np.eye(len(train)),np.eye(p)[support_labels]-1/p)
    return (interaction_kernel(test,train,degree)@weights+1/p).argmax(-1)


def probabilities(h, weights, bias, p):
    logits=(h@weights.T+bias).reshape(len(h),4,p)
    values=np.exp(logits-logits.max(-1,keepdims=True))
    return values/values.sum(-1,keepdims=True)


def run():
    torch.set_num_threads(4)
    root=Path('results/factor_kernel_readout');root.mkdir(exist_ok=True)
    signature={'analysis_sha256':sha(__file__), 'alpha':1., 'degrees':[1,2,3,4],
               'status':'post hoc exploratory after ordinary neural and categorical-subset outcomes are known; before these kernel endpoints',
               'encodings':['learned source-head probabilities','initial source-head probabilities','direct physical input one-hot factors'],
               'selection':'all degrees, targets, budgets, worlds, initializations and groups; no test-based policy selection',
               'scope':'transductive new-task labels; generic factor-interaction hypothesis class supplied to readout',
               'fit_inputs':'source probabilities and target support labels only; no target coefficients, composition orders or field arithmetic'}
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['signature']==signature
    else:atomic_json(path,{'registered_utc':datetime.now(timezone.utc).isoformat(),'signature':signature})
    all_rows=[]
    for source_path,transfer_path,provider in (('configs/field_symmetry.json','configs/field_symmetry_transfer.json',world),
                                              ('configs/field_matched_support.json','configs/field_matched_support_transfer.json',controlled_world)):
        source=json.loads(Path(source_path).read_text());transfer=json.loads(Path(transfer_path).read_text())
        cohort=Path(source['output']).name;targets=np.array(transfer['targets']);p=source['p']
        for w,m in product(source['world_seeds'],source['model_seeds']):
            inputs,_,encoded,basis=provider(p,source['dimension'],w)
            latent=inputs@basis.T % p;labels=latent@targets.T % p
            torch.manual_seed(m);initial=SourceModel(encoded.shape[1],source['hidden'],source['features'],p)
            with torch.no_grad():random_logits=initial(torch.tensor(encoded)).numpy().astype(np.float64)
            random_values=np.exp(random_logits-random_logits.max(-1,keepdims=True));random_values/=random_values.sum(-1,keepdims=True)
            designs=[('random_heads',random_values),('physical_factors',np.eye(p)[inputs])]
            for group in source['groups']:
                record=json.loads(Path(source['output']).joinpath(f'{group}_w{w}_m{m}.json').read_text())
                assert record['status']=='complete' and record['source_gate_passed']
                checkpoint=Path(source['output'])/'checkpoints'/f'{group}_w{w}_m{m}.pt'
                assert sha(checkpoint)==record['checkpoint_sha256']
                state=torch.load(checkpoint,map_location='cpu',weights_only=True)
                h=np.load(Path(source['output'])/f'{group}_w{w}_m{m}_step{source["steps"]}_features.npy')
                factors=probabilities(h,state['heads.weight'].numpy().astype(np.float64),state['heads.bias'].numpy().astype(np.float64),p)
                designs.append((group,factors))
            for group,factors in designs:
                output=root/f'{cohort}_{group}_w{w}_m{m}.json'
                if output.exists():
                    cached=json.loads(output.read_text());assert cached['protocol']==signature
                    all_rows.extend(cached['rows']);continue
                rows=[]
                for t in range(len(targets)):
                    supports,test=splits(labels[:,t],p,w+90000+t,transfer['budgets'],transfer['test_per_class'])
                    order=composition_size(group_sources(group,p),targets[t],p) if group in source['groups'] else None
                    for budget,degree in product(transfer['budgets'],signature['degrees']):
                        support=supports[budget]
                        prediction=ridge_readout(factors,labels[support,t],support,test,degree,p,signature['alpha'])
                        rows.append({'cohort':cohort,'group':group,'world_seed':w,'model_seed':m,'target_id':t,
                                     'budget':budget,'degree':degree,'target_composition_size':order,
                                     'target_physical_complexity':int(np.count_nonzero(targets[t]@basis % p)),
                                     'test_accuracy':float(np.mean(prediction==labels[test,t]))})
                atomic_json(output,{'status':'complete','protocol':signature,'rows':rows});all_rows.extend(rows)
            print(json.dumps({'cohort':cohort,'world_seed':w,'model_seed':m,'kernel_models_complete':len(designs)}),flush=True)
    write_rows(root/'endpoints.csv',all_rows)
    summarize(root,all_rows,signature)


def summarize(root,rows,protocol):
    lookup={(r['cohort'],r['group'],r['world_seed'],r['model_seed'],r['target_id'],r['budget'],r['degree']):r for r in rows}
    assert len(lookup)==len(rows)
    paired=[]
    for row in rows:
        if row['group'] in ('random_heads','physical_factors'):continue
        common=(row['cohort'],row['world_seed'],row['model_seed'],row['target_id'],row['budget'],row['degree'])
        random=lookup[(common[0],'random_heads')+common[1:]]['test_accuracy']
        physical=lookup[(common[0],'physical_factors')+common[1:]]['test_accuracy']
        paired.append({**row,'random_head_accuracy':random,'physical_factor_accuracy':physical,
                       'gain_over_random_heads':row['test_accuracy']-random,'gain_over_physical_factors':row['test_accuracy']-physical})
    summaries=[]
    for cohort,budget,degree in product(sorted({r['cohort'] for r in rows}),(25,50),protocol['degrees']):
        for order in (2,3,4):
            cell=[r for r in paired if (r['cohort'],r['budget'],r['degree'],r['target_composition_size'])==(cohort,budget,degree,order)]
            if cell:summaries.append({'cohort':cohort,'budget':budget,'degree':degree,'composition_order':order,'cells':len(cell),
                                      'source_accuracy':float(np.mean([r['test_accuracy'] for r in cell])),
                                      'physical_accuracy':float(np.mean([r['physical_factor_accuracy'] for r in cell])),
                                      'gain_over_physical_factors':float(np.mean([r['gain_over_physical_factors'] for r in cell]))})
    write_rows(root/'paired_gains.csv',paired);write_rows(root/'order_summary.csv',summaries)
    atomic_json(root/'summary.json',{'status':'complete','endpoints':len(rows),'paired_source_endpoints':len(paired),
                                    'all_degrees_retained':True,'no_target_test_selection':True,
                                    'registered_protocol':protocol,'order_summary':summaries})
    plot(root,summaries)
    table=''.join(f"<tr><td>{r['cohort']}</td><td>{r['budget']}</td><td>{r['degree']}</td><td>{r['composition_order']}</td><td>{100*r['source_accuracy']:.1f}%</td><td>{100*r['physical_accuracy']:.1f}%</td><td>{100*r['gain_over_physical_factors']:+.1f} pp</td></tr>" for r in summaries)
    (root/'report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>通用源因子交互核读出</title><style>body{{max-width:1200px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}aside{{background:#f2f4f7;padding:16px}}</style>
<h1>固定的通用因子交互核：检查读出假设的作用</h1><p>这是在普通神经读出及类别查表结果之后追加的事后探索，{len(rows)} 个端点全部保留。四个源头的五分类概率形成四个因子；核同时包含截至一、二、三或四阶的所有因子交互，岭惩罚固定为 1。拟合只使用源头概率与目标支持标签，不输入目标系数、组合阶数或有限域运算。</p>
<p>三个对照表示为已训练源头概率、相同初始化的未训练源头概率，以及原始四个物理输入的类别独热因子；它们使用相同的目标支持/测试、交互次数和岭惩罚。未训练概率因子的信息几何与硬类别因子不同，因此直接物理因子是额外的实际参照。所有输入在源预训练中均已暴露。</p><img src="factor_kernel.png">
<table><tr><th>源实验</th><th>标签</th><th>核最高阶</th><th>实际目标组合阶数</th><th>训练源因子</th><th>物理输入因子</th><th>差值</th></tr>{table}</table>
<aside>交互核是额外提供给目标读出的类别因子假设类。任何收益都不能替代原始隐藏层线性/MLP/微调的结论，也不能称为模型自发学会代数组合。数学组合阶数仅在评估后分组，目标及其物理系数支撑在不同阶数中组成不同。三个输入基与三个初始化交叉，物理因子对照在不同初始化间重复，不能当成独立数据集。没有按测试选择最好的核阶数。</aside>
<p><a href="protocol.json">追加分析协议</a> · <a href="endpoints.csv">所有源与控制端点</a> · <a href="paired_gains.csv">严格配对差值</a> · <a href="order_summary.csv">全部阶数组合</a> · <a href="factor_kernel.pdf">PDF 图</a></p></html>''')
    print(json.dumps({'endpoints':len(rows),'paired_source_endpoints':len(paired)}))


def plot(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for i,cohort in enumerate(('field_symmetry','field_matched_support')):
        for j,budget in enumerate((25,50)):
            ax=axes[i,j]
            for order in (2,3,4):
                cell=sorted([r for r in rows if (r['cohort'],r['budget'],r['composition_order'])==(cohort,budget,order)],key=lambda r:r['degree'])
                ax.plot([r['degree'] for r in cell],[r['source_accuracy'] for r in cell],'o-',label=f'target order {order}')
            ax.axhline(.2,color='gray',linestyle=':');ax.set(xticks=[1,2,3,4],xlabel='Maximum interaction degree',ylabel='Target accuracy',title=f'{cohort}; {budget} labels',ylim=(0,1.04));ax.legend(fontsize=8)
    for suffix in ('png','pdf'):fig.savefig(root/f'factor_kernel.{suffix}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
