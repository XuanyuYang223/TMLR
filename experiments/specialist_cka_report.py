"""Readable report of all frozen specialist controls and the small ablation."""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np

from .analysis import write_csv
from .longrun_engine import atomic_json

ROOT=Path('results/specialist_cka_controls')
ADJUST=Path('results/specialist_input_adjustment')
ABLATION=Path('results/frozen_operator_constraint_ablation')
NAMES={'descent_inverse':'下降数—逆下降数（取逆）','peak_complement':'峰—谷（补排列）',
    'fixed_complement':'不动点—反不动点（补排列）','exceedance_inverse':'超越数—不足数（取逆）',
    'subsequence_complement':'LIS—LDS（补排列）','run_complement':'递增段—递减段（补排列）',
    'record_complement':'左向最大记录—最小记录（补排列）','decomposition_complement':'全局下降—分解块（补排列）'}


def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


def stats(rows,key):
    values=[float(r[key]) for r in rows if r[key] not in ['',None]]
    if not values:return {'count':0,'mean':None,'positive':0}
    return {'count':len(values),'mean':float(np.mean(values)),'min':min(values),'max':max(values),'positive':sum(v>0 for v in values)}


def table(rows,columns):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(title)+'</th>' for key,title in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(row.get(key,'')))+'</td>' for key,title in columns)+'</tr>' for row in rows)+'</tbody></table>'


def fmt(x):return '未定义' if x is None else f'{x:+.4f}'


def run():
    contrasts=read_csv(ROOT/'contrasts.csv');null=read_csv(ROOT/'conditional_null.csv')
    adjusted=read_csv(ADJUST/'per_source_contrasts.csv');ablation=read_csv(ABLATION/'per_source_results.csv')
    protocol=json.loads((ROOT/'protocol.json').read_text());sig=protocol['signature'];rels=[r['pair_id'] for r in sig['relations']]
    summary={'scope':sig['plan']['inference_scope'],'primary_position':'task-free ONE_END, final_norm',
        'test_anchors':2688,'original_source_checkpoints':48,'source_seeds':[17,42,101],'source_retraining':0,
        'core_protocol_registered_utc':protocol['registered_utc'],'controls':{},'initialization_increments':{},'conditional_permutation_controls':{}}
    for branch in ['prefix_final_norm','query_final_norm','output_prob','answer_onehot','answer_scalar']:
        summary['controls'][branch]={}
        for control in ['raw','length','answer_strata','confidence']:
            summary['controls'][branch][control]={cond:stats([r for r in contrasts if r['branch']==branch and r['control']==control and r['condition']==cond],'correct_minus_wrong') for cond in ['trained','matched_init','independent_init']}
    for branch in ['prefix_final_norm','query_final_norm']:
        summary['initialization_increments'][branch]={}
        for control in ['raw','length','answer_strata','confidence']:
            rows=[r for r in contrasts if r['branch']==branch and r['control']==control]
            for cond in ['matched_init','independent_init']:
                result=[]
                for x in rows:
                    if x['condition']!='trained':continue
                    y=next(y for y in rows if y['condition']==cond and y['relation']==x['relation'] and y['seed']==x['seed'])
                    result.append({'relation':x['relation'],'seed':x['seed'],'increment':float(x['correct_minus_wrong'])-float(y['correct_minus_wrong'])})
                summary['initialization_increments'][branch][control+'_'+cond]=stats(result,'increment')
        a=[r for r in null if r['branch']==branch and r['condition']=='trained']
        summary['conditional_permutation_controls'][branch]={'shuffle_contrast':stats(a,'shuffle_mean'),
            'excess_above_shuffle_mean':stats(a,'contrast_minus_shuffle_mean'),
            'above_all_32_shuffles':sum(float(x['observed_contrast'])>float(x['shuffle_max']) for x in a)}
    summary['supplemental_input_adjustment']={'scope':json.loads((ADJUST/'protocol.json').read_text())['signature']['plan']['registration_scope'],'controls':{}}
    for branch in ['prefix_final_norm','query_final_norm']:
        summary['supplemental_input_adjustment']['controls'][branch]={}
        for control in ['length','answer_strata']:
            selected=[r for r in adjusted if r['branch']==branch and r['control']==control]
            summary['supplemental_input_adjustment']['controls'][branch][control]={co:stats([r for r in selected if r['condition']==co],'input_adjusted_correct_minus_wrong') for co in ['trained','matched_init','independent_init']}
            summary['supplemental_input_adjustment']['controls'][branch][control]['ordinary_per_length']={co:stats([r for r in selected if r['condition']==co],'ordinary_correct_minus_wrong') for co in ['trained','matched_init','independent_init']}
            for cond in ['matched_init','independent_init']:
                diffs=[];ordinary_diffs=[]
                for x in selected:
                    if x['condition']=='trained':
                        y=next(y for y in selected if y['relation']==x['relation'] and y['seed']==x['seed'] and y['condition']==cond)
                        diffs.append({'increment':float(x['input_adjusted_correct_minus_wrong'])-float(y['input_adjusted_correct_minus_wrong'])})
                        ordinary_diffs.append({'increment':float(x['ordinary_correct_minus_wrong'])-float(y['ordinary_correct_minus_wrong'])})
                summary['supplemental_input_adjustment']['controls'][branch][control]['increment_vs_'+cond]=stats(diffs,'increment')
                summary['supplemental_input_adjustment']['controls'][branch][control]['ordinary_increment_vs_'+cond]=stats(ordinary_diffs,'increment')
    summary['constraint_ablation']={}
    for method in dict.fromkeys(x['method'] for x in ablation):
        summary['constraint_ablation'][method]={word:{key:stats([x for x in ablation if x['method']==method and x['word']==word],key) for key in ['answer_accuracy','both_correct','neither_correct','orientation_excess','displacement_nmse']} for word in ['ci','ici']}
        summary['constraint_ablation'][method]['word_average']={key:stats([x for x in ablation if x['method']==method],key)['mean'] for key in ['answer_accuracy','both_correct','displacement_nmse']}
    per_relation=[];per_seed=[]
    def cell(rel,seed,branch,control,condition='trained'):
        return next(x for x in contrasts if x['relation']==rel and x['seed']==str(seed) and x['branch']==branch and x['control']==control and x['condition']==condition)
    for rel in rels:
        audit=json.loads((ROOT/'evaluations'/f'{rel}_s17.json').read_text())['stratum_audit']
        raw=[float(cell(rel,s,'prefix_final_norm','raw')['correct_minus_wrong']) for s in [17,42,101]]
        matched=[float(cell(rel,s,'prefix_final_norm','answer_strata')['correct_minus_wrong']) for s in [17,42,101]]
        init=[float(cell(rel,s,'prefix_final_norm','answer_strata','matched_init')['correct_minus_wrong']) for s in [17,42,101]]
        out=[float(cell(rel,s,'output_prob','answer_strata')['correct_minus_wrong']) for s in [17,42,101]]
        adj=[float(x['input_adjusted_correct_minus_wrong']) for x in adjusted if x['relation']==rel and x['branch']=='prefix_final_norm' and x['control']=='answer_strata' and x['condition']=='trained']
        adj_init=[float(x['input_adjusted_correct_minus_wrong']) for x in adjusted if x['relation']==rel and x['branch']=='prefix_final_norm' and x['control']=='answer_strata' and x['condition']=='matched_init']
        per_relation.append({'relation':rel,'name':NAMES[rel],'matched_anchors':audit['retained'],
            'raw_prefix_mean':float(np.mean(raw)),'answer_matched_prefix_mean':float(np.mean(matched)),
            'answer_matched_prefix_seed17':matched[0],'answer_matched_prefix_seed42':matched[1],'answer_matched_prefix_seed101':matched[2],
            'answer_matched_increment_vs_matched_init_mean':float(np.mean(np.array(matched)-init)),
            'answer_matched_output_mean':float(np.mean(out)),'per_length_input_adjusted_prefix_mean':float(np.mean(adj)),
            'per_length_input_adjusted_increment_vs_matched_init_mean':float(np.mean(adj)-np.mean(adj_init))})
        for s in [17,42,101]:
            row={'relation':rel,'seed':s,'matched_anchors':audit['retained']}
            for view in ['prefix_final_norm','query_final_norm','output_prob']:
                for control in ['raw','length','answer_strata']:
                    row[view+'_'+control]=float(cell(rel,s,view,control)['correct_minus_wrong'])
            row['prefix_matched_init']=float(cell(rel,s,'prefix_final_norm','answer_strata','matched_init')['correct_minus_wrong'])
            row['prefix_independent_init']=float(cell(rel,s,'prefix_final_norm','answer_strata','independent_init')['correct_minus_wrong'])
            per_seed.append(row)
    write_csv(ROOT/'per_relation_summary.csv',per_relation);write_csv(ROOT/'per_relation_seed_summary.csv',per_seed)
    summary['per_relation']=per_relation
    accuracy=read_csv(ROOT/'source_accuracy.csv');task_accuracy=[]
    for task in dict.fromkeys(x['task'] for x in accuracy):
        values=[float(x['test_numeric_accuracy']) for x in accuracy if x['task']==task]
        task_accuracy.append({'task':task,'mean':float(np.mean(values)),'min':min(values),'max':max(values)})
    summary['source_task_accuracy']=task_accuracy
    atomic_json(ROOT/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    for ax,branch,title in zip(axes,['prefix_final_norm','query_final_norm'],['Pooled lengths: task-free ONE_END','Pooled lengths: task query (=)']):
        for co,color,offset in [('trained','#2166ac',-.15),('matched_init','#b35806',0),('independent_init','#777777',.15)]:
            for j,rel in enumerate(rels):
                values=[float(x['correct_minus_wrong']) for x in contrasts if x['relation']==rel and x['branch']==branch and x['control']=='answer_strata' and x['condition']==co]
                ax.scatter([j+offset]*3,values,color=color,s=18,alpha=.65)
                ax.scatter(j+offset,np.mean(values),color=color,marker='D',s=35,label=co if j==0 else None)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(8),[r.replace('_','\n') for r in rels],fontsize=7)
        ax.set(title=title,ylabel='Correct - wrong CKA, common answer strata');ax.legend(fontsize=8)
    for ext in ['png','pdf']:fig.savefig(ROOT/f'answer_matched_cka.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    for ax,branch,title in zip(axes,['prefix_final_norm','query_final_norm'],['Per-length input adjustment: ONE_END','Per-length input adjustment: query']):
        for co,color,offset in [('trained','#2166ac',-.15),('matched_init','#b35806',0),('independent_init','#777777',.15)]:
            for j,rel in enumerate(rels):
                values=[float(x['input_adjusted_correct_minus_wrong']) for x in adjusted if x['relation']==rel and x['branch']==branch and x['control']=='answer_strata' and x['condition']==co]
                ax.scatter([j+offset]*3,values,color=color,s=18,alpha=.65)
                ax.scatter(j+offset,np.mean(values),color=color,marker='D',s=35,label=co if j==0 else None)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(8),[r.replace('_','\n') for r in rels],fontsize=7)
        ax.set(title=title,ylabel='Correct - wrong, mean of 21 lengths');ax.legend(fontsize=8)
    for ext in ['png','pdf']:fig.savefig(ROOT/f'per_length_input_adjustment.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    methods=list(summary['constraint_ablation'])
    for i,word in enumerate(['ci','ici']):
        for j,m in enumerate(methods):
            values=[100*float(x['answer_accuracy']) for x in ablation if x['method']==m and x['word']==word]
            ax.scatter([j+(i-.5)*.16]*3,values,color=['#2166ac','#b35806'][i],alpha=.65,label=word.upper() if j==0 else None)
    ax.axhline(50,color='black',ls='--',lw=.8);ax.set_xticks(range(5),[m.replace('_','\n') for m in methods],fontsize=8)
    ax.set(ylabel='Collision answer accuracy (%)',title='Frozen-source constraint ablation, 3 source seeds');ax.legend()
    for ext in ['png','pdf']:fig.savefig(ROOT/f'constraint_ablation.{ext}',dpi=180)
    plt.close(fig)
    relation_display=[{'name':r['name'],'anchors':r['matched_anchors'],'raw':fmt(r['raw_prefix_mean']),
        'seeds':' / '.join(fmt(r[f'answer_matched_prefix_seed{s}']) for s in [17,42,101]),
        'increment':fmt(r['answer_matched_increment_vs_matched_init_mean']),'output':fmt(r['answer_matched_output_mean']),
        'adjust':fmt(r['per_length_input_adjusted_prefix_mean']),
        'adjust_increment':fmt(r['per_length_input_adjusted_increment_vs_matched_init_mean'])} for r in per_relation]
    global_display=[]
    for branch,name in [('prefix_final_norm','任务前 ONE_END'),('query_final_norm','查询位置'),('output_prob','31 类数值输出'),('answer_onehot','真实答案 onehot')]:
        for control,cname in [('raw','原始'),('length','长度中心化'),('answer_strata','共同答案分层'),('confidence','FIT 置信度回归')]:
            val=summary['controls'][branch][control]['trained']
            if val['count'] or control=='answer_strata':global_display.append({'branch':name,'control':cname,'mean':fmt(val['mean']),'positive':f"{val['positive']}/{val['count']}" if val['count'] else '常量，CKA 未定义'})
    ab_display=[]
    for method in methods:
        row={'method':method}
        for word in ['ci','ici']:
            row[word]=f"{100*summary['constraint_ablation'][method][word]['answer_accuracy']['mean']:.2f}%"
            row[word+'_both']=f"{100*summary['constraint_ablation'][method][word]['both_correct']['mean']:.2f}%"
        ab_display.append(row)
    seed_display=[{'relation':NAMES[r['relation']],'seed':r['seed'],'prefix_raw':fmt(r['prefix_final_norm_raw']),
        'prefix_length':fmt(r['prefix_final_norm_length']),'prefix_answer':fmt(r['prefix_final_norm_answer_strata']),
        'query_answer':fmt(r['query_final_norm_answer_strata']),'output_answer':fmt(r['output_prob_answer_strata']),
        'init':fmt(r['prefix_matched_init'])} for r in per_seed]
    adj=summary['supplemental_input_adjustment']['controls']['prefix_final_norm']['answer_strata']['trained']
    length_comparison=[]
    for branch,label in [('prefix_final_norm','任务前 ONE_END'),('query_final_norm','查询位置')]:
        values=summary['supplemental_input_adjustment']['controls'][branch]['answer_strata']
        for kind,title in [('ordinary_per_length','逐长度普通 CKA'),('adjusted','逐长度输入调整 CKA')]:
            co=values[kind] if kind=='ordinary_per_length' else values
            inc=values['ordinary_increment_vs_matched_init'] if kind=='ordinary_per_length' else values['increment_vs_matched_init']
            length_comparison.append({'branch':label,'method':title,'trained':fmt(co['trained']['mean']),
                'matched':fmt(co['matched_init']['mean']),'independent':fmt(co['independent_init']['mean']),
                'increment':fmt(inc['mean']),'positive':f"{inc['positive']}/{inc['count']}"})
    summary['interpretation']={'original_correct_transform_signal_reproduced':True,
        'pooled_answer_matched_signal_positive':True,
        'uniform_training_added_relation_geometry_supported':False,
        'reason':'Per-length ordinary and invariant-input-adjusted CKA both exceed matched initialization in only 6/24 answer-matched cells, all six from the two inverse relations. Complement signals also occur strongly in random networks. Pooled length centering does not equal per-length CKA averaging.',
        'output_independent_algebraic_mechanism_confirmed':False,'stable_hidden_composition_confirmed':False}
    atomic_json(ROOT/'summary.json',summary)
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>原始跨模型 CKA：新测试与答案控制</title>
<style>body{{max-width:1250px;margin:36px auto;padding:0 22px;font:16px/1.7 system-ui;color:#202733}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}}th,td{{border:1px solid #d7dde5;padding:7px;text-align:left}}th{{background:#edf2f7}}img{{max-width:100%}}.note{{padding:14px;background:#f1f5f9;border-left:4px solid #64748b}}a{{color:#2166ac}}</style>
<h1>回到原始跨模型 CKA：新测试与答案控制</h1>
<p><strong>原始 CKA 现象复现，但初始化与逐长度控制揭示了关键边界。</strong>八条关系、三个种子，48 个已有单性质检查点；没有重新训练源模型。混合长度的正确变换优势为 24/24 正值，答案匹配后仍然如此；然而逐长度分别计算后，训练模型仅有 6/24 个答案匹配单元超过初始化，全部来自两条取逆关系。六条补排列关系在随机网络中也有更强的对齐，不能统一解释为训练学会了关系。</p>
<p>主要量为 ΔCKA = CKA(正确变换) − CKA(原协议错误变换)。错误对照保持原来的选择：取逆关系对照补排列，补排列关系对照取逆；未变换输入另行报告。条件与权重相同，只改变数学对齐。</p>
<p>核心比较规则于 {protocol['registered_utc']} 保存，早于新数据、表征与 CKA。新的 2,688 个测试锚点，另有 672 FIT 和 336 验证锚点。已扫描原始 1,600 万条父数据输入；候选的全部八个轨道状态均排除原始数据及此前本地输入。输入读取器在初次注册后、任何数据选择和模型推断之前修正了紧凑 JSON 格式；失败注册保留，随后重新保存协议。</p>
<h2>每条关系与所有种子</h2>
<p>共同分层键为长度、左侧未变换答案、右侧 e/C/I 三个答案；保留至少三个锚点的分层，各分支在同一分层内减均值。所有条件使用完全相同的保留行。确定性真实答案编码在每个分层中变成常量，所以条件 CKA 未定义。最后一列是另行补充的逐长度输入调整均值，不能与 pooled CKA 当作同一统计量直接相减。</p>
{table(relation_display,[('name','关系'),('anchors','匹配锚点'),('raw','原始 ONE_END Δ 均值'),('seeds','混合长度答案匹配 Δ：17 / 42 / 101'),('increment','混合长度相对初始化增量'),('output','匹配后输出 Δ 均值'),('adjust','逐长度输入调整 Δ'),('adjust_increment','逐长度调整相对初始化增量')])}
<img src="answer_matched_cka.png"><p><a href="answer_matched_cka.pdf">下载 PDF 图</a> · <a href="per_relation_seed_summary.csv">逐关系 × 种子 CSV</a> · <a href="contrasts.csv">所有分支与控制对照</a> · <a href="scores.csv">三个变换的 CKA 原值</a></p>
{table(seed_display,[('relation','关系'),('seed','种子'),('prefix_raw','ONE_END 原始 Δ'),('prefix_length','长度中心化 Δ'),('prefix_answer','答案匹配 Δ'),('query_answer','查询匹配 Δ'),('output_answer','输出匹配 Δ'),('init','同种子初始化匹配 Δ')])}
<h2>信号体现在哪里</h2>
{table(global_display,[('branch','分支'),('control','控制'),('mean','Δ 均值'),('positive','正值单元')])}
<p>混合长度的任务前表征在答案匹配后平均 Δ 为 {summary['controls']['prefix_final_norm']['answer_strata']['trained']['mean']:.4f}，其中 23/24 高于同种子初始化；独立种子初始化对照也为 23/24。训练模型的 24/24 个条件对照都高于事先固定的 32 个组内打乱对照。打乱保留真实答案与分层，却破坏具体输入对应。这说明信号不只是确定性真实答案编码，但尚不能区分训练得到的关系结构与输入对应的自然几何。下方逐长度控制改变了初始化比较的结论。这些单元共享任务、输入与种子，不能当成 24 个独立试验。</p>
<p>真实答案与模型输出分布本身能复现很大的原始正确变换优势。因此原始 CKA 不能直接证明内部代数机制。输出条件几何仍可能包含置信度与其他信息，隐藏 Δ 高于输出 Δ 也不是输出独立性的证明。固定 FIT-only 置信度回归是补充分析，不能作为因果中介分解。原始与匹配分析的样本集合及中心化方式不同，其下降幅度不能直接解释为答案解释了多少比例。</p>
<h2>输入结构控制与补充去混淆分析</h2>
<p>输入编码基线显示 CKA 对编码选择敏感，例如长度中心化的数值位置输入对补排列 CKA=1，而置换矩阵编码对取逆 CKA=1。这也是加入初始化与组内打乱控制的原因。完整输入基线见 <a href="input_baselines.csv">CSV</a>。</p>
<p>另在每个长度上使用位置置换矩阵的线性输入核。该输入 Gram 在 e/C/I 下严格相同，包括共同答案中心化之后。按 <a href="https://proceedings.neurips.cc/paper_files/paper/2022/file/79cbf4f96c2bcc67267421154da689dd-Paper-Conference.pdf">Cui 等 2022</a> 的回归与 PSD 投影思路去除这一输入核，然后分别报告每个长度与三个种子。此补充方案在查看核心 CKA 与输入基线后、计算新调整结果前保存，采用已中心化的核；不声称复现论文全部预处理，也不声称排除了所有输入混淆。</p>
<p>共同答案条件下，任务前输入调整 Δ 的逐长度均值为 {adj['mean']:.4f}，正值 {adj['positive']}/{adj['count']}；详见 <a href="../specialist_input_adjustment/per_source_contrasts.csv">逐关系种子</a>、<a href="../specialist_input_adjustment/per_length_scores.csv">逐长度原值</a>、<a href="../specialist_input_adjustment/protocol.json">补充协议</a>。</p>
{table(length_comparison,[('branch','位置'),('method','答案匹配后的统计方式'),('trained','训练 Δ 均值'),('matched','同种子初始化 Δ'),('independent','独立初始化 Δ'),('increment','训练相对同种子初始化增量'),('positive','增量正值')])}
<img src="per_length_input_adjustment.png"><p><a href="per_length_input_adjustment.pdf">逐长度对照 PDF 图</a></p>
<p><strong>逐长度的普通 CKA 与输入调整 CKA 都只有 6/24 高于初始化。</strong>这六个单元恰好是下降数—逆下降数和超越数—不足数两条取逆关系的三个种子；六条补排列关系则都低于初始化。独立初始化对照给出同样的 6/24。变化主要来自按长度分别计算再平均，而不是核回归本身：先减去长度均值、再把所有长度汇总为一项 CKA，并不等于对每个长度的 CKA 求平均。因此不能只报告混合长度的 23/24，再概括成普遍训练收益。取逆关系值得继续追踪，但仍需同一变换下的无关任务对照和更完整的输出控制来证明任务关系特异性。</p>
<h2>冻结源模型的组合约束消融</h2>
<p>另用三个已指定的新源种子 8111/9127/10141，冻结同一模型与原数值读出。拟合仅使用 e↔C 与 e↔I 已知边；超参数仅按可见状态验证误差选择，400 次固定更新。新碰撞测试含 1,344 对；每对三个已知答案相同，CI/ICI 隐藏答案不同。算子施加范数约束、仿射往返约束或两者联合；同时报告原始算子和同预算无约束重拟合，避免把额外优化预算误判为约束收益。借鉴 <a href="https://proceedings.mlr.press/v119/azencot20a/azencot20a.pdf">Azencot 等 2020</a> 一致性思想，但这里直接约束 C/I 的自反关系，没有训练 Koopman 自编码器。</p>
{table(ab_display,[('method','方法'),('ci','CI 准确率'),('ici','ICI 准确率'),('ci_both','CI 双正确'),('ici_both','ICI 双正确')])}
<img src="constraint_ablation.png"><p>双正确指碰撞对的两个输入都正确；CI/ICI 是分别计分的两个计算路径，不是双正确的两个成员。50% 标线只适用于此碰撞选择下的确定性已知标量答案预测器，不是神经网络的信息上限。往返约束相对相同预算无约束重拟合仅有极小增量；范数与联合约束在这组固定选择规则下变差。结论限于提供已知结构约束后的组合泛化。</p>
<p><a href="../frozen_operator_constraint_ablation/per_source_results.csv">所有种子 × 方法 × CI/ICI</a> · <a href="../frozen_operator_constraint_ablation/protocol.json">消融协议</a> · <a href="constraint_ablation.pdf">PDF 图</a></p>
<h2>学习程度、证据边界与核验</h2>
<p>原始源任务的学习程度不同，尤其 peaks 与 recoils 的长排列准确率偏低；本轮全部保留，没有按新结果筛选。各任务平均数是已提取变换条件与种子的描述汇总，并非平衡的同等学习程度对照。逐任务与条件准确率见 <a href="source_accuracy.csv">CSV</a>。</p>
{table([{'task':r['task'],'mean':f"{100*r['mean']:.2f}%",'min':f"{100*r['min']:.2f}%",'max':f"{100*r['max']:.2f}%"} for r in task_accuracy],[('task','源任务'),('mean','平均数值准确率'),('min','最低条件'),('max','最高条件')])}
<div class="note">本轮支持的具体结论是：正确数学变换能提高跨模型 CKA，但效应依赖关系类型、输入几何及长度汇总方式。两条取逆关系在逐长度、答案匹配与初始化控制下显示一致的训练增量；补排列的原始正信号不足以识别训练形成的结构。仍不能写成完整代数表示、输出独立机制，或稳定推断未见复合关系。所有结果均条件于这八条关系、三个种子、已有数据世界和指定分析方式。</div>
<p>已按独立样本 Gram 公式核验 1,944 个核心 CKA 原值；原始前向抽查所有前缀层和数值输出；消融 30 个端点、全部 48 个候选可见验证分数及冻结读出单独复核。核验数量代表覆盖面，不是独立效应复现。<a href="verification.json">CKA 核验</a> · <a href="extraction_reference_check.json">原始前向对照</a> · <a href="../frozen_operator_constraint_ablation/verification.json">消融核验</a> · <a href="../specialist_input_adjustment/verification.json">输入调整核验</a> · <a href="summary.json">机器可读汇总</a> · <a href="protocol.json">核心协议</a> · <a href="completion.json">完成与哈希记录</a></p>
</html>'''
    (ROOT/'report.html').write_text(report)
    print(json.dumps({'report':str(ROOT/'report.html'),'primary_answer_matched':summary['controls']['prefix_final_norm']['answer_strata']['trained'],
        'input_adjusted':adj,'ablation_mean':{m:summary['constraint_ablation'][m]['word_average'] for m in methods}},indent=2))


if __name__=='__main__':run()
