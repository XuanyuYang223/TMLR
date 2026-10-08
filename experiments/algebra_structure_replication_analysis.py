"""Separate blind source-seed results from the reused exploratory pilot."""
import csv
from datetime import datetime,timezone
import html
import json
from pathlib import Path

import numpy as np

from .algebra_noninvertible import polynomial_law_audit
from .longrun_engine import atomic_json
from .permworld_combinations import sha


def table(rows,columns):
    text='<table><tr>'+''.join(f'<th>{html.escape(label)}</th>' for _,label in columns)+'</tr>'
    for row in rows:
        text+='<tr>'+''.join('<td>'+html.escape(f'{row[k]:.4f}' if isinstance(row[k],float) else str(row[k]))+'</td>' for k,_ in columns)+'</tr>'
    return text+'</table>'


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def record_existing_checks(root):
    directory=Path('results/algebra_structure/direct/native')
    rows=[json.loads(p.read_text())['results']['source_query_concat'] for p in directory.glob('position_four_s*_trained.json')]
    records=json.loads(Path('results/algebra_structure/new_combinations/probes/novel_records_trained.json').read_text())['results']['source_query_concat']
    audit={'created_utc':datetime.now(timezone.utc).isoformat(),
        'identity_prediction':'identity predicts zero displacement; normalized squared displacement error is exactly 1 when true action is noninvariant',
        'position_four_inverse_displacement_energy_mean':float(np.mean([r['generators'][2]['action_displacement_energy'] for r in rows])),
        'position_four_inverse_displacement_nmse_mean':float(np.mean([r['generators'][2]['full_space_displacement_nmse'] for r in rows])),
        'position_four_two_inverse_latent_consistency_nmse_mean':float(np.mean([r['laws'][2]['consistency_nmse'] for r in rows])),
        'records_generator_shuffled_fit_nmse':[r['shuffled_fit_displacement_nmse'] for r in records['generators']],
        'records_composite_shuffled_fit_nmse':[r['shuffled_fit_displacement_nmse'] for r in records['composites']],
        'records_two_inverse_latent_consistency_nmse':records['laws'][2]['consistency_nmse'],
        'composition':'Only rho_c/rho_r/rho_i are regression fits. All four composites multiply these independent generator maps; no composite regression.',
        'row_convention':'first C then I: z rho_c rho_i predicts T_i(T_c(x))',
        'projection_correction':'The implemented projection fixes dimension and sets latent coordinate 3 to zero. Dimension-changing coordinate deletion does not generally satisfy P^2=P.',
        'polynomial_relations':polynomial_law_audit()}
    atomic_json(root/'existing_checks.json',audit);return audit


def run():
    plan=json.loads(Path('configs/algebra_structure_replication.json').read_text());root=Path(plan['output'])
    records=[json.loads(p.read_text()) for p in (root/'probes').glob('*.json') if not p.name.startswith('teacher_')]
    assert len(records)==24
    lookup={(r['group'],r['seed'],r['model_status']):r for r in records};rows=[];actions=[]
    for r in records:
        for view,result in r['results'].items():
            e=result['explicit_checks']
            rows.append({'group':r['group'],'seed':r['seed'],'model_status':r['model_status'],'role':r['replication_role'],
                'view':view,'source_accuracy':r['source_accuracy'],**e,
                'wrong_order_gap':result['wrong_order'][0]['gap_wrong_minus_correct'],
                'test_pca_energy_fraction':result['test_pca_energy_fraction']})
            for kind in ('generators','composites'):
                for a in result[kind]:
                    actions.append({'group':r['group'],'seed':r['seed'],'model_status':r['model_status'],
                        'role':r['replication_role'],'view':view,'kind':kind,'word':a.get('generator',a.get('word')),
                        'displacement_nmse':a['full_space_displacement_nmse'],
                        'shuffled_fit_displacement_nmse':a['shuffled_fit_displacement_nmse'],
                        'displacement_energy':a['action_displacement_energy'],'status':a['status']})
    write_csv(root/'endpoints.csv',rows);write_csv(root/'actions.csv',actions)
    summaries=[]
    for group in plan['groups']:
        for view in plan['views']+[v+'_full_width' for v in plan['full_width_sensitivity']]:
            trained=[lookup[group['id'],s,'trained']['results'][view] for s in plan['source_seeds']]
            random=[lookup[group['id'],s,'random']['results'][view] for s in plan['source_seeds']]
            mean=lambda rs,k:float(np.mean([r['explicit_checks'][k] for r in rs]))
            summaries.append({'group':group['id'],'view':view,'new_source_seeds':len(trained),
                'source_accuracy':float(np.mean([lookup[group['id'],s,'trained']['source_accuracy'] for s in plan['source_seeds']])),
                'generator_nmse':mean(trained,'generator_nmse'),'random_generator_nmse':mean(random,'generator_nmse'),
                'composite_nmse':mean(trained,'composite_nmse'),'random_composite_nmse':mean(random,'composite_nmse'),
                'generator_shuffled_nmse':mean(trained,'generator_shuffled_nmse'),
                'composite_shuffled_nmse':mean(trained,'composite_shuffled_nmse'),
                'two_inverse_reconstruction_nmse':mean(trained,'two_inverse_full_space_reconstruction_nmse'),
                'inverse_nmse':float(np.mean([r['generators'][2]['full_space_displacement_nmse'] for r in trained])),
                'random_inverse_nmse':float(np.mean([r['generators'][2]['full_space_displacement_nmse'] for r in random])),
                'inverse_displacement_energy':float(np.mean([r['generators'][2]['action_displacement_energy'] for r in trained])),
                'wrong_order_gap':float(np.mean([r['wrong_order'][0]['gap_wrong_minus_correct'] for r in trained])),
                'all_generators_better_than_identity_seeds':sum(all(a['full_space_displacement_nmse']<1 for a in r['generators']) for r in trained),
                'generator_better_than_random_seeds':sum(a['explicit_checks']['generator_nmse']<b['explicit_checks']['generator_nmse'] for a,b in zip(trained,random)),
                'composite_better_than_random_seeds':sum(a['explicit_checks']['composite_nmse']<b['explicit_checks']['composite_nmse'] for a,b in zip(trained,random))})
    write_csv(root/'group_summary.csv',summaries)
    primary=[r for r in summaries if r['view']==plan['primary_landmark']]
    rec=next(r for r in primary if r['group']=='novel_records');controls=[r for r in primary if r['group']!='novel_records']
    seed_comparisons=[]
    for s in plan['source_seeds']:
        r=lookup['novel_records',s,'trained']['results'][plan['primary_landmark']]
        vals={g['id']:lookup[g['id'],s,'trained']['results'][plan['primary_landmark']]['explicit_checks'] for g in plan['groups']}
        seed_comparisons.append({'seed':s,'records_generator_nmse':vals['novel_records']['generator_nmse'],
            'records_composite_nmse':vals['novel_records']['composite_nmse'],
            'generator_beats_both_controls':all(vals['novel_records']['generator_nmse']<vals[g]['generator_nmse'] for g in ('novel_minima_pair','novel_peak_mixed')),
            'composite_beats_both_controls':all(vals['novel_records']['composite_nmse']<vals[g]['composite_nmse'] for g in ('novel_minima_pair','novel_peak_mixed')),
            'positive_wrong_order_gap':r['wrong_order'][0]['gap_wrong_minus_correct']>0,
            'two_inverse_reconstruction_nmse':r['explicit_checks']['two_inverse_full_space_reconstruction_nmse']})
    hypotheses={
        'lower_mean_generator_than_both_controls':all(rec['generator_nmse']<r['generator_nmse'] for r in controls),
        'lower_mean_composite_than_both_controls':all(rec['composite_nmse']<r['composite_nmse'] for r in controls),
        'lower_generator_and_composite_than_identity_shuffled_random':rec['generator_nmse']<min(1.,rec['generator_shuffled_nmse'],rec['random_generator_nmse']) and rec['composite_nmse']<min(1.,rec['composite_shuffled_nmse'],rec['random_composite_nmse']),
        'positive_wrong_order_gap':rec['wrong_order_gap']>0}
    old_checks=record_existing_checks(root)
    source_scores=[]
    for seed in plan['source_seeds']:
        for group in plan['groups']:
            record=json.loads((root/'source'/f"{group['id']}_s{seed}.json").read_text())
            for a in record['source_audit']:
                source_scores.append({'group':group['id'],'seed':seed,**a})
    write_csv(root/'source_scores.csv',source_scores)
    summary={'reported_utc':datetime.now(timezone.utc).isoformat(),'registered_protocol_sha256':sha(root/'protocol.json'),
        'new_source_models':9,'new_source_seeds':plan['source_seeds'],'pilot_retested_separately':plan['pilot_seed'],
        'primary_new_seed_summaries':primary,'all_component_summaries':summaries,
        'per_new_seed_comparisons':seed_comparisons,'fixed_predictions':hypotheses,
        'limitations':['三个源种子使用同一个旧训练世界和一个新探针世界；本轮复核已选定的组合，没有测试未见任务组合。',
            '三组的联合标签统计和学习成绩仍不同，尚未单独识别代数关系的因果效应。',
            '数字读出零空间只排除一个线性读出子空间；仍可能保留非线性答案编码或其它读出信息。',
            '查询拼接结合四个任务提示条件下的隐藏状态，不能推广为每一个隐藏状态都具有相同结构。',
            '精确任务答案编码自身已有封闭置换关系；神经现象本身不能确认新的内部计算机制。',
            '复用的 1009 试验不计入三个新源种子检验。']}
    atomic_json(root/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    labels={'novel_records':'Four directional records','novel_minima_pair':'Peaks/valleys + minima','novel_peak_mixed':'Peak + three records'}
    for ax,key in zip(axes,('generator_nmse','composite_nmse')):
        for i,g in enumerate(plan['groups']):
            vals=[lookup[g['id'],s,'trained']['results'][plan['primary_landmark']]['explicit_checks'][key] for s in plan['source_seeds']]
            ax.bar(i,np.mean(vals),color=['#2563a8','#8999ad','#b4bfcd'][i]);ax.scatter(np.repeat(i,len(vals)),vals,c='black',s=20,zorder=3)
        ax.axhline(1,c='#999',linestyle=':',label='Identity');ax.set_xticks(range(3),[labels[g['id']] for g in plan['groups']],rotation=15,ha='right',fontsize=8)
        ax.set_ylim(bottom=0);ax.set_ylabel('Full-space displacement NMSE');ax.set_title(key.replace('_',' '))
    fig.savefig(root/'replication.png',dpi=180);fig.savefig(root/'replication.pdf');plt.close(fig)
    css='body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 18px;color:#172334}p{line-height:1.6}table{border-collapse:collapse;font-size:14px;margin:20px 0}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}img{max-width:100%}.note{padding:16px;background:#eef4fb}'
    body=f'<h1>代数结构：新源种子复核</h1><p class="note">四方向记录组在三个新源种子的查询表征平均结果上，支持 {sum(hypotheses.values())}/4 项预先固定的预测。逐种子一致性、读出空间边界与源成绩差异见下表；这还不足以证明整个网络形成了完整群表示。</p><p>三个新源种子 2027/3037/4051 ×三个四任务组合，共九个源模型；每模型固定 20,000 步、256 万标签。探针输入是新的 588 个完整八操作轨道，与 129,864 个旧输入不重叠。1009 作为已知试验单独重测，不计入新源种子结果。</p>'
    body+='<p class="note">恒等预测为零变化，其归一化误差为 1。分母是真实隐藏变化量，而不是原始向量大小。只用拟合集计算长度均值和 PCA；验证集选岭强度。生成元分别拟合，复合词未拟合；行向量先 C 后 I 使用 zρ_Cρ_I。全部结果保留，不按源成绩或几何结果剔除种子。</p><img src="replication.png">'
    null=next(r for r in summaries if r['group']=='novel_records' and r['view']=='source_query_numeric_null')
    prefix=next(r for r in summaries if r['group']=='novel_records' and r['view']=='ONE_END')
    body+=f'<p>结构最清楚地出现在任务查询表征。数字读出零空间的生成元/复合误差为 {null["generator_nmse"]:.3f}/{null["composite_nmse"]:.3f}，仍低于匹配随机模型 {null["random_generator_nmse"]:.3f}/{null["random_composite_nmse"]:.3f}，但不能排除非线性答案编码。无任务提示前缀为 {prefix["generator_nmse"]:.3f}/{prefix["composite_nmse"]:.3f}，高于随机模型 {prefix["random_generator_nmse"]:.3f}/{prefix["random_composite_nmse"]:.3f}；整体优势没有推广到这个位置。</p>'
    body+=table(primary,[('group','组合'),('source_accuracy','源任务准确率'),('generator_nmse','生成元误差'),('random_generator_nmse','随机生成元误差'),
        ('composite_nmse','未拟合复合误差'),('random_composite_nmse','随机复合误差'),('generator_shuffled_nmse','错误配对误差'),
        ('two_inverse_reconstruction_nmse','两次取逆恢复误差'),('wrong_order_gap','错误顺序 − 正确顺序')])
    body+='<p>测量前固定的四项预测：</p>'+table([{'prediction':k,'supported':v} for k,v in hypotheses.items()],[('prediction','预测'),('supported','新种子结果')])
    body+='<p>逐源种子：</p>'+table(seed_comparisons,[('seed','种子'),('records_generator_nmse','记录组生成元误差'),
        ('records_composite_nmse','记录组复合误差'),('generator_beats_both_controls','生成元优于两对照'),
        ('composite_beats_both_controls','复合优于两对照'),('two_inverse_reconstruction_nmse','两次取逆恢复误差')])
    body+='<p>取逆单项与真实变化量：</p>'+table(primary,[('group','组合'),('inverse_nmse','取逆预测误差'),('random_inverse_nmse','随机取逆误差'),('inverse_displacement_energy','真实取逆变化能量/中心化表征能量')])
    body+='<p>两次取逆恢复误差以中心化原表征能量归一化；取逆预测以真实取逆变化量归一化，两者分母不同。恒等变换也能做到两次恢复，因此恢复误差必须与真实像预测、错误配对和错误顺序一起判断。这里的留出泛化是相同长度范围内的新完整轨道，不是未见长度。</p>'
    body+='<h2>提取位置与数字读出空间</h2><p>查询向量是四个任务各自的等号位置隐藏向量拼接。数字读出空间由 31 个数字类别的中心化权重张成，每个任务块秩为 30。零空间保留数字类别 logit 差为零的方向，但可能仍含非线性答案编码。另报告数字 logits 与无任务提示的 ONE_END。完整维度敏感性不使用测试集搜索维度。</p>'
    body+=table(summaries,[('group','组合'),('view','表征/空间'),('generator_nmse','生成元误差'),('composite_nmse','复合误差'),
        ('two_inverse_reconstruction_nmse','两次取逆恢复误差'),('wrong_order_gap','顺序判别差')])
    teachers=[]
    for group in plan['groups']:
        record=json.loads((root/'probes'/f"teacher_{group['id']}.json").read_text())
        for view,r in record['results'].items():
            teachers.append({'group':group['id'],'view':view,
                'generator_nmse':float(np.mean([a['full_space_displacement_nmse'] for a in r['generators'] if a['status']=='complete'])),
                'composite_nmse':float(np.mean([a['full_space_displacement_nmse'] for a in r['composites'] if a['status']=='complete']))})
    body+='<h2>任务答案编码对照</h2><p>以下把真实任务标签直接编码成数字或 one-hot，未使用神经模型。四方向记录任务本身在这些操作下封闭，精确答案已有可预测的关系；因此查询表征中出现关系不能单独证明一种新的内部计算机制。长度不超过 7 的全部排列已核验 17,736 个输入—操作对应，具体映射见<a href="record_task_closure.json">任务闭合核验</a>；有限穷举本身不作为对任意长度的证明。</p>'
    body+=table(teachers,[('group','组合'),('view','真实任务编码'),('generator_nmse','生成元误差'),('composite_nmse','复合误差')])
    body+='<h2>数学定义修正及补充算子检验</h2><p>此前实验的 P 是固定维度坐标置零，满足 P²=P；真正删除坐标会改变维度，不能使用这个等式。多项式截断 P_k 也保留固定系数空间，求导 D 与截断满足 D P_k=P_(k−1)D（k≥1）。在次数 0 至 8 上，375 项整数矩阵规律已精确核验；本轮没有训练多项式神经模型。另复用了有限域模型做<a href="polynomial/report.html">求导—截断补充探针</a>：P 组整体优势未复现，不能作为独立领域的正面确认。</p>'
    body+='<h2>结论边界</h2><ul>'+''.join(f'<li>{html.escape(x)}</li>' for x in summary['limitations'])+'</ul>'
    body+='<p><a href="summary.json">完整汇总</a> · <a href="actions.csv">逐操作结果</a> · <a href="source_scores.csv">逐任务源成绩</a> · <a href="protocol.json">事前协议</a> · <a href="verification.json">核验</a> · <a href="existing_checks.json">上一轮口径审计</a>。</p>'
    (root/'report.html').write_text(f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>Structural replication</title><style>{css}</style>{body}</html>')
    print(json.dumps({'primary':primary,'predictions':hypotheses,'per_seed':seed_comparisons},indent=2))


if __name__=='__main__':run()
