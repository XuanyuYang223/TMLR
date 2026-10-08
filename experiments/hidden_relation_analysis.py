"""Report answer-only explanations separately from relation-holdout behavior."""
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from .algebra_structure_replication_analysis import table,write_csv
from .hidden_relation_train import load_plan
from .longrun_engine import atomic_json
from .permworld_combinations import sha


def register_claim_gate():
    plan,_,root=load_plan();path=root/'claim_gate.json'
    signature={'analysis_code_sha256':sha(__file__),'source_protocol_sha256':sha(root/'protocol.json'),
        'claims':['Native correct-relation collision CI and ICI accuracy both exceed 0.5 on source-seed averages.',
            'Native correct-relation CI has positive pair-both-correct rate.',
            'Native correct-relation average of CI/ICI beats native shuffled and ordinary identity operators.',
            'Stronger interpretation additionally requires correct native CI/ICI to beat ordinary posthoc generator probes and mean source validation accuracy >=0.9.'],
        'scope':'Added stricter interpretation gate after source training began, before any hidden evaluation. It supplements, rather than replaces, the source protocol forecasts.'}
    if path.exists():assert json.loads(path.read_text())['signature']==signature
    else:
        assert not list((root/'evaluations').glob('*.json'))
        atomic_json(path,{'registered_utc':datetime.now(timezone.utc).isoformat(),'signature':signature,'hidden_evaluation_records':0})


def run():
    plan,_,root=load_plan();register_claim_gate();records=[json.loads(p.read_text()) for p in (root/'evaluations').glob('*.json')];assert len(records)==9
    rows=[]
    for r in records:
        for method in r['methods']:
            for a in method['results']:
                rows.append({'condition':r['condition'],'seed':r['seed'],'view':method['view'],'method':method['method'],
                    'source_accuracy':float(np.mean(r['source_validation_accuracy'])),
                    **{k:a[k] for k in ['split','word','displacement_nmse','answer_accuracy','direct_transformed_input_accuracy','pair_both_correct']}})
    write_csv(root/'endpoints.csv',rows);summaries=[]
    for condition in plan['conditions']:
        for method in ['native_operators','posthoc_correct_generators','posthoc_shuffled_generators','identity']:
            for split in ['iid','answer_collisions']:
                for word in ['c','i','ci','ici','ic','cc','ii','cici']:
                    selected=[r for r in rows if r['condition']==condition and r['method']==method and r['view']=='query' and r['split']==split and r['word']==word]
                    if not selected:continue
                    keys=['source_accuracy','displacement_nmse','answer_accuracy','direct_transformed_input_accuracy','pair_both_correct']
                    summaries.append({'condition':condition,'method':method,'split':split,'word':word,'source_seeds':len(selected),
                        **{k:float(np.mean([r[k] for r in selected if r[k] is not None])) if any(r[k] is not None for r in selected) else None for k in keys},
                        'accuracy_min_seed':min(r['answer_accuracy'] for r in selected),'accuracy_max_seed':max(r['answer_accuracy'] for r in selected)})
    write_csv(root/'group_summary.csv',summaries)
    get=lambda c,m,w:next(r for r in summaries if r['condition']==c and r['method']==m and r['split']=='answer_collisions' and r['word']==w)
    primary=[get(c,'native_operators',w) for c in plan['conditions'] for w in plan['heldout_primary_words']]
    averaged=lambda c,m:float(np.mean([get(c,m,w)['answer_accuracy'] for w in plan['heldout_primary_words']]))
    cavg=averaged('correct_relations','native_operators')
    predictions={
        'both_native_hidden_words_above_answer_ceiling':all(get('correct_relations','native_operators',w)['answer_accuracy']>.5 for w in plan['heldout_primary_words']),
        'native_CI_positive_pair_both_correct':get('correct_relations','native_operators','ci')['pair_both_correct']>0,
        'native_correct_beats_shuffled_and_ordinary_identity':cavg>max(averaged('shuffled_relations','native_operators'),averaged('ordinary','native_operators')),
        'stronger_gate_correct_beats_ordinary_posthoc':cavg>averaged('ordinary','posthoc_correct_generators') and get('correct_relations','native_operators','ci')['source_accuracy']>=.9}
    per_seed=[r for r in rows if r['condition']=='correct_relations' and r['method']=='native_operators' and r['view']=='query' and r['split']=='answer_collisions' and r['word'] in plan['heldout_primary_words']]
    baseline=json.loads((root/'answer_controls/summary.json').read_text())
    limitations=['新增实验显式监督 C/I 状态对应，并提供算子组合方式；不能据此声称普通任务训练自行发现了组合规则。',
        '复合输入仅用于评估目标与直接输入诊断；组合预测从原输入隐藏向量出发，不使用隐藏复合输入或其答案。',
        '碰撞测试按已知答案相同且隐藏答案不同来选取，结论针对这个条件分布；另报告未筛选的 iid 轨道。',
        '每组只有三个源种子、一个训练世界；逐模型区间仅对测试排列对重采样，不覆盖新训练世界的不确定性。',
        '普通模型的事后探针使用另一个拟合集中的已知生成元对应；须与源训练时学得的算子区分。',
        '线性答案残差只去掉加性编码，仍可能保留非线性答案交互，不能宣称排除了全部答案信息。']
    summary={'reported_utc':datetime.now(timezone.utc).isoformat(),'primary':primary,'all_query_summaries':summaries,
        'correct_condition_per_seed':per_seed,'fixed_predictions':predictions,'answer_controls':baseline,
        'stronger_interpretation_supported':all(predictions.values()),'limitations':limitations,
        'answer_only_accuracy_ceiling':.5,'answer_only_pair_both_correct_ceiling':0.,
        'source_seeds':plan['source_seeds'],'new_source_models':9,'new_source_label_exposures_per_model':1920000}
    atomic_json(root/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    selections=[('ordinary','native_operators','Ordinary / identity'),('correct_relations','native_operators','Correct learned relations'),
        ('shuffled_relations','native_operators','Shuffled learned relations'),('ordinary','posthoc_correct_generators','Ordinary / fitted generators')]
    for ax,key in zip(axes,['answer_accuracy','pair_both_correct']):
        for index,(c,m,label) in enumerate(selections):
            values=[r[key] for r in rows if r['condition']==c and r['method']==m and r['view']=='query' and r['split']=='answer_collisions' and r['word']=='ci']
            ax.bar(index,np.mean(values),color=['#8999ad','#2563a8','#d49063','#7c7ab8'][index]);ax.scatter([index]*3,values,c='black',s=18,zorder=3)
        ax.axhline(.5 if key=='answer_accuracy' else 0,color='#555',linestyle=':',label='Answer-only ceiling')
        ax.set_xticks(range(4),[x[2] for x in selections],rotation=18,ha='right',fontsize=8);ax.set_ylim(0,1);ax.set_title('CI '+key.replace('_',' '))
    fig.savefig(root/'hidden_answers.png',dpi=180);fig.savefig(root/'hidden_answers.pdf');plt.close(fig)
    css='body{font:16px system-ui;max-width:1300px;margin:32px auto;padding:0 18px;color:#172334}p,li{line-height:1.6}table{border-collapse:collapse;font-size:13px;margin:20px 0}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}img{max-width:100%}.note{background:#eef4fb;padding:16px}'
    body=f'<h1>答案解释与真正留出关系</h1><p class="note">答案＋任务提示基线能近乎精确重现旧四方向记录组的代数几何。新的显式关系学习条件在隐藏 CI/ICI 答案测试中，支持 {sum(predictions.values())}/4 项预先保存的检验。更强解释门槛：{all(predictions.values())}。完整正负结果如下。</p>'
    body+='<h2>旧结果的答案基线</h2><p>基线只访问真实任务答案和任务身份，没有排列输入或神经训练。每个答案类别映射到固定随机向量，再加任务提示向量；使用与旧实验相同的轨道、生成元拟合和复合评估。这是特权答案对照，用来检验解释是否充分，不是可部署的隐藏答案预测器。</p>'
    body+=table(baseline['rows'],[('group','旧任务组'),('kind','基线/残差'),('generator_nmse','生成元误差'),('composite_nmse','复合误差'),('hidden_reconstruction_nmse','线性答案重建隐藏误差')])
    body+='<p>答案基线也重现了四记录优于两个非闭合任务集合的排序。旧几何不能单独识别额外计算机制。去掉线性加性答案预测后，残余复合误差上升；这仍未排除非线性答案编码。</p>'
    body+='<h2>新训练与关系留出</h2><p>九个全新模型：普通任务训练、正确关系训练、错误配对关系训练 ×三个源种子。每模型 20,000 步，192 万个相同源标签曝光。只监督 LR-max 在原状态、C 状态、I 状态的答案，对应原排列的 LR-max、LR-min、RL-min。源训练的完整八状态轨道只开放三个状态，其余五个状态和对应边均排除；验证也只看三个已知状态。</p>'
    body+='<p>关系条件另监督四条有向对应：e↔C 与 e↔I，不添加新源答案。两个仿射算子分别学习，不施加群恒等式。隐藏 CI 由 hA_C A_I（含偏置）预测，ICI 同样由生成元复合；然后使用固定数字读出。复合方式由实验框架提供，算子是否预测真实隐藏状态与答案由留出检验决定。</p>'
    body+='<p>测试包含 168 个未筛选的新轨道，以及 1,344 对碰撞排列。每对长度及三个已知答案完全相同，隐藏 CI 与 ICI 答案不同。因此只用已知答案、任务与长度的确定性预测器准确率至多 50%，两项同时正确率为 0。CI/ICI 都对应缺失的 RL-max 答案，但它们的变换输入和隐藏向量仍不同。</p><img src="hidden_answers.png">'
    body+=table(primary,[('condition','训练条件'),('word','隐藏词'),('source_accuracy','可见源准确率'),('answer_accuracy','组合答案准确率'),('pair_both_correct','一对全部正确'),('direct_transformed_input_accuracy','直接输入复合排列诊断'),('displacement_nmse','真实隐藏位移误差')])
    body+='<p>正确关系条件逐源种子：</p>'+table(per_seed,[('seed','源种子'),('word','隐藏词'),('answer_accuracy','组合准确率'),('pair_both_correct','一对全部正确'),('displacement_nmse','隐藏位移误差')])
    body+='<p>预先保存的检验：</p>'+table([{'prediction':k,'supported':v} for k,v in predictions.items()],[('prediction','检验'),('supported','支持')])
    body+='<h2>事后探针、错误关系和 iid 测试</h2><p>事后探针只从 336 个拟合轨道及 84 个验证轨道的已知三状态拟合 e↔C、e↔I；均值和参数选择也不访问隐藏状态。它与源训练中学得的算子单列。直接输入诊断会访问复合排列，是预测器的能力上界参考，不能代替组合推断。</p>'
    selected=[r for r in summaries if r['word'] in ['c','i','ci','ici']]
    body+=table(selected,[('condition','训练条件'),('method','算子来源'),('split','测试'),('word','词'),('answer_accuracy','答案准确率'),('pair_both_correct','一对全对'),('displacement_nmse','隐藏位移误差')])
    body+='<h2>结论范围</h2><ul>'+''.join('<li>'+x+'</li>' for x in limitations)+'</ul>'
    body+='<p><a href="summary.json">汇总</a> · <a href="protocol.json">源训练协议</a> · <a href="claim_gate.json">隐藏评估前解释门槛</a> · <a href="evaluation_protocol.json">预测口径</a> · <a href="data_verification.json">留出与碰撞核验</a> · <a href="verification.json">完整核验</a> · <a href="endpoints.csv">全部端点</a> · <a href="tests.log">测试</a>。</p>'
    (root/'report.html').write_text(f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>Hidden relations</title><style>{css}</style>{body}</html>')
    print(json.dumps({'primary':primary,'predictions':predictions,'stronger_interpretation':all(predictions.values())},indent=2))


if __name__=='__main__':run()
