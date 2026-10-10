"""Frozen exploratory summaries and self-contained HTML report."""
from collections import defaultdict
import html
import json
import shutil
import time
import numpy as np
from inventory import ROOT,WS,csv_read,csv_write,js,save,sha

PRIMARY=['linear_cka','knn_overlap','procrustes']
FIELDS=['initial_correct','initial_wrong','initial_identity','trained_correct','trained_wrong','trained_identity',
        'B0','Bt','delta_correct','delta_wrong','delta',
        'initial_correct_raw_error','initial_wrong_raw_error','initial_identity_raw_error',
        'trained_correct_raw_error','trained_wrong_raw_error','trained_identity_raw_error']
KEYS=['domain','cohort','relation','family','kind','branch','control','metric']


def number(x):return float(x) if x not in ['',None] else None

def complete_mean(values):
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def table(rows,fields,limit=None):
    def fmt(x):
        if x is None or x=='':return '未定义／缺失'
        if isinstance(x,(float,np.floating)):return f'{x:.4f}'
        return html.escape(str(x))
    rows=rows[:limit] if limit else rows
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+fmt(r.get(k))+'</td>' for k in fields)+'</tr>' for r in rows)+'</tbody></table></div>'


def run():
    started=time.monotonic();freeze=js(ROOT/'protocol_freeze.json')
    assert sha(ROOT/'summary_protocol.json')==freeze['summary_protocol_sha256']
    rows=csv_read(ROOT/'permutation_metric_results.csv')+csv_read(ROOT/'field_metric_results.csv')
    csv_write('metric_results.csv',rows)
    group=defaultdict(list)
    for r in rows:group[tuple(r[k] for k in KEYS)+ (r['source_unit'],r['seed'],r['world'])].append(r)
    units=[]
    for key,rs in group.items():
        expected=21 if rs[0]['domain']=='PermWorld' else 1
        assert len(rs)==expected
        out=dict(zip(KEYS+['source_unit','seed','world'],key),planned_lengths=expected)
        for f in FIELDS:
            v=[number(r[f]) for r in rs];out[f]=complete_mean(v)
            if f in ['Bt','delta']:out[f+'_defined_lengths']=sum(x is not None for x in v)
        for f in ['Bt','delta']:
            v=[number(r[f]) for r in rs];good=[x for x in v if x is not None]
            out[f+'_positive_lengths']=sum(x>0 for x in good)
            out[f+'_negative_lengths']=sum(x<0 for x in good)
            out[f+'_length_median']=float(np.median(good)) if len(good)==expected else None
            out[f+'_length_min']=min(good) if good else None;out[f+'_length_max']=max(good) if good else None
        units.append(out)
    csv_write('source_metric_summary.csv',units)
    rg=defaultdict(list)
    for r in units:rg[tuple(r[k] for k in KEYS)].append(r)
    relations=[]
    for key,rs in rg.items():
        out=dict(zip(KEYS,key),planned_sources=len(rs))
        for f in FIELDS:
            v=[r[f] for r in rs if r[f] is not None]
            out[f+'_sources']=len(v);out[f+'_mean']=float(np.mean(v)) if v else None
            out[f+'_median']=float(np.median(v)) if v else None
            if f in ['Bt','delta']:
                out[f+'_min']=min(v) if v else None;out[f+'_max']=max(v) if v else None
                out[f+'_source_sd']=float(np.std(v,ddof=1)) if len(v)>1 else None
                out[f+'_positive_sources']=sum(x>0 for x in v)
        relations.append(out)
    csv_write('relation_metric_summary.csv',relations)
    # Direction comparisons use the same source units for every primary metric.
    ag=defaultdict(dict)
    for r in units:
        if r['metric'] in PRIMARY:ag[tuple(r[k] for k in KEYS[:-1])+ (r['source_unit'],)].update({r['metric']:r})
    agreements=[]
    for key,ms in ag.items():
        out=dict(zip(KEYS[:-1]+['source_unit'],key))
        for f in ['Bt','delta']:
            good=all(m in ms and ms[m][f] is not None for m in PRIMARY)
            out[f+'_all_three_defined']=good
            for m in PRIMARY:out[f+'_'+m]=ms.get(m,{}).get(f)
            signs=[int(np.sign(ms[m][f])) for m in PRIMARY] if good else []
            out[f+'_all_three_direction_agree']=len(set(signs))==1 if good else None
            out[f+'_all_three_positive']=all(s>0 for s in signs) if good else None
        agreements.append(out)
    csv_write('direction_agreement.csv',agreements)
    # Equal relation means within a cohort; all shared families removed together.
    lg=defaultdict(list)
    for r in relations:
        if r['kind'] not in ['cross_task','cross_task_set']:continue
        lg[r['domain'],r['cohort'],r['branch'],r['control'],r['metric']].append(r)
    leave=[]
    for key,rs in lg.items():
        for family in ['none']+sorted({r['family'] for r in rs}):
            remain=[r for r in rs if family=='none' or r['family']!=family]
            row=dict(zip(['domain','cohort','branch','control','metric'],key),excluded_family=family,remaining_relations=len(remain))
            for f in ['Bt','delta','delta_correct','delta_wrong']:
                row[f]=complete_mean([r[f+'_mean'] for r in remain])
            leave.append(row)
    csv_write('leave_one_family.csv',leave)
    # Reuse old input and full-cache drift controls; never overwrite originals.
    parent=WS/'diagnostics/permworld_cka_20261009'
    reused=['input_length_baselines.csv','k_summary.csv','k_length_cka.csv',
            'k_pooled_control.csv','k_input_bootstrap.csv','input_bootstrap_summary.csv','permutation_summary.csv']
    for f in set(reused):shutil.copyfile(parent/f,ROOT/('inherited_'+f))
    save('inherited_control_audit.json',{'files':{f:dict(source=str(parent/f),sha256=sha(parent/f)) for f in set(reused)},
        'scope':'Historical controls retain their original estimator and uncertainty scope; not recomputed new-metric input uncertainty.'})
    # Conditional numeric answer accuracy, explicitly separate from full sequence scoring.
    data=dict(np.load(ROOT/'evaluation_inputs.npz'));tasks=list(data['tasks']);accuracy=[]
    records=js(ROOT/'source_forward_audit.json')['records']
    for r in records:
        if r['condition']!='trained':continue
        path=ROOT/'features'/f'{r["cohort"]}_{r["task"]}_s{r["seed"]}_trained.npz'
        with np.load(path) as a:
            pred=a['output_prob'][:,0].argmax(-1)
        gold=data['labels'][:,0,tasks.index(r['task'])]
        for n in range(10,31):
            ids=(data['lengths']==n)&(data['split']==1)
            accuracy.append(dict(cohort=r['cohort'],task=r['task'],seed=r['seed'],length=n,anchors=int(ids.sum()),
                accuracy=float((pred[ids]==gold[ids]).mean()),scope='Conditional numeric0..30 argmax, not full-vocabulary or sequence accuracy'))
    csv_write('source_answer_accuracy.csv',accuracy)
    # Approved additional training list is intentionally empty.
    csv_write('additional_training_manifest.csv',[],fields=['category','task','cohort','source_unit','encoding','reason','status'])
    future=[]
    future.append(dict(category='A',unique_tasks=0,source_units=0,encodings=0,models=0,reason='All current selected comparisons have existing complete weights; additional analysis only',status='completed_this_round'))
    future.append(dict(category='B_optional',unique_tasks=1,source_units=1,encodings=1,models=1,reason='Optional left_to_right_minima seed17 in six-hour matched protocol to fill min_record_reverse; could be unnecessary if relation omitted',status='candidate_not_scheduled; old-world protocol and exact seed alone do not establish new-world independence'))
    future.append(dict(category='C_candidate',unique_tasks=6,source_units='U_unfrozen',encodings=1,models='6*U',reason='Actual step0 plus new world/initialization source units; descents,recoils,peaks,valleys,Lmax,Rmax',status='not_authorized_to_run_this_round; U must follow precision/budget protocol'))
    future.append(dict(category='C_cost_scenario',unique_tasks=6,source_units=5,encodings=1,models=30,reason='Illustrative count only, not power or precision guarantee; independent engineering preflight excluded',status='candidate_only'))
    csv_write('future_training_candidates.csv',future)
    fwd=js(ROOT/'source_forward_audit.json');pa=js(ROOT/'permutation_evaluation_audit.json');fa=js(ROOT/'field_evaluation_audit.json');ka=js(ROOT/'k_evaluation_audit.json')
    cost=[dict(activity='existing_PermWorld_forward_and_cache',seconds=fwd['elapsed_seconds'],new_source_models=0,forward_sequences=fwd['forward_sequences_this_run'],
        reused_sequences=fwd['reused_sequences_this_run'],budget_type='GPU+cache IO wall time, RTX5070; reconstruction creates no trained source'),
        dict(activity='PermWorld_metric_analysis_PCA_orthogonal_fits',seconds=pa['elapsed_seconds'],new_source_models=0,forward_sequences=0,reused_sequences=0,budget_type='CPU analysis only; many single-purpose analytical fits'),
        dict(activity='F5_reuse_validation_and_metric_analysis',seconds=fa['elapsed_seconds'],new_source_models=0,forward_sequences=90*625,reused_sequences=90*625,budget_type='GPU feature audit + CPU maps;625 source-seen inputs'),
        dict(activity='Property32_cached_metrics',seconds=ka['elapsed_seconds'],new_source_models=0,forward_sequences=0,reused_sequences=50*4096,budget_type='CPU cached analysis only'),
        dict(activity='new_source_training',seconds=0,new_source_models=0,forward_sequences=0,reused_sequences=0,budget_type='No optimizer/source updates or paused resumes'),
        dict(activity='deferred_encoding768_plan',seconds='',new_source_models=0,forward_sequences=0,reused_sequences=0,budget_type='Old158–190GPU-hour allowance is a deferred scenario, not measured training in this task')]
    csv_write('cost_estimate.csv',cost)
    inventory=csv_read(ROOT/'training_inventory.csv');cohorts=csv_read(ROOT/'cohort_inventory_summary.csv')
    grade={g:sum(r['initialization_grade']==g for r in inventory) for g in ['A','B','C','D']}
    source_main=[r for r in relations if r['branch'] in ['prefix_final_norm','encoder_final'] and r['control']=='answer_strata' and r['metric'] in PRIMARY and r['kind']=='cross_task']
    ordinary_main=[r for r in relations if r['branch'] in ['prefix_final_norm','encoder_final'] and r['control']=='within_length' and r['metric'] in PRIMARY and r['kind'] in ['cross_task','cross_task_set']]
    direction_main=[r for r in agreements if r['domain']=='PermWorld' and r['branch']=='prefix_final_norm' and r['kind']=='cross_task' and r['control']=='answer_strata']
    counts={f:dict(available=sum(r[f+'_all_three_defined'] for r in direction_main),same=sum(r[f+'_all_three_direction_agree'] is True for r in direction_main),
        positive=sum(r[f+'_all_three_positive'] is True for r in direction_main),total=len(direction_main)) for f in ['Bt','delta']}
    save('summary_audit.json',dict(rows=len(rows),source_metric_summary_rows=len(units),relation_metric_summary_rows=len(relations),
        new_source_training=0,initialization_inventory_grades=grade,answer_control_direction_agreement=counts,
        elapsed_seconds=time.monotonic()-started))
    plots(relations)
    report(relations,units,agreements,leave,cohorts,cost,counts,grade)
    print('summary/report complete',len(rows),'metric cells',counts,flush=True)


def plots(relations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old=csv_read(ROOT/'inherited_k_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),sharey=True)
    for rep in ['r0','r1','r2','r3','r4']:
        rr=sorted((r for r in old if r['replicate']==rep),key=lambda r:int(r['k']))
        for ax,field in zip(axes,['intermodel_cka','mean_self_initial_cka']):
            ax.plot([int(r['k']) for r in rr],[float(r[field]) for r in rr],marker='o',linestyle='--' if rep in ['r3','r4'] else '-',label=rep+(' subset' if rep in ['r3','r4'] else ' source'))
            ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],[1,2,4,8,16]);ax.set_xlabel('Number of tasks');ax.set_ylim(0,1)
    axes[0].set_title('Between pools: original full-cache length means');axes[1].set_title('Own reconstructed init: original length means')
    axes[0].set_ylabel('Linear CKA');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'k_drift_reused.png',dpi=160);plt.close(fig)
    # All relation IDs retained; do not choose a subset with positive values.
    for cohort in ['specialist16','native_confirmation_single','six_hour_session_single']:
        rr=[r for r in relations if r['cohort']==cohort and r['kind']=='cross_task' and r['branch']=='prefix_final_norm' and r['control']=='answer_strata' and r['metric'] in PRIMARY]
        ids=list(dict.fromkeys(r['relation'] for r in rr));fig,axes=plt.subplots(1,3,figsize=(13, max(3,len(ids)*.3)),sharey=True)
        for ax,m in zip(axes,PRIMARY):
            lookup={r['relation']:r for r in rr if r['metric']==m};vals=[lookup[i]['delta_mean'] for i in ids]
            ax.barh(np.arange(len(ids)),[v or 0 for v in vals],color=['#147d64' if v is not None and v>0 else '#b85749' for v in vals])
            for pos,v in enumerate(vals):
                if v is None:ax.text(0,pos,'undefined',fontsize=8)
            ax.axvline(0,color='black',linewidth=.8);ax.set_title(m);ax.set_xlabel('Delta (metric-specific units)')
        axes[0].set_yticks(np.arange(len(ids)),ids);axes[0].invert_yaxis();fig.suptitle(cohort+' / pre-task / exact answer strata');fig.tight_layout();fig.savefig(ROOT/f'delta_{cohort}.png',dpi=160);plt.close(fig)


def report(relations,units,agreements,leave,cohorts,cost,counts,grade):
    primary=[r for r in relations if r['domain']=='PermWorld' and r['kind']=='cross_task' and r['branch']=='prefix_final_norm' and r['control']=='answer_strata' and r['metric'] in PRIMARY]
    compact=[]
    for r in primary:
        compact.append(dict(cohort=r['cohort'],relation=r['relation'],metric=r['metric'],sources=r['planned_sources'],defined=r['delta_sources'],
            B0=r['B0_mean'],Bt=r['Bt_mean'],dc=r['delta_correct_mean'],dw=r['delta_wrong_mean'],Delta=r['delta_mean'],median=r['delta_median'],
            min=r['delta_min'],max=r['delta_max'],positive_sources=r['delta_positive_sources']))
    missing=[r for r in csv_read(ROOT/'relation_task_graph.csv') if r['id']=='min_record_reverse']
    oldtrend=csv_read(WS/'diagnostics/permworld_cka_20261009/k_trend.csv')
    field=[r for r in relations if r['domain']=='F5' and r['branch']=='encoder_final' and r['control']=='within_length' and r['metric'] in PRIMARY]
    controls=[r for r in relations if r['branch'] in ['output_prob','answer_onehot','answer_scalar'] and r['control']=='within_length' and r['metric'] in PRIMARY]
    loo=[r for r in leave if r['branch']=='prefix_final_norm' and r['control']=='answer_strata' and r['metric'] in PRIMARY]
    s='''PermWorld 库存与已有模型探索评估（2026-10-09）
本轮新增源模型训练0；无暂停任务续跑、无发布。624逻辑库存条目与582不同权重容器SHA不能解释为独立重复。
原48单任务+50个k位置：98逻辑位置、94不同checkpoint，已逐文件SHA核对。
k16为24端点，其中6重复使用旧v3权重、18新增历史训练。小PermWorld1200步pilot24、F5 rank3 72/rank4 45/matched27已分队列登记。
本轮关系：19条数学证书，18条有匹配完成权重；min_record_reverse缺同批次Lmin/Rmin。62关系×源配对，82不同PermWorld源位置；F5仅统一rank4的45权重；k复用50位置。
初始化：PermWorld参照C，构造器重建未证明原权重身份；F5 encoder可用真实历史step0激活B，仅限625保存输入和encoder位置，初始化输出读出仍C。未为清单加载所有state_dict；未测参数哈希显式空白。
普通/答案分层、真实答案、条件数字输出概率都保留。主点是任务token前ONE_END；query分支单列。固定长度1/21；缺失不重加权。CKA、kNN(k10)、8维拟合PCA后正交Procrustes在看到新结果前冻结。
F5全部625输入源训练见过；250/375划分仅用于分析映射拟合/测试。真实四答案唯一标识每个输入，严格答案控制无重复组，全部未定义；未放宽控制。域、架构与输入编码同时变化。
统计：探索性，无确认性p值或等价性主张。源范围/标准差描述固定世界的初始化变化；关系共享任务和轨道，不新增独立关系数量。旧输入bootstrap仅条件于旧固定模型/CKA协议；未替新指标制造输入抽样区间。
最小追加训练：当前交付0，additional_training_manifest只有表头。可选补缺1个Lmin旧协议seed17（非新世界确认）；若要真实step0与新数据确认，候选6任务×U×E，优先E1，U未冻结；U5=30只是成本情景。未来先独立开发预跑，不能按几何结果选择预算。
编码768模型为未执行计划。现库存足以扩关系/换指标/第二实例；只有明确要估计编码干预，且现探索确定代表关系、精度与资源目标后才另冻结缩小方案。此次不调用编码prepare/train/evaluate。
F17仅库存登记，本轮不改论文、结果或发布产物，不充当自然几何确认。完整保护哈希核验见audit.json。
复现：inventory.py -> relations.py -> prepare_eval.py -> evaluate.py --stage features -> finalize_inventory.py -> evaluate.py --stage permutation-metrics/field-metrics；evaluate_k.py -> summarize.py -> verify_results.py。固定协议/输入已有文件不得在结果后覆盖。
'''
    s+='\n实测主结果：原两条取逆关系descents/recoils与exceedances/deficiencies，在答案控制后三种指标的Delta均为正、各三个源均同向。CKA Delta分别0.588965/0.641461，dc分别-0.033851/+0.020471、dw分别-0.622816/-0.620990，主要来自错误配对下降。六小时队列的Lmax/Rmin取逆仅一源正向，不能称为多源稳定复现。\n'
    s+='新增反转/补值关系的平均Bt多数正，但Delta为负，不能解释为没有学习；最大纪录反转的native seed1009/3037训练后CKA偏好略负，另外两指标略正。双升/双降的kNN在两长度样本不足，固定汇总未定义。\n'
    s+='两条旧取逆关系的3源×21长度中，三指标和debiased CKA的Delta均正，不由某单一长度驱动。最大纪录反转的native seed1009/3037在debiased CKA中Bt变正，说明弱训练后偏好依赖指标版本；Delta仍负。\n'
    s+='F5前两组关系CKA/Procrustes平均Delta正而kNN负；其他源也有符号变化。缺严格答案控制及源未见输入结果，第二实例不能支持统一机制结论。\n'
    s+='旧全缓存k系列中五条曲线k16的初始化保留都高于k1，R0/R2/R4单调，R1/R3不单调。新轨道指标完整29长度汇总未定义。\n'
    (ROOT/'reuse_and_missing_report.txt').write_text(s)
    body='<h1>PermWorld：训练库存与最小补充计划</h1><p class="lead">已复用完成权重，扩展关系与度量，并完成 F5 第二实例的探索评估。新增源训练：<strong>0</strong>。本报告不作数学机制或确认性推断。</p>'
    body+='<h2>1. 实际可用库存与初始化证据</h2><p>下表是逻辑条目和不同权重容器，不是独立实验数。原48+50位置去重为94 checkpoint；k16的6个端点复用旧v3权重。F17容器只盘点。所有本轮选中完成权重均在本地可访问，不需下载release或重训。</p>'
    body+=table(cohorts,['cohort','logical_positions','completed_records','accessible_weights','distinct_file_sha256','reused_endpoints','partial','plans_only'])
    body+='<p>完整状态、配置、哈希、缓存和缺口见 <a href="training_inventory.csv">training_inventory.csv</a>。PermWorld初始化均为C：按原构造器/词表/RNG路径重建，原始权重身份未获验证。F5 encoder使用实际step-0激活B；初始输出头重建为C。'+html.escape(str(grade))+'。未选中的状态未为填表而全部加载，参数哈希缺项明确保留。</p>'
    body+='<h2>2. 哪些新增关系无需源训练</h2><p>穷举S1–S7和固定大n随机样本，共8,601排列×19条公式均通过。18条有匹配完成权重，组成62个关系/源配对。新增反转的LIS/LDS、run、fixed/anti-fixed，左/右最大纪录，双升/双降；取逆的左最大/右最小纪录；补值的双升/双降均可复用。三个取逆自不变性单列；缺匹配cohort的左/右最小纪录关系不跨批拼接。</p><p>补值也是部分反转任务对的正确变换，因此不能作为它们的错误对照；错误变换已按oracle反例在读取指标前固定。共享任务/群作用关系带依赖标签，18条不是18个独立性质族。</p>'
    body+='<p><a href="relation_task_graph.csv">关系覆盖图</a> · <a href="relationship_certificates.json">数学证书</a> · <a href="metric_protocol.json">冻结度量协议</a> · <a href="summary_protocol.json">冻结汇总规则</a></p>'
    body+='<h2>3. 三类度量是否一致，Δ来自哪里</h2><p>每长度独立计算，固定1/21汇总。主点为输入结束、任务提示前final-layer状态。B0=初始化正确−错误；Bt=训练正确−错误；dc/dw为各自训练变化；Δ=Bt−B0=dc−dw。Procrustes score=−NRMSE，原始误差保留。正Δ不能单独解释为正确表征趋同。</p>'
    body+='<p>答案分层主位置：三指标都可定义的源/关系单元，Bt方向一致 '+str(counts['Bt']['same'])+'/'+str(counts['Bt']['available'])+'；Δ方向一致 '+str(counts['delta']['same'])+'/'+str(counts['delta']['available'])+'。这是共享关系/源的描述性覆盖计数，不能当作独立重复。分歧与负值全部保留。</p>'
    body+='<p><strong>最稳的正增量是原两条取逆关系</strong>：descents/recoils与exceedances/deficiencies三个源在三类指标上均为正Δ。各3源×21长度中三指标与debiased CKA的Δ均正，不由单一长度驱动。CKA均值为0.589/0.641，dc分别约−0.034/+0.020、dw约−0.623/−0.621，因此主要表现为错误配对相似度下降。六小时的左最大/右最小纪录取逆也正向，但仅一源。这里超过的是重建初始化参照C，不能确认实际原始初始权重的增量。</p><p>新增反转/补值的平均Bt多数正而Δ负；初始化起点较高，不能据Δ负否定学习。native左/右最大纪录反转seed1009与3037的Bt：普通CKA略负，邻域/Procrustes略正；debiased CKA也变为正，故这一弱训练后偏好依赖指标版本，Δ仍负。双升/双降的邻域指标在两个长度样本不足，严格21长度汇总未定义。</p>'
    body+=table(compact,['cohort','relation','metric','sources','defined','B0','Bt','dc','dw','Delta','median','min','max','positive_sources'])
    for c in ['specialist16','native_confirmation_single','six_hour_session_single']:body+=f'<img src="delta_{c}.png" alt="{c} all relation Delta">'
    body+='<p>普通CKA/debiased敏感性、全部源与逐长度数据见 <a href="metric_results.csv">metric_results.csv</a>、<a href="source_metric_summary.csv">源级汇总</a>、<a href="direction_agreement.csv">共同源方向比较</a>。不比较不同指标数值大小。8维PCA Procrustes只评价所选子空间，不能推断完整隐藏状态可线性预测。</p>'
    body+='<details><summary>留一依赖族：保留所有结果</summary>'+table(loo,['cohort','metric','excluded_family','remaining_relations','Bt','delta','delta_correct','delta_wrong'])+'</details>'
    body+='<h2>4. 答案、输出、输入与取点控制</h2><p>严格答案分层对fit/test分别去组均值，是oracle测量控制；不是无需答案的可部署预测器。真值答案几何为静态基线，没有虚构初始化或标签学习Δ。输出分支仅数字0–30条件概率，保留误差/置信度结构。query另列，不能代替任务前表征；未拼接为source_query_concat。</p><p>历史输入kernel控制和CKA置换/输入bootstrap已复用，其范围见 <a href="inherited_control_audit.json">继承控制审计</a>。这些原协议控制不能直接充作新kNN或Procrustes的输入抽样区间，也不能覆盖未测的反转输入kernel。隐藏Δ减输出Δ不识别额外机制。</p>'
    body+='<details><summary>全部输出与真值基线（长度分层）</summary>'+table(controls,['domain','cohort','relation','branch','metric','B0_mean','Bt_mean','delta_correct_mean','delta_wrong_mean','delta_mean','delta_sources'])+'</details>'
    body+='<h2>5. k任务数系列：漂移是辅助线索</h2><p>复用50逻辑位置。R0–R2为不同源种子；R3/R4固定seed17任务子集敏感性。两池重建初始参数哈希相同，但不能据此验证历史原始共享初始化。下图继承完整缓存的固定逐长度CKA，与原报告完全相同；不是新轨道切分指标。</p><img src="k_drift_reused.png" alt="Inter-pool CKA and own-init retention">'
    body+=table(oldtrend,['replicate','metric','k1','k16','k16_minus_k1','increasing_adjacent_steps','adjacent_steps'])
    body+='<p>旧逐长度CKA中，五条初始化保留曲线的k16都高于k1；R0/R2/R4逐步增加，R1/R3存在回落。这是固定世界/协议的描述性趋势，R3/R4尤其不是新源种子。</p>'
    body+='<p>新度量将分析fit/test按完整轨道SHA奇偶固定划分。n=2没有test、n=3没有fit，固定1/29且完整长度要求使新汇总未定义；没有事后改成只平均长排列。逐长度kNN/Procrustes/CKA仍在 <a href="k_metric_results.csv">k_metric_results.csv</a>，覆盖见 <a href="k_metric_summary.csv">k_metric_summary.csv</a>。kNN排除查询自身和同一输入对象的重复行；其他对象重复行仍按原缓存抽样权重。高池间CKA与高初始化保留同时出现仅提供保留初始几何的线索，不能精确分解来源。</p>'
    body+='<h2>6. 第二领域：F5能复用到什么程度</h2><p>仅选统一rank4普通训练批次45模型（3输入basis×3初始化×5组），36个P/Mk比较，不混144条异质源。正确变换是任务基的逆；错误为预定相邻基的逆。全部625输入源训练见过，250 fit/375 test仅是分析映射留出；动作保持第一坐标，fit/test轨道交集0。原始step0和最终encoder激活共90份独立重算，最大差0。</p>'
    body+=table(field,['relation','metric','planned_sources','B0_mean','Bt_mean','delta_correct_mean','delta_wrong_mean','delta_mean','delta_median','delta_min','delta_max'])
    body+='<p>四个关系的平均Bt在三种度量中均为正，但Δ并不全同向：P/M1和P/M2的CKA/Procrustes为正，kNN分别为−0.0368/−0.0061。P/M3、P/M4的平均Δ三者正向，但不少源仍反向；P/M3的Procrustes平均Δ仅0.0010。第二实例体现了度量和输入basis/初始化敏感性，不能写为统一跨域复现。</p>'
    body+='<p>完整rank4答案元组唯一标识输入，严格答案分层全部未定义。这限制了排除输出关联的能力；不放宽控制制造正结果。F5与PermWorld同时改变数学域、MLP/Transformer和输入表示，结果只作为第二实例，不能归因为领域差异。F5坐标重解释为多项式不算第三实例，F13/F17干预任务不作自然几何确认。</p>'
    body+='<h2>7. 最小新训练与768编码计划</h2><p>当前必须追加：0个。<a href="additional_training_manifest.csv">新增训练清单</a>为空（仅表头）。可选配置补缺：同六小时协议的1个Lmin seed17填补反转纪录关系；不补也能完成当前范围。若要原始初始化身份/独立新世界确认，候选6任务×U×E，先E=1；U未冻结，U=5的30模型只是成本情景，另加独立工程开发单元、不计正式样本。六任务仅覆盖代表关系；若两条原取逆关系都作为确认主要比较，还须加exceedances/deficiencies，变为8U（U5时40）。<a href="minimal_confirmation_draft.json">未来最小确认草案</a>保留这些未决项，尚非可启动的冻结协议。</p><p>768模型是未执行编码计划。库存已经支持当前“扩关系→换指标→第二实例”，不需先完成大规模编码干预。若未来做编码干预，它应解决“改变的是初始化B0，还是训练后Bt”这一具体未决问题，并拆开Gamma0/Gammat/GammaDelta；Delta翻转而Bt近似不能写为学习关系偏好翻转。只有明确该问题、代表关系、精度和预算后才另冻小规模协议。此轮不调用编码prepare/train/evaluate、不续训partial。</p>'
    body+=table(csv_read(ROOT/'future_training_candidates.csv'),['category','unique_tasks','source_units','encodings','models','reason','status'])
    body+='<h2>8. 成本、复现与停止</h2>'+table(cost,['activity','seconds','forward_sequences','reused_sequences','new_source_models','budget_type'])
    body+='<p>全部本轮产物位于独立目录。<a href="audit.json">完整验证与保护审计</a> · <a href="reuse_and_missing_report.txt">简短文字报告</a>。复现脚本不调用优化器或源训练入口；解析PCA/正交映射拟合预算单列。旧CKA报告、F17论文/结果/发布哈希核验后保持不变。本轮完成后停止。</p>'
    (ROOT/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PermWorld inventory and exploratory metrics</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1300px;margin:36px auto;padding:0 22px;color:#24313b}h1,h2{line-height:1.3}h2{margin-top:38px}.lead{background:#e8f3ee;padding:16px;border-radius:8px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px;width:100%;margin:14px 0}th,td{border:1px solid #dce2e5;padding:7px;text-align:left;white-space:nowrap}th{background:#eef2f5}img{max-width:100%;margin:18px 0}a{color:#1456a0}details{margin:18px 0}</style>'+body+'</html>')


if __name__=='__main__':run()
