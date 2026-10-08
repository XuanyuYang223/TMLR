"""Geometry-first analysis of the predeclared coverage intervention."""
import argparse
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .inverse_alignment_coverage import initialize
from .inverse_functional_alignment import configure, now, extract
from .inverse_functional_auxiliary import scores, cka
from .inverse_functional_report import statistics, table
from .longrun_attention import accelerate
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .specialist_cka_controls import api

LABELS = {'ordinary_exposure':'普通训练＋输入曝光','support_alignment':'只对齐原标签集','coverage_alignment':'扩大覆盖的正确对齐',
          'coverage_matched_mismatch':'扩大覆盖的预测答案错配','coverage_distillation':'扩大覆盖的输出蒸馏','coverage_distillation_alignment':'扩大覆盖的蒸馏＋正确对齐'}
SHORT = ['Exposure only','Support alignment','Coverage alignment','Matched mismatch','Distillation','Distill + alignment']
RULES = {'primary':'Fixed 1200 update; per-length first128 fresh rows, five lengths equally weighted. Accuracy of the same model is secondary to the geometry question.',
    'sensitivity':'Existing validation-selected checkpoint, all conditions retained; initialization CKA and answer-stratified pairing controls.',
    'pool_diagnostic':'For each length first128 rows of the unlabeled pool; raw CKA using exactly the same sample count as fresh test. Extract every final/selected model and initialization; no oracle pool answers.',
    'uncertainty':'Exact empirical percentile intervals: 6^6 paired-repeat draws conditional on teachers and 3^3 teacher-cluster draws. All signs shown. Descriptive intervals, no multiplicity-adjusted population significance.',
    'interpretation':'A positive raw CKA contrast alone does not establish relation specificity or output independence. Compare correct to exposure-only, support-only and matched mismatch, then accuracy and distillation controls. No test-based retuning.'}


def register(root):
    file = root/'analysis_protocol.json'
    if file.exists():
        recorded = json.loads(file.read_text()); assert recorded['rules']==RULES and recorded['code_sha256']==sha(__file__); return
    assert not (root/'test_opened.json').exists()
    atomic_json(file,{'registered_utc':now(),'rules':RULES,'code_sha256':sha(__file__),'test_outcomes_observed':False})


def contrasts(delta,reps):
    s = statistics(delta,reps)
    return {'mean':s['mean_pp'],'paired_deltas':s['paired_deltas_pp'],'positive_replicates':s['positive_replicates'],
        'conditional_95':s['paired_conditional_bootstrap_95_pp'],'teacher_cluster_95':s['three_teacher_cluster_bootstrap_95_pp'],
        'teacher_cluster_means':s['teacher_cluster_deltas_pp']}


def pool_features(plan,root,sig):
    device = configure(); _,tokens,_,TrainConfig,factory = api(plan); old = Path(plan['previous_study'])
    for rep in sig['replicates']:
        source = next(s for s in sig['sources'] if s['seed']==rep['source_seed']); cp = torch.load(source['checkpoint'],weights_only=True,map_location='cpu'); cfg = TrainConfig.from_value(cp['config']); del cp
        pool = dict(np.load(root/'dataset'/rep['id']/'dataset.npz')); ix = np.concatenate([np.flatnonzero(pool['lengths']==n)[:128] for n in plan['lengths']])
        subset = {k:v[ix] for k,v in pool.items()}
        for condition in ['initialization']+plan['conditions']:
            for endpoint in (['initialization'] if condition=='initialization' else ['final','selected']):
                name = rep['id']+'_'+condition; file = root/'evaluations'/f'{name}_{endpoint}_pool.npz'
                if file.exists(): continue
                if condition=='initialization': path = old/'initializations'/f"{rep['id']}.pt"
                else:
                    record = json.loads((root/'training'/f'{name}.json').read_text()); step = 1200 if endpoint=='final' else record['selected']['step']; path = root/'checkpoints'/f'{name}_u{step}.pt'
                cp = torch.load(path,weights_only=True,map_location='cpu'); model = accelerate(factory(cfg)); model.load_state_dict(cp if condition=='initialization' else cp['model']); del cp; model.to(device)
                out = extract(model,subset,plan['target_task'],tokens,0); np.savez_compressed(file,hidden=out['hidden'],pool_indices=ix); del model


def analyze(plan,root,sig):
    register(root); assert json.loads((root/'state.json').read_text())['status']=='evaluation_complete'
    pool_features(plan,root,sig); data = dict(np.load(root/'dataset/test/dataset.npz')); rows = []
    for rep in sig['replicates']:
        teacher = np.load(root/'teacher'/f"test_s{rep['source_seed']}.npz")['hidden']; poolteacher = np.load(root/'teacher'/f"{rep['id']}.npz")['hidden']
        for condition in ['initialization']+plan['conditions']:
            for endpoint in (['initialization'] if condition=='initialization' else ['final','selected']):
                name = rep['id']+'_'+condition
                if condition=='initialization': hidden = np.load(root/'evaluations'/f'{name}.npz')['hidden']; accuracy = None
                else:
                    hidden = np.load(root/'evaluations'/f'{name}.npz')[endpoint+'_hidden']; accuracy = json.loads((root/'evaluations'/f'{name}.json').read_text())['results'][endpoint]['accuracy']
                geometry = scores(hidden,teacher,data['lengths'],data['labels'],plan['lengths'])
                pf = dict(np.load(root/'evaluations'/f'{name}_{endpoint}_pool.npz')); pool = np.load(root/'dataset'/rep['id']/'dataset.npz'); lengths = pool['lengths'][pf['pool_indices']]; ph = poolteacher[pf['pool_indices']]
                pool_scores = {str(n):cka(pf['hidden'][lengths==n],ph[lengths==n]) for n in plan['lengths']}
                mean = {key:float(np.mean([r[key] for r in geometry])) for key in ['correct_cka','matched_correct_cka','matched_wrong_cka','answer_residual_correct_cka','answer_residual_wrong_cka']}
                mean['matched_contrast'] = mean['matched_correct_cka']-mean['matched_wrong_cka']; mean['answer_residual_contrast'] = mean['answer_residual_correct_cka']-mean['answer_residual_wrong_cka']
                mean['pool_cka'] = float(np.mean(list(pool_scores.values())))
                rows.append({'replicate':rep['id'],'condition':condition,'endpoint':endpoint,'accuracy':accuracy,'geometry':geometry,'pool_geometry':pool_scores,'mean':mean})
    atomic_json(root/'geometry.json',{'completed_utc':now(),'records':rows,'pool_oracle_answers_read':False})
    means = {}; comparisons = {}
    for endpoint in ['final','selected']:
        means[endpoint] = {}
        for condition in plan['conditions']:
            subset = [r for r in rows if r['condition']==condition and r['endpoint']==endpoint]
            means[endpoint][condition] = {'accuracy':float(np.mean([r['accuracy'] for r in subset])),**{k:float(np.mean([r['mean'][k] for r in subset])) for k in subset[0]['mean']}}
        comparisons[endpoint] = {}
        for contrast in plan['primary_contrasts']:
            a,b = contrast.split('-'); aa = [next(r for r in rows if r['replicate']==rep['id'] and r['condition']==a and r['endpoint']==endpoint) for rep in sig['replicates']]; bb = [next(r for r in rows if r['replicate']==rep['id'] and r['condition']==b and r['endpoint']==endpoint) for rep in sig['replicates']]
            comparisons[endpoint][contrast] = {'cka':contrasts([x['mean']['correct_cka']-y['mean']['correct_cka'] for x,y in zip(aa,bb)],sig['replicates']),
                'accuracy_pp':contrasts([100*(x['accuracy']-y['accuracy']) for x,y in zip(aa,bb)],sig['replicates']),
                'residual_relation_contrast':contrasts([x['mean']['answer_residual_contrast']-y['mean']['answer_residual_contrast'] for x,y in zip(aa,bb)],sig['replicates'])}
    init = [r for r in rows if r['condition']=='initialization']; initialization = {k:float(np.mean([r['mean'][k] for r in init])) for k in init[0]['mean']}
    summary = {'created_utc':now(),'fixed_final_primary':True,'means':means,'contrasts':comparisons,'initialization':initialization,
        'labels_including_validation':256,'unlabeled_pool_size':4096,'paired_repeats_reused':6,'independent_sources':3,'new_test_examples':2560,
        'spontaneous_discovery_tested':False,'output_independent_geometry_confirmed':False}
    atomic_json(root/'summary.json',summary); return summary


def report(plan,root,sig):
    summary = json.loads((root/'summary.json').read_text()); geometry = json.loads((root/'geometry.json').read_text())['records']
    final = summary['means']['final']; comparisons = summary['contrasts']['final']
    fig,axs = plt.subplots(1,2,figsize=(14,5),constrained_layout=True); colors = {17:'#3073ae',42:'#b56230',101:'#557d40'}
    for i,rep in enumerate(sig['replicates']):
        rows = [next(r for r in geometry if r['replicate']==rep['id'] and r['condition']==c and r['endpoint']=='final') for c in plan['conditions']]
        for ax,values in zip(axs,[[r['mean']['correct_cka'] for r in rows],[100*r['accuracy'] for r in rows]]):
            ax.plot(np.arange(6),values,'.-',alpha=.65,color=colors[rep['source_seed']],label=f"{rep['id']} / teacher {rep['source_seed']}")
    axs[0].axhline(summary['initialization']['correct_cka'],color='gray',linestyle='--',label='Mean target initialization')
    axs[0].set_ylabel('Fresh-test length-averaged CKA'); axs[0].set_title('Primary geometry: fixed update 1200')
    axs[1].set_ylabel('Fresh-test target accuracy (%)'); axs[1].set_title('Same fixed-update models')
    for ax in axs: ax.set_xticks(np.arange(6),SHORT,rotation=25,ha='right'); ax.grid(axis='y',alpha=.2)
    axs[0].legend(fontsize=7,ncol=2)
    fig.savefig(root/'coverage_geometry_accuracy.png',dpi=180); fig.savefig(root/'coverage_geometry_accuracy.pdf'); plt.close(fig)
    rows = [[LABELS[c],f"{final[c]['pool_cka']:.4f}",f"{final[c]['correct_cka']:.4f}",f"{final[c]['answer_residual_contrast']:+.4f}",f"{100*final[c]['accuracy']:.2f}%",f"{100*summary['means']['selected'][c]['accuracy']:.2f}%"] for c in plan['conditions']]
    cr = []
    for contrast in plan['primary_contrasts']:
        a,b = contrast.split('-'); stats = comparisons[contrast]; g = stats['cka']; acc = stats['accuracy_pp']
        cr.append([LABELS[a]+' − '+LABELS[b],f"{g['mean']:+.4f}",f"{g['positive_replicates']}/6",f"{g['teacher_cluster_95'][0]:+.4f} 至 {g['teacher_cluster_95'][1]:+.4f}",f"{acc['mean']:+.2f}",f"{acc['positive_replicates']}/6",f"{acc['teacher_cluster_95'][0]:+.2f} 至 {acc['teacher_cluster_95'][1]:+.2f}"])
    positive = lambda s: all(x>0 for x in s['teacher_cluster_means'].values())
    main = [comparisons[c] for c in plan['primary_contrasts'][:3]]
    if all(positive(s['cka']) for s in main):
        interpretation = '正确对齐相对普通曝光、只对齐原标签集、预测答案错配，在三个教师簇均值上都有正的新输入CKA差值。这是本设置下对齐泛化的候选证据；仍须对照答案残差与蒸馏，不能据此确认输出独立的代数机制。'
    else:
        interpretation = '未同时得到正确对齐相对普通曝光、只对齐原标签集、预测答案错配的三个教师簇一致正优势。增加输入覆盖是否形成关系特异的可推广对齐，仍不能直接确认。单个平均CKA上升不能代替全部对照。'
    accstat = comparisons['coverage_alignment-ordinary_exposure']['accuracy_pp']
    interpretation += ' 正确对齐相对普通曝光的平均准确率差为 '+f"{accstat['mean']:+.2f}"+' 个百分点；功能收益须结合逐次差值和输出蒸馏对照判断。'
    content = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>增加无标签取逆约束覆盖</title><style>body{font-family:system-ui;max-width:1350px;margin:32px auto;padding:0 18px;line-height:1.7}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ddd;padding:8px}th{background:#edf2f7}img{width:100%}.scroll{overflow:auto}.callout{background:#eef4fa;padding:16px;border-left:4px solid #3978ad}pre{background:#f2f4f7;padding:12px;overflow:auto}</style><body><h1>增加无标签排列覆盖：取逆对齐能否推广？</h1>'
    content += '<p class="callout">'+escape(interpretation)+'</p><p>主要端点固定1200步；新测试共2560例。每长度128例CKA，五长度等权。六次配对重复复用上一轮标签与初始化，并非六次新的源模型训练。先看几何，再看同一模型的准确率；没有根据本测试选择权重、预算或任务。</p><img src="coverage_geometry_accuracy.png" alt="六次固定最终步数的几何与准确率">'
    content += '<h2>固定最终步数的主要结果</h2>'+table(['条件','无标签池CKA','新测试CKA','答案均值去除后关系CKA差','新测试准确率','验证选定准确率'],rows)
    content += '<h2>预先固定的配对比较</h2><p>几何差值是CKA单位；准确率差值是百分点。所列区间为3³次教师簇经验bootstrap的描述性95%区间。完整6⁶次条件区间与所有逐次差值见下方及summary.json。共享历史数据世界和仅三个教师限制了外推；未作多重比较校正。</p><div class="scroll">'+table(['比较','CKA均值差','CKA正重复','教师簇区间','准确率均值差','准确率正重复','教师簇区间'],cr)+'</div>'
    for metric, title in [('correct_cka','逐次新测试CKA'),('accuracy','逐次新测试准确率')]:
        rr = []
        for rep in sig['replicates']:
            values = [next(r for r in geometry if r['replicate']==rep['id'] and r['condition']==c and r['endpoint']=='final') for c in plan['conditions']]
            rr.append([rep['id']+f" / 教师 {rep['source_seed']}"]+[f"{r['mean'][metric]:.4f}" if metric!='accuracy' else f"{100*r['accuracy']:.2f}%" for r in values])
        content += '<h2>'+title+'</h2>'+table(['重复']+list(LABELS.values()),rr)
    content += '<h2>标签、曝光与监督的区别</h2><p>数学规则 recoils(x)=descents(x⁻¹) 由实验给出。每次沿用192个训练和64个验证目标标签及原随机初始化；新增4096个无标签排列，真实答案未生成或用于训练。各条件每步看到同长度的32个标签输入和32个无标签输入，均训练1200步，使用同一批次顺序、随机种子、优化器与学习率。</p><p>普通曝光的新输入损失权重为零，计算这些输入本身不构成学习目标。“只对齐原标签集”也计算新输入，但几何梯度仅来自标签批次；扩大覆盖条件在合格的合并批次施加1−线性CKA，固定权重0.3。蒸馏对合并批次使用教师完整输出、温度2、权重1。所有损失项在所有条件都计算，按固定条件赋权。</p><p>错配在原标签集按真实答案＋长度匹配，在无标签池按冻结教师预测答案＋长度匹配；单例在所有覆盖几何条件共同排除。教师输出属于新增的预训练源知识，输出蒸馏会提供伪监督；标签预算不变不能理解为有效监督信息完全不变。预测答案匹配控制也不能单独排除全部输出信息。</p>'
    exposure = []
    for rep in sig['replicates']:
        rec = json.loads((root/'training'/f"{rep['id']}_ordinary_exposure.json").read_text())
        exposure.append([rep['id'],rec['distinct_unlabeled_exposed'],rec['unlabeled_available'],rec['labeled_forward_examples'],rec['unlabeled_forward_examples']])
    content += '<h2>实际曝光</h2>'+table(['重复','独立新输入已曝光','可用新输入','标签前向次数','无标签前向次数'],exposure)+'<p>同一次重复中六种条件具有相同初始化、输入及曝光数。原标签/验证集是复用数据；新池与新测试完整八状态轨道与所有原始1600万条输入、530040个旧本地输入及其他新队列分离。</p>'
    teacher_rows = []
    data = np.load(root/'dataset/test/dataset.npz')
    for source in sig['sources']:
        logits = np.load(root/'teacher'/f"test_s{source['seed']}.npz")['logits']; hit = logits.argmax(-1)==data['labels']
        teacher_rows.append([source['seed'],f'{100*hit.mean():.2f}%']+[f"{100*hit[data['lengths']==n].mean():.2f}%" for n in plan['lengths']])
    content += '<h2>冻结源模型的新测试表现</h2>'+table(['源种子','总体']+plan['lengths'],teacher_rows)+'<p>源模型在本轮没有再训练，原独立可靠性审计已通过。新测试源成绩在全部目标拟合结束后才计算；没有据此筛选来源或调整方案。</p>'
    content += '<h2>辅助几何与敏感性</h2><p>初始化的新测试CKA均值为 '+f"{summary['initialization']['correct_cka']:.4f}"+'。答案匹配/去均值操作只在测试测量中使用真实答案，保留至少3例的共同答案层；这不会向训练提供真实无标签答案，也不会消除全部输出信息。无标签池CKA与新测试CKA均每长度128例，避免样本数不同导致的CKA比较误读。验证选定端点固定使用原64个验证标签，不选择CKA；全部结果保留在geometry.json和summary.json。</p>'
    verification = json.loads((root/'verification.json').read_text()) if (root/'verification.json').exists() else {'status':'pending'}
    content += '<h2>独立核验</h2><pre>'+escape(json.dumps(verification,ensure_ascii=False,indent=2))+'</pre>'
    links = [('protocol.json','主规则与注册时间'),('analysis_protocol.json','分析规则'),('dataset/audit.json','数据审计'),('summary.json','全部配对统计'),('geometry.json','逐次逐长度几何'),('verification.json','独立核验'),('completion.json','完成与哈希清单'),('coverage_geometry_accuracy.pdf','可导出图PDF')]
    content += '<p>'+' · '.join('<a href="'+p+'">'+escape(label)+'</a>' for p,label in links)+'</p><pre>OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_alignment_coverage train\nOPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_alignment_coverage evaluate\nOPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_coverage_analysis analyze\nOPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_coverage_verify\nOPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_coverage_analysis report\nOPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_coverage_finalize</pre></body></html>'
    (root/'report.html').write_text(content)
    print(json.dumps({'means':summary['means']['final'],'contrasts':summary['contrasts']['final']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage',choices=['register','analyze','report']); stage = parser.parse_args().stage; plan,root,sig = initialize()
    if stage=='register': register(root)
    elif stage=='analyze': analyze(plan,root,sig)
    else: report(plan,root,sig)
