"""Build a separate, evidence-linked review without altering completed studies."""
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now
from .two_step_relation_final_report import table

ROOT=Path('results/algebra_relation_review')
NAMES={'both_correct':'双正确','c_correct_i_wrong':'仅补排列正确','c_wrong_i_correct':'仅取逆正确',
       'both_wrong':'双错配','a_correct_b_wrong':'仅A正确','a_wrong_b_correct':'仅B正确',
       'no_geometry':'无几何约束','output_space':'输出空间诊断'}


def read(path):return json.loads(Path(path).read_text())


def run():
    ROOT.mkdir(exist_ok=True)
    main=read('results/two_step_relation_dose_confirmation/summary.json')
    diagnostic=read('results/relation_error_localization/results.json')
    baseline=read('results/two_step_relation_output_only20/results.json')
    baseline_verify=read('results/two_step_relation_output_only20/statistics_verification.json')
    diagnostic_verify=read('results/relation_error_localization_verification/verification.json')
    factors=[r for r in main['means'] if (r['view'],r['split'],r['word'])==('hidden','collisions','ci')]
    b=[r for r in baseline['records'] if (r['split'],r['word'])==('collisions','ci')]
    factors.append({'condition':'no_geometry','accuracy':float(np.mean([r['accuracy'] for r in b])),
                    'pair_both_correct':float(np.mean([r['pair_both_correct'] for r in b])),
                    'cka':float(np.mean([r['cka'] for r in b]))})
    d=[r for r in diagnostic['means'] if (r['endpoint_epochs'],r['split'])==(20,'collisions')]
    frozen=read('results/two_step_relation_factorial/summary.json')
    prior=[r for r in frozen['means'] if (r['word'],r['split'],r['condition'])==('ci','collisions','both_correct')]
    pieces=['<h1>关系正确性、复合收益与跨领域检验</h1><p>'+html.escape(now())+'</p>',
        '<p><strong>本轮结论：PermWorld存在局部关系正确性收益；矩阵与多项式没有复现稳定复合优势。原生或单步预测准确，仍不保证中间表征可以连续使用。不能据此宣称普通训练自发形成了可推广的代数机制。</strong></p>',
        '<p>当前主张：在正确输出监督相同的条件下，关系配对的正确性可能产生额外的复合收益；必须分别检验关系类型、生成元泛化和串联稳定性。现有结果不证明自发发现代数，也不证明独立于输出的机制。</p>',
        '<h2>PermWorld：20轮预算一致的功能比较</h2>',
        table(['几何条件','碰撞复合准确率','配对双正确','CKA'],[[NAMES[r['condition']],
              f'{100*r["accuracy"]:.2f}%',f'{100*r["pair_both_correct"]:.2f}%',f'{r["cka"]:.4f}'] for r in factors]),
        '<p>双正确−双错配：+6.33个百分点；双正确−仅补排列正确：+0.78个百分点，后者区间跨零。补排列正确性贡献较大，不把双正确必须最好当作预设目标。六次拟合复用三个源模型。</p>',
        '<p>双正确−20轮无几何：+14.31个百分点，三源条件bootstrap区间[12.82,16.03]。这只重采样三个源、固定测试输入；新增20轮基线登记在主20轮结果已观察之后，零系数沿用先前定义，无结果调参。</p>',
        '<p><a href="../two_step_relation_output_only20/statistics_verification.json">20轮基线独立核对</a> · <a href="../two_step_relation_dose_confirmation/summary.json">四条件完整结果</a></p>',
        '<figure><img src="permworld_diagnostic.svg" alt="PermWorld五种条件的复合准确率，以及使用真实中间信息的修复诊断" style="width:100%"><figcaption>黑点：三个来源的均值。右图修复使用真实中间表征，不计为隐藏推断成绩。<a href="permworld_diagnostic.pdf">独立PDF图</a></figcaption></figure>',
        '<h2>误差定位：预测中间状态与真实中间状态</h2>',
        table(['条件','单步C','单步I','串联CI','真实C后接I','修复C读出方向','修复C读出零空间'],[[NAMES[r['condition']]]+
              [f'{100*r[k]:.2f}%' for k in ['C_accuracy','I_accuracy','composed_accuracy','true_intermediate_accuracy',
                 'repair_C_readout_rows_accuracy','repair_C_readout_null_accuracy']] for r in d]),
        '<p>真实中间表征及修复均为oracle诊断，不能作为自行推断成绩。独立最小二乘复核48个真实模型缓存：修复零空间保留第一步全部31个数字logits，最大改变'+
        f'{diagnostic_verify["max_all31_first_step_logits_change"]:.2e}'+'，同时复现最终预测变化。第一步答案正确仍不保证中间表征可供第二步使用。</p>',
        '<p>平方误差=第一步传播项+第二步真中间误差项+交叉项。报告中的零空间能量比例排除了交叉项，不能解释为因果贡献百分比；线性读出零空间仍可含有其他答案信息。</p>',
        '<p><a href="../relation_error_localization/report.html">完整定位诊断</a> · <a href="../relation_error_localization_verification/verification.json">独立不变性验证</a></p>',
        '<h2>已有输出空间对照的解释范围</h2>',
        table(['旧冻结实验空间','复合准确率','真实中间准确率','配对双正确'],[[r['view'],f'{100*r["accuracy"]:.2f}%',
              f'{100*r["teacher_forced_accuracy"]:.2f}%',f'{100*r["pair_both_correct"]:.2f}%'] for r in prior]),
        '<p>这些旧算子实验固定编码器，不能当作联合训练、参数量匹配的因果对照。新领域输出组与其他组共享编码器训练和曝光，但算子参数量更小；须同时展示其单步成绩。</p>',
        '<h2>跨领域共用协议与选择记录</h2>',
        '<p>F13、四维数值输入、128维普通GELU编码器、3个源种子、6次适配，每组1024个支持输入、900轮；四配对条件、无几何、输出空间共六组。输出监督始终正确，五次原生查询及整轮边际曝光一致。每个领域36组全部完成才打开复合测试。</p>',
        '<p>初始one-hot/数值编码器的矩阵生成元训练成绩99.8%，留出第二生成元7.8%，未通过门槛。仅用可见状态验证选择数值坐标GELU模型，两个独立可见验证集通过后固定；多项式使用相同架构。旧PermWorld为Transformer与双向监督，不能描述成完全相同架构或损失。</p>',
        '<p><a href="../algebra_relation_feasibility/results.json">仅可见状态的编码选择</a> · <a href="../../configs/algebra_relation_common_v3.json">固定共用配置</a></p>']
    cross=[]
    for domain,label in [('matrix','有限矩阵群'),('polynomial','不可逆多项式')]:
        folder=Path('results/algebra_relation_v3')/domain
        pieces.append('<h2>'+label+'</h2>')
        if not (folder/'state.json').exists():
            pieces.append('<p>尚未启动，顺序等待前一领域完成。</p>');cross.append({'domain':domain,'status':'not_started'});continue
        state=read(folder/'state.json')
        if state['status']!='complete':
            pieces.append('<p>状态：'+html.escape(state['status'])+'。复合结果不填入；可见任务未通过门槛时不打开复合测试。</p>')
            cross.append({'domain':domain,**state});continue
        summary=read(folder/'summary.json')
        verified=Path('results/algebra_relation_v3')/f'{domain}_verification/verification.json'
        assert verified.exists(),str(verified)
        v=read(verified);rows=[r for r in summary['means'] if r['split']=='collisions']
        pieces.append(table(['条件','碰撞AB准确率','配对双正确','预测误差（各自空间）','CKA','真实中间诊断'],
            [[NAMES[r['condition']],f'{100*r["accuracy"]:.2f}%',f'{100*r["pair_both_correct"]:.2f}%',
              f'{r["prediction_nmse"]:.4f}',f'{r["cka"]:.4f}',f'{100*r["true_intermediate_accuracy"]:.2f}%'] for r in rows]))
        primitive=[]
        for condition in summary['means']:
            if condition['split']!='collisions':continue
            matches=[r for r in v['primitive_grades'] if r['condition']==condition['condition']]
            primitive.append([NAMES[condition['condition']]]+[f'{100*np.mean([r["test_native"][j] for r in matches]):.2f}%' for j in range(3)]+
                [f'{100*np.mean([r["test_generators"][j] for r in matches]):.2f}%' for j in range(2)])
        pieces.append(table(['条件','原生e','原生A','原生B','算子A','算子B'],primitive))
        pieces.append(table(['双正确对照差值','百分点','三源条件95%区间'],[[NAMES[r['condition']],f'{r["mean_pp"]:.3f}',
             '['+', '.join(f'{x:.3f}' for x in r['three_source_bootstrap_95_pp'])+']'] for r in summary['contrasts'] if r['split']=='collisions']))
        input_file=verified.parent/'input_uncertainty.json'
        if input_file.exists():
            uncertainty=read(input_file)
            chosen=[r for r in uncertainty['records'] if r['split']=='collisions' and r['metric']=='accuracy']
            pieces.append(table(['双正确对照：整对输入与来源重采样','百分点','95%区间'],[
                [NAMES[r['comparison'].replace('both_correct-','')],f'{r["mean_pp"]:.3f}',
                 '['+', '.join(f'{x:.3f}' for x in r['source_and_whole_input_bootstrap95_pp'])+']'
                 if r['source_and_whole_input_bootstrap95_pp'] is not None else '连通块不足，不给出输入区间'] for r in chosen]))
            pieces.append('<p>输入不确定性为补充诊断，定义在矩阵汇总结果观察后、多项式复合结果观察前；不改变模型、测试集或终点。矩阵双正确−双错配的来源＋整对输入区间跨零，不能把较窄的固定输入来源区间当作稳定推广证据。</p>')
        statistics_file=verified.parent/'primary_statistics.json'
        if statistics_file.exists():
            statistics=read(statistics_file)
            chosen=[r for r in statistics['contrasts'] if r['split']=='collisions' and r['metric']=='pair_both_correct'
                    and r['contrast'].startswith('both_correct-')]
            pieces.append(table(['配对双正确增量','百分点','三源固定输入区间'],[[NAMES[r['contrast'].replace('both_correct-','')],
                f'{r["mean"]:.3f}','['+', '.join(f'{x:.3f}' for x in r['three_source_bootstrap_95'])+']'] for r in chosen]))
            laws=[r for r in statistics['auxiliary_law_means'] if r['split']=='collisions']
            law_keys=['A_squared_identity','B_cubed_identity','ABA_equals_B_squared'] if domain=='matrix' else ['T_D_commute','D_fourth_zero_state','T_thirteenth_identity']
            pieces.append(table(['条件']+law_keys,[[NAMES[r['condition']]]+[f'{r[k]:.4f}' for k in law_keys] for r in laws]))
            pieces.append('<p>辅助规律误差使用原生表征的中心化方差归一化。恒等操作的真实位移为零，原始逐终点数据中的位移NMSE没有可解释分母，不用于该规律的比较。原先登记的主AB误差分母非零。</p>')
            pieces.append('<p><a href="../algebra_relation_v3/'+domain+'_verification/primary_statistics.json">三项主要终点、因素主效应与交互</a> · <a href="../algebra_relation_v3/'+domain+'_verification/input_uncertainty.json">输入依赖与补充区间</a></p>')
        correct=next(r for r in summary['means'] if r['split']=='collisions' and r['condition']=='both_correct')
        order=[r for r in summary['records'] if r['condition']=='both_correct' and r['split']=='collisions' and r['word']=='wrong_order_against_ab']
        pieces.append('<p>双正确：直接读取真实AB输入时准确率'+f'{100*correct["direct_native_accuracy"]:.2f}%'+'；生成元自行串联时'+
                      f'{100*correct["accuracy"]:.2f}%'+'。错误顺序对同一AB目标为'+f'{100*np.mean([r["accuracy"] for r in order]):.2f}%'+'。</p>')
        localization_file=verified.parent/'error_localization.json'
        if localization_file.exists():
            local=read(localization_file)
            chosen=[r for r in local['means'] if r['split']=='collisions']
            pieces.append(table(['条件','原串联','真实中间','修复读出零空间','修复读出方向','第一步平方误差传播倍率'],[
                [NAMES[r['condition']]]+[f'{100*r[k]:.2f}%' for k in ['composed_accuracy','true_intermediate_accuracy',
                 'oracle_null_repair_accuracy','oracle_row_repair_accuracy']]+[f'{r["empirical_first_error_amplification"]:.2f}'] for r in chosen]))
            pieces.append('<p>这是事后中间状态定位：修复使用真实中间隐藏向量，不能当作推断成绩；第一步全部数字输出保持不变。传播倍率指平方误差之比，不是向量范数倍率。</p>')
            pieces.append('<p><a href="../algebra_relation_v3/'+domain+'_verification/error_localization.json">新领域误差定位的逐模型结果</a></p>')
        pieces.append('<p>全部双正确来源的生成元是否均达到解释门槛：'+str(summary['all_correct_source_generators_pass_gate'])+
                      '。每个来源复用两个适配池；门槛失败会限制对该领域复合失败的解释。</p>')
        strata_file=verified.parent/'source_strata.json'
        if strata_file.exists():
            strata=read(strata_file)
            pieces.append(table(['源种子序号','已知A','已知B','达到门槛','双正确复合','真实中间复合'],[
                [r['source_index'],f'{100*r["known_generators"][0]:.2f}%',f'{100*r["known_generators"][1]:.2f}%',
                 str(r['passed_known_gate']),f'{100*r["collision_means"]["accuracy"]:.2f}%',
                 f'{100*r["collision_means"]["true_intermediate_accuracy"]:.2f}%'] for r in strata['sources']]))
            pieces.append('<p>来源分层使用事先固定的可见门槛，不替换全部六次拟合的主结果。通过门槛的来源：'+
                          html.escape(str(strata['passed_source_indices']))+'。</p>')
            prior=next(r for r in strata['source_label_prior_baselines'] if r['split']=='collisions')
            pieces.append('<p>仅由原生源标签众数固定的常量基线：'+f'{100*prior["source_mode_accuracy"]:.2f}%'+'；均匀随机猜测的期望：7.69%。众数不使用任何源复合标签或测试标签来选择。</p>')
            pieces.append('<p><a href="../algebra_relation_v3/'+domain+'_verification/source_strata.json">完整来源分层与原生标签先验</a></p>')
        dep=v['collision_block_dependence']
        pieces.append('<p>碰撞输入涉及'+str(dep['split_blocks'])+'个划分块，配对连接后有'+str(dep['connected_components'])+
          '个连通分量。上述区间只重采样三个来源，不能当作全部测试样本独立的区间。隐藏与输出NMSE在不同空间计算，不能直接比较绝对数值。</p>')
        pieces.append('<p>矩阵两操作不交换；多项式平移与求导可交换，多项式不作顺序区分。50%上限仅适用于三个真实可见答案的确定性编码，不适用于完整概率输出或隐藏向量。</p>')
        pieces.append('<p><a href="../algebra_relation_v3/'+domain+'/report.html">逐领域报告</a> · <a href="../algebra_relation_v3/'+domain+'/summary.json">全部逐模型终点</a> · <a href="../algebra_relation_v3/'+domain+'_verification/verification.json">独立复核与单步成绩</a></p>')
        cross.append({'domain':domain,'status':'complete','means':summary['means'],'contrasts':summary['contrasts'],
                      'all_correct_source_generators_pass_gate':summary['all_correct_source_generators_pass_gate'],
                      'joint_input_intervals':uncertainty['records'] if input_file.exists() else [],
                      'collision_dependence':dep})
    pieces+=['<h2>论文与下一步边界</h2><p>论文草稿已按“同输出监督下的正确关系干预”组织。所有结果以实际完成记录填入；多项式报告全部主结果，并明确部分来源没有通过生成元解释门槛。既有群表征方法已能预测操作序列（<a href="https://proceedings.mlr.press/v202/keurti23a.html">Keurti等，ICML2023</a>），本研究不声称首次发现群结构。</p>',
             '<p><a href="../../paper/main.tex">论文LaTeX草稿</a> · <a href="../../paper/references.bib">核实的原始研究引用</a></p>']
    (ROOT/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:16px/1.7 system-ui;max-width:1250px;margin:35px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px;text-align:left}h2{margin-top:35px}</style>'+''.join(pieces))
    atomic_json(ROOT/'results.json',{'updated_utc':now(),'permworld20_means':factors,'localization':d,'cross_domain':cross,
             'baseline_independently_verified':baseline_verify['status']=='complete','localization_verified':diagnostic_verify['status']=='complete'})
    labels=['Both correct','C only correct','I only correct','Both incorrect','No geometry']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    axes[0].bar(np.arange(5),[100*r['accuracy'] for r in factors],color=['#2563eb','#60a5fa','#f59e0b','#ef4444','#9ca3af'])
    axes[0].set_xticks(np.arange(5),labels,rotation=25,ha='right');axes[0].set_ylabel('Collision composition accuracy (%)');axes[0].set_ylim(0,100)
    axes[0].set_title('Matched20-epoch permutation study')
    original=read('results/two_step_relation_dose_confirmation/evaluation_records.json')['records']
    for pos,r in enumerate(factors):
        pool=b if r['condition']=='no_geometry' else [x for x in original if
            x['condition']==r['condition'] and x['word']=='ci' and x['split']=='collisions']
        source_values=[100*np.mean([x['accuracy'] for x in pool if x['replicate'] in [f'n{j}',f'n{j+3}']]) for j in range(3)]
        axes[0].scatter(np.repeat(pos,3)+np.linspace(-.08,.08,3),source_values,color='#111827',s=18,zorder=3)
        axes[0].text(pos,100*r['accuracy']+3,f'{100*r["accuracy"]:.1f}',ha='center',fontsize=9)
    correct=next(r for r in d if r['condition']=='both_correct')
    keys=['composed_accuracy','repair_C_readout_rows_accuracy','repair_C_readout_null_accuracy','true_intermediate_accuracy']
    axes[1].bar(np.arange(4),[100*correct[k] for k in keys],color=['#2563eb','#a78bfa','#8b5cf6','#6d28d9'])
    axes[1].set_xticks(np.arange(4),['Predicted C','Oracle row repair','Oracle null repair','Oracle true C'],rotation=25,ha='right')
    axes[1].set_ylim(0,100);axes[1].set_title('Intermediate-state diagnosis (oracle uses truth)')
    diag_rows=[r for r in diagnostic['records'] if (r['endpoint_epochs'],r['condition'],r['split'])==(20,'both_correct','collisions')]
    for pos,key in enumerate(keys):
        source_values=[100*np.mean([x[key] for x in diag_rows if x['replicate'] in [f'n{j}',f'n{j+3}']]) for j in range(3)]
        axes[1].scatter(np.repeat(pos,3)+np.linspace(-.08,.08,3),source_values,color='#111827',s=18,zorder=3)
        axes[1].text(pos,100*correct[key]+3,f'{100*correct[key]:.1f}',ha='center',fontsize=9)
    fig.supxlabel('Dots: means for each of three pretrained source seeds; oracle repairs use true intermediate representations.',fontsize=9)
    for ext in ['pdf','svg','png']:fig.savefig(ROOT/f'permworld_diagnostic.{ext}',dpi=180)
    plt.close(fig)
    tex=[]
    for result in cross:
        tex.append('\\paragraph{'+result['domain'].capitalize()+'.}')
        if result['status']!='complete':
            tex.append('Execution status: '+result['status'].replace('_',' ')+'; no compound outcome is reported.');continue
        tex.append('\\begin{center}\\begin{tabular}{lrr}\\toprule Condition & Accuracy(\\%) & Double correct(\\%)\\\\\\midrule')
        for r in result['means']:
            if r['split']=='collisions':tex.append(r['condition'].replace('_',' ')+f' & {100*r["accuracy"]:.2f} & {100*r["pair_both_correct"]:.2f}'+'\\\\')
        tex.append('\\bottomrule\\end{tabular}\\end{center}')
        tex.append('All 36 models have the same 900-epoch budget. Source-cluster uncertainty conditions on the shared test inputs. Output-space results are information diagnostics with fewer parameters.')
        if result['domain']=='matrix':
            tex.append('Correct-vs-incorrect collision accuracy differs by0.52 percentage points; supplementary source-plus-whole-pair resampling gives[-1.56,2.73] percentage points. Thus a stable additional functional benefit is not established. Single-step scores are high, and true-intermediate prediction reaches99.22\\%, whereas inferred composition reaches9.44\\%.')
        else:
            tex.append('The all-source correct group reaches8.27\\% versus8.20\\% for incorrect geometry. Translation generator scores are91.80\\%,87.11\\%,87.50\\% across sources; only the first meets the previously fixed90\\% threshold. Its composition accuracy is7.42\\%. Thus neither stable relation-specific benefit nor robust irreversible composition is established. Differentiation at the true intermediate achieves99.87\\% overall.')
            tex.append('The128 collision pairs connect39 derivative blocks into one connected component. They cannot be treated as128 independent block-level repetitions; source intervals condition on this fixed test set.')
    Path('paper/generated_cross_domain.tex').write_text('\n'.join(tex)+'\n')
    atomic_json(ROOT/'completion.json',{'completed_utc':now(),'status':'complete' if all(r['status'] in ['complete','known_pilot_failed'] for r in cross) else 'in_progress',
        'artifact_sha256':{str(p):sha(p) for p in ROOT.iterdir() if p.is_file() and p.name not in ['completion.json','final_verification.json','delivery.json']},
        'review_code_sha256':sha(__file__)})
    print(json.dumps({'report':str(ROOT/'report.html'),'domains':[(r['domain'],r['status']) for r in cross]}),flush=True)


if __name__=='__main__':run()
