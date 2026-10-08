"""Present mechanism diagnosis, additional contrasts and numerical limits."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .longrun_engine import atomic_json
from .operator_capacity_confirmation import crossed_interval
from .operator_capacity_verify import replay
from .operator_mechanism_diagnostic import single_objective, nested_interval
from .permworld_combinations import sha
from .two_step_relation_factorial import now

ROOT=Path('results/operator_mechanism_diagnostic')
PARENT=Path('results/operator_capacity_confirmation')


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def run():
    config=json.loads(Path('configs/operator_mechanism_diagnostic.json').read_text())
    results=json.loads((ROOT/'results.json').read_text());fits=json.loads((ROOT/'fit_summary.json').read_text())
    verification=json.loads((ROOT/'independent_verification.json').read_text())
    assert verification['status']=='passed' and fits['all_six_convex_gap_certificates_pass']
    tests=(ROOT/'pytest.log').read_text();assert 'passed' in tests and 'failed' not in tests.lower()
    # Describe all six additional optimization contrasts, rather than selecting favorable ones.
    coefficient_sets={
        'optimized_linear_correct_minus_wrong':{'convex_kd_geometry_correct':1,'convex_kd_geometry_wrong':-1},
        'nonlinear_correct_minus_wrong':{'nonlinear_correct':1,'nonlinear_wrong':-1},
        'remaining_nonlinear_relation_interaction':{'nonlinear_correct':1,'nonlinear_wrong':-1,
            'convex_kd_geometry_correct':-1,'convex_kd_geometry_wrong':1},
        'optimization_gain_correct':{'convex_kd_geometry_correct':1,'linear_correct':-1},
        'optimization_gain_wrong':{'convex_kd_geometry_wrong':1,'linear_wrong':-1},
        'recovered_relation_advantage':{'convex_kd_geometry_correct':1,'convex_kd_geometry_wrong':-1,
            'linear_correct':-1,'linear_wrong':1}}
    additional=[]
    for test in ['existing','fresh']:
        td=[dict(np.load(PARENT/'dataset/test.npz' if test=='existing' else ROOT/'datasets'/f's{i}_fresh.npz')) for i in range(3)]
        for name,coefficients in coefficient_sets.items():
            values=[]
            for i in range(3):
                total=np.zeros(128)
                for condition,weight in coefficients.items():
                    hit=np.load(ROOT/'evaluations'/f'{test}_s{i}_{condition}.npz')['compound_hit']
                    total+=weight*np.array([hit[td[i]['pair_ids']==p].mean() for p in range(128)])
                values.append(100*total)
            values=np.array(values)
            ci=(crossed_interval if test=='existing' else nested_interval)(values,config['bootstrap_samples'],config['bootstrap_seed'])
            additional.append({'test':test,'contrast':name,'mean_pp':float(values.mean()),
                'source_effects_pp':values.mean(1).tolist(),'bootstrap_95_pp':ci})
    atomic_json(ROOT/'additional_optimization_contrasts.json',{'scope':'Additional posthoc contrasts after diagnostics, no refitting or compound-based selection. All six specified contrasts reported.',
        'records':additional})
    originals=[]
    precision=[]
    for i in range(3):
        z=dict(np.load(PARENT/'states'/f's{i}_known.npz'));scale=float(np.var(z['train'][:,1],axis=0).mean())
        w,bw=z['w'].astype(float),z['bias_w'].astype(float)
        for condition in config['conditions']:
            state=torch.load(PARENT/'predictors'/f's{i}_{condition}.pt',map_location='cpu',weights_only=True)['state_dict']
            for split in ['train','validation']:
                hidden=z[split].astype(float)
                pred=replay(hidden[:,0],state,condition.startswith('nonlinear'))
                originals.append({'source':i,'condition':condition,'split':split,
                                  **single_objective(pred,hidden[:,1],w,bw,scale,config)})
        for test in ['existing','fresh']:
            hp=PARENT/'states'/f's{i}_test.npz' if test=='existing' else ROOT/'states'/f's{i}_fresh.npz'
            h=np.load(hp)['hidden'][:,0].astype(np.float32)
            td=np.load(PARENT/'dataset/test.npz' if test=='existing' else ROOT/'datasets'/f's{i}_fresh.npz')
            x=np.column_stack([h,np.ones(len(h),np.float32)]);use=td['split']==1
            for pairing in ['correct','wrong']:
                for kind,theta in dict(np.load(ROOT/'fits'/f's{i}_{pairing}.npz')).items():
                    first32=x@theta.astype(np.float32);first64=x.astype(float)@theta
                    assert first32.dtype==np.float32
                    y32=((first32@z['b']+z['bias_b'])@z['w'].T+z['bias_w']).argmax(1)
                    y64=((first64@z['b'].astype(float)+z['bias_b'].astype(float))@w.T+bw).argmax(1)
                    precision.append({'source':i,'test':test,'kind':kind,'pairing':pairing,
                        'float32_accuracy':float((y32[use]==td['labels'][use,3]).mean()),
                        'float64_accuracy':float((y64[use]==td['labels'][use,3]).mean()),
                        'first_max_rounding_difference':float(np.abs(first32-first64).max())})
    atomic_json(ROOT/'original_known_scores.json',{'scope':'Unchanged old predictor known-only train/validation scores; no selection.', 'records':originals})
    atomic_json(ROOT/'precision_audit.json',{'scope':'Posthoc evaluation-only float32 versus float64 arithmetic; no fitting or selection.', 'records':precision})

    means={(r['test'],r['condition']):r for r in results['means'] if r['split']=='collisions'}
    fresh={c:r for (t,c),r in means.items() if t=='fresh'}
    effects={r['contrast']:r for r in results['contrasts'] if r['test']=='fresh' and r['endpoint']=='accuracy'}
    def interval(r):
        return f'{r["mean_pp"]:+.2f} [{r["bootstrap_95_pp"][0]:+.2f}, {r["bootstrap_95_pp"][1]:+.2f}]'
    labels={'linear_correct':'原同预算线性正确','linear_wrong':'原同预算线性错配',
        'nonlinear_correct':'原同预算 GELU 正确','nonlinear_wrong':'原同预算 GELU 错配',
        'swap_linear_correct_null_nonlinear_correct':'线性读出分量＋正确 GELU 零空间',
        'swap_linear_correct_null_nonlinear_wrong':'线性读出分量＋错配 GELU 零空间',
        'swap_nonlinear_correct_null_linear_correct':'GELU 读出分量＋线性零空间',
        'convex_kd_geometry_correct':'充分拟合线性：原 KD＋几何目标',
        'state_ols_correct':'线性状态 MSE 全局最小解',
        'validation_ridge_state_correct':'单步验证选择的正则状态线性解'}
    selected=['linear_correct','swap_linear_correct_null_nonlinear_correct','swap_linear_correct_null_nonlinear_wrong',
              'nonlinear_correct','swap_nonlinear_correct_null_linear_correct','convex_kd_geometry_correct',
              'state_ols_correct','validation_ridge_state_correct']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True,gridspec_kw={'width_ratios':[1.4,1]})
    short=['Original\nlinear','Correct GELU\nnull donor','Wrong GELU\nnull donor','Original\nGELU',
           'GELU with\nlinear null','Converged\nlinear KD','Global affine\nstate OLS','Validation\nridge state']
    colors=['#779ab7','#c0672a','#b5c4cf','#c0672a','#d5ad90','#32658b','#32658b','#a9c0d0']
    for j,c in enumerate(selected):
        values=[100*r['accuracy'] for r in results['records'] if r['test']=='fresh' and r['condition']==c and r['split']=='collisions']
        axes[0].bar(j,np.mean(values),color=colors[j],width=.65)
        axes[0].scatter(j+np.array([-.12,0,.12]),values,s=20,color='#222222',zorder=3)
        axes[0].text(j,max(values)+1.2,f'{np.mean(values):.2f}',ha='center',fontsize=8)
    axes[0].set_xticks(range(len(selected)),short,fontsize=7)
    axes[0].set_ylabel('Fresh collision AB accuracy (%)');axes[0].set_ylim(0,35)
    axes[0].set_title('A. Predicted-state swaps and separately budgeted affine diagnosis')
    effect_names=['correct_null_swap_gain','correct_vs_wrong_null_donor','remove_gelu_correct_null','convex_linear_vs_gelu']
    for j,name in enumerate(effect_names):
        row=effects[name];lo,hi=row['bootstrap_95_pp'];mu=row['mean_pp']
        axes[1].errorbar(j,mu,yerr=[[mu-lo],[hi-mu]],fmt='o',capsize=4,color='#27577a')
        axes[1].scatter(j+np.array([-.12,0,.12]),row['source_effects_pp'],marker='x',s=25,color='#b75127')
    axes[1].axhline(0,color='gray',lw=1);axes[1].set_ylabel('Paired accuracy difference (percentage points)')
    axes[1].set_xticks(range(4),['Correct-null\nswap gain','Correct − wrong\nnull donor','Loss after\nremoving GELU null','Converged\nlinear − GELU'],fontsize=8)
    axes[1].set_title('B. Fresh per-source inputs; nested source/pair 95% intervals')
    for ext in ['png','svg','pdf']:fig.savefig(ROOT/f'mechanism_diagnosis.{ext}',dpi=180)
    plt.close(fig)
    conclusion='预测零空间分量的替换支持一条局部功能路径；充分拟合显著缩小原线性与 GELU 的复合差距，提示优化／数值条件影响。GELU 的状态拟合仍更好，但剩余复合准确率差尚不明确，不能把主收益全部归因于线性表达限制。'
    text='<!doctype html><html lang="zh"><meta charset="utf-8"><title>算子机制诊断</title><style>body{font:16px/1.75 system-ui;max-width:1280px;margin:32px auto;padding:0 20px;color:#18232d}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{border:1px solid #d7dee3;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%}.lead{padding:18px;background:#eef4f7;border-left:4px solid #4682b4}</style><h1>GELU 为什么扩大正确配对收益：机制与优化诊断</h1>'
    text+='<p class="lead">'+conclusion+'</p><p>保留上一轮 12 个同预算模型及报告。本轮重用三个冻结来源，完成 6 个正确／错配充分拟合诊断，每个保存 OLS、原 KD＋几何凸优化和单步验证选择的 ridge 状态解，共 18 个线性诊断解。参数化与额外求解预算不同，不能当作另一轮同预算比较。</p>'
    text+='<img src="mechanism_diagnosis.svg" alt="新输入上的预测零空间替换及充分拟合线性对照">'
    text+='<h2>预测分量替换与功能结果</h2>'+table(['条件','旧碰撞 AB','新碰撞 AB','新配对双正确','新单步 A'],
        [[labels[c],f'{100*means[("existing",c)]["accuracy"]:.2f}%',f'{100*fresh[c]["accuracy"]:.2f}%',
          f'{100*fresh[c]["pair_both_correct"]:.2f}%',f'{100*fresh[c]["first_accuracy"]:.2f}%'] for c in selected])
    text+=f'<p>零空间替换只使用两个模型自己的预测：同一坐标系中的读出分量来自受体，零空间来自供体；没有真实中间状态参与替换。全部第一步 logits 的最大变化为 {results["max_swap_first_logit_change"]:.3g}，softmax 概率最大变化为 {results["max_swap_first_probability_change"]:.3g}，分类全部保持。正确供体带来新输入上的 {interval(effects["correct_null_swap_gain"])} pp；错配供体增量为 {interval(effects["wrong_null_swap_gain"])} pp。反向替换正确 GELU 的零空间后，成绩降低 {interval(effects["remove_gelu_correct_null"])} pp。</p>'
    text+='<h2>误差经过冻结第二算子后的影响</h2>'+table(['条件','总状态 NMSE','零空间 NMSE','读出方向 NMSE','下游零空间误差','下游总误差','零空间越过正确间隔'],
        [[labels[c]]+[f'{fresh[c][k]:.5f}' for k in ['hidden_nmse','null_nmse','row_nmse','downstream_null_normalized','downstream_total_normalized']]+
         [f'{100*fresh[c]["null_margin_crossing_fraction"]:.2f}%'] for c in ['linear_correct','nonlinear_correct','convex_kd_geometry_correct','state_ols_correct']])
    text+='<p>这里 e=预测第一状态−真实第一状态，d_null=e P_null B Wᵀ。下游误差先去除每行共同 logit 平移，再以真实中间状态经 B 的输出分数能量归一化。真实中间状态仅用于诊断。总下游平方误差包含读出与零空间分量的交叉项，不能把两个能量直接解释成贡献百分比。充分拟合线性与 GELU 的零空间总 NMSE接近，GELU 的下游误差和间隔越过率更低，说明误差方向也需考虑。</p>'
    text+='<h2>充分拟合：优化收益与残余差距</h2>'+table(['预定诊断','新测试均值 [描述性 95% 区间] pp'],
        [['充分拟合线性 − GELU',interval(effects['convex_linear_vs_gelu'])],['OLS 状态线性 − GELU',interval(effects['ols_linear_vs_gelu'])],
         ['验证选择的 ridge 状态线性 − GELU',interval(effects['ridge_linear_vs_gelu'])]])
    text+=table(['补充事后对比','新测试均值 [描述性 95% 区间] pp'],[[r['contrast'],interval(r)] for r in additional if r['test']=='fresh'])
    known=[]
    for kind in ['linear_correct','nonlinear_correct','state_ols','convex_kd_geometry','validation_ridge_state']:
        if kind in ['linear_correct','nonlinear_correct']:
            train=np.mean([r['true_state_nmse'] for r in originals if r['condition']==kind and r['split']=='train'])
            val=np.mean([r['true_state_nmse'] for r in originals if r['condition']==kind and r['split']=='validation'])
        else:
            chosen=[r['scores'][kind] for r in fits['records'] if r['pairing']=='correct']
            train=np.mean([r['train']['true_state_nmse'] for r in chosen]);val=np.mean([r['validation']['true_state_nmse'] for r in chosen])
        known.append([kind,f'{train:.5f}',f'{val:.5f}'])
    text+=table(['正确组','单步训练 NMSE','单步验证 NMSE'],known)
    text+='<p>OLS 给出这批训练样本上无约束仿射族的最小状态平方误差，GELU 在训练和验证上仍有更低的状态误差。这是当前数据／表征下的拟合差距；不能据此断言所有线性关系推断理论上不可能。凸优化拟合原始 KD＋几何的数据目标，读出零空间由 OLS精确解出，读出行空间使用白化后的双精度 L-BFGS。未复制因子化 AdamW 的参数衰减；参数化、精度及预算变化都可能有作用。复合结果的区间跨零不能作为两个方法等价的检验。</p>'
    solver_rows=[]
    for r in fits['records']:
        solver_rows.append([r['source'],r['pairing'],f'{r["solver"]["condition_number"]:.3g}',r['solver']['lbfgs_iterations'],
            f'{r["solver"]["training_optimality_gap_upper_bound"]:.3g}',r['ridge_selected_lambda'],
            f'{r["scores"]["convex_kd_geometry"]["affine_coefficient_norm"]:.3g}'])
    text+=table(['源','配对','特征条件数','L-BFGS 迭代','训练最优性 gap 上界','验证选 ridge λ','充分拟合系数范数'],solver_rows)
    text+='<p>六个凸解均通过预定的训练最优性误差上界，独立解析梯度复算也通过。条件数约 10⁸，无正则解系数达到 10⁶，可能利用浮点舍入产生的近 LayerNorm 约束方向，因此单独保留正则化与推断精度检查，不把无正则解当作稳健部署方案。</p>'
    precision_rows=[]
    for test in ['existing','fresh']:
        for kind in ['state_ols','convex_kd_geometry','validation_ridge_state']:
            chosen=[r for r in precision if r['test']==test and r['kind']==kind and r['pairing']=='correct']
            precision_rows.append([test,kind,f'{100*np.mean([r["float64_accuracy"] for r in chosen]):.2f}%',
                                   f'{100*np.mean([r["float32_accuracy"] for r in chosen]):.2f}%'])
    text+=table(['测试','正确线性解','float64 复合','float32 复合'],precision_rows)
    text+='<h2>复核范围</h2><p>每源新测试包含 128 个 IID 输入、128 对碰撞样本，共 384 个整组轨道；三个新测试共 1,152 个不同轨道。每个模型的新输入排除其来源、适应训练、验证轨道，以及三轮历史测试。新输入可以是另一个源的训练输入，因此是相对被评估模型的留出。冻结来源重用，不是新来源复现。所有线性拟合只使用旧单步训练和验证信息，完整结束后才打开新复合测试；本轮不再根据复合结果调参。</p>'
    text+='<p>零空间仅针对固定第一步的全部线性 logits，不排除更广义的答案信息；混合表征也可能不在编码器流形上。这些干预说明当前复合计算对该分量敏感，不证明网络自发发现代数关系。旧测试使用配对的交叉源／样本 bootstrap，新测试按源独立样本分层重采样；三个复用源的区间均仅用于描述。</p>'
    text+=f'<p>{verification["checks"]:,} 项独立检查通过；原 83 项父研究产物哈希保持；测试记录：{html.escape(tests.strip())}。</p>'
    text+='<p><a href="results.json">逐模型和干预数据</a> · <a href="fit_summary.json">充分拟合与验证选择</a> · <a href="protocol.json">诊断前方案</a> · <a href="independent_verification.json">独立复算</a> · <a href="mechanism_diagnosis.pdf">导出图 PDF</a> · <a href="../operator_capacity_confirmation/report.html">原同预算确认</a></p></html>'
    (ROOT/'report.html').write_text(text)
    paths=[p for p in ROOT.rglob('*') if p.is_file() and p.name not in ['state.json','delivery.json','report_build.log']]
    paths += [Path(__file__),Path('experiments/operator_mechanism_diagnostic.py'),Path('experiments/operator_mechanism_verify.py'),
              Path('configs/operator_mechanism_diagnostic.json'),Path('tests/test_operator_mechanism.py')]
    atomic_json(ROOT/'delivery.json',{'status':'complete','completed_utc':now(),'reused_sources':3,'affine_diagnostic_solutions':18,
        'fresh_test_orbits':1152,'conclusion':conclusion,'pytest':tests.strip(),'independent_verification_checks':verification['checks'],
        'report':str(ROOT/'report.html'),'original_equal_budget_study_unchanged':True,
        'artifact_sha256':{str(p):sha(p) for p in paths}})
    atomic_json(ROOT/'state.json',{'status':'complete','updated_utc':now()})
    print(json.dumps({'status':'complete','conclusion':conclusion,'fresh_core_contrasts':effects}),flush=True)


if __name__=='__main__':run()
