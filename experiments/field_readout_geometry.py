"""Post hoc diagnostics of source-head subspaces and feature scale dependence.

This analysis was added after the registered full-hidden ordering failed.
It changes no trained checkpoint, target policy, or saved forecast.
"""
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from .algebra import world
from .analysis import linear_cka
from .field_symmetry import audit_group, group_sources, spectral_power
from .longrun_engine import atomic_json
from .matched_field import controlled_world
from .models import SourceModel
from .permworld_combinations import sha
from .six_hour_report import write_rows


def class_contrast_basis(weights, p):
    weights = np.asarray(weights, dtype=np.float64).reshape(4, p, -1)
    contrasts = (weights-weights.mean(1, keepdims=True)).reshape(4*p, -1)
    _, singular, right = np.linalg.svd(contrasts, full_matrices=False)
    return right[singular > singular[0]*1e-10], contrasts


def standardized(h, ids):
    return (h-h[ids].mean(0))/np.maximum(h[ids].std(0), 1e-4)


def run():
    torch.set_num_threads(4)
    root = Path('results/field_readout_geometry'); root.mkdir(exist_ok=True)
    protocol = {'created_utc':datetime.now(timezone.utc).isoformat(), 'code_sha256':sha(__file__),
                'status':'post hoc after observing registered full-hidden ordering fails in the first cohort',
                'representations':['full hidden','support-style standardized hidden','source-class-contrast row space',
                                   'source-class-contrast null space','class-centered source logits','source probabilities','decoded source one-hot'],
                'gauge_policy':'eight fixed random positive diagonal scales in [1/8,8], reused across all models; inverse scaling of source heads preserves logits',
                'selection':'all 72 source models and all variants, no target readout refitting or forecast modification',
                'scope':'diagnostic of extraction geometry and coordinate scale dependence, not a new prospective replication'}
    atomic_json(root/'protocol.json', protocol)
    rng = np.random.default_rng(20261005)
    scales = [2**rng.uniform(-3, 3, 64) for _ in range(8)]
    rows, gauge_rows, parameter_checks = [], [], []
    for config_path, provider in (('configs/field_symmetry.json',world), ('configs/field_matched_support.json',controlled_world)):
        config = json.loads(Path(config_path).read_text()); source = Path(config['output'])
        cohort = source.name
        for w in config['world_seeds']:
            inputs, _, encoded, basis = provider(config['p'], config['dimension'], w)
            latent = inputs@basis.T % config['p']
            lookup = {tuple(z):i for i,z in enumerate(latent)}
            swapped = latent.copy(); swapped[:,[2,3]] = swapped[:,[3,2]]
            ids = np.array([lookup[tuple(z)] for z in swapped])
            for m, g in product(config['model_seeds'],config['groups']):
                record = json.loads((source/f'{g}_w{w}_m{m}.json').read_text())
                assert record['status'] == 'complete' and record['source_gate_passed']
                checkpoint = source/'checkpoints'/f'{g}_w{w}_m{m}.pt'
                assert sha(checkpoint) == record['checkpoint_sha256']
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                h = np.load(source/f'{g}_w{w}_m{m}_step{config["steps"]}_features.npy')
                weights = state['heads.weight'].numpy().astype(np.float64)
                bias = state['heads.bias'].numpy().astype(np.float64)
                row_basis, contrasts = class_contrast_basis(weights,config['p'])
                relevant = h@row_basis.T
                null = h-relevant@row_basis
                assert np.max(abs(null@contrasts.T)) < 1e-10
                logits = (h@weights.T+bias).reshape(625,4,config['p'])
                centered_logits = logits-logits.mean(-1,keepdims=True)
                exponential = np.exp(logits-logits.max(-1,keepdims=True))
                probability = exponential/exponential.sum(-1,keepdims=True)
                codes = logits.argmax(-1)
                one_hot = np.eye(config['p'])[codes].reshape(625,-1)
                expected = audit_group(g,config['p'])['predicted_categorical_cka']
                assert abs(linear_cka(one_hot,one_hot[ids])-expected) < 1e-12
                support = np.arange(25)
                variants = {'full_hidden':h, 'standardized_hidden':standardized(h,support),
                            'source_contrast_rowspace':relevant,'source_contrast_nullspace':null,
                            'centered_logits':centered_logits.reshape(625,-1),
                            'source_probabilities':probability.reshape(625,-1),'decoded_one_hot':one_hot}
                total_variance = np.square(h-h.mean(0)).sum()
                frequency = {tuple(a*s % config['p']) for s in group_sources(g,config['p']) for a in range(1,config['p'])}
                common = {'cohort':cohort,'group':g,'world_seed':w,'model_seed':m,
                          'ideal_code_cka':expected,'contrast_rank':len(row_basis)}
                for name, feature in variants.items():
                    power = spectral_power(feature,latent,config['p'])
                    rows.append({**common,'representation':name,'cka':linear_cka(feature,feature[ids]),
                                 'source_character_energy_fraction':float(sum(power[k] for k in frequency)/power.sum()),
                                 'variance_relative_to_full_hidden':float(np.square(feature-feature.mean(0)).sum()/total_variance)})
                for k, scale in enumerate(scales):
                    changed = h*scale
                    compensated_logits = (changed@(weights/scale).T+bias).reshape(logits.shape)
                    error = float(abs(compensated_logits-logits).max())
                    assert error < 1e-10 and np.array_equal(compensated_logits.argmax(-1),codes)
                    standard_error = float(abs(standardized(changed,support)-standardized(h,support)).max())
                    assert standard_error < 1e-10
                    power = spectral_power(changed,latent,config['p'])
                    gauge_rows.append({**common,'scale_id':k,'original_cka':record['curve'][-1]['transformed_hidden_cka'],
                                       'scaled_cka':linear_cka(changed,changed[ids]),'logit_max_abs_error':error,
                                       'standardized_feature_max_abs_error':standard_error,
                                       'source_character_energy_fraction':float(sum(power[z] for z in frequency)/power.sum())})
                if w == config['world_seeds'][0] and m == config['model_seeds'][0] and g == 'P':
                    model = SourceModel(encoded.shape[1],config['hidden'],config['features'],config['p'])
                    model.load_state_dict(state)
                    with torch.no_grad():
                        tensor = torch.tensor(encoded); original_logits = model(tensor)
                        scale = torch.tensor(scales[0],dtype=model.heads.weight.dtype)
                        model.encoder.norm.weight.mul_(scale); model.encoder.norm.bias.mul_(scale)
                        model.heads.weight.div_(scale)
                        actual_logits = model(tensor)
                    torch.testing.assert_close(actual_logits,original_logits,atol=2e-5,rtol=2e-5)
                    assert torch.equal(actual_logits.argmax(-1),original_logits.argmax(-1))
                    parameter_checks.append({'cohort':cohort,'logit_max_abs_error':float((actual_logits-original_logits).abs().max()),
                                             'source_decisions_identical':True})
    summaries, ordered = [], []
    for cohort in sorted({r['cohort'] for r in rows}):
        groups = sorted({r['group'] for r in rows if r['cohort']==cohort})
        for name in variants:
            for group in groups:
                cells = [r for r in rows if (r['cohort'],r['representation'],r['group'])==(cohort,name,group)]
                summaries.append({'cohort':cohort,'representation':name,'group':group,'models':len(cells),
                                  'cka':float(np.mean([r['cka'] for r in cells])),
                                  'source_character_energy_fraction':float(np.mean([r['source_character_energy_fraction'] for r in cells]))})
            settings = sorted({(r['world_seed'],r['model_seed']) for r in rows if r['cohort']==cohort})
            for w,m in settings:
                cell = {r['group']:r['cka'] for r in rows if (r['cohort'],r['representation'],r['world_seed'],r['model_seed'])==(cohort,name,w,m)}
                middle = np.mean([cell[g] for g in ('M1','M4') if g in cell]); lower = np.mean([cell[g] for g in ('M2','M3') if g in cell])
                ordered.append({'cohort':cohort,'representation':name,'world_seed':w,'model_seed':m,
                                'P':cell['P'],'middle':float(middle),'lower':float(lower),'ordered':bool(cell['P']>middle>lower)})
    write_rows(root/'representations.csv',rows);write_rows(root/'group_summary.csv',summaries)
    write_rows(root/'paired_orderings.csv',ordered);write_rows(root/'equivalent_scalings.csv',gauge_rows)
    summary = {'protocol':protocol,'models':len(rows)//len(variants),'representation_endpoints':len(rows),'scaling_endpoints':len(gauge_rows),
               'parameter_equivalence_checks':parameter_checks,
               'ordered_settings':{f'{c}_{n}':sum(r['ordered'] for r in ordered if (r['cohort'],r['representation'])==(c,n)) for c in sorted({r['cohort'] for r in rows}) for n in variants},
               'mean_within_model_scaling_cka_range':float(np.mean([max(r['scaled_cka'] for r in gauge_rows if (r['cohort'],r['group'],r['world_seed'],r['model_seed'])==key)-min(r['scaled_cka'] for r in gauge_rows if (r['cohort'],r['group'],r['world_seed'],r['model_seed'])==key) for key in sorted({(r['cohort'],r['group'],r['world_seed'],r['model_seed']) for r in gauge_rows})])),
               'all_source_logits_and_decisions_preserved':True,'support_standardized_features_preserved':True}
    atomic_json(root/'summary.json',summary)
    plot(root,summaries)
    table=''.join(f"<tr><td>{c}</td><td>{n}</td><td>{count}/9</td></tr>" for key,count in summary['ordered_settings'].items() for c,n in [next((c,key[len(c)+1:]) for c in ('field_symmetry','field_matched_support') if key.startswith(c+'_'))])
    (root/'report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>源读出子空间与 CKA 尺度诊断</title>
<style>body{{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui;color:#18202b}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}aside{{background:#f2f4f7;padding:16px}}</style>
<h1>完整隐藏几何与源读出所用几何分别检查</h1>
<p>这是在观察原登记隐藏排序失败后追加的事后分析，共 {summary['models']} 个源模型。它不能替代原登记检验；全部表示方式与模型保留，未重训目标头或修改任何已保存的迁移预测。</p>
<img src="readout_geometry.png"><p>将源头每个任务的类别权重减去类别均值，得到真正影响分类的权重行空间。用正交投影将隐藏表征分为该行空间和其零空间；零空间不影响源类别 logit 差。另报告源 logits、softmax 概率和硬答案编码。硬答案编码的 CKA 等于理想核是源准确率 100% 的直接后果，不是新的学习机制发现。</p>
<table><tr><th>源实验</th><th>表示方式</th><th>满足登记三层排序</th></tr>{table}</table>
<h2>行为相同的参数变换</h2><p>给隐藏各维乘正数 D，同时给源头对应权重列乘 D⁻¹。可直接通过 LayerNorm 的仿射参数实现；分类输出保持不变。八个固定尺度变换下，每个模型的原始 CKA 最大减最小，平均为 {summary['mean_within_model_scaling_cka_range']:.4f}。源输出误差和实际模型参数检查已保存。</p>
<p>本次目标冻结读出使用支持集逐维标准化。在没有触及标准差下限时，正尺度变换被标准化精确抵消。这里用固定 25 个输入检查了该等价关系；没有重新拟合目标头。原始 CKA 和未标准化的字符方差比例会受尺度影响，不能仅凭它们的数值推出读出能力变化。</p>
<aside>源读出子空间投影使用已训练源头，属于有额外信息的诊断。不同表示之间指标改变并非新的因果证据，也不保证目标任务有用。所有输入已用于源训练。原假设不成立的结果保留在独立报告中。</aside>
<p><a href="protocol.json">事后分析范围</a> · <a href="representations.csv">逐模型结果</a> · <a href="paired_orderings.csv">完整排序</a> · <a href="equivalent_scalings.csv">行为等价变换</a> · <a href="summary.json">核验与汇总</a> · <a href="readout_geometry.pdf">PDF 图</a></p></html>''')
    print(json.dumps({k:v for k,v in summary.items() if k!='protocol'},indent=2))


def plot(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,cohort in zip(axes,('field_symmetry','field_matched_support')):
        for name in ('full_hidden','standardized_hidden','source_contrast_rowspace','source_contrast_nullspace','source_probabilities'):
            cells=sorted([r for r in rows if (r['cohort'],r['representation'])==(cohort,name)],key=lambda r:r['group'])
            ax.plot([r['group'] for r in cells],[r['cka'] for r in cells],'o-',label=name)
        ax.set(title=cohort,ylabel='CKA under latent-coordinate swap',ylim=(0,1.04));ax.legend(fontsize=7)
    for suffix in ('png','pdf'):fig.savefig(root/f'readout_geometry.{suffix}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
