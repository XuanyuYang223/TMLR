"""Score all frozen new-target forecasts; retain the earlier mixed-target study."""
from datetime import datetime
from itertools import product
import json
from pathlib import Path

import numpy as np

from .controlled_target_followup import future_design,select_targets
from .field_forecast import VARIANTS,predict
from .field_forecast_report import match_outcomes,metrics
from .field_symmetry_transfer_report import summarize
from .permworld_combinations import sha
from .six_hour_report import write_rows


def run():
    root=Path('results/controlled_target_followup');saved=json.loads((root/'forecasts.json').read_text())
    config=saved['protocol']['transfer_config'];source=saved['protocol']['source_config']
    behavior=Path(config['output']);metadata=json.loads((behavior/'metadata.json').read_text())
    assert saved['protocol']['code_sha256']==sha('experiments/controlled_target_followup.py')
    assert saved['protocol']['original_predictor_sha256']==sha('results/field_forecast/forecasts.json')
    assert datetime.fromisoformat(saved['frozen_utc']).timestamp()< (behavior/'metadata.json').stat().st_mtime
    old=json.loads(Path('configs/field_matched_support_transfer.json').read_text())
    candidates,targets=select_targets(source,old)
    assert candidates==saved['protocol']['all_candidates'] and targets==saved['protocol']['selected']
    summary=summarize(behavior)
    rows=[]
    for path in behavior.glob('*_w*_m*.json'):
        record=json.loads(path.read_text());assert record['fingerprint']==metadata['fingerprint']
        if record['status']=='complete':rows.extend(record['rows'])
    evaluated=match_outcomes(saved['forecasts'],rows)
    old_config=json.loads(Path('configs/field_symmetry_transfer.json').read_text());old_source=json.loads(Path(old_config['source_config']).read_text())
    scores=[]
    for mode,budget,variant in product(('linear','mlp','categorical_subset'),config['budgets'],VARIANTS):
        subset=[r for r in evaluated if (r['mode'],r['budget'],r['variant'])==(mode,budget,variant)]
        if not subset:continue
        np.testing.assert_allclose(predict(saved['fitted_models'][f'{mode}_{budget}_{variant}'],future_design(subset,variant,old_config,old_source)),[r['predicted_outcome'] for r in subset],atol=1e-12,rtol=0)
        scores.append({'mode':mode,'budget':budget,'variant':variant,**metrics(subset)})
    gain_summary=[]
    baseline={(r['world_seed'],r['model_seed'],r['target_id'],r['budget'],r['mode']):r['accuracy'] for r in rows if r['group']=='random'}
    for mode,budget in product(('linear','mlp'),config['budgets']):
        subset=[r for r in rows if r['group']!='random' and (r['mode'],r['budget'])==(mode,budget)]
        gains=[r['accuracy']-baseline[r['world_seed'],r['model_seed'],r['target_id'],r['budget'],r['mode']] for r in subset]
        if gains:gain_summary.append({'mode':mode,'budget':budget,'cells':len(gains),'source_accuracy':float(np.mean([r['accuracy'] for r in subset])),'mean_gain':float(np.mean(gains))})
    result={'behavior_conditions':summary['completed_conditions'],'endpoints':summary['endpoints'],'forecasts':len(saved['forecasts']),
            'evaluated_forecasts':len(evaluated),'scores':scores,'neural_gains':gain_summary,
            'timing_and_fitted_forecasts_verified':True,'new_targets_exclude_old_targets_and_source_scalars':True,
            'physical_support_minimum_three_in_all_bases':True,'all_targets_and_variants_retained':True}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');write_rows(root/'evaluated_forecasts.csv',evaluated);write_rows(root/'scores.csv',scores);write_rows(root/'neural_gains.csv',gain_summary)
    table=''.join(f"<tr><td>{r['mode']}</td><td>{r['budget']}</td><td>{100*r['source_accuracy']:.2f}%</td><td>{100*r['mean_gain']:+.2f} pp</td><td>{r['cells']}</td></tr>" for r in gain_summary)
    score_table=''.join(f"<tr><td>{r['mode']}</td><td>{r['budget']}</td><td>{r['variant']}</td><td>{r['r2']:.3f}</td><td>{r['mae']:.4f}</td></tr>" for r in scores)
    (root/'report.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>同编码器的新目标边界检查</title><style>body{{max-width:1200px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}aside{{background:#f2f4f7;padding:16px}}</style>
<h1>冻结编码器，追加物理支撑至少三维的新目标</h1><p>这项后续试验在观察先前混合目标池的负收益之后设计。原目标池的完整负结果保留在<a href="../field_matched_support_transfer/report.html">原报告</a>。没有重训或挑选源模型，使用同一批 27 个编码器及 9 个初始编码器。新增八个目标全部是新的标量方向，在三个输入基中均至少依赖三个物理坐标；四个目标的最小源组合阶数为二，另四个为三，均在三个源组合间变化。</p>
<p>目标只从 40 个满足数学条件的候选中按固定种子选择，不参考新目标的神经结果。全部 {summary['completed_conditions']}/36 个行为条件目前给出 {summary['endpoints']} 个端点；固定 200 更新和学习率 0.01，所有源与随机读出均为 FP32 CPU。支持与测试规则同原试验。</p>
<table><tr><th>读出</th><th>标签</th><th>源编码器准确率</th><th>配对随机收益</th><th>端点</th></tr>{table}</table>
<p>{len(saved['forecasts'])} 条预测于 {saved['frozen_utc']} 保存，早于本次读出。复用原预测器的全部拟合权重，不在此前独立输入基或本次结果上再拟合。未知目标身份使用八个原目标固定效应的平均值，未知输入基沿用三个原基效应的平均值。数学定义是分析预测器的额外信息，不输入目标神经头。</p>
<table><tr><th>读出</th><th>标签</th><th>预测规则</th><th>新目标 R²</th><th>MAE</th></tr>{score_table}</table>
<aside>这是一项针对已观察失败条件的探索性边界检查，不能用它替换原目标池或声称无条件复现。物理目标复杂度仍在三、四维之间变化，源组合公式、编码器和世界均已知，未见的是目标身份。源训练见过全部 625 输入。这三个输入基与三个初始化为相关交叉重复；未报告独立数据集显著性。</aside>
<p><a href="protocol.json">新目标选择及后续协议</a> · <a href="forecasts.json">事先保存的预测与原权重</a> · <a href="scores.csv">完整评分</a> · <a href="evaluated_forecasts.csv">逐预测核对</a> · <a href="../controlled_target_transfer/report.html">全部行为与组合阶数对照</a> · <a href="summary.json">核验与汇总</a></p></html>''')
    print(json.dumps({k:result[k] for k in ('behavior_conditions','endpoints','neural_gains')},indent=2))


if __name__=='__main__':run()
