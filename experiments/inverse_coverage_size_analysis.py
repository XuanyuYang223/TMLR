"""Report and independently replay the pre-inspection equal-count supplement."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .inverse_alignment_coverage import initialize
from .inverse_functional_alignment import configure, now
from .inverse_functional_auxiliary import scores
from .inverse_functional_verify import direct, gram
from .inverse_functional_report import table
from .inverse_coverage_analysis import contrasts
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .specialist_cka_controls import api


def run():
    plan,root,sig=initialize();folder=root/'size_control';registered=json.loads((folder/'protocol.json').read_text());assert registered['code_sha256']==sha('experiments/inverse_coverage_size_control.py')
    device=configure();_,tokens,_,TrainConfig,factory=api(plan);old=Path(plan['previous_study']);data=dict(np.load(root/'dataset/test/dataset.npz'));base=json.loads((root/'geometry.json').read_text())['records'];rows=[];errors=[];validations=0;endpoints=0;checks=0
    sample=np.concatenate([np.flatnonzero(data['lengths']==n)[::128]for n in plan['lengths']])
    for rep in sig['replicates']:
        record=json.loads((folder/f"{rep['id']}_training.json").read_text());assert record['status']=='complete'
        s=dict(np.load(old/'dataset'/rep['id']/'dataset.npz'));mask=np.load(old/'dataset'/rep['id']/'mismatch.npz')['eligible'];ls=np.load(root/'training'/f"{rep['id']}_schedule.npz")['labeled']
        np.testing.assert_array_equal(np.asarray(record['geometry_rows_per_step']),mask[ls].sum(-1));assert record['schedule_sha256']==sha(root/'training'/f"{rep['id']}_schedule.npz")
        assert record['initialization_sha256']==sha(old/'initializations'/f"{rep['id']}.pt")
        selected=min(record['curve'],key=lambda g:(-g['accuracy'],g['cross_entropy'],g['step']));assert selected==record['selected']
        source=next(s for s in sig['sources']if s['seed']==rep['source_seed']);cp=torch.load(source['checkpoint'],weights_only=True,map_location='cpu');cfg=TrainConfig.from_value(cp['config']);del cp
        eval_file=folder/f"{rep['id']}_evaluation.json";evaluation=json.loads(eval_file.read_text());arrays=dict(np.load(eval_file.with_suffix('.npz')));assert sha(eval_file.with_suffix('.npz'))==evaluation['archive_sha256']
        for grade in record['curve']:
            step=grade['step'];path=folder/f"{rep['id']}_u{step}.pt";assert sha(path)==record['candidate_sha256'][str(step)];cp=torch.load(path,weights_only=True,map_location='cpu');model=factory(cfg);model.load_state_dict(cp['model']);del cp;model.to(device)
            val=np.flatnonzero(s['split']==1);v=direct(model,s,plan['target_task'],tokens,0,val)['logits'];assert float((v.argmax(-1)==s['labels'][val]).mean())==grade['accuracy']
            np.testing.assert_allclose(float(F.cross_entropy(torch.tensor(v),torch.tensor(s['labels'][val]))),grade['cross_entropy'],atol=5e-5,rtol=5e-5);validations+=1
            for endpoint,epstep in [('final',1200),('selected',selected['step'])]:
                if step!=epstep:continue
                out=direct(model,data,plan['target_task'],tokens,0,sample)
                for branch in ['logits','hidden']:
                    expected=arrays[endpoint+'_'+branch][sample];np.testing.assert_allclose(out[branch],expected,atol=3e-4,rtol=3e-4);errors.append(float(np.max(np.abs(out[branch]-expected))))
                hit=arrays[endpoint+'_logits'].argmax(-1)==data['labels'];assert float(hit.mean())==evaluation['results'][endpoint]['accuracy'];endpoints+=1
            del model
        teacher=np.load(root/'teacher'/f"test_s{rep['source_seed']}.npz")['hidden']
        for endpoint in ['final','selected']:
            geometry=scores(arrays[endpoint+'_hidden'],teacher,data['lengths'],data['labels'],plan['lengths'])
            for row in geometry:
                ix=np.flatnonzero(data['lengths']==row['length'])[:128];actual=gram(arrays[endpoint+'_hidden'][ix],teacher[ix]);np.testing.assert_allclose(actual,row['correct_cka'],atol=2e-12);checks+=1
            rows.append({'replicate':rep['id'],'endpoint':endpoint,'geometry':geometry,'cka':float(np.mean([r['correct_cka']for r in geometry])),
                'accuracy':evaluation['results'][endpoint]['accuracy'],'geometry_rows_match_support_every_step':True})
    comparisons={};means={}
    for endpoint in ['final','selected']:
        a=[next(r for r in rows if r['replicate']==rep['id']and r['endpoint']==endpoint)for rep in sig['replicates']]
        b=[next(r for r in base if r['replicate']==rep['id']and r['endpoint']==endpoint and r['condition']=='support_alignment')for rep in sig['replicates']]
        comparisons[endpoint]={'cka':contrasts([x['cka']-y['mean']['correct_cka']for x,y in zip(a,b)],sig['replicates']),
            'accuracy_pp':contrasts([100*(x['accuracy']-y['accuracy'])for x,y in zip(a,b)],sig['replicates'])}
        means[endpoint]={'cka':float(np.mean([r['cka']for r in a])),'accuracy':float(np.mean([r['accuracy']for r in a]))}
    result={'completed_utc':now(),'registration_after_base_test_computation_started':True,'registered_before_agent_inspection_of_base_numerical_outcomes':True,
        'means':means,'contrasts_vs_support_alignment':comparisons,'records':rows,'source_models_retrained':0}
    atomic_json(folder/'summary.json',result);atomic_json(folder/'verification.json',{'status':'passed','validation_candidates_replayed':validations,'accuracy_endpoints_checked':endpoints,'raw_cka_gram_checks':checks,'maximum_replay_error':max(errors),'all_7200_geometry_row_counts_match_support':True})
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>几何行数匹配控制</title><style>body{font-family:system-ui;max-width:1100px;margin:35px auto;padding:15px;line-height:1.7}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}</style><body><h1>几何行数匹配：补充控制</h1><p>该控制在基础测试已开始计算、但代理尚未查看数值结果时，从代码审查发现的问题提出并登记；它不是原六条件预注册的一部分。新增六次拟合，固定同样的初始化、64行前向、192训练/64验证标签、批次顺序、1200步和损失权重。每步无标签几何损失使用的行数严格等于“只对齐原标签集”当步合格行数，分别覆盖4096与192个可用输入。</p>'
    tab=[]
    for endpoint in ['final','selected']:
        c=comparisons[endpoint];tab.append([endpoint,f"{means[endpoint]['cka']:.4f}",f"{100*means[endpoint]['accuracy']:.2f}%",f"{c['cka']['mean']:+.4f}",f"{c['cka']['positive_replicates']}/6",f"{c['accuracy_pp']['mean']:+.2f}",f"{c['accuracy_pp']['positive_replicates']}/6"])
    html+=table(['端点','新输入CKA','准确率','相对标签对齐CKA差','正重复','准确率差/百分点','正重复'],tab)
    html+='<p>配对差、三教师簇描述性区间、每次结果保留在summary.json。独立核验全部18个验证候选和12个测试端点，并检查全部7200步的几何行数与标签对齐一致。</p><p><a href="../report.html">主报告</a> · <a href="protocol.json">登记与范围</a> · <a href="summary.json">完整统计</a> · <a href="verification.json">核验</a></p></body></html>'
    (folder/'report.html').write_text(html);print(json.dumps({'means':means,'contrasts':comparisons},ensure_ascii=False,indent=2))


if __name__=='__main__':run()
