"""Seal coverage results while preserving the previous completed assay."""
import json
from pathlib import Path
import re

from .inverse_alignment_coverage import initialize
from .inverse_functional_alignment import now
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .specialist_cka_finalize import Links


def run():
    plan,root,sig=initialize();old=Path(plan['previous_study']);assert sha(old/'completion.json')==sig['prior_completion_sha256']
    previous=json.loads((old/'completion.json').read_text())
    for path,digest in previous['artifact_sha256'].items():assert sha(path)==digest,path
    for path,digest in sig['references_sha256'].items():assert sha(path)==digest,path
    for path,digest in sig['upstream_sha256'].items():assert sha(Path(plan['upstream_python'])/'neurips_permutations'/path)==digest
    analysis=json.loads((root/'analysis_protocol.json').read_text());assert analysis['code_sha256']==sha('experiments/inverse_coverage_analysis.py')
    registered=json.loads((root/'verification_protocol.json').read_text());assert registered['code_sha256']==sha('experiments/inverse_coverage_verify.py')
    opened=json.loads((root/'test_opened.json').read_text());assert analysis['registered_utc']<opened['opened_utc'] and registered['registered_utc']<opened['opened_utc']
    verification=json.loads((root/'verification.json').read_text());assert verification['status']=='passed'
    assert verification['accuracy_endpoints_checked']==72 and verification['validation_candidates_replayed']==108
    assert verification['independent_gram_scores']==2340 and verification['bootstrap_bounds_independently_checked']==96
    assert len(list((root/'training').glob('r*_*.json')))==36
    assert len(list((root/'checkpoints').glob('*_u*.pt')))==108
    for file in (root/'evaluations').glob('r*_*.json'):
        assert sha(file.with_suffix('.npz'))==json.loads(file.read_text())['archive_sha256']
    for file in (root/'training').glob('r*_*.json'):
        record=json.loads(file.read_text());assert record['status']=='complete'
        for step,digest in record['candidate_sha256'].items():assert sha(root/'checkpoints'/f'{file.stem}_u{step}.pt')==digest
        assert sha(root/'checkpoints'/f'{file.stem}_latest.pt')==record['latest_sha256']
    supplement=json.loads((root/'size_control/summary.json').read_text());sv=json.loads((root/'size_control/verification.json').read_text());assert sv['status']=='passed'
    assert sv['validation_candidates_replayed']==18 and sv['accuracy_endpoints_checked']==12 and sv['raw_cka_gram_checks']==60
    sp=json.loads((root/'size_control/protocol.json').read_text());assert sp['code_sha256']==sha('experiments/inverse_coverage_size_control.py')
    assert sp['base_test_computation_started'] and not sp['base_numerical_test_outcomes_inspected_by_agent']
    assert sp['registered_utc']<json.loads((root/'size_control/test_opened.json').read_text())['opened_utc']
    report=(root/'report.html').read_text();start,end='<!-- equal-count supplement start -->','<!-- equal-count supplement end -->'
    if start in report:report=report[:report.index(start)]+report[report.index(end)+len(end):]
    sm=supplement['means']['final'];sd=supplement['contrasts_vs_support_alignment']['final']
    section=start+'<h2>几何行数匹配：额外六次控制</h2><p>原覆盖条件使用至多64行几何批次，标签对齐使用至多32行。额外六次拟合逐步使用与标签对齐完全相同的几何行数，但换成无标签池输入；初始化、64行前向、标签、批次顺序和更新预算不变。该控制在基础测试开始计算后、代理查看数值结果前，基于代码审查登记；不是原六条件预注册的一部分。</p><p>固定1200步的无标签行数匹配控制，新测试CKA为 '+f"{sm['cka']:.4f}"+'，准确率为 '+f"{100*sm['accuracy']:.2f}%"+'。相对标签对齐，CKA差为 '+f"{sd['cka']['mean']:+.4f}"+'（'+str(sd['cka']['positive_replicates'])+'/6为正），准确率差为 '+f"{sd['accuracy_pp']['mean']:+.2f}"+' 个百分点。主六条件36模型与该补充6模型分别报告，总计42模型。</p><p><a href="size_control/report.html">完整行数匹配控制</a> · <a href="size_control/protocol.json">登记及时间顺序</a> · <a href="size_control/summary.json">全部配对统计</a> · <a href="size_control/verification.json">18验证候选与12测试端点核验</a></p>'+end
    assert '<h2>标签、曝光与监督的区别</h2>' in report
    report=report.replace('<h2>标签、曝光与监督的区别</h2>',section+'<h2>标签、曝光与监督的区别</h2>');(root/'report.html').write_text(report)
    priors=json.loads((root/'length_priors/summary.json').read_text());pp=json.loads((root/'length_priors/protocol.json').read_text())
    assert pp['exploratory_post_test'] and pp['code_sha256']==sha('experiments/inverse_coverage_priors.py')
    assert priors['independent_count_and_accuracy_checks']==72 and not priors['new_oracle_unlabeled_answers_used']
    for entry in priors['records']:
        path=root/'length_priors'/f"{entry['replicate']}_{entry['condition']}.npz";assert sha(path)==entry['prediction_archive_sha256']
    ps,pe='<!-- length-prior diagnostic start -->','<!-- length-prior diagnostic end -->'
    if ps in report:report=report[:report.index(ps)]+report[report.index(pe)+len(pe):]
    prior_section=ps+'<h2>仅长度的答案先验：事后解释控制</h2><p>主结果揭晓后追加，不属于预注册主要比较。只用192训练标签估计各长度最常见答案，平均准确率为 '+f"{100*priors['means']['labeled_length_mode']:.2f}%"+'；只用4096无标签输入的教师预测估计各长度众数，为 '+f"{100*priors['means']['teacher_length_mode']:.2f}%"+'。预测不依赖排列内容，也不使用验证或测试答案拟合。它们限制了将蒸馏相对普通神经训练的增益解释为输入相关数学规律学习的结论，不能证明模型完全没有输入信息。</p><p><a href="length_priors/report.html">完整长度先验诊断</a> · <a href="length_priors/protocol.json">事后范围</a> · <a href="length_priors/summary.json">独立计数核验与全部重复</a></p>'+pe
    report=report.replace('<h2>辅助几何与敏感性</h2>',prior_section+'<h2>辅助几何与敏感性</h2>');(root/'report.html').write_text(report)
    log=Path('results/inverse_alignment_coverage_tests.log').read_text();tests=int(re.search(r'(\d+) passed',log)[1]);assert tests==133 and 'failed' not in log
    links=Links();links.feed((root/'report.html').read_text())
    for rel in links.paths:
        if rel!='completion.json':assert (root/rel).is_file(),rel
    summary=json.loads((root/'summary.json').read_text());files=[p for p in root.rglob('*')if p.is_file()and p.name!='completion.json']
    files+=[Path(p)for p in ['configs/inverse_alignment_coverage.json','experiments/inverse_alignment_coverage.py','experiments/inverse_coverage_analysis.py','experiments/inverse_coverage_verify.py','experiments/inverse_coverage_finalize.py','tests/test_inverse_alignment_coverage.py',
        'experiments/inverse_coverage_size_control.py','experiments/inverse_coverage_size_analysis.py','tests/test_inverse_coverage_size_control.py',
        'experiments/inverse_coverage_priors.py','results/inverse_alignment_coverage_priors.log',
        'results/inverse_alignment_coverage_size_train.log','results/inverse_alignment_coverage_size_evaluate.log','results/inverse_alignment_coverage_size_analysis.log',
        'results/inverse_alignment_coverage_data.log','results/inverse_alignment_coverage_teachers.log','results/inverse_alignment_coverage_train.log','results/inverse_alignment_coverage_evaluate.log','results/inverse_alignment_coverage_analysis.log','results/inverse_alignment_coverage_verify.log','results/inverse_alignment_coverage_report.log','results/inverse_alignment_coverage_tests.log']]
    completion={'status':'complete','completed_utc':now(),'target_models_trained':42,'original_registered_target_models':36,'supplemental_equal_count_target_models':6,'source_models_retrained':0,
        'reused_paired_support_initialization_repeats':6,'oracle_target_labels_including_validation_per_repeat':256,'new_unlabeled_inputs_per_repeat':4096,'new_test_examples':2560,
        'all_36_training_runs_finished_before_test_opening':True,'fixed_update_primary':1200,'new_unlabeled_oracle_answers_used':False,
        'previous_artifact_hashes_preserved':len(previous['artifact_sha256']),'source_checkpoint_hashes_verified':3,'project_tests_passed':tests,
        'accuracy_endpoints_independently_checked':72,'validation_checkpoints_independently_replayed':108,'independent_gram_scores':2340,
        'supplemental_accuracy_endpoints_checked':12,'supplemental_validation_candidates_replayed':18,'supplemental_raw_cka_gram_checks':60,
        'supplement_registered_after_base_test_computation_before_agent_numeric_inspection':True,
        'length_prior_post_test_diagnostic':True,'length_prior_accuracy':priors['means'],'length_prior_independent_checks':72,
        'bootstrap_bounds_independently_checked':96,'original_inputs_independently_rescanned':16000000,
        'main_means':summary['means']['final'],'main_contrasts':summary['contrasts']['final'],'supplement_equal_count_results':supplement['means'],'report_links_checked':len(links.paths),
        'spontaneous_algebra_discovery_confirmed':False,'output_independent_algebraic_mechanism_confirmed':False,
        'artifact_sha256':{str(p):sha(p)for p in sorted(set(files))}}
    atomic_json(root/'completion.json',completion)
    for rel in links.paths:assert (root/rel).is_file(),rel
    print(json.dumps({k:v for k,v in completion.items()if k!='artifact_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':run()
