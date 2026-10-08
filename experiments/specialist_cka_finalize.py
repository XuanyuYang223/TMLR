"""Authenticate completed analyses, prior artifacts and local report links."""
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re

from .longrun_engine import atomic_json
from .permworld_combinations import sha

ROOT=Path('results/specialist_cka_controls')
ADJUST=Path('results/specialist_input_adjustment')
ABLATION=Path('results/frozen_operator_constraint_ablation')


class Links(HTMLParser):
    def __init__(self):super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):
        self.paths.extend(value for key,value in attrs if key in ['href','src'] and value and not value.startswith(('http:','https:','#')))


def run():
    previous=json.loads(Path('results/output_information_followup/completion.json').read_text())
    for filename,digest in previous['artifact_sha256'].items():assert sha(filename)==digest,filename
    protocols={p:json.loads((p/'protocol.json').read_text())['signature'] for p in [ROOT,ADJUST,ABLATION]}
    assert sha('experiments/specialist_cka_controls.py')==protocols[ROOT]['code_sha256']
    assert sha('configs/specialist_cka_controls.json')==protocols[ROOT]['config_sha256']
    assert sha('experiments/specialist_input_adjustment.py')==protocols[ADJUST]['code_sha256']
    assert sha('configs/specialist_input_adjustment.json')==protocols[ADJUST]['config_sha256']
    assert sha('experiments/frozen_operator_constraint_ablation.py')==protocols[ABLATION]['code_sha256']
    assert sha('configs/frozen_operator_constraint_ablation.json')==protocols[ABLATION]['config_sha256']
    for root in [ROOT,ADJUST,ABLATION]:assert json.loads((root/'verification.json').read_text())['status']=='passed'
    for source in protocols[ROOT]['sources']+protocols[ABLATION]['sources']:assert sha(source['checkpoint'])==source['checkpoint_sha256']
    for metadata in (ROOT/'features').glob('*.json'):
        assert sha(metadata.with_suffix('.npz'))==json.loads(metadata.read_text())['archive_sha256']
    assert len(list((ROOT/'features').glob('*.json')))==120
    assert len(list((ROOT/'evaluations').glob('*.json')))==24
    assert json.loads((ROOT/'state.json').read_text())['status']=='analyses_complete'
    assert json.loads((ADJUST/'state.json').read_text())['status']=='complete'
    assert json.loads((ABLATION/'state.json').read_text())['status']=='complete'
    log=Path('results/specialist_cka_controls_tests.log').read_text()
    count=int(re.search(r'(\d+) passed',log)[1]);assert count==123 and 'failed' not in log
    links=Links();links.feed((ROOT/'report.html').read_text())
    for rel in links.paths:
        if rel=='completion.json':continue
        assert (ROOT/rel).is_file(),rel
    summary=json.loads((ROOT/'summary.json').read_text())
    assert summary['controls']['prefix_final_norm']['answer_strata']['trained']['positive']==24
    assert summary['supplemental_input_adjustment']['controls']['prefix_final_norm']['answer_strata']['increment_vs_matched_init']['positive']==6
    files=[]
    for folder in [ROOT,ADJUST,ABLATION]:files.extend(p for p in folder.rglob('*') if p.is_file() and p.name!='completion.json')
    files.extend(Path(p) for p in [
        'configs/specialist_cka_controls.json','configs/specialist_input_adjustment.json','configs/frozen_operator_constraint_ablation.json',
        'experiments/specialist_cka_controls.py','experiments/specialist_cka_resume.py','experiments/specialist_cka_verify.py',
        'experiments/specialist_cka_report.py','experiments/specialist_cka_finalize.py','experiments/specialist_input_adjustment.py',
        'experiments/specialist_input_adjustment_verify.py','experiments/frozen_operator_constraint_ablation.py',
        'experiments/frozen_operator_constraint_verify.py','tests/test_specialist_cka_controls.py',
        'tests/test_specialist_input_adjustment.py','tests/test_frozen_operator_constraint_ablation.py','results/specialist_cka_controls_tests.log'])
    artifacts={str(p):sha(p) for p in sorted(set(files))}
    record={'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),'source_models_retrained':0,
        'original_specialist_source_checkpoints_unchanged':48,'additional_frozen_operator_source_checkpoints_unchanged':3,
        'test_anchors':2688,'fresh_collision_pairs':1344,'project_tests_passed':count,
        'verified_core_gram_scores':1944,'verified_input_adjusted_spectral_scores':2592,'verified_constraint_endpoints':30,
        'previous_completed_artifact_hashes_preserved':len(previous['artifact_sha256']),
        'core_pooled_answer_matched_contrasts_positive':24,'per_length_answer_matched_increments_over_initialization_positive':6,
        'per_length_positive_increment_scope':'Both predeclared inverse relations, all three source seeds; none of the six complement relations exceed initialization.',
        'supplemental_analysis_registered_after_core_results':True,
        'output_independent_algebraic_mechanism_confirmed':False,'stable_hidden_composition_confirmed':False,
        'report_local_links_verified':len(links.paths),'artifact_sha256':artifacts}
    atomic_json(ROOT/'completion.json',record)
    for rel in links.paths:assert (ROOT/rel).is_file(),rel
    print(json.dumps({k:v for k,v in record.items() if k!='artifact_sha256'},indent=2))


if __name__=='__main__':run()
