"""Resume against the saved exclusion catalog, without adding later datasets.

The initial registrar discovers archives once. Subsequent independent studies
may add archives; they must not alter this already generated evaluation cohort.
"""
import argparse
import json
from pathlib import Path

from .permworld_combinations import sha
from .specialist_cka_controls import CONFIG, create_data, features, evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['data','features','evaluate'])
    stage=parser.parse_args().stage
    plan=json.loads(CONFIG.read_text());root=Path(plan['output'])
    sig=json.loads((root/'protocol.json').read_text())['signature']
    assert plan==sig['plan'] and sha(CONFIG)==sig['config_sha256']
    assert sha('experiments/specialist_cka_controls.py')==sig['code_sha256']
    for source in sig['sources']:
        assert sha(source['checkpoint'])==source['checkpoint_sha256']
        assert sha(source['marker'])==source['marker_sha256']
    for entry in sig['excluded_archives']:assert sha(entry['path'])==entry['sha256']
    if (root/'dataset.npz').exists():
        assert sha(root/'dataset.npz')==json.loads((root/'dataset_audit.json').read_text())['sha256']
    for meta in (root/'features').glob('*.json'):
        assert sha(meta.with_suffix('.npz'))==json.loads(meta.read_text())['archive_sha256']
    {'data':create_data,'features':features,'evaluate':evaluate}[stage](plan,root,sig)


if __name__=='__main__':main()
